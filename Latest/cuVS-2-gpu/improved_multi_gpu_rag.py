#!/usr/bin/env python3
"""
Improved Multi-GPU RAG Implementation with Parallel Execution
Addresses key issues: parallel GPU ops, top-2K retrieval, better error handling
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from enum import Enum
import gc
import psutil
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Supported index types"""
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    CAGRA = "cagra"
    FAISS_FLAT = "faiss_flat"
    FAISS_IVF = "faiss_ivf"

@dataclass
class SearchConfig:
    """Configuration for search operations"""
    top_k: int = 2000  # Changed from 5 to 2000 for better recall
    search_batch_size: int = 100
    num_queries: int = 100
    enable_recall_eval: bool = True
    recall_k_values: List[int] = None

    def __post_init__(self):
        if self.recall_k_values is None:
            self.recall_k_values = [1, 5, 10, 50, 100, 500, 1000, 2000]

@dataclass
class GPUConfig:
    """GPU configuration and resource management"""
    device_id: int
    memory_limit_gb: float = 40.0  # A100 has 40GB
    reserved_memory_gb: float = 2.0  # Reserve some memory for safety

    @property
    def device_str(self):
        return f"cuda:{self.device_id}"

    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            free_memory = torch.cuda.mem_get_info(self.device_id)[0] / 1024**3
            return free_memory
        return 0.0

    def can_allocate(self, size_gb: float) -> bool:
        """Check if we can allocate the requested memory"""
        available = self.get_available_memory()
        return available > (size_gb + self.reserved_memory_gb)

class CUDAMemoryManager:
    """Manages CUDA memory and handles OOM errors"""

    @staticmethod
    @contextmanager
    def managed_allocation(gpu_config: GPUConfig, operation: str):
        """Context manager for safe GPU memory allocation"""
        try:
            initial_memory = gpu_config.get_available_memory()
            logger.info(f"[GPU {gpu_config.device_id}] Starting {operation} with {initial_memory:.2f} GB available")
            yield
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[GPU {gpu_config.device_id}] OOM during {operation}: {e}")
            # Clear cache and retry
            torch.cuda.empty_cache()
            gc.collect()
            raise
        except Exception as e:
            logger.error(f"[GPU {gpu_config.device_id}] Error during {operation}: {e}")
            raise
        finally:
            final_memory = gpu_config.get_available_memory()
            memory_used = initial_memory - final_memory
            logger.info(f"[GPU {gpu_config.device_id}] Completed {operation}, used {memory_used:.2f} GB")

class ParallelIndexBuilder:
    """Handles parallel index building across multiple GPUs"""

    def __init__(self, num_gpus: int = None):
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.gpu_configs = [GPUConfig(i) for i in range(self.num_gpus)]
        self.executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        logger.info(f"Initialized ParallelIndexBuilder with {self.num_gpus} GPUs")

    def build_index_on_gpu(self, gpu_config: GPUConfig, embeddings: torch.Tensor,
                          index_type: IndexType, params: Dict) -> Tuple[Any, float]:
        """Build index on specific GPU - runs in parallel thread"""
        import pylibraft
        from cuvs.neighbors import ivf_flat, ivf_pq, cagra

        pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())

        start_time = time.time()

        with CUDAMemoryManager.managed_allocation(gpu_config, f"building {index_type.value} index"):
            torch.cuda.set_device(gpu_config.device_id)

            # Move embeddings to GPU if not already there
            if not embeddings.is_cuda or embeddings.device.index != gpu_config.device_id:
                embeddings = embeddings.to(gpu_config.device_str)

            # Build index based on type
            if index_type == IndexType.IVF_FLAT:
                index_params = ivf_flat.IndexParams(
                    n_lists=params.get('n_lists', min(256, embeddings.shape[0] // 1000 + 1))
                )
                index = ivf_flat.build(index_params, embeddings)
            elif index_type == IndexType.IVF_PQ:
                index_params = ivf_pq.IndexParams(
                    n_lists=params.get('n_lists', min(512, embeddings.shape[0] // 500 + 1)),
                    pq_dim=params.get('pq_dim', 96),
                    pq_bits=params.get('pq_bits', 8)
                )
                index = ivf_pq.build(index_params, embeddings)
            elif index_type == IndexType.CAGRA:
                index_params = cagra.IndexParams(
                    intermediate_graph_degree=params.get('intermediate_graph_degree', 128),
                    graph_degree=params.get('graph_degree', 64)
                )
                index = cagra.build(index_params, embeddings)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")

            build_time = time.time() - start_time
            logger.info(f"[GPU {gpu_config.device_id}] Built {index_type.value} index in {build_time:.2f}s")

            return index, build_time

    def build_indices_parallel(self, embedding_parts: List[torch.Tensor],
                              index_type: IndexType, params: Dict = None) -> Dict:
        """Build indices in parallel across all GPUs"""
        if params is None:
            params = {}

        futures = []
        for i, embeddings in enumerate(embedding_parts[:self.num_gpus]):
            gpu_config = self.gpu_configs[i]
            future = self.executor.submit(
                self.build_index_on_gpu, gpu_config, embeddings, index_type, params
            )
            futures.append((i, future))

        # Collect results
        gpu_indexes = {}
        build_times = {}
        failed_gpus = []

        for gpu_id, future in futures:
            try:
                index, build_time = future.result(timeout=300)  # 5 minute timeout
                gpu_indexes[gpu_id] = index
                build_times[gpu_id] = build_time
            except Exception as e:
                logger.error(f"Failed to build index on GPU {gpu_id}: {e}")
                failed_gpus.append(gpu_id)

        total_build_time = sum(build_times.values())
        avg_build_time = np.mean(list(build_times.values())) if build_times else 0

        return {
            'indexes': gpu_indexes,
            'build_times': build_times,
            'total_time': total_build_time,
            'avg_time': avg_build_time,
            'failed_gpus': failed_gpus,
            'success': len(failed_gpus) == 0
        }

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class ParallelSearchEngine:
    """Handles parallel search across multiple GPU indexes"""

    def __init__(self, gpu_indexes: Dict[int, Any], index_type: IndexType,
                 search_config: SearchConfig):
        self.gpu_indexes = gpu_indexes
        self.index_type = index_type
        self.search_config = search_config
        self.num_gpus = len(gpu_indexes)
        self.executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        logger.info(f"Initialized ParallelSearchEngine with {self.num_gpus} indexes")

    def search_on_gpu(self, gpu_id: int, index: Any, query: torch.Tensor,
                     k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search on specific GPU - runs in parallel thread"""
        from cuvs.neighbors import ivf_flat, ivf_pq, cagra

        torch.cuda.set_device(gpu_id)

        # Move query to correct GPU
        if not query.is_cuda or query.device.index != gpu_id:
            query = query.to(f'cuda:{gpu_id}')

        # Ensure query has correct shape
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Perform search based on index type
        if self.index_type == IndexType.IVF_FLAT:
            search_params = ivf_flat.SearchParams()
            distances, indices = ivf_flat.search(search_params, index, query, k)
        elif self.index_type == IndexType.IVF_PQ:
            search_params = ivf_pq.SearchParams()
            distances, indices = ivf_pq.search(search_params, index, query, k)
        elif self.index_type == IndexType.CAGRA:
            search_params = cagra.SearchParams()
            distances, indices = cagra.search(search_params, index, query, k)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return distances, indices

    def parallel_search(self, query: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Perform parallel search across all GPU indexes"""
        k = self.search_config.top_k

        # Submit search tasks to all GPUs in parallel
        futures = []
        for gpu_id, index in self.gpu_indexes.items():
            future = self.executor.submit(
                self.search_on_gpu, gpu_id, index, query, k * 2  # Get 2x for merging
            )
            futures.append(future)

        # Collect results from all GPUs
        all_distances = []
        all_indices = []

        for future in as_completed(futures):
            try:
                distances, indices = future.result(timeout=10)
                if distances.size > 0:
                    all_distances.extend(distances.flatten())
                    all_indices.extend(indices.flatten())
            except Exception as e:
                logger.error(f"Search failed on GPU: {e}")
                continue

        # Merge and sort results to get global top-k
        if all_distances:
            all_distances = np.array(all_distances)
            all_indices = np.array(all_indices)

            # Sort by distance and get top-k
            sorted_idx = np.argsort(all_distances)[:k]
            final_distances = all_distances[sorted_idx]
            final_indices = all_indices[sorted_idx]

            return final_distances, final_indices
        else:
            return np.array([]), np.array([])

    def batch_search(self, queries: List[torch.Tensor]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform batch search for multiple queries"""
        results = []

        # Process queries in batches for efficiency
        batch_size = self.search_config.search_batch_size
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            # Search each query in parallel
            batch_futures = []
            for query in batch:
                future = self.executor.submit(self.parallel_search, query)
                batch_futures.append(future)

            # Collect batch results
            for future in as_completed(batch_futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch search failed: {e}")
                    results.append((np.array([]), np.array([])))

        return results

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class RecallEvaluator:
    """Evaluates recall metrics for retrieval results"""

    @staticmethod
    def calculate_recall_at_k(retrieved: np.ndarray, relevant: np.ndarray, k: int) -> float:
        """Calculate recall@k metric"""
        if len(relevant) == 0:
            return 1.0 if len(retrieved) == 0 else 0.0

        # Take top k retrieved items
        top_k = retrieved[:k] if len(retrieved) >= k else retrieved

        # Calculate intersection with relevant items
        relevant_retrieved = np.intersect1d(top_k, relevant)

        # Recall = relevant retrieved / total relevant
        recall = len(relevant_retrieved) / len(relevant)
        return recall

    @staticmethod
    def evaluate_recall_multiple_k(retrieved: np.ndarray, relevant: np.ndarray,
                                  k_values: List[int]) -> Dict[int, float]:
        """Calculate recall at multiple k values"""
        recalls = {}
        for k in k_values:
            if k <= len(retrieved):
                recalls[k] = RecallEvaluator.calculate_recall_at_k(retrieved, relevant, k)
            else:
                recalls[k] = RecallEvaluator.calculate_recall_at_k(retrieved, relevant, len(retrieved))
        return recalls

    @staticmethod
    def generate_synthetic_ground_truth(num_queries: int, index_size: int,
                                       relevant_per_query: int = 100) -> Dict[int, np.ndarray]:
        """Generate synthetic ground truth for testing"""
        np.random.seed(42)
        ground_truth = {}

        for i in range(num_queries):
            # Generate random relevant indices
            relevant_indices = np.random.choice(
                index_size,
                size=min(relevant_per_query, index_size),
                replace=False
            )
            ground_truth[i] = relevant_indices

        return ground_truth

def get_memory_stats() -> Dict:
    """Get current memory statistics"""
    stats = {
        'ram_gb': psutil.Process().memory_info().rss / 1024**3,
        'cpu_percent': psutil.cpu_percent()
    }

    if torch.cuda.is_available():
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = torch.cuda.mem_get_info(i)[0] / 1024**3
            total = torch.cuda.mem_get_info(i)[1] / 1024**3

            gpu_stats.append({
                'gpu_id': i,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free,
                'total_gb': total,
                'used_percent': (allocated / total) * 100
            })
        stats['gpu_stats'] = gpu_stats

    return stats

def print_memory_status(label: str = ""):
    """Print current memory status"""
    stats = get_memory_stats()
    logger.info(f"{label} - RAM: {stats['ram_gb']:.2f} GB, CPU: {stats['cpu_percent']:.1f}%")

    if 'gpu_stats' in stats:
        for gpu in stats['gpu_stats']:
            logger.info(
                f"  GPU {gpu['gpu_id']}: {gpu['allocated_gb']:.2f}/{gpu['total_gb']:.2f} GB "
                f"({gpu['used_percent']:.1f}% used)"
            )

# Main execution function for testing
def main():
    """Main function for testing the implementation"""
    logger.info("Starting improved multi-GPU RAG implementation test")

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("No GPU available. This implementation requires CUDA.")
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s)")

    # Print initial memory status
    print_memory_status("Initial")

    # Configuration
    search_config = SearchConfig(
        top_k=2000,  # Using 2K for better recall
        search_batch_size=10,
        num_queries=10,
        enable_recall_eval=True
    )

    logger.info(f"Configuration: top_k={search_config.top_k}")

    # Create synthetic data for testing
    logger.info("Creating synthetic test data...")
    embedding_dim = 768
    num_vectors_per_gpu = 100000  # Start small for testing

    embedding_parts = []
    for i in range(num_gpus):
        embeddings = torch.randn(
            num_vectors_per_gpu, embedding_dim,
            dtype=torch.float32, device=f'cuda:{i}'
        )
        embedding_parts.append(embeddings)
        logger.info(f"Created {num_vectors_per_gpu} vectors on GPU {i}")

    print_memory_status("After data creation")

    # Test parallel index building
    logger.info("Testing parallel index building...")
    builder = ParallelIndexBuilder(num_gpus)

    for index_type in [IndexType.IVF_FLAT]:  # Start with one type
        logger.info(f"\nTesting {index_type.value} index...")

        result = builder.build_indices_parallel(embedding_parts, index_type)

        if result['success']:
            logger.info(f"Successfully built {len(result['indexes'])} indexes")
            logger.info(f"Average build time: {result['avg_time']:.2f}s")

            # Test parallel search
            logger.info("Testing parallel search...")
            search_engine = ParallelSearchEngine(
                result['indexes'], index_type, search_config
            )

            # Create test queries
            test_queries = [
                torch.randn(embedding_dim, dtype=torch.float32)
                for _ in range(5)
            ]

            # Time the search
            start_time = time.time()
            results = search_engine.batch_search(test_queries)
            search_time = time.time() - start_time

            logger.info(f"Searched {len(test_queries)} queries in {search_time:.2f}s")
            logger.info(f"Average search time: {(search_time/len(test_queries))*1000:.2f}ms")

            # Evaluate recall if enabled
            if search_config.enable_recall_eval:
                logger.info("Evaluating recall metrics...")
                evaluator = RecallEvaluator()
                ground_truth = evaluator.generate_synthetic_ground_truth(
                    len(test_queries),
                    num_vectors_per_gpu * num_gpus,
                    relevant_per_query=100
                )

                recall_scores = []
                for i, (distances, indices) in enumerate(results):
                    if len(indices) > 0:
                        recalls = evaluator.evaluate_recall_multiple_k(
                            indices,
                            ground_truth[i],
                            search_config.recall_k_values
                        )
                        recall_scores.append(recalls)

                # Average recall across queries
                if recall_scores:
                    avg_recalls = {}
                    for k in search_config.recall_k_values:
                        avg_recalls[k] = np.mean([r.get(k, 0) for r in recall_scores])

                    logger.info("Average Recall@K:")
                    for k, recall in avg_recalls.items():
                        logger.info(f"  Recall@{k}: {recall:.4f}")
        else:
            logger.error(f"Failed to build indexes: {result['failed_gpus']}")

    print_memory_status("Final")
    logger.info("Test completed")

if __name__ == "__main__":
    main()