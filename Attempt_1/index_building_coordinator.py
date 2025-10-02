"""
Index Building Coordinator for Multi-GPU cuVS Implementation

This module provides robust index building coordination across multiple GPUs with
proper error handling, rollback capabilities, and resource tracking to prevent
indexing errors in multi-GPU vector search operations.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from gpu_resource_manager import GPUResourceManager
from embedding_distribution_manager import EmbeddingDistributionManager, DistributedEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import cuVS modules
try:
    from cuvs.neighbors import ivf_flat, ivf_pq, cagra
    CUVS_AVAILABLE = True
except ImportError:
    logger.warning("cuVS not available. Index building will be simulated.")
    CUVS_AVAILABLE = False


@dataclass
class IndexBuildResult:
    """Result of building an index on a specific GPU"""
    gpu_id: int
    index: Optional[Any]
    build_time: float
    success: bool
    error_message: Optional[str] = None
    memory_usage_bytes: int = 0
    
    def __post_init__(self):
        """Validate IndexBuildResult after initialization"""
        if self.gpu_id < 0:
            raise ValueError(f"gpu_id must be non-negative, got {self.gpu_id}")
        if self.build_time < 0:
            raise ValueError(f"build_time must be non-negative, got {self.build_time}")
        if self.success and self.index is None:
            raise ValueError("index cannot be None when success is True")
        if not self.success and self.error_message is None:
            raise ValueError("error_message cannot be None when success is False")


@dataclass
class IndexBuildConfig:
    """Configuration for index building"""
    index_type: str  # 'ivf_flat', 'ivf_pq', 'cagra'
    index_params: Dict[str, Any]
    search_params: Optional[Dict[str, Any]] = None
    parallel_build: bool = True
    max_retries: int = 2
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Validate IndexBuildConfig after initialization"""
        valid_types = ['ivf_flat', 'ivf_pq', 'cagra']
        if self.index_type not in valid_types:
            raise ValueError(f"index_type must be one of {valid_types}, got {self.index_type}")
        if not isinstance(self.index_params, dict):
            raise ValueError("index_params must be a dictionary")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")


@dataclass
class CoordinatedIndexBuild:
    """Container for coordinated index build results"""
    build_results: List[IndexBuildResult]
    total_build_time: float
    success: bool
    failed_gpus: List[int]
    successful_gpus: List[int]
    config: IndexBuildConfig
    
    def __post_init__(self):
        """Validate CoordinatedIndexBuild after initialization"""
        if not self.build_results:
            raise ValueError("build_results cannot be empty")
        if self.total_build_time < 0:
            raise ValueError(f"total_build_time must be non-negative, got {self.total_build_time}")
        
        # Validate consistency between results and GPU lists
        result_gpus = {result.gpu_id for result in self.build_results}
        failed_set = set(self.failed_gpus)
        successful_set = set(self.successful_gpus)
        
        if failed_set | successful_set != result_gpus:
            raise ValueError("failed_gpus and successful_gpus must match build_results GPU IDs")
        if failed_set & successful_set:
            raise ValueError("failed_gpus and successful_gpus cannot overlap")


class IndexBuildingCoordinator:
    """
    Coordinate index building across multiple GPUs with proper error handling.
    
    This class addresses index building issues in multi-GPU cuVS:
    - Parallel index building with proper error handling
    - Rollback capabilities for failed operations
    - Resource tracking and cleanup for GPU indices
    - Validation methods for successful index builds
    """
    
    def __init__(self, gpu_manager: GPUResourceManager):
        """
        Initialize IndexBuildingCoordinator.
        
        Args:
            gpu_manager: GPU Resource Manager instance
        """
        self.gpu_manager = gpu_manager
        self.built_indices: Dict[int, Any] = {}
        self.build_history: List[CoordinatedIndexBuild] = []
        self._active_builds: Dict[int, bool] = {}
        
    def build_indices_parallel(self, 
                             distributed_embeddings: DistributedEmbeddings,
                             config: IndexBuildConfig) -> CoordinatedIndexBuild:
        """
        Build indices on multiple GPUs in parallel with proper error handling.
        
        Args:
            distributed_embeddings: DistributedEmbeddings container
            config: IndexBuildConfig with build parameters
            
        Returns:
            CoordinatedIndexBuild: Results of the coordinated build operation
            
        Raises:
            ValueError: If distributed_embeddings or config are invalid
            RuntimeError: If build coordination fails
        """
        # Validate inputs
        if not isinstance(distributed_embeddings, DistributedEmbeddings):
            raise ValueError("distributed_embeddings must be a DistributedEmbeddings instance")
        if not isinstance(config, IndexBuildConfig):
            raise ValueError("config must be an IndexBuildConfig instance")
            
        logger.info(f"Starting parallel index building with {config.index_type} on {len(distributed_embeddings.parts)} GPUs")
        
        # Clear any existing indices for the GPUs we're about to use
        gpu_ids = [part.gpu_id for part in distributed_embeddings.parts]
        self._cleanup_existing_indices(gpu_ids)
        
        # Mark GPUs as having active builds
        for gpu_id in gpu_ids:
            self._active_builds[gpu_id] = True
            
        build_results = []
        start_time = time.time()
        
        try:
            if config.parallel_build and len(distributed_embeddings.parts) > 1:
                # Parallel building
                build_results = self._build_parallel(distributed_embeddings, config)
            else:
                # Sequential building
                build_results = self._build_sequential(distributed_embeddings, config)
                
            total_build_time = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in build_results if r.success]
            failed_results = [r for r in build_results if not r.success]
            
            successful_gpus = [r.gpu_id for r in successful_results]
            failed_gpus = [r.gpu_id for r in failed_results]
            
            # Store successful indices
            for result in successful_results:
                self.built_indices[result.gpu_id] = result.index
                
            # Create coordinated build result
            coordinated_result = CoordinatedIndexBuild(
                build_results=build_results,
                total_build_time=total_build_time,
                success=len(failed_results) == 0,
                failed_gpus=failed_gpus,
                successful_gpus=successful_gpus,
                config=config
            )
            
            # Store in history
            self.build_history.append(coordinated_result)
            
            if failed_gpus:
                logger.warning(f"Index building failed on GPUs: {failed_gpus}")
                # Cleanup failed builds
                self.cleanup_failed_builds(failed_gpus)
            else:
                logger.info(f"Successfully built indices on all {len(successful_gpus)} GPUs in {total_build_time:.2f}s")
                
            return coordinated_result
            
        except Exception as e:
            logger.error(f"Index building coordination failed: {str(e)}")
            # Cleanup any partial builds
            self._cleanup_existing_indices(gpu_ids)
            raise RuntimeError(f"Index building coordination failed: {str(e)}")
        finally:
            # Clear active build flags
            for gpu_id in gpu_ids:
                self._active_builds.pop(gpu_id, None)
                
    def _build_parallel(self, 
                       distributed_embeddings: DistributedEmbeddings,
                       config: IndexBuildConfig) -> List[IndexBuildResult]:
        """
        Build indices in parallel using ThreadPoolExecutor.
        
        Args:
            distributed_embeddings: DistributedEmbeddings container
            config: IndexBuildConfig with build parameters
            
        Returns:
            List[IndexBuildResult]: Results from parallel builds
        """
        build_results = []
        
        with ThreadPoolExecutor(max_workers=len(distributed_embeddings.parts)) as executor:
            # Submit build tasks
            future_to_part = {}
            for part in distributed_embeddings.parts:
                future = executor.submit(self._build_single_index, part, config)
                future_to_part[future] = part
                
            # Collect results with timeout handling
            for future in as_completed(future_to_part.keys(), timeout=config.timeout_seconds):
                try:
                    result = future.result()
                    build_results.append(result)
                    logger.info(f"GPU {result.gpu_id} build completed: success={result.success}, time={result.build_time:.2f}s")
                except Exception as e:
                    part = future_to_part[future]
                    logger.error(f"GPU {part.gpu_id} build failed with exception: {str(e)}")
                    build_results.append(IndexBuildResult(
                        gpu_id=part.gpu_id,
                        index=None,
                        build_time=0.0,
                        success=False,
                        error_message=str(e)
                    ))
                    
        return build_results
        
    def _build_sequential(self, 
                         distributed_embeddings: DistributedEmbeddings,
                         config: IndexBuildConfig) -> List[IndexBuildResult]:
        """
        Build indices sequentially.
        
        Args:
            distributed_embeddings: DistributedEmbeddings container
            config: IndexBuildConfig with build parameters
            
        Returns:
            List[IndexBuildResult]: Results from sequential builds
        """
        build_results = []
        
        for part in distributed_embeddings.parts:
            result = self._build_single_index(part, config)
            build_results.append(result)
            logger.info(f"GPU {result.gpu_id} build completed: success={result.success}, time={result.build_time:.2f}s")
            
            # If this is a critical failure and we want to stop, we could break here
            # For now, continue with all parts to get complete failure information
            
        return build_results
        
    def _build_single_index(self, embedding_part, config: IndexBuildConfig) -> IndexBuildResult:
        """
        Build a single index on one GPU with retry logic.
        
        Args:
            embedding_part: EmbeddingPart with tensor and GPU assignment
            config: IndexBuildConfig with build parameters
            
        Returns:
            IndexBuildResult: Result of the build operation
        """
        gpu_id = embedding_part.gpu_id
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying index build on GPU {gpu_id} (attempt {attempt + 1}/{config.max_retries + 1})")
                    
                # Validate GPU is still available
                if not self.gpu_manager.validate_gpu_index(gpu_id):
                    raise RuntimeError(f"GPU {gpu_id} is no longer available")
                    
                start_time = time.time()
                
                # Build index based on type
                # Only use CUDA context if CUDA is available and we're not simulating
                if torch.cuda.is_available() and CUVS_AVAILABLE:
                    with torch.cuda.device(gpu_id):
                        index = self._create_index(embedding_part.tensor, config)
                else:
                    # For simulation or CPU-only environments
                    index = self._create_index(embedding_part.tensor, config)
                    
                build_time = time.time() - start_time
                
                # Get memory usage
                try:
                    memory_info = self.gpu_manager.get_gpu_memory_info(gpu_id)
                    memory_usage = memory_info.get('allocated', 0)
                except Exception:
                    memory_usage = 0
                
                # Validate the built index
                if not self.validate_index_build(gpu_id, index, embedding_part.tensor):
                    raise RuntimeError("Index validation failed")
                    
                return IndexBuildResult(
                    gpu_id=gpu_id,
                    index=index,
                    build_time=build_time,
                    success=True,
                    memory_usage_bytes=memory_usage
                )
                
            except Exception as e:
                error_msg = f"GPU {gpu_id} build attempt {attempt + 1} failed: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == config.max_retries:
                    # Final attempt failed
                    return IndexBuildResult(
                        gpu_id=gpu_id,
                        index=None,
                        build_time=0.0,
                        success=False,
                        error_message=error_msg
                    )
                    
                # Wait before retry
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
                # Clean up GPU memory before retry
                try:
                    self.gpu_manager.cleanup_gpu_resources([gpu_id])
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup failed on GPU {gpu_id}: {str(cleanup_error)}")
                    
        # This should never be reached due to the loop structure, but included for safety
        return IndexBuildResult(
            gpu_id=gpu_id,
            index=None,
            build_time=0.0,
            success=False,
            error_message="Unexpected error in build loop"
        )
        
    def _create_index(self, embeddings: torch.Tensor, config: IndexBuildConfig) -> Any:
        """
        Create an index based on the specified type and parameters.
        
        Args:
            embeddings: Tensor with embeddings for this GPU
            config: IndexBuildConfig with build parameters
            
        Returns:
            Built index object
            
        Raises:
            ValueError: If index type is not supported
            RuntimeError: If index building fails
        """
        if not CUVS_AVAILABLE:
            # Simulate index building for testing
            logger.warning("cuVS not available, simulating index build")
            time.sleep(0.1)  # Simulate build time
            return {"type": config.index_type, "size": embeddings.shape[0], "dim": embeddings.shape[1]}
            
        try:
            if config.index_type == 'ivf_flat':
                # Calculate appropriate n_lists based on data size
                n_lists = config.index_params.get('n_lists', max(1, min(256, embeddings.shape[0] // 1000 + 1)))
                params = ivf_flat.IndexParams(n_lists=n_lists)
                return ivf_flat.build(params, embeddings)
                
            elif config.index_type == 'ivf_pq':
                # IVF-PQ parameters
                n_lists = config.index_params.get('n_lists', max(1, min(256, embeddings.shape[0] // 1000 + 1)))
                pq_bits = config.index_params.get('pq_bits', 8)
                pq_dim = config.index_params.get('pq_dim', min(64, embeddings.shape[1] // 4))
                params = ivf_pq.IndexParams(n_lists=n_lists, pq_bits=pq_bits, pq_dim=pq_dim)
                return ivf_pq.build(params, embeddings)
                
            elif config.index_type == 'cagra':
                # CAGRA parameters
                intermediate_graph_degree = config.index_params.get('intermediate_graph_degree', 64)
                graph_degree = config.index_params.get('graph_degree', 32)
                params = cagra.IndexParams(
                    intermediate_graph_degree=intermediate_graph_degree,
                    graph_degree=graph_degree
                )
                return cagra.build(params, embeddings)
                
            else:
                raise ValueError(f"Unsupported index type: {config.index_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to create {config.index_type} index: {str(e)}")
            
    def validate_index_build(self, gpu_id: int, index: Any, original_embeddings: torch.Tensor) -> bool:
        """
        Validate that an index was built successfully.
        
        Args:
            gpu_id: GPU ID where index was built
            index: Built index object
            original_embeddings: Original embeddings used to build the index
            
        Returns:
            bool: True if index is valid, False otherwise
        """
        try:
            # Basic validation
            if index is None:
                logger.error(f"Index on GPU {gpu_id} is None")
                return False
                
            # Validate GPU is still accessible
            if not self.gpu_manager.validate_gpu_index(gpu_id):
                logger.error(f"GPU {gpu_id} is no longer accessible")
                return False
                
            # For simulated indices (when cuVS not available)
            if not CUVS_AVAILABLE:
                if isinstance(index, dict):
                    expected_size = original_embeddings.shape[0]
                    expected_dim = original_embeddings.shape[1]
                    return (index.get('size') == expected_size and 
                           index.get('dim') == expected_dim)
                return False
                
            # For real cuVS indices, we could perform more sophisticated validation
            # such as a small test search, but for now we just check that the index exists
            # and is not obviously corrupted
            
            # Try to access basic properties (this will vary by index type)
            try:
                # This is a basic check - in practice, you might want to perform
                # a small test search to ensure the index is functional
                _ = str(index)  # Basic object access
                return True
            except Exception as e:
                logger.error(f"Index validation failed on GPU {gpu_id}: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error during index validation on GPU {gpu_id}: {str(e)}")
            return False
            
    def cleanup_failed_builds(self, failed_gpu_ids: List[int]) -> None:
        """
        Clean up resources from failed index builds.
        
        Args:
            failed_gpu_ids: List of GPU IDs where builds failed
        """
        logger.info(f"Cleaning up failed builds on GPUs: {failed_gpu_ids}")
        
        for gpu_id in failed_gpu_ids:
            try:
                # Remove from built indices if present
                if gpu_id in self.built_indices:
                    del self.built_indices[gpu_id]
                    
                # Clear active build flag
                self._active_builds.pop(gpu_id, None)
                
            except Exception as e:
                logger.warning(f"Error during cleanup of GPU {gpu_id}: {str(e)}")
                
        # Clean up GPU memory
        try:
            self.gpu_manager.cleanup_gpu_resources(failed_gpu_ids)
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {str(e)}")
            
    def _cleanup_existing_indices(self, gpu_ids: List[int]) -> None:
        """
        Clean up existing indices on specified GPUs.
        
        Args:
            gpu_ids: List of GPU IDs to clean up
        """
        for gpu_id in gpu_ids:
            if gpu_id in self.built_indices:
                try:
                    del self.built_indices[gpu_id]
                    logger.info(f"Cleaned up existing index on GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up existing index on GPU {gpu_id}: {str(e)}")
                    
    def get_built_indices(self) -> Dict[int, Any]:
        """
        Get dictionary of successfully built indices.
        
        Returns:
            Dict mapping GPU ID to index object
        """
        return self.built_indices.copy()
        
    def get_index_for_gpu(self, gpu_id: int) -> Optional[Any]:
        """
        Get index for a specific GPU.
        
        Args:
            gpu_id: Target GPU ID
            
        Returns:
            Index object or None if not found
        """
        return self.built_indices.get(gpu_id)
        
    def has_active_builds(self) -> bool:
        """
        Check if there are any active builds in progress.
        
        Returns:
            bool: True if builds are active, False otherwise
        """
        return any(self._active_builds.values())
        
    def get_active_build_gpus(self) -> List[int]:
        """
        Get list of GPUs with active builds.
        
        Returns:
            List of GPU IDs with active builds
        """
        return [gpu_id for gpu_id, active in self._active_builds.items() if active]
        
    def get_build_summary(self) -> Dict[str, Any]:
        """
        Get summary of build history and current state.
        
        Returns:
            Dict with build summary information
        """
        total_builds = len(self.build_history)
        successful_builds = sum(1 for build in self.build_history if build.success)
        
        gpu_success_rates = {}
        for build in self.build_history:
            for result in build.build_results:
                if result.gpu_id not in gpu_success_rates:
                    gpu_success_rates[result.gpu_id] = {'success': 0, 'total': 0}
                gpu_success_rates[result.gpu_id]['total'] += 1
                if result.success:
                    gpu_success_rates[result.gpu_id]['success'] += 1
                    
        return {
            'total_coordinated_builds': total_builds,
            'successful_coordinated_builds': successful_builds,
            'current_built_indices': len(self.built_indices),
            'active_builds': len([gpu_id for gpu_id, active in self._active_builds.items() if active]),
            'gpu_success_rates': {
                gpu_id: rates['success'] / rates['total'] if rates['total'] > 0 else 0
                for gpu_id, rates in gpu_success_rates.items()
            }
        }
        
    def cleanup_all_indices(self) -> None:
        """Clean up all built indices and resources."""
        logger.info("Cleaning up all indices and resources")
        
        gpu_ids_to_cleanup = list(self.built_indices.keys())
        
        # Clear built indices
        self.built_indices.clear()
        
        # Clear active builds
        self._active_builds.clear()
        
        # Clean up GPU memory
        if gpu_ids_to_cleanup:
            try:
                self.gpu_manager.cleanup_gpu_resources(gpu_ids_to_cleanup)
            except Exception as e:
                logger.warning(f"Error during final GPU cleanup: {str(e)}")
                
        # Force garbage collection
        gc.collect()
        
    def __str__(self) -> str:
        """String representation of IndexBuildingCoordinator"""
        return (f"IndexBuildingCoordinator(built_indices={len(self.built_indices)}, "
                f"active_builds={len([gpu_id for gpu_id, active in self._active_builds.items() if active])}, "
                f"gpu_manager={self.gpu_manager})")
        
    def __repr__(self) -> str:
        """Detailed representation of IndexBuildingCoordinator"""
        return (f"IndexBuildingCoordinator(gpu_manager={repr(self.gpu_manager)}, "
                f"built_indices={list(self.built_indices.keys())}, "
                f"build_history={len(self.build_history)} builds)")