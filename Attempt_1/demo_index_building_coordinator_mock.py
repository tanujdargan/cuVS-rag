#!/usr/bin/env python3
"""
Mock demonstration of IndexBuildingCoordinator for Multi-GPU cuVS Implementation

This script demonstrates the IndexBuildingCoordinator with mocked GPU resources
to show functionality even when CUDA is not available.
"""

import torch
import numpy as np
import logging
from unittest.mock import Mock, patch
from gpu_resource_manager import GPUResourceManager
from embedding_distribution_manager import EmbeddingDistributionManager, EmbeddingPart, DistributedEmbeddings
from index_building_coordinator import (
    IndexBuildingCoordinator,
    IndexBuildConfig,
    IndexBuildResult,
    CoordinatedIndexBuild
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_gpu_manager():
    """Create a mock GPU manager for demonstration"""
    mock_manager = Mock(spec=GPUResourceManager)
    mock_manager.get_available_gpu_ids.return_value = [0, 1]
    mock_manager.get_available_gpu_count.return_value = 2
    mock_manager.validate_gpu_index.return_value = True
    mock_manager.get_safe_device_string.side_effect = lambda gpu_id: f'cuda:{gpu_id}'
    mock_manager.distribute_workload.return_value = [(0, 0, 500), (1, 500, 1000)]
    mock_manager.get_gpu_memory_info.return_value = {
        'allocated': 1024 * 1024,
        'reserved': 2048 * 1024,
        'total': 16 * 1024 * 1024 * 1024,
        'free': 14 * 1024 * 1024 * 1024
    }
    mock_manager.cleanup_gpu_resources.return_value = None
    return mock_manager


def create_mock_distributed_embeddings():
    """Create mock distributed embeddings for demonstration"""
    tensor1 = torch.randn(500, 128)
    tensor2 = torch.randn(500, 128)
    
    parts = [
        EmbeddingPart(gpu_id=0, tensor=tensor1, start_index=0, end_index=500),
        EmbeddingPart(gpu_id=1, tensor=tensor2, start_index=500, end_index=1000)
    ]
    
    return DistributedEmbeddings(
        parts=parts,
        total_size=1000,
        embedding_dim=128
    )


def main():
    """Demonstrate IndexBuildingCoordinator functionality with mocked GPUs"""
    print("=" * 70)
    print("IndexBuildingCoordinator Mock Demonstration")
    print("=" * 70)
    
    try:
        # Create mock GPU manager
        print("\n1. Creating mock GPU Resource Manager...")
        gpu_manager = create_mock_gpu_manager()
        print(f"   Mock available GPUs: {gpu_manager.get_available_gpu_ids()}")
        print(f"   Mock GPU count: {gpu_manager.get_available_gpu_count()}")
        
        # Initialize Index Building Coordinator
        print("\n2. Initializing Index Building Coordinator...")
        coordinator = IndexBuildingCoordinator(gpu_manager)
        
        # Create mock distributed embeddings
        print("\n3. Creating mock distributed embeddings...")
        distributed_embeddings = create_mock_distributed_embeddings()
        print(f"   Created distributed embeddings with {len(distributed_embeddings.parts)} parts")
        
        for i, part in enumerate(distributed_embeddings.parts):
            print(f"   GPU {part.gpu_id}: {part.tensor.shape[0]} vectors [{part.start_index}-{part.end_index})")
        
        # Configure index building
        print("\n4. Configuring index building...")
        configs = [
            IndexBuildConfig(
                index_type='ivf_flat',
                index_params={'n_lists': 10},
                parallel_build=False,
                max_retries=1
            ),
            IndexBuildConfig(
                index_type='ivf_pq',
                index_params={'n_lists': 10, 'pq_bits': 8, 'pq_dim': 32},
                parallel_build=True,
                max_retries=2
            ),
            IndexBuildConfig(
                index_type='cagra',
                index_params={'intermediate_graph_degree': 32, 'graph_degree': 16},
                parallel_build=True,
                max_retries=1
            )
        ]
        
        # Patch CUDA availability and cuVS availability for simulation
        with patch('torch.cuda.is_available', return_value=False), \
             patch('index_building_coordinator.CUVS_AVAILABLE', False):
            
            # Build indices with different configurations
            for i, config in enumerate(configs):
                print(f"\n5.{i+1}. Building {config.index_type.upper()} indices...")
                print(f"     Parallel build: {config.parallel_build}")
                print(f"     Max retries: {config.max_retries}")
                
                try:
                    # Build indices
                    result = coordinator.build_indices_parallel(distributed_embeddings, config)
                    
                    # Display results
                    print(f"     Build result: {'SUCCESS' if result.success else 'FAILED'}")
                    print(f"     Total build time: {result.total_build_time:.2f}s")
                    print(f"     Successful GPUs: {result.successful_gpus}")
                    print(f"     Failed GPUs: {result.failed_gpus}")
                    
                    # Show individual GPU results
                    for build_result in result.build_results:
                        status = "SUCCESS" if build_result.success else "FAILED"
                        print(f"       GPU {build_result.gpu_id}: {status} ({build_result.build_time:.2f}s)")
                        if not build_result.success:
                            print(f"         Error: {build_result.error_message}")
                    
                    # Validate indices
                    print(f"     Built indices: {len(coordinator.get_built_indices())}")
                    for gpu_id, index in coordinator.get_built_indices().items():
                        # Find the corresponding part for validation
                        part = next(p for p in distributed_embeddings.parts if p.gpu_id == gpu_id)
                        validation_result = coordinator.validate_index_build(
                            gpu_id, index, part.tensor
                        )
                        print(f"       GPU {gpu_id} validation: {'PASS' if validation_result else 'FAIL'}")
                    
                except Exception as e:
                    print(f"     Error building indices: {str(e)}")
                
                # Clean up indices for next test
                coordinator.cleanup_all_indices()
                print(f"     Cleaned up indices")
        
        # Demonstrate error handling
        print(f"\n6. Demonstrating error handling...")
        
        # Test with invalid configuration
        try:
            invalid_config = IndexBuildConfig(
                index_type='invalid_type',
                index_params={}
            )
        except ValueError as e:
            print(f"   Caught expected validation error: {str(e)}")
        
        # Test with invalid input
        try:
            coordinator.build_indices_parallel("invalid", configs[0])
        except ValueError as e:
            print(f"   Caught expected input validation error: {str(e)}")
        
        # Test with GPU failure simulation
        print(f"\n7. Simulating GPU failure...")
        gpu_manager.validate_gpu_index.side_effect = lambda gpu_id: gpu_id == 0  # Only GPU 0 works
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('index_building_coordinator.CUVS_AVAILABLE', False):
            
            result = coordinator.build_indices_parallel(distributed_embeddings, configs[0])
            print(f"   Build with GPU failure: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"   Successful GPUs: {result.successful_gpus}")
            print(f"   Failed GPUs: {result.failed_gpus}")
        
        # Reset GPU manager for summary
        gpu_manager.validate_gpu_index.return_value = True
        
        # Show coordinator summary
        print(f"\n8. Coordinator Summary:")
        summary = coordinator.get_build_summary()
        print(f"   Total coordinated builds: {summary['total_coordinated_builds']}")
        print(f"   Successful builds: {summary['successful_coordinated_builds']}")
        print(f"   Current built indices: {summary['current_built_indices']}")
        print(f"   Active builds: {summary['active_builds']}")
        
        if summary['gpu_success_rates']:
            print(f"   GPU success rates:")
            for gpu_id, rate in summary['gpu_success_rates'].items():
                print(f"     GPU {gpu_id}: {rate:.1%}")
        
        print(f"\n9. Final cleanup...")
        coordinator.cleanup_all_indices()
        print(f"   All resources cleaned up")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise
    
    print(f"\n{'=' * 70}")
    print("IndexBuildingCoordinator mock demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()