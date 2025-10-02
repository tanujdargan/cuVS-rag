#!/usr/bin/env python3
"""
Demonstration of IndexBuildingCoordinator for Multi-GPU cuVS Implementation

This script demonstrates how to use the IndexBuildingCoordinator to build
indices across multiple GPUs with proper error handling and resource management.
"""

import torch
import numpy as np
import logging
from gpu_resource_manager import GPUResourceManager
from embedding_distribution_manager import EmbeddingDistributionManager
from index_building_coordinator import (
    IndexBuildingCoordinator,
    IndexBuildConfig,
    IndexBuildResult,
    CoordinatedIndexBuild
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate IndexBuildingCoordinator functionality"""
    print("=" * 60)
    print("IndexBuildingCoordinator Demonstration")
    print("=" * 60)
    
    try:
        # Initialize GPU Resource Manager
        print("\n1. Initializing GPU Resource Manager...")
        gpu_manager = GPUResourceManager()
        print(f"   Available GPUs: {gpu_manager.get_available_gpu_ids()}")
        print(f"   GPU count: {gpu_manager.get_available_gpu_count()}")
        
        if gpu_manager.get_available_gpu_count() == 0:
            print("   No GPUs available. Running in simulation mode.")
        
        # Initialize Embedding Distribution Manager
        print("\n2. Initializing Embedding Distribution Manager...")
        embedding_manager = EmbeddingDistributionManager(gpu_manager)
        
        # Initialize Index Building Coordinator
        print("\n3. Initializing Index Building Coordinator...")
        coordinator = IndexBuildingCoordinator(gpu_manager)
        
        # Create sample embeddings
        print("\n4. Creating sample embeddings...")
        num_vectors = 1000
        embedding_dim = 128
        embeddings = torch.randn(num_vectors, embedding_dim)
        print(f"   Created {num_vectors} embeddings with dimension {embedding_dim}")
        
        # Distribute embeddings across GPUs
        print("\n5. Distributing embeddings across GPUs...")
        try:
            distributed_embeddings = embedding_manager.distribute_embeddings(embeddings)
            print(f"   Successfully distributed embeddings across {len(distributed_embeddings.parts)} GPUs")
            
            for i, part in enumerate(distributed_embeddings.parts):
                print(f"   GPU {part.gpu_id}: {part.tensor.shape[0]} vectors [{part.start_index}-{part.end_index})")
                
        except Exception as e:
            print(f"   Error distributing embeddings: {str(e)}")
            return
        
        # Configure index building
        print("\n6. Configuring index building...")
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
        
        # Build indices with different configurations
        for i, config in enumerate(configs):
            print(f"\n7.{i+1}. Building {config.index_type.upper()} indices...")
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
                    validation_result = coordinator.validate_index_build(
                        gpu_id, index, distributed_embeddings.parts[gpu_id].tensor
                    )
                    print(f"       GPU {gpu_id} validation: {'PASS' if validation_result else 'FAIL'}")
                
            except Exception as e:
                print(f"     Error building indices: {str(e)}")
            
            # Clean up indices for next test
            coordinator.cleanup_all_indices()
            print(f"     Cleaned up indices")
        
        # Demonstrate error handling
        print(f"\n8. Demonstrating error handling...")
        
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
        
        # Show coordinator summary
        print(f"\n9. Coordinator Summary:")
        summary = coordinator.get_build_summary()
        print(f"   Total coordinated builds: {summary['total_coordinated_builds']}")
        print(f"   Successful builds: {summary['successful_coordinated_builds']}")
        print(f"   Current built indices: {summary['current_built_indices']}")
        print(f"   Active builds: {summary['active_builds']}")
        
        if summary['gpu_success_rates']:
            print(f"   GPU success rates:")
            for gpu_id, rate in summary['gpu_success_rates'].items():
                print(f"     GPU {gpu_id}: {rate:.1%}")
        
        print(f"\n10. Final cleanup...")
        coordinator.cleanup_all_indices()
        embedding_manager.cleanup_distribution()
        gpu_manager.cleanup_gpu_resources()
        print(f"    All resources cleaned up")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise
    
    print(f"\n{'=' * 60}")
    print("IndexBuildingCoordinator demonstration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()