"""
Unit tests for IndexBuildingCoordinator

Tests cover index building coordination, error recovery, resource tracking,
and validation methods for the multi-GPU cuVS implementation.
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import TimeoutError

from index_building_coordinator import (
    IndexBuildingCoordinator,
    IndexBuildResult,
    IndexBuildConfig,
    CoordinatedIndexBuild
)
from gpu_resource_manager import GPUResourceManager
from embedding_distribution_manager import EmbeddingDistributionManager, EmbeddingPart, DistributedEmbeddings


class TestIndexBuildResult:
    """Test IndexBuildResult data class"""
    
    def test_valid_success_result(self):
        """Test creating a valid successful result"""
        mock_index = {"type": "ivf_flat", "size": 1000}
        result = IndexBuildResult(
            gpu_id=0,
            index=mock_index,
            build_time=1.5,
            success=True,
            memory_usage_bytes=1024
        )
        
        assert result.gpu_id == 0
        assert result.index == mock_index
        assert result.build_time == 1.5
        assert result.success is True
        assert result.error_message is None
        assert result.memory_usage_bytes == 1024
        
    def test_valid_failure_result(self):
        """Test creating a valid failure result"""
        result = IndexBuildResult(
            gpu_id=1,
            index=None,
            build_time=0.0,
            success=False,
            error_message="GPU memory exhausted"
        )
        
        assert result.gpu_id == 1
        assert result.index is None
        assert result.build_time == 0.0
        assert result.success is False
        assert result.error_message == "GPU memory exhausted"
        
    def test_invalid_gpu_id(self):
        """Test validation of negative GPU ID"""
        with pytest.raises(ValueError, match="gpu_id must be non-negative"):
            IndexBuildResult(
                gpu_id=-1,
                index=None,
                build_time=0.0,
                success=False,
                error_message="test"
            )
            
    def test_invalid_build_time(self):
        """Test validation of negative build time"""
        with pytest.raises(ValueError, match="build_time must be non-negative"):
            IndexBuildResult(
                gpu_id=0,
                index=None,
                build_time=-1.0,
                success=False,
                error_message="test"
            )
            
    def test_success_without_index(self):
        """Test validation that success requires index"""
        with pytest.raises(ValueError, match="index cannot be None when success is True"):
            IndexBuildResult(
                gpu_id=0,
                index=None,
                build_time=1.0,
                success=True
            )
            
    def test_failure_without_error_message(self):
        """Test validation that failure requires error message"""
        with pytest.raises(ValueError, match="error_message cannot be None when success is False"):
            IndexBuildResult(
                gpu_id=0,
                index=None,
                build_time=0.0,
                success=False
            )


class TestIndexBuildConfig:
    """Test IndexBuildConfig data class"""
    
    def test_valid_config(self):
        """Test creating a valid config"""
        config = IndexBuildConfig(
            index_type='ivf_flat',
            index_params={'n_lists': 100},
            search_params={'nprobe': 10},
            parallel_build=True,
            max_retries=3,
            timeout_seconds=60.0
        )
        
        assert config.index_type == 'ivf_flat'
        assert config.index_params == {'n_lists': 100}
        assert config.search_params == {'nprobe': 10}
        assert config.parallel_build is True
        assert config.max_retries == 3
        assert config.timeout_seconds == 60.0
        
    def test_invalid_index_type(self):
        """Test validation of invalid index type"""
        with pytest.raises(ValueError, match="index_type must be one of"):
            IndexBuildConfig(
                index_type='invalid_type',
                index_params={}
            )
            
    def test_invalid_index_params(self):
        """Test validation of non-dict index_params"""
        with pytest.raises(ValueError, match="index_params must be a dictionary"):
            IndexBuildConfig(
                index_type='ivf_flat',
                index_params="not a dict"
            )
            
    def test_invalid_max_retries(self):
        """Test validation of negative max_retries"""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            IndexBuildConfig(
                index_type='ivf_flat',
                index_params={},
                max_retries=-1
            )
            
    def test_invalid_timeout(self):
        """Test validation of non-positive timeout"""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            IndexBuildConfig(
                index_type='ivf_flat',
                index_params={},
                timeout_seconds=0.0
            )


class TestCoordinatedIndexBuild:
    """Test CoordinatedIndexBuild data class"""
    
    def test_valid_coordinated_build(self):
        """Test creating a valid coordinated build result"""
        mock_index = {"type": "ivf_flat"}
        results = [
            IndexBuildResult(0, mock_index, 1.0, True),
            IndexBuildResult(1, None, 0.0, False, "error")
        ]
        config = IndexBuildConfig('ivf_flat', {})
        
        coordinated = CoordinatedIndexBuild(
            build_results=results,
            total_build_time=2.5,
            success=False,
            failed_gpus=[1],
            successful_gpus=[0],
            config=config
        )
        
        assert len(coordinated.build_results) == 2
        assert coordinated.total_build_time == 2.5
        assert coordinated.success is False
        assert coordinated.failed_gpus == [1]
        assert coordinated.successful_gpus == [0]
        
    def test_empty_build_results(self):
        """Test validation of empty build results"""
        config = IndexBuildConfig('ivf_flat', {})
        
        with pytest.raises(ValueError, match="build_results cannot be empty"):
            CoordinatedIndexBuild(
                build_results=[],
                total_build_time=0.0,
                success=True,
                failed_gpus=[],
                successful_gpus=[],
                config=config
            )
            
    def test_inconsistent_gpu_lists(self):
        """Test validation of inconsistent GPU lists"""
        mock_index = {"type": "ivf_flat"}
        results = [IndexBuildResult(0, mock_index, 1.0, True)]
        config = IndexBuildConfig('ivf_flat', {})
        
        with pytest.raises(ValueError, match="failed_gpus and successful_gpus must match"):
            CoordinatedIndexBuild(
                build_results=results,
                total_build_time=1.0,
                success=True,
                failed_gpus=[],
                successful_gpus=[0, 1],  # GPU 1 not in results
                config=config
            )


class TestIndexBuildingCoordinator:
    """Test IndexBuildingCoordinator class"""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Create a mock GPU resource manager"""
        manager = Mock(spec=GPUResourceManager)
        manager.get_available_gpu_ids.return_value = [0, 1]
        manager.validate_gpu_index.return_value = True
        manager.get_gpu_memory_info.return_value = {
            'allocated': 1024 * 1024,
            'reserved': 2048 * 1024,
            'total': 16 * 1024 * 1024 * 1024,
            'free': 14 * 1024 * 1024 * 1024
        }
        manager.cleanup_gpu_resources.return_value = None
        return manager
        
    @pytest.fixture
    def coordinator(self, mock_gpu_manager):
        """Create IndexBuildingCoordinator with mock GPU manager"""
        return IndexBuildingCoordinator(mock_gpu_manager)
        
    @pytest.fixture
    def sample_distributed_embeddings(self):
        """Create sample distributed embeddings"""
        # Create mock tensors - use CPU tensors for testing
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
        
    @pytest.fixture
    def sample_config(self):
        """Create sample index build config"""
        return IndexBuildConfig(
            index_type='ivf_flat',
            index_params={'n_lists': 10},
            parallel_build=False,  # Use sequential for simpler testing
            max_retries=1
        )
        
    def test_initialization(self, mock_gpu_manager):
        """Test coordinator initialization"""
        coordinator = IndexBuildingCoordinator(mock_gpu_manager)
        
        assert coordinator.gpu_manager == mock_gpu_manager
        assert coordinator.built_indices == {}
        assert coordinator.build_history == []
        assert coordinator._active_builds == {}
        
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_build_indices_sequential_success(self, mock_cuda_available, coordinator, sample_distributed_embeddings, sample_config):
        """Test successful sequential index building"""
        result = coordinator.build_indices_parallel(sample_distributed_embeddings, sample_config)
        
        assert isinstance(result, CoordinatedIndexBuild)
        assert result.success is True
        assert len(result.successful_gpus) == 2
        assert len(result.failed_gpus) == 0
        assert len(coordinator.built_indices) == 2
        assert 0 in coordinator.built_indices
        assert 1 in coordinator.built_indices
        
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_build_indices_parallel_success(self, mock_cuda_available, coordinator, sample_distributed_embeddings):
        """Test successful parallel index building"""
        config = IndexBuildConfig(
            index_type='ivf_flat',
            index_params={'n_lists': 10},
            parallel_build=True,
            max_retries=1
        )
        
        result = coordinator.build_indices_parallel(sample_distributed_embeddings, config)
        
        assert isinstance(result, CoordinatedIndexBuild)
        assert result.success is True
        assert len(result.successful_gpus) == 2
        assert len(result.failed_gpus) == 0
        
    def test_build_indices_invalid_input(self, coordinator, sample_config, sample_distributed_embeddings):
        """Test build with invalid input"""
        with pytest.raises(ValueError, match="distributed_embeddings must be a DistributedEmbeddings instance"):
            coordinator.build_indices_parallel("invalid", sample_config)
            
        with pytest.raises(ValueError, match="config must be an IndexBuildConfig instance"):
            coordinator.build_indices_parallel(sample_distributed_embeddings, "invalid")
            
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_build_with_gpu_failure(self, mock_cuda_available, coordinator, sample_distributed_embeddings, sample_config, mock_gpu_manager):
        """Test build with GPU failure"""
        # Make GPU 1 validation fail
        def validate_side_effect(gpu_id):
            return gpu_id == 0  # Only GPU 0 is valid
            
        mock_gpu_manager.validate_gpu_index.side_effect = validate_side_effect
        
        result = coordinator.build_indices_parallel(sample_distributed_embeddings, sample_config)
        
        assert result.success is False
        assert 0 in result.successful_gpus
        assert 1 in result.failed_gpus
        assert len(coordinator.built_indices) == 1  # Only GPU 0 succeeded
        
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_build_with_retry_success(self, mock_cuda_available, coordinator, sample_distributed_embeddings, mock_gpu_manager):
        """Test build with retry that eventually succeeds"""
        config = IndexBuildConfig(
            index_type='ivf_flat',
            index_params={'n_lists': 10},
            parallel_build=False,
            max_retries=2
        )
        
        # Make validation fail first time, succeed second time for GPU 1
        call_count = {'count': 0}
        def validate_side_effect(gpu_id):
            if gpu_id == 0:
                return True
            else:  # gpu_id == 1
                call_count['count'] += 1
                return call_count['count'] > 1  # Fail first call, succeed after
                
        mock_gpu_manager.validate_gpu_index.side_effect = validate_side_effect
        
        result = coordinator.build_indices_parallel(sample_distributed_embeddings, config)
        
        assert result.success is True
        assert len(result.successful_gpus) == 2
        assert len(result.failed_gpus) == 0
        
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    def test_validate_index_build_success(self, coordinator):
        """Test successful index validation"""
        mock_index = {"type": "ivf_flat", "size": 1000, "dim": 128}
        embeddings = torch.randn(1000, 128)
        
        result = coordinator.validate_index_build(0, mock_index, embeddings)
        assert result is True
        
    @patch('index_building_coordinator.CUVS_AVAILABLE', False)
    def test_validate_index_build_failure(self, coordinator):
        """Test index validation failure"""
        embeddings = torch.randn(1000, 128)
        
        # Test with None index
        result = coordinator.validate_index_build(0, None, embeddings)
        assert result is False
        
        # Test with invalid simulated index
        invalid_index = {"type": "ivf_flat", "size": 500, "dim": 128}  # Wrong size
        result = coordinator.validate_index_build(0, invalid_index, embeddings)
        assert result is False
        
    def test_cleanup_failed_builds(self, coordinator):
        """Test cleanup of failed builds"""
        # Add some mock indices
        coordinator.built_indices[0] = {"mock": "index"}
        coordinator.built_indices[1] = {"mock": "index"}
        coordinator._active_builds[0] = True
        coordinator._active_builds[1] = True
        
        coordinator.cleanup_failed_builds([1])
        
        assert 0 in coordinator.built_indices
        assert 1 not in coordinator.built_indices
        assert 0 in coordinator._active_builds
        assert 1 not in coordinator._active_builds
        
    def test_get_built_indices(self, coordinator):
        """Test getting built indices"""
        mock_indices = {0: {"mock": "index1"}, 1: {"mock": "index2"}}
        coordinator.built_indices = mock_indices
        
        result = coordinator.get_built_indices()
        assert result == mock_indices
        assert result is not coordinator.built_indices  # Should be a copy
        
    def test_get_index_for_gpu(self, coordinator):
        """Test getting index for specific GPU"""
        mock_index = {"mock": "index"}
        coordinator.built_indices[0] = mock_index
        
        assert coordinator.get_index_for_gpu(0) == mock_index
        assert coordinator.get_index_for_gpu(1) is None
        
    def test_has_active_builds(self, coordinator):
        """Test checking for active builds"""
        assert coordinator.has_active_builds() is False
        
        coordinator._active_builds[0] = True
        assert coordinator.has_active_builds() is True
        
        coordinator._active_builds[0] = False
        assert coordinator.has_active_builds() is False
        
    def test_get_active_build_gpus(self, coordinator):
        """Test getting active build GPUs"""
        coordinator._active_builds[0] = True
        coordinator._active_builds[1] = False
        coordinator._active_builds[2] = True
        
        active_gpus = coordinator.get_active_build_gpus()
        assert set(active_gpus) == {0, 2}
        
    def test_get_build_summary(self, coordinator):
        """Test getting build summary"""
        # Add some mock build history
        mock_results = [
            IndexBuildResult(0, {"mock": "index"}, 1.0, True),
            IndexBuildResult(1, None, 0.0, False, "error")
        ]
        config = IndexBuildConfig('ivf_flat', {})
        
        coordinator.build_history = [
            CoordinatedIndexBuild(mock_results, 2.0, False, [1], [0], config)
        ]
        coordinator.built_indices = {0: {"mock": "index"}}
        coordinator._active_builds = {2: True}
        
        summary = coordinator.get_build_summary()
        
        assert summary['total_coordinated_builds'] == 1
        assert summary['successful_coordinated_builds'] == 0
        assert summary['current_built_indices'] == 1
        assert summary['active_builds'] == 1
        assert summary['gpu_success_rates'][0] == 1.0
        assert summary['gpu_success_rates'][1] == 0.0
        
    def test_cleanup_all_indices(self, coordinator, mock_gpu_manager):
        """Test cleanup of all indices"""
        coordinator.built_indices = {0: {"mock": "index1"}, 1: {"mock": "index2"}}
        coordinator._active_builds = {0: True, 1: False}
        
        coordinator.cleanup_all_indices()
        
        assert coordinator.built_indices == {}
        assert coordinator._active_builds == {}
        mock_gpu_manager.cleanup_gpu_resources.assert_called_once_with([0, 1])
        
    def test_string_representations(self, coordinator):
        """Test string representations"""
        coordinator.built_indices = {0: {"mock": "index"}}
        coordinator._active_builds = {1: True}
        
        str_repr = str(coordinator)
        assert "built_indices=1" in str_repr
        assert "active_builds=1" in str_repr
        
        repr_str = repr(coordinator)
        assert "IndexBuildingCoordinator" in repr_str
        assert "built_indices=[0]" in repr_str


class TestIndexBuildingIntegration:
    """Integration tests for index building with real GPU operations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_gpu_integration(self):
        """Test with real GPU operations (if available)"""
        gpu_manager = GPUResourceManager()
        if len(gpu_manager.get_available_gpu_ids()) == 0:
            pytest.skip("No GPUs available for integration test")
            
        coordinator = IndexBuildingCoordinator(gpu_manager)
        
        # Create small test embeddings
        embeddings = torch.randn(100, 64)
        embedding_manager = EmbeddingDistributionManager(gpu_manager)
        distributed = embedding_manager.distribute_embeddings(embeddings)
        
        config = IndexBuildConfig(
            index_type='ivf_flat',
            index_params={'n_lists': 2},
            parallel_build=False,
            max_retries=1
        )
        
        # This test will use simulated indices if cuVS is not available
        result = coordinator.build_indices_parallel(distributed, config)
        
        assert isinstance(result, CoordinatedIndexBuild)
        assert len(result.build_results) > 0
        
        # Cleanup
        coordinator.cleanup_all_indices()
        embedding_manager.cleanup_distribution()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])