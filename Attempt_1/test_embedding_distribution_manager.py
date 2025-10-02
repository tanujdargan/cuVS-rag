"""
Unit tests for EmbeddingDistributionManager

Tests for safe embedding distribution across multiple GPUs with proper
bounds checking, validation, and error handling.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from embedding_distribution_manager import (
    EmbeddingDistributionManager, 
    EmbeddingPart, 
    DistributedEmbeddings
)
from gpu_resource_manager import GPUResourceManager


class TestEmbeddingPart:
    """Test EmbeddingPart data model"""
    
    def test_valid_embedding_part_creation(self):
        """Test creating a valid EmbeddingPart"""
        tensor = torch.randn(10, 128)
        part = EmbeddingPart(
            gpu_id=0,
            tensor=tensor,
            start_index=0,
            end_index=10
        )
        
        assert part.gpu_id == 0
        assert torch.equal(part.tensor, tensor)
        assert part.start_index == 0
        assert part.end_index == 10
    
    def test_embedding_part_validation_negative_start_index(self):
        """Test EmbeddingPart validation with negative start_index"""
        tensor = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="start_index must be non-negative"):
            EmbeddingPart(
                gpu_id=0,
                tensor=tensor,
                start_index=-1,
                end_index=10
            )
    
    def test_embedding_part_validation_invalid_end_index(self):
        """Test EmbeddingPart validation with invalid end_index"""
        tensor = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="end_index .* must be greater than start_index"):
            EmbeddingPart(
                gpu_id=0,
                tensor=tensor,
                start_index=10,
                end_index=5
            )
    
    def test_embedding_part_validation_negative_gpu_id(self):
        """Test EmbeddingPart validation with negative gpu_id"""
        tensor = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="gpu_id must be non-negative"):
            EmbeddingPart(
                gpu_id=-1,
                tensor=tensor,
                start_index=0,
                end_index=10
            )
            
    def test_tensor_size_mismatch(self):
        """Test EmbeddingPart with tensor size not matching index range"""
        tensor = torch.randn(50, 128)  # Size 50
        
        with pytest.raises(ValueError, match="Tensor size.*doesn't match index range"):
            EmbeddingPart(
                gpu_id=0,
                tensor=tensor,
                start_index=0,
                end_index=100  # Range is 100, but tensor is 50
            )


class TestDistributedEmbeddings:
    """Test DistributedEmbeddings data model"""
    
    def test_valid_distributed_embeddings(self):
        """Test creating valid DistributedEmbeddings"""
        part1 = EmbeddingPart(0, torch.randn(50, 128), 0, 50)
        part2 = EmbeddingPart(1, torch.randn(50, 128), 50, 100)
        
        dist_emb = DistributedEmbeddings(
            parts=[part1, part2],
            total_size=100,
            embedding_dim=128
        )
        
        assert len(dist_emb.parts) == 2
        assert dist_emb.total_size == 100
        assert dist_emb.embedding_dim == 128
        
    def test_empty_parts_list(self):
        """Test DistributedEmbeddings with empty parts list"""
        with pytest.raises(ValueError, match="parts list cannot be empty"):
            DistributedEmbeddings(
                parts=[],
                total_size=100,
                embedding_dim=128
            )
            
    def test_invalid_total_size(self):
        """Test DistributedEmbeddings with invalid total_size"""
        part = EmbeddingPart(0, torch.randn(50, 128), 0, 50)
        
        with pytest.raises(ValueError, match="total_size must be positive"):
            DistributedEmbeddings(
                parts=[part],
                total_size=0,
                embedding_dim=128
            )
            
    def test_inconsistent_embedding_dim(self):
        """Test DistributedEmbeddings with inconsistent embedding dimensions"""
        part1 = EmbeddingPart(0, torch.randn(50, 128), 0, 50)
        part2 = EmbeddingPart(1, torch.randn(50, 64), 50, 100)  # Different dim
        
        with pytest.raises(ValueError, match="has embedding_dim.*expected"):
            DistributedEmbeddings(
                parts=[part1, part2],
                total_size=100,
                embedding_dim=128
            )
            
    def test_gap_in_parts(self):
        """Test DistributedEmbeddings with gap between parts"""
        part1 = EmbeddingPart(0, torch.randn(50, 128), 0, 50)
        part2 = EmbeddingPart(1, torch.randn(40, 128), 60, 100)  # Gap from 50-60
        
        with pytest.raises(ValueError, match="Gap or overlap detected"):
            DistributedEmbeddings(
                parts=[part1, part2],
                total_size=100,
                embedding_dim=128
            )
            
    def test_overlap_in_parts(self):
        """Test DistributedEmbeddings with overlapping parts"""
        part1 = EmbeddingPart(0, torch.randn(60, 128), 0, 60)
        part2 = EmbeddingPart(1, torch.randn(50, 128), 50, 100)  # Overlap from 50-60
        
        with pytest.raises(ValueError, match="Gap or overlap detected"):
            DistributedEmbeddings(
                parts=[part1, part2],
                total_size=100,
                embedding_dim=128
            )


class TestEmbeddingDistributionManager:
    """Test EmbeddingDistributionManager class"""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Create a mock GPU resource manager"""
        gpu_manager = Mock(spec=GPUResourceManager)
        gpu_manager.get_available_gpu_ids.return_value = [0, 1]
        gpu_manager.validate_gpu_index.side_effect = lambda x: x in [0, 1]
        gpu_manager.get_safe_device_string.side_effect = lambda x: f'cuda:{x}'
        gpu_manager.distribute_workload.return_value = [(0, 0, 50), (1, 50, 100)]
        gpu_manager.cleanup_gpu_resources.return_value = None
        return gpu_manager
        
    @pytest.fixture
    def distribution_manager(self, mock_gpu_manager):
        """Create EmbeddingDistributionManager with mock GPU manager"""
        return EmbeddingDistributionManager(mock_gpu_manager)
        
    def test_init(self, mock_gpu_manager):
        """Test EmbeddingDistributionManager initialization"""
        manager = EmbeddingDistributionManager(mock_gpu_manager)
        assert manager.gpu_manager == mock_gpu_manager
        assert manager.current_distribution is None
        
    def test_distribute_embeddings_valid(self, distribution_manager, mock_gpu_manager):
        """Test successful embedding distribution"""
        embeddings = torch.randn(100, 128)
        
        # Mock tensor.to() method to avoid CUDA issues
        def mock_to_side_effect(device_string):
            # Return the tensor itself (simulating successful GPU transfer)
            return embeddings[:50] if '0' in device_string else embeddings[50:]
            
        with patch.object(torch.Tensor, 'to', side_effect=mock_to_side_effect):
            result = distribution_manager.distribute_embeddings(embeddings)
            
            assert isinstance(result, DistributedEmbeddings)
            assert result.total_size == 100
            assert result.embedding_dim == 128
            assert len(result.parts) == 2
            
            # Verify GPU manager calls
            mock_gpu_manager.get_available_gpu_ids.assert_called_once()
            
    def test_distribute_embeddings_invalid_tensor(self, distribution_manager):
        """Test distribution with invalid tensor"""
        # Test non-tensor input
        with pytest.raises(TypeError, match="embeddings must be a torch.Tensor"):
            distribution_manager.distribute_embeddings("not a tensor")
            
        # Test 1D tensor
        with pytest.raises(ValueError, match="embeddings must be 2D tensor"):
            distribution_manager.distribute_embeddings(torch.randn(100))
            
        # Test empty tensor
        with pytest.raises(ValueError, match="embeddings tensor cannot be empty"):
            distribution_manager.distribute_embeddings(torch.empty(0, 128))
            
    def test_distribute_embeddings_no_gpus(self, mock_gpu_manager):
        """Test distribution when no GPUs are available"""
        mock_gpu_manager.get_available_gpu_ids.return_value = []
        manager = EmbeddingDistributionManager(mock_gpu_manager)
        
        embeddings = torch.randn(100, 128)
        
        with pytest.raises(RuntimeError, match="No GPUs available"):
            manager.distribute_embeddings(embeddings)
            
    def test_distribute_embeddings_invalid_target_gpu(self, distribution_manager, mock_gpu_manager):
        """Test distribution with invalid target GPU"""
        embeddings = torch.randn(100, 128)
        mock_gpu_manager.validate_gpu_index.return_value = False
        
        with pytest.raises(ValueError, match="Target GPU.*is not available"):
            distribution_manager.distribute_embeddings(embeddings, target_gpus=[5])
            
    def test_validate_distribution_valid(self, distribution_manager, mock_gpu_manager):
        """Test validation of valid distribution"""
        # Create mock embedding parts with proper device string
        tensor1 = Mock()
        tensor1.device = Mock()
        tensor1.device.__str__ = Mock(return_value='cuda:0')
        tensor1.shape = (50, 128)
        tensor1.dtype = torch.float32
        
        tensor2 = Mock()
        tensor2.device = Mock()
        tensor2.device.__str__ = Mock(return_value='cuda:1')
        tensor2.shape = (50, 128)
        tensor2.dtype = torch.float32
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        
        # Mock GPU manager methods
        mock_gpu_manager.get_available_gpu_ids.return_value = [0, 1]
        
        result = distribution_manager.validate_distribution(dist_emb)
        assert result is True
        
    def test_validate_distribution_invalid_gpu(self, distribution_manager, mock_gpu_manager):
        """Test validation with invalid GPU"""
        tensor = Mock()
        tensor.device.type = 'cuda'
        tensor.device.index = 0
        tensor.size.side_effect = lambda dim: 100 if dim == 0 else 128
        
        part = EmbeddingPart(0, tensor, 0, 100)
        dist_emb = DistributedEmbeddings([part], 100, 128)
        
        # Mock GPU validation to return False for the specific GPU
        mock_gpu_manager.validate_gpu_index.side_effect = lambda x: False if x == 0 else True
        
        result = distribution_manager.validate_distribution(dist_emb)
        assert result is False
        
    def test_validate_distribution_wrong_device(self, distribution_manager, mock_gpu_manager):
        """Test validation with tensor on wrong device"""
        tensor = Mock()
        tensor.device.type = 'cpu'  # Wrong device type
        tensor.device.index = 0
        tensor.size.side_effect = lambda dim: 100 if dim == 0 else 128
        
        part = EmbeddingPart(0, tensor, 0, 100)
        dist_emb = DistributedEmbeddings([part], 100, 128)
        
        mock_gpu_manager.validate_gpu_index.return_value = True
        
        result = distribution_manager.validate_distribution(dist_emb)
        assert result is False
        
    def test_get_total_memory_usage(self, distribution_manager):
        """Test calculating total memory usage"""
        tensor1 = torch.randn(50, 128)
        tensor2 = torch.randn(50, 128)
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        
        memory_usage = distribution_manager.get_total_memory_usage(dist_emb)
        
        assert isinstance(memory_usage, dict)
        assert 0 in memory_usage
        assert 1 in memory_usage
        assert memory_usage[0] > 0
        assert memory_usage[1] > 0
        
    def test_get_embedding_part_by_gpu(self, distribution_manager):
        """Test getting embedding part by GPU ID"""
        tensor1 = torch.randn(50, 128)
        tensor2 = torch.randn(50, 128)
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        
        # Test finding existing GPU
        result = distribution_manager.get_embedding_part_by_gpu(dist_emb, 1)
        assert result == part2
        
        # Test non-existing GPU
        result = distribution_manager.get_embedding_part_by_gpu(dist_emb, 5)
        assert result is None
        
    def test_cleanup_distribution(self, distribution_manager, mock_gpu_manager):
        """Test cleaning up distributed embeddings"""
        tensor1 = torch.randn(50, 128)
        tensor2 = torch.randn(50, 128)
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        distribution_manager.current_distribution = dist_emb
        
        distribution_manager.cleanup_distribution(dist_emb)
        
        # Verify GPU cleanup was called
        mock_gpu_manager.cleanup_gpu_resources.assert_called_once_with([0, 1])
        
        # Verify current distribution was cleared
        assert distribution_manager.current_distribution is None
        
    def test_get_distribution_summary(self, distribution_manager):
        """Test getting distribution summary"""
        tensor1 = torch.randn(50, 128)
        tensor2 = torch.randn(50, 128)
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        
        summary = distribution_manager.get_distribution_summary(dist_emb)
        
        assert summary['total_embeddings'] == 100
        assert summary['embedding_dimension'] == 128
        assert summary['num_gpus'] == 2
        assert summary['gpu_ids'] == [0, 1]
        assert len(summary['part_sizes']) == 2
        assert 'memory_usage_bytes' in summary
        assert 'memory_usage_mb' in summary
        
    def test_redistribute_if_needed_no_change(self, distribution_manager, mock_gpu_manager):
        """Test redistribution when no change is needed"""
        # Create mock tensors instead of real ones
        tensor1 = Mock()
        tensor1.device.type = 'cuda'
        tensor1.device.index = 0
        tensor1.size.side_effect = lambda dim: 50 if dim == 0 else 128
        tensor1.cpu.return_value = torch.randn(50, 128)
        
        tensor2 = Mock()
        tensor2.device.type = 'cuda'
        tensor2.device.index = 1
        tensor2.size.side_effect = lambda dim: 50 if dim == 0 else 128
        tensor2.cpu.return_value = torch.randn(50, 128)
        
        part1 = EmbeddingPart(0, tensor1, 0, 50)
        part2 = EmbeddingPart(1, tensor2, 50, 100)
        
        dist_emb = DistributedEmbeddings([part1, part2], 100, 128)
        
        # Mock validation to return True and available GPUs to include both GPUs
        mock_gpu_manager.get_available_gpu_ids.return_value = [0, 1]
        with patch.object(distribution_manager, 'validate_distribution', return_value=True):
            result = distribution_manager.redistribute_if_needed(dist_emb)
            
        # Should return the same distribution
        assert result == dist_emb
        
    def test_string_representations(self, distribution_manager, mock_gpu_manager):
        """Test string representations of the manager"""
        str_repr = str(distribution_manager)
        assert "EmbeddingDistributionManager" in str_repr
        
        repr_str = repr(distribution_manager)
        assert "EmbeddingDistributionManager" in repr_str
        assert "has_current_distribution=False" in repr_str


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])