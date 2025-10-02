"""
Unit tests for GPU Resource Manager

Tests GPU discovery, validation, safe indexing, and resource management
to ensure the multi-GPU cuVS implementation works correctly.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from gpu_resource_manager import GPUResourceManager, GPUConfig, MultiGPUConfig


class TestGPUResourceManager(unittest.TestCase):
    """Test cases for GPU Resource Manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_gpu_props = MagicMock()
        self.mock_gpu_props.name = "Tesla T4"
        self.mock_gpu_props.total_memory = 16 * 1024**3  # 16GB
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_gpu_discovery_no_cuda(self, mock_device_count, mock_is_available):
        """Test GPU discovery when CUDA is not available"""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        manager = GPUResourceManager()
        
        self.assertEqual(len(manager.available_gpus), 0)
        self.assertEqual(len(manager.gpu_configs), 0)
        self.assertEqual(manager.get_available_gpu_count(), 0)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.device')
    def test_gpu_discovery_single_gpu(self, mock_device, mock_empty_cache, mock_memory_allocated, 
                                     mock_get_props, mock_device_count, mock_is_available):
        """Test GPU discovery with single GPU"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_props.return_value = self.mock_gpu_props
        mock_memory_allocated.return_value = 1024**3  # 1GB allocated
        mock_device.return_value.__enter__ = MagicMock()
        mock_device.return_value.__exit__ = MagicMock()
        
        manager = GPUResourceManager()
        
        self.assertEqual(len(manager.available_gpus), 1)
        self.assertEqual(manager.available_gpus[0], 0)
        self.assertEqual(len(manager.gpu_configs), 1)
        
        gpu_config = manager.gpu_configs[0]
        self.assertEqual(gpu_config.gpu_id, 0)
        self.assertEqual(gpu_config.device_name, "Tesla T4")
        self.assertTrue(gpu_config.is_available)
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.device')
    def test_gpu_discovery_multiple_gpus(self, mock_device, mock_empty_cache, mock_memory_allocated,
                                        mock_get_props, mock_device_count, mock_is_available):
        """Test GPU discovery with multiple GPUs"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        mock_get_props.return_value = self.mock_gpu_props
        mock_memory_allocated.return_value = 1024**3  # 1GB allocated
        mock_device.return_value.__enter__ = MagicMock()
        mock_device.return_value.__exit__ = MagicMock()
        
        manager = GPUResourceManager()
        
        self.assertEqual(len(manager.available_gpus), 2)
        self.assertEqual(manager.available_gpus, [0, 1])
        self.assertEqual(len(manager.gpu_configs), 2)
        self.assertEqual(manager.get_available_gpu_count(), 2)
        
    def test_validate_gpu_index_negative(self):
        """Test validation of negative GPU index"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        self.assertFalse(manager.validate_gpu_index(-1))
        
    def test_validate_gpu_index_not_available(self):
        """Test validation of GPU index not in available list"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        self.assertFalse(manager.validate_gpu_index(2))
        
    @patch('torch.cuda.is_available')
    def test_validate_gpu_index_no_cuda(self, mock_is_available):
        """Test validation when CUDA is not available"""
        mock_is_available.return_value = False
        
        manager = GPUResourceManager()
        manager.available_gpus = [0]  # Simulate having GPUs in list but CUDA unavailable
        
        self.assertFalse(manager.validate_gpu_index(0))
        
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_validate_gpu_index_exceeds_count(self, mock_device_count, mock_is_available):
        """Test validation when GPU index exceeds device count"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1, 2]  # Simulate inconsistent state
        
        self.assertFalse(manager.validate_gpu_index(2))
        
    def test_validate_gpu_index_valid(self):
        """Test validation of valid GPU index"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            self.assertTrue(manager.validate_gpu_index(0))
            self.assertTrue(manager.validate_gpu_index(1))
            
    def test_get_safe_device_string_valid(self):
        """Test getting device string for valid GPU"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(manager.get_safe_device_string(0), 'cuda:0')
            self.assertEqual(manager.get_safe_device_string(1), 'cuda:1')
            
    def test_get_safe_device_string_invalid(self):
        """Test getting device string for invalid GPU raises error"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with self.assertRaises(ValueError) as context:
            manager.get_safe_device_string(2)
            
        self.assertIn("Invalid GPU index: 2", str(context.exception))
        self.assertIn("Available GPUs: [0, 1]", str(context.exception))
        
    def test_distribute_workload_even_strategy(self):
        """Test even workload distribution"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1, 2]
        
        # Test with evenly divisible workload
        distribution = manager.distribute_workload(300, strategy='even')
        expected = [(0, 0, 100), (1, 100, 200), (2, 200, 300)]
        self.assertEqual(distribution, expected)
        
        # Test with remainder
        distribution = manager.distribute_workload(301, strategy='even')
        expected = [(0, 0, 101), (1, 101, 201), (2, 201, 301)]
        self.assertEqual(distribution, expected)
        
    def test_distribute_workload_memory_based_strategy(self):
        """Test memory-based workload distribution"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        manager.gpu_memory_info = {
            0: {'available': 8 * 1024**3},  # 8GB
            1: {'available': 16 * 1024**3}  # 16GB
        }
        
        distribution = manager.distribute_workload(300, strategy='memory_based')
        
        # GPU 1 should get twice as much work as GPU 0 (16GB vs 8GB)
        self.assertEqual(len(distribution), 2)
        gpu0_items = distribution[0][2] - distribution[0][1]  # end - start
        gpu1_items = distribution[1][2] - distribution[1][1]
        
        # GPU 1 should get approximately twice as many items
        self.assertAlmostEqual(gpu1_items / gpu0_items, 2.0, delta=0.1)
        
        # Total should equal input
        total_distributed = sum(end - start for _, start, end in distribution)
        self.assertEqual(total_distributed, 300)
        
    def test_distribute_workload_no_gpus(self):
        """Test workload distribution with no GPUs raises error"""
        manager = GPUResourceManager()
        manager.available_gpus = []
        
        with self.assertRaises(RuntimeError) as context:
            manager.distribute_workload(100)
            
        self.assertIn("No GPUs available", str(context.exception))
        
    def test_distribute_workload_invalid_items(self):
        """Test workload distribution with invalid item count"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with self.assertRaises(ValueError):
            manager.distribute_workload(0)
            
        with self.assertRaises(ValueError):
            manager.distribute_workload(-10)
            
    def test_distribute_workload_unknown_strategy(self):
        """Test workload distribution with unknown strategy"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with self.assertRaises(ValueError) as context:
            manager.distribute_workload(100, strategy='unknown')
            
        self.assertIn("Unknown distribution strategy", str(context.exception))
        
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.device')
    def test_cleanup_gpu_resources(self, mock_device, mock_synchronize, mock_empty_cache):
        """Test GPU resource cleanup"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            manager.cleanup_gpu_resources([0, 1])
            
            # Should call empty_cache and synchronize for each GPU
            self.assertEqual(mock_empty_cache.call_count, 2)
            self.assertEqual(mock_synchronize.call_count, 2)
            
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.device')
    def test_cleanup_all_gpu_resources(self, mock_device, mock_synchronize, mock_empty_cache):
        """Test cleanup of all GPU resources"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        mock_device.return_value.__enter__ = MagicMock()
        mock_device.return_value.__exit__ = MagicMock()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            manager.cleanup_gpu_resources()  # No specific GPUs provided
            
            self.assertEqual(mock_empty_cache.call_count, 2)
            self.assertEqual(mock_synchronize.call_count, 2)
            
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device')
    def test_get_gpu_memory_info(self, mock_device, mock_get_props, mock_memory_reserved, mock_memory_allocated):
        """Test getting GPU memory information"""
        manager = GPUResourceManager()
        manager.available_gpus = [0]
        
        mock_memory_allocated.return_value = 2 * 1024**3  # 2GB
        mock_memory_reserved.return_value = 4 * 1024**3   # 4GB
        mock_get_props.return_value = self.mock_gpu_props
        mock_device.return_value.__enter__ = MagicMock()
        mock_device.return_value.__exit__ = MagicMock()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            
            memory_info = manager.get_gpu_memory_info(0)
            
            self.assertEqual(memory_info['allocated'], 2 * 1024**3)
            self.assertEqual(memory_info['reserved'], 4 * 1024**3)
            self.assertEqual(memory_info['total'], 16 * 1024**3)
            self.assertEqual(memory_info['free'], 12 * 1024**3)  # total - reserved
            
    def test_get_gpu_memory_info_invalid_gpu(self):
        """Test getting memory info for invalid GPU"""
        manager = GPUResourceManager()
        manager.available_gpus = [0]
        
        with self.assertRaises(ValueError):
            manager.get_gpu_memory_info(1)
            
    def test_get_multi_gpu_config(self):
        """Test getting multi-GPU configuration"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        manager.gpu_configs = [
            GPUConfig(0, "Tesla T4", 16*1024**3, 15*1024**3, True),
            GPUConfig(1, "Tesla T4", 16*1024**3, 15*1024**3, True)
        ]
        
        config = manager.get_multi_gpu_config('even')
        
        self.assertIsInstance(config, MultiGPUConfig)
        self.assertEqual(len(config.available_gpus), 2)
        self.assertEqual(config.primary_gpu, 0)
        self.assertEqual(config.distribution_strategy, 'even')
        
    def test_get_multi_gpu_config_no_gpus(self):
        """Test getting multi-GPU config with no GPUs"""
        manager = GPUResourceManager()
        manager.available_gpus = []
        manager.gpu_configs = []
        
        config = manager.get_multi_gpu_config()
        
        self.assertEqual(config.primary_gpu, -1)
        self.assertEqual(len(config.available_gpus), 0)
        
    def test_validate_tensor_distribution_valid(self):
        """Test validation of valid tensor distribution"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        # Mock tensors on correct devices
        mock_tensor_0 = MagicMock()
        mock_tensor_0.device.index = 0
        mock_tensor_1 = MagicMock()
        mock_tensor_1.device.index = 1
        
        tensor_parts = [mock_tensor_0, mock_tensor_1]
        
        self.assertTrue(manager.validate_tensor_distribution(tensor_parts))
        
    def test_validate_tensor_distribution_count_mismatch(self):
        """Test validation with tensor count mismatch"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        mock_tensor = MagicMock()
        tensor_parts = [mock_tensor]  # Only 1 tensor for 2 GPUs
        
        self.assertFalse(manager.validate_tensor_distribution(tensor_parts))
        
    def test_validate_tensor_distribution_wrong_device(self):
        """Test validation with tensor on wrong device"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        
        mock_tensor_0 = MagicMock()
        mock_tensor_0.device.index = 0
        mock_tensor_1 = MagicMock()
        mock_tensor_1.device.index = 2  # Wrong device
        
        tensor_parts = [mock_tensor_0, mock_tensor_1]
        
        self.assertFalse(manager.validate_tensor_distribution(tensor_parts))
        
    def test_string_representations(self):
        """Test string representations of GPU Resource Manager"""
        manager = GPUResourceManager()
        manager.available_gpus = [0, 1]
        manager.gpu_configs = [MagicMock(), MagicMock()]
        
        str_repr = str(manager)
        self.assertIn("available_gpus=[0, 1]", str_repr)
        self.assertIn("gpu_count=2", str_repr)
        
        repr_str = repr(manager)
        self.assertIn("available_gpus=[0, 1]", repr_str)
        self.assertIn("gpu_configs=2", repr_str)


class TestGPUConfig(unittest.TestCase):
    """Test cases for GPUConfig dataclass"""
    
    def test_gpu_config_creation(self):
        """Test GPUConfig creation and attributes"""
        config = GPUConfig(
            gpu_id=0,
            device_name="Tesla T4",
            total_memory=16 * 1024**3,
            available_memory=15 * 1024**3,
            is_available=True
        )
        
        self.assertEqual(config.gpu_id, 0)
        self.assertEqual(config.device_name, "Tesla T4")
        self.assertEqual(config.total_memory, 16 * 1024**3)
        self.assertEqual(config.available_memory, 15 * 1024**3)
        self.assertTrue(config.is_available)


class TestMultiGPUConfig(unittest.TestCase):
    """Test cases for MultiGPUConfig dataclass"""
    
    def test_multi_gpu_config_creation(self):
        """Test MultiGPUConfig creation and attributes"""
        gpu_configs = [
            GPUConfig(0, "Tesla T4", 16*1024**3, 15*1024**3, True),
            GPUConfig(1, "Tesla T4", 16*1024**3, 15*1024**3, True)
        ]
        
        config = MultiGPUConfig(
            available_gpus=gpu_configs,
            primary_gpu=0,
            distribution_strategy='even'
        )
        
        self.assertEqual(len(config.available_gpus), 2)
        self.assertEqual(config.primary_gpu, 0)
        self.assertEqual(config.distribution_strategy, 'even')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)