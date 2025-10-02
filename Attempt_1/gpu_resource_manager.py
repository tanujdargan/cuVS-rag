"""
GPU Resource Manager for Multi-GPU cuVS Implementation

This module provides robust GPU resource management with proper indexing,
validation, and error handling to prevent out-of-bounds errors in multi-GPU
vector search operations.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Configuration information for a single GPU"""
    gpu_id: int
    device_name: str
    total_memory: int
    available_memory: int
    is_available: bool


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU setup"""
    available_gpus: List[GPUConfig]
    primary_gpu: int
    distribution_strategy: str  # 'even', 'memory_based', 'custom'


class GPUResourceManager:
    """
    Centralized management of GPU resources with safe indexing and validation.
    
    This class addresses the core issues in the multi-GPU cuVS implementation:
    - Proper GPU discovery and validation
    - Safe GPU index mapping
    - Resource lifecycle management
    - Error recovery and cleanup
    """
    
    def __init__(self):
        """Initialize GPU Resource Manager with automatic GPU discovery"""
        self.available_gpus: List[int] = []
        self.gpu_memory_info: Dict[int, Dict] = {}
        self.gpu_configs: List[GPUConfig] = []
        self._discover_gpus()
        
    def _discover_gpus(self) -> None:
        """Discover and validate available GPUs"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. No GPUs detected.")
                return
                
            gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {gpu_count} GPU(s)")
            
            for gpu_id in range(gpu_count):
                try:
                    # Test GPU accessibility
                    with torch.cuda.device(gpu_id):
                        # Get GPU properties
                        props = torch.cuda.get_device_properties(gpu_id)
                        total_memory = props.total_memory
                        
                        # Get current memory usage
                        torch.cuda.empty_cache()  # Clear cache for accurate reading
                        allocated_memory = torch.cuda.memory_allocated(gpu_id)
                        available_memory = total_memory - allocated_memory
                        
                        # Create GPU config
                        gpu_config = GPUConfig(
                            gpu_id=gpu_id,
                            device_name=props.name,
                            total_memory=total_memory,
                            available_memory=available_memory,
                            is_available=True
                        )
                        
                        self.gpu_configs.append(gpu_config)
                        self.available_gpus.append(gpu_id)
                        self.gpu_memory_info[gpu_id] = {
                            'total': total_memory,
                            'available': available_memory,
                            'allocated': allocated_memory
                        }
                        
                        logger.info(f"GPU {gpu_id}: {props.name} - "
                                  f"Total: {total_memory / 1024**3:.1f}GB, "
                                  f"Available: {available_memory / 1024**3:.1f}GB")
                        
                except Exception as e:
                    logger.warning(f"GPU {gpu_id} is not accessible: {str(e)}")
                    # Create unavailable GPU config for tracking
                    gpu_config = GPUConfig(
                        gpu_id=gpu_id,
                        device_name="Unknown",
                        total_memory=0,
                        available_memory=0,
                        is_available=False
                    )
                    self.gpu_configs.append(gpu_config)
                    
        except Exception as e:
            logger.error(f"Failed to discover GPUs: {str(e)}")
            
    def validate_gpu_index(self, gpu_id: int) -> bool:
        """
        Validate that a GPU index is available and accessible.
        
        Args:
            gpu_id: GPU index to validate
            
        Returns:
            bool: True if GPU is valid and available, False otherwise
        """
        if gpu_id < 0:
            logger.error(f"Invalid GPU index: {gpu_id} (negative index)")
            return False
            
        if gpu_id not in self.available_gpus:
            logger.error(f"GPU {gpu_id} is not in available GPUs list: {self.available_gpus}")
            return False
            
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False
            
        if gpu_id >= torch.cuda.device_count():
            logger.error(f"GPU {gpu_id} exceeds available GPU count: {torch.cuda.device_count()}")
            return False
            
        return True
        
    def get_safe_device_string(self, gpu_id: int) -> str:
        """
        Get a safe device string for the given GPU ID.
        
        Args:
            gpu_id: GPU index
            
        Returns:
            str: Device string (e.g., 'cuda:0') or 'cpu' if GPU is invalid
            
        Raises:
            ValueError: If GPU index is invalid and no fallback is desired
        """
        if self.validate_gpu_index(gpu_id):
            return f'cuda:{gpu_id}'
        else:
            raise ValueError(f"Invalid GPU index: {gpu_id}. Available GPUs: {self.available_gpus}")
            
    def get_available_gpu_count(self) -> int:
        """Get the number of available GPUs"""
        return len(self.available_gpus)
        
    def get_available_gpu_ids(self) -> List[int]:
        """Get list of available GPU IDs"""
        return self.available_gpus.copy()
        
    def distribute_workload(self, total_items: int, strategy: str = 'even') -> List[Tuple[int, int, int]]:
        """
        Distribute workload across available GPUs.
        
        Args:
            total_items: Total number of items to distribute
            strategy: Distribution strategy ('even', 'memory_based')
            
        Returns:
            List of tuples (gpu_id, start_index, end_index) for each GPU
        """
        if not self.available_gpus:
            raise RuntimeError("No GPUs available for workload distribution")
            
        if total_items <= 0:
            raise ValueError(f"Invalid total_items: {total_items}")
            
        gpu_count = len(self.available_gpus)
        distribution = []
        
        if strategy == 'even':
            # Distribute evenly across GPUs
            items_per_gpu = total_items // gpu_count
            remainder = total_items % gpu_count
            
            start_idx = 0
            for i, gpu_id in enumerate(self.available_gpus):
                # Add one extra item to first 'remainder' GPUs
                current_items = items_per_gpu + (1 if i < remainder else 0)
                end_idx = start_idx + current_items
                
                distribution.append((gpu_id, start_idx, end_idx))
                start_idx = end_idx
                
        elif strategy == 'memory_based':
            # Distribute based on available memory
            total_memory = sum(self.gpu_memory_info[gpu_id]['available'] 
                             for gpu_id in self.available_gpus)
            
            start_idx = 0
            for gpu_id in self.available_gpus:
                memory_ratio = self.gpu_memory_info[gpu_id]['available'] / total_memory
                items_for_gpu = int(total_items * memory_ratio)
                
                # Ensure we don't exceed total_items
                if start_idx + items_for_gpu > total_items:
                    items_for_gpu = total_items - start_idx
                    
                end_idx = start_idx + items_for_gpu
                distribution.append((gpu_id, start_idx, end_idx))
                start_idx = end_idx
                
                if start_idx >= total_items:
                    break
                    
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")
            
        # Validate distribution
        total_distributed = sum(end - start for _, start, end in distribution)
        if total_distributed != total_items:
            logger.warning(f"Distribution mismatch: {total_distributed} != {total_items}")
            
        return distribution
        
    def cleanup_gpu_resources(self, gpu_ids: Optional[List[int]] = None) -> None:
        """
        Clean up GPU resources and memory.
        
        Args:
            gpu_ids: List of GPU IDs to clean up. If None, clean all available GPUs.
        """
        target_gpus = gpu_ids if gpu_ids is not None else self.available_gpus
        
        for gpu_id in target_gpus:
            if self.validate_gpu_index(gpu_id):
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.info(f"Cleaned up GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup GPU {gpu_id}: {str(e)}")
                    
        # Force garbage collection
        gc.collect()
        
    def get_gpu_memory_info(self, gpu_id: int) -> Dict[str, int]:
        """
        Get current memory information for a specific GPU.
        
        Args:
            gpu_id: GPU index
            
        Returns:
            Dict with memory information
        """
        if not self.validate_gpu_index(gpu_id):
            raise ValueError(f"Invalid GPU index: {gpu_id}")
            
        try:
            with torch.cuda.device(gpu_id):
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                
                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'free': total - reserved
                }
        except Exception as e:
            logger.error(f"Failed to get memory info for GPU {gpu_id}: {str(e)}")
            return {'allocated': 0, 'reserved': 0, 'total': 0, 'free': 0}
            
    def get_multi_gpu_config(self, strategy: str = 'even') -> MultiGPUConfig:
        """
        Get multi-GPU configuration.
        
        Args:
            strategy: Distribution strategy
            
        Returns:
            MultiGPUConfig object
        """
        primary_gpu = self.available_gpus[0] if self.available_gpus else -1
        
        return MultiGPUConfig(
            available_gpus=self.gpu_configs.copy(),
            primary_gpu=primary_gpu,
            distribution_strategy=strategy
        )
        
    def validate_tensor_distribution(self, tensor_parts: List[torch.Tensor]) -> bool:
        """
        Validate that tensor parts match available GPU configuration.
        
        Args:
            tensor_parts: List of tensor parts
            
        Returns:
            bool: True if distribution is valid
        """
        if len(tensor_parts) != len(self.available_gpus):
            logger.error(f"Tensor parts count ({len(tensor_parts)}) != available GPUs ({len(self.available_gpus)})")
            return False
            
        for i, tensor_part in enumerate(tensor_parts):
            expected_gpu = self.available_gpus[i]
            if tensor_part.device.index != expected_gpu:
                logger.error(f"Tensor part {i} is on GPU {tensor_part.device.index}, expected GPU {expected_gpu}")
                return False
                
        return True
        
    def __str__(self) -> str:
        """String representation of GPU Resource Manager"""
        return f"GPUResourceManager(available_gpus={self.available_gpus}, gpu_count={len(self.available_gpus)})"
        
    def __repr__(self) -> str:
        """Detailed representation of GPU Resource Manager"""
        return (f"GPUResourceManager(available_gpus={self.available_gpus}, "
                f"gpu_configs={len(self.gpu_configs)}, "
                f"cuda_available={torch.cuda.is_available()})")