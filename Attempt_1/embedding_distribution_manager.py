"""
Embedding Distribution Manager for Multi-GPU cuVS Implementation

This module provides safe embedding distribution across multiple GPUs with proper
bounds checking, validation, and error handling to prevent indexing errors.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging
from gpu_resource_manager import GPUResourceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPart:
    """Represents a part of embeddings assigned to a specific GPU"""
    gpu_id: int
    tensor: torch.Tensor
    start_index: int
    end_index: int
    
    def __post_init__(self):
        """Validate EmbeddingPart after initialization"""
        if self.start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {self.start_index}")
        if self.end_index <= self.start_index:
            raise ValueError(f"end_index ({self.end_index}) must be greater than start_index ({self.start_index})")
        if self.gpu_id < 0:
            raise ValueError(f"gpu_id must be non-negative, got {self.gpu_id}")
        if self.tensor.size(0) != (self.end_index - self.start_index):
            raise ValueError(f"Tensor size ({self.tensor.size(0)}) doesn't match index range ({self.end_index - self.start_index})")


@dataclass
class DistributedEmbeddings:
    """Container for embeddings distributed across multiple GPUs"""
    parts: List[EmbeddingPart]
    total_size: int
    embedding_dim: int
    
    def __post_init__(self):
        """Validate DistributedEmbeddings after initialization"""
        if not self.parts:
            raise ValueError("parts list cannot be empty")
        if self.total_size <= 0:
            raise ValueError(f"total_size must be positive, got {self.total_size}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
            
        # Validate that all parts have consistent embedding dimension
        for i, part in enumerate(self.parts):
            if part.tensor.size(1) != self.embedding_dim:
                raise ValueError(f"Part {i} has embedding_dim {part.tensor.size(1)}, expected {self.embedding_dim}")
                
        # Validate that parts cover the full range without gaps or overlaps
        sorted_parts = sorted(self.parts, key=lambda p: p.start_index)
        expected_start = 0
        for i, part in enumerate(sorted_parts):
            if part.start_index != expected_start:
                raise ValueError(f"Gap or overlap detected at part {i}: expected start {expected_start}, got {part.start_index}")
            expected_start = part.end_index
            
        if expected_start != self.total_size:
            raise ValueError(f"Parts don't cover full range: expected {self.total_size}, got {expected_start}")


class EmbeddingDistributionManager:
    """
    Manages safe distribution of embeddings across multiple GPUs.
    
    This class addresses embedding distribution issues in multi-GPU cuVS:
    - Safe embedding chunking with bounds checking
    - Validation of embedding parts against available GPUs
    - Redistribution when GPU availability changes
    - Proper error handling and recovery
    """
    
    def __init__(self, gpu_manager: GPUResourceManager):
        """
        Initialize EmbeddingDistributionManager.
        
        Args:
            gpu_manager: GPU Resource Manager instance
        """
        if not isinstance(gpu_manager, GPUResourceManager):
            raise TypeError("gpu_manager must be an instance of GPUResourceManager")
            
        self.gpu_manager = gpu_manager
        self.current_distribution: Optional[DistributedEmbeddings] = None
        
    def distribute_embeddings(self, embeddings: torch.Tensor, target_gpus: Optional[List[int]] = None) -> DistributedEmbeddings:
        """
        Distribute embeddings across available GPUs with proper bounds checking.
        
        Args:
            embeddings: Input embeddings tensor (N x D)
            target_gpus: Optional list of specific GPU IDs to use. If None, uses all available GPUs.
            
        Returns:
            DistributedEmbeddings: Container with embedding parts distributed across GPUs
            
        Raises:
            ValueError: If embeddings are invalid or target_gpus are not available
            RuntimeError: If no GPUs are available for distribution
        """
        # Validate input embeddings
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D tensor (N x D), got shape {embeddings.shape}")
        if embeddings.size(0) == 0:
            raise ValueError("embeddings tensor cannot be empty")
            
        total_embeddings = embeddings.size(0)
        embedding_dim = embeddings.size(1)
        
        # Determine target GPUs
        if target_gpus is None:
            target_gpus = self.gpu_manager.get_available_gpu_ids()
        else:
            # Validate target GPUs
            for gpu_id in target_gpus:
                if not self.gpu_manager.validate_gpu_index(gpu_id):
                    raise ValueError(f"Invalid target GPU: {gpu_id}")
                    
        if not target_gpus:
            raise RuntimeError("No GPUs available for embedding distribution")
            
        logger.info(f"Distributing {total_embeddings} embeddings across {len(target_gpus)} GPUs")
        
        # Get workload distribution
        try:
            distribution = self.gpu_manager.distribute_workload(total_embeddings, strategy='even')
            # Filter distribution to only include target GPUs
            filtered_distribution = [(gpu_id, start, end) for gpu_id, start, end in distribution if gpu_id in target_gpus]
            
            if not filtered_distribution:
                raise RuntimeError("No valid distribution found for target GPUs")
                
        except Exception as e:
            logger.error(f"Failed to get workload distribution: {str(e)}")
            raise RuntimeError(f"Failed to distribute workload: {str(e)}")
            
        # Create embedding parts
        embedding_parts = []
        
        try:
            for gpu_id, start_idx, end_idx in filtered_distribution:
                if start_idx >= end_idx:
                    logger.warning(f"Skipping empty range for GPU {gpu_id}: [{start_idx}, {end_idx})")
                    continue
                    
                # Validate bounds
                if start_idx < 0 or end_idx > total_embeddings:
                    raise ValueError(f"Invalid range [{start_idx}, {end_idx}) for {total_embeddings} embeddings")
                    
                # Extract embedding chunk
                embedding_chunk = embeddings[start_idx:end_idx].clone()
                
                # Move to target GPU
                device_string = self.gpu_manager.get_safe_device_string(gpu_id)
                embedding_chunk = embedding_chunk.to(device_string)
                
                # Create embedding part
                embedding_part = EmbeddingPart(
                    gpu_id=gpu_id,
                    tensor=embedding_chunk,
                    start_index=start_idx,
                    end_index=end_idx
                )
                
                embedding_parts.append(embedding_part)
                logger.info(f"Created embedding part for GPU {gpu_id}: [{start_idx}, {end_idx}) -> {embedding_chunk.shape}")
                
        except Exception as e:
            logger.error(f"Failed to create embedding parts: {str(e)}")
            # Cleanup any partial allocations
            self._cleanup_embedding_parts(embedding_parts)
            raise RuntimeError(f"Failed to distribute embeddings: {str(e)}")
            
        # Create distributed embeddings container
        try:
            distributed_embeddings = DistributedEmbeddings(
                parts=embedding_parts,
                total_size=total_embeddings,
                embedding_dim=embedding_dim
            )
            
            # Validate the distribution
            if not self.validate_distribution(distributed_embeddings):
                self._cleanup_embedding_parts(embedding_parts)
                raise RuntimeError("Distribution validation failed")
                
            self.current_distribution = distributed_embeddings
            logger.info(f"Successfully distributed embeddings across {len(embedding_parts)} GPUs")
            return distributed_embeddings
            
        except Exception as e:
            logger.error(f"Failed to create distributed embeddings: {str(e)}")
            self._cleanup_embedding_parts(embedding_parts)
            raise
            
    def validate_distribution(self, distributed_embeddings: DistributedEmbeddings) -> bool:
        """
        Validate that embedding distribution is correct and safe.
        
        Args:
            distributed_embeddings: Distributed embeddings to validate
            
        Returns:
            bool: True if distribution is valid, False otherwise
        """
        try:
            # Validate basic structure
            if not distributed_embeddings.parts:
                logger.error("No embedding parts found")
                return False
                
            # Validate GPU assignments
            for i, part in enumerate(distributed_embeddings.parts):
                if not self.gpu_manager.validate_gpu_index(part.gpu_id):
                    logger.error(f"Part {i} assigned to invalid GPU {part.gpu_id}")
                    return False
                    
                # Validate tensor is on correct device
                expected_device = f"cuda:{part.gpu_id}"
                if str(part.tensor.device) != expected_device:
                    logger.error(f"Part {i} tensor is on {part.tensor.device}, expected {expected_device}")
                    return False
                    
            # Validate index ranges
            sorted_parts = sorted(distributed_embeddings.parts, key=lambda p: p.start_index)
            expected_start = 0
            
            for i, part in enumerate(sorted_parts):
                if part.start_index != expected_start:
                    logger.error(f"Index gap at part {i}: expected {expected_start}, got {part.start_index}")
                    return False
                    
                if part.end_index <= part.start_index:
                    logger.error(f"Invalid range for part {i}: [{part.start_index}, {part.end_index})")
                    return False
                    
                expected_start = part.end_index
                
            if expected_start != distributed_embeddings.total_size:
                logger.error(f"Total size mismatch: expected {distributed_embeddings.total_size}, got {expected_start}")
                return False
                
            # Validate tensor shapes
            for i, part in enumerate(distributed_embeddings.parts):
                expected_rows = part.end_index - part.start_index
                if part.tensor.size(0) != expected_rows:
                    logger.error(f"Part {i} tensor has {part.tensor.size(0)} rows, expected {expected_rows}")
                    return False
                    
                if part.tensor.size(1) != distributed_embeddings.embedding_dim:
                    logger.error(f"Part {i} has embedding_dim {part.tensor.size(1)}, expected {distributed_embeddings.embedding_dim}")
                    return False
                    
            logger.info("Distribution validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Distribution validation failed with exception: {str(e)}")
            return False
            
    def redistribute_if_needed(self, distributed_embeddings: DistributedEmbeddings) -> DistributedEmbeddings:
        """
        Redistribute embeddings if GPU availability has changed.
        
        Args:
            distributed_embeddings: Current distributed embeddings
            
        Returns:
            DistributedEmbeddings: Redistributed embeddings or original if no change needed
        """
        # Check if current GPUs are still available
        current_gpus = [part.gpu_id for part in distributed_embeddings.parts]
        available_gpus = self.gpu_manager.get_available_gpu_ids()
        
        unavailable_gpus = [gpu_id for gpu_id in current_gpus if gpu_id not in available_gpus]
        
        if not unavailable_gpus:
            logger.info("All GPUs still available, no redistribution needed")
            return distributed_embeddings
            
        logger.warning(f"GPUs {unavailable_gpus} are no longer available, redistributing...")
        
        # Collect all embeddings back to CPU
        try:
            all_embeddings = self._collect_embeddings_to_cpu(distributed_embeddings)
            
            # Redistribute with currently available GPUs
            return self.distribute_embeddings(all_embeddings, target_gpus=available_gpus)
            
        except Exception as e:
            logger.error(f"Failed to redistribute embeddings: {str(e)}")
            raise RuntimeError(f"Redistribution failed: {str(e)}")
            
    def _collect_embeddings_to_cpu(self, distributed_embeddings: DistributedEmbeddings) -> torch.Tensor:
        """
        Collect distributed embeddings back to a single CPU tensor.
        
        Args:
            distributed_embeddings: Distributed embeddings to collect
            
        Returns:
            torch.Tensor: Combined embeddings on CPU
        """
        # Sort parts by start index
        sorted_parts = sorted(distributed_embeddings.parts, key=lambda p: p.start_index)
        
        # Collect tensors to CPU
        cpu_tensors = []
        for part in sorted_parts:
            cpu_tensor = part.tensor.cpu()
            cpu_tensors.append(cpu_tensor)
            
        # Concatenate all parts
        combined_embeddings = torch.cat(cpu_tensors, dim=0)
        
        # Validate combined shape
        expected_shape = (distributed_embeddings.total_size, distributed_embeddings.embedding_dim)
        if combined_embeddings.shape != expected_shape:
            raise RuntimeError(f"Combined embeddings shape {combined_embeddings.shape} != expected {expected_shape}")
            
        return combined_embeddings
        
    def _cleanup_embedding_parts(self, embedding_parts: List[EmbeddingPart]) -> None:
        """
        Clean up embedding parts and free GPU memory.
        
        Args:
            embedding_parts: List of embedding parts to clean up
        """
        gpu_ids_to_cleanup = set()
        
        for part in embedding_parts:
            try:
                # Move tensor to CPU to free GPU memory
                part.tensor = part.tensor.cpu()
                gpu_ids_to_cleanup.add(part.gpu_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup embedding part on GPU {part.gpu_id}: {str(e)}")
                
        # Clean up GPU resources
        if gpu_ids_to_cleanup:
            self.gpu_manager.cleanup_gpu_resources(list(gpu_ids_to_cleanup))
            
    def get_embedding_part_by_gpu(self, distributed_embeddings: DistributedEmbeddings, gpu_id: int) -> Optional[EmbeddingPart]:
        """
        Get embedding part for a specific GPU.
        
        Args:
            distributed_embeddings: Distributed embeddings container
            gpu_id: Target GPU ID
            
        Returns:
            EmbeddingPart or None if not found
        """
        for part in distributed_embeddings.parts:
            if part.gpu_id == gpu_id:
                return part
        return None
        
    def get_total_gpu_memory_usage(self, distributed_embeddings: DistributedEmbeddings) -> Dict[int, int]:
        """
        Get memory usage for each GPU in the distribution.
        
        Args:
            distributed_embeddings: Distributed embeddings container
            
        Returns:
            Dict mapping GPU ID to memory usage in bytes
        """
        memory_usage = {}
        
        for part in distributed_embeddings.parts:
            # Calculate tensor memory usage
            tensor_bytes = part.tensor.numel() * part.tensor.element_size()
            memory_usage[part.gpu_id] = tensor_bytes
            
        return memory_usage
        
    def cleanup_current_distribution(self) -> None:
        """Clean up the current distribution and free GPU memory."""
        if self.current_distribution is not None:
            self._cleanup_embedding_parts(self.current_distribution.parts)
            self.current_distribution = None
            logger.info("Cleaned up current distribution")
            
    def __str__(self) -> str:
        """String representation of EmbeddingDistributionManager"""
        current_parts = len(self.current_distribution.parts) if self.current_distribution else 0
        return f"EmbeddingDistributionManager(current_parts={current_parts})"
        
    def __repr__(self) -> str:
        """Detailed representation of EmbeddingDistributionManager"""
        return (f"EmbeddingDistributionManager(gpu_manager={self.gpu_manager}, "
                f"current_distribution={self.current_distribution is not None})")