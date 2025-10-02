# Design Document

## Overview

The multi-GPU cuVS implementation currently suffers from critical indexing errors that prevent successful vector search operations. The primary issue is inconsistent GPU index management between embedding distribution, index building, and search operations. This design addresses these issues through robust GPU resource management, proper bounds checking, and consistent indexing patterns.

## Architecture

### Current Problem Analysis

The current implementation has several architectural flaws:

1. **Inconsistent GPU Indexing**: The code assumes GPU indices match array indices without validation
2. **Missing Bounds Checking**: No verification that requested GPU indices exist
3. **Fragile Resource Management**: GPU cleanup and memory management lacks proper error handling
4. **Unsafe Array Operations**: Direct array access without bounds validation

### Proposed Architecture

The solution implements a **GPU Resource Manager** pattern with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Resource Manager                     │
├─────────────────────────────────────────────────────────────┤
│  • GPU Discovery & Validation                               │
│  • Safe Index Mapping                                       │
│  • Resource Lifecycle Management                            │
│  • Error Recovery & Cleanup                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Multi-GPU Operations Layer                  │
├─────────────────────────────────────────────────────────────┤
│  • Embedding Distribution Manager                           │
│  • Index Building Coordinator                               │
│  • Search Result Aggregator                                 │
│  • Memory Management Controller                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    cuVS Operations                          │
├─────────────────────────────────────────────────────────────┤
│  • IVF-FLAT Index Operations                                │
│  • IVF-PQ Index Operations                                  │
│  • CAGRA Index Operations                                   │
│  • Search Parameter Management                              │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. GPU Resource Manager

**Purpose**: Centralized management of GPU resources and safe indexing

**Key Methods**:
```python
class GPUResourceManager:
    def __init__(self):
        self.available_gpus: List[int]
        self.gpu_memory_info: Dict[int, Dict]
        
    def validate_gpu_index(self, gpu_id: int) -> bool
    def get_safe_device_string(self, gpu_id: int) -> str
    def distribute_workload(self, total_items: int) -> List[Tuple[int, int]]
    def cleanup_gpu_resources(self, gpu_ids: List[int]) -> None
```

**Responsibilities**:
- Discover and validate available GPUs
- Provide safe GPU index mapping
- Manage GPU memory allocation tracking
- Handle resource cleanup and error recovery

### 2. Embedding Distribution Manager

**Purpose**: Safe distribution of embeddings across available GPUs

**Key Methods**:
```python
class EmbeddingDistributionManager:
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        
    def distribute_embeddings(self, embeddings: torch.Tensor, target_gpus: List[int]) -> List[torch.Tensor]
    def validate_distribution(self, embedding_parts: List[torch.Tensor]) -> bool
    def redistribute_if_needed(self, embedding_parts: List[torch.Tensor]) -> List[torch.Tensor]
```

**Responsibilities**:
- Safely chunk embeddings for multi-GPU distribution
- Validate that embedding parts match available GPUs
- Handle redistribution when GPU availability changes

### 3. Index Building Coordinator

**Purpose**: Coordinate index building across multiple GPUs with proper error handling

**Key Methods**:
```python
class IndexBuildingCoordinator:
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        self.built_indices: Dict[int, Any] = {}
        
    def build_indices_parallel(self, embedding_parts: List[torch.Tensor], index_type: str) -> Dict[int, Any]
    def validate_index_build(self, gpu_id: int, index: Any) -> bool
    def cleanup_failed_builds(self, failed_gpu_ids: List[int]) -> None
```

**Responsibilities**:
- Build indices on multiple GPUs with proper error handling
- Track successful and failed index builds
- Provide rollback capabilities for failed operations

### 4. Search Result Aggregator

**Purpose**: Safely aggregate search results from multiple GPUs

**Key Methods**:
```python
class SearchResultAggregator:
    def __init__(self, gpu_manager: GPUResourceManager):
        self.gpu_manager = gpu_manager
        
    def perform_distributed_search(self, query: torch.Tensor, indices: Dict[int, Any], k: int) -> Tuple[np.ndarray, np.ndarray]
    def merge_search_results(self, results: List[Tuple[np.ndarray, np.ndarray]], k: int) -> Tuple[np.ndarray, np.ndarray]
    def validate_search_results(self, results: List[Tuple[np.ndarray, np.ndarray]]) -> bool
```

**Responsibilities**:
- Execute searches across multiple GPUs safely
- Merge and rank results from all GPUs
- Handle empty or invalid search results gracefully

## Data Models

### GPU Configuration Model
```python
@dataclass
class GPUConfig:
    gpu_id: int
    device_name: str
    total_memory: int
    available_memory: int
    is_available: bool
    
@dataclass
class MultiGPUConfig:
    available_gpus: List[GPUConfig]
    primary_gpu: int
    distribution_strategy: str  # 'even', 'memory_based', 'custom'
```

### Embedding Distribution Model
```python
@dataclass
class EmbeddingPart:
    gpu_id: int
    tensor: torch.Tensor
    start_index: int
    end_index: int
    
@dataclass
class DistributedEmbeddings:
    parts: List[EmbeddingPart]
    total_size: int
    embedding_dim: int
```

### Search Result Model
```python
@dataclass
class SearchResult:
    distances: np.ndarray
    indices: np.ndarray
    gpu_id: int
    query_time: float
    
@dataclass
class AggregatedSearchResult:
    final_distances: np.ndarray
    final_indices: np.ndarray
    total_query_time: float
    gpu_results: List[SearchResult]
```

## Error Handling

### GPU Index Validation
- **Pre-operation Validation**: All GPU indices validated before any operations
- **Runtime Bounds Checking**: Array access operations include bounds validation
- **Graceful Degradation**: System continues with available GPUs if some fail

### Memory Management
- **Allocation Tracking**: Track all GPU memory allocations
- **Automatic Cleanup**: Ensure cleanup on both success and failure paths
- **Memory Pressure Handling**: Detect and respond to out-of-memory conditions

### Search Operation Safety
- **Result Validation**: Validate search results before merging
- **Empty Result Handling**: Handle cases where GPUs return no results
- **Data Type Consistency**: Ensure consistent data types across GPU operations

## Testing Strategy

### Unit Tests
1. **GPU Resource Manager Tests**
   - Test GPU discovery and validation
   - Test safe index mapping with various GPU configurations
   - Test resource cleanup under normal and error conditions

2. **Embedding Distribution Tests**
   - Test embedding chunking with different sizes and GPU counts
   - Test redistribution when GPU availability changes
   - Test validation of distributed embeddings

3. **Index Building Tests**
   - Test parallel index building across multiple GPUs
   - Test error handling when individual GPU builds fail
   - Test cleanup of partial builds

4. **Search Aggregation Tests**
   - Test result merging with various result sizes
   - Test handling of empty results from individual GPUs
   - Test global top-k selection accuracy

### Integration Tests
1. **End-to-End Multi-GPU Pipeline**
   - Test complete workflow from embedding distribution to search results
   - Test with different vector sizes and GPU configurations
   - Test error recovery and graceful degradation

2. **Memory Management Integration**
   - Test memory usage patterns across different workloads
   - Test cleanup effectiveness after operations
   - Test behavior under memory pressure

3. **Performance Validation**
   - Verify that fixes don't significantly impact performance
   - Test scaling behavior with different GPU counts
   - Validate search result accuracy across implementations

### Error Scenario Tests
1. **GPU Failure Simulation**
   - Test behavior when individual GPUs become unavailable
   - Test recovery from partial operation failures
   - Test system behavior with changing GPU availability

2. **Memory Exhaustion Tests**
   - Test behavior when GPU memory is exhausted
   - Test cleanup and recovery from memory allocation failures
   - Test graceful degradation under memory pressure

3. **Data Corruption Tests**
   - Test handling of corrupted embedding data
   - Test validation of search results
   - Test error propagation and reporting