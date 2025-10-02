# Improved Multi-GPU RAG Implementation

## Overview
This improved implementation addresses critical performance issues in the original multi-GPU RAG system:

### Key Improvements
1. **Parallel GPU Execution**: Uses ThreadPoolExecutor for concurrent GPU operations
2. **Top-2K Retrieval**: Changed from top-5 to top-2000 for better recall benchmarking
3. **Enhanced Error Handling**: Proper CUDA OOM handling and retry logic
4. **FAISS Comparison**: Benchmark against FAISS for performance validation
5. **Memory Management**: Better GPU memory tracking and cleanup

## Files
- `improved_multi_gpu_rag.py`: Main improved implementation with parallel execution
- `colab_a100_test.ipynb`: Google Colab notebook for A100 testing
- `submit_narval_job.sh`: SLURM job script for Narval cluster
- `build_instructions_for_FAISS.md`: Instructions for building FAISS on Narval

## Performance Improvements

### Before (Sequential)
```python
# Sequential GPU execution
for i, part_embedding in enumerate(parts):
    index = build_index(part_embedding)  # Blocks until complete
```

### After (Parallel)
```python
# Parallel GPU execution
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    futures = [executor.submit(build_index, part) for part in parts]
    results = [f.result() for f in futures]  # All GPUs work simultaneously
```

### Expected Performance Gains
- **40-60% faster** index building with parallel GPU execution
- **Better GPU utilization** (both GPUs working simultaneously)
- **Improved recall evaluation** with top-2K retrieval
- **Production-ready** error handling and recovery

## Usage

### Google Colab (A100)
1. Open `colab_a100_test.ipynb` in Google Colab
2. Select Runtime > Change runtime type > A100 GPU
3. Run all cells to test the implementation
4. Results will be saved as CSV files

### Narval Cluster
1. Edit `submit_narval_job.sh`:
   - Change `--account=def-yourgroup` to your allocation
   - Update email address for notifications

2. Submit the job:
```bash
sbatch submit_narval_job.sh
```

3. Monitor job status:
```bash
squeue -u $USER
```

4. Check results:
```bash
cat rag_parallel_*.out
cat narval_benchmark_results.csv
```

### Local Testing
```bash
# Install dependencies
pip install torch pylibraft-cu12 cuvs-cu12 sentence-transformers numpy pandas matplotlib

# Run the improved implementation
python improved_multi_gpu_rag.py
```

## Benchmark Results (Expected)

| Index Type | Build Time | Search Time (top-2K) | Memory Usage |
|------------|------------|---------------------|--------------|
| IVF-Flat   | ~5s        | ~15-20ms           | ~15GB        |
| IVF-PQ     | ~8s        | ~10-15ms           | ~8GB         |
| CAGRA      | ~12s       | ~5-10ms            | ~20GB        |
| FAISS-Flat | ~3s        | ~25-30ms           | ~15GB        |
| FAISS-IVF  | ~6s        | ~20-25ms           | ~12GB        |

## Configuration

### Search Configuration
```python
SearchConfig(
    top_k=2000,           # Number of neighbors to retrieve
    search_batch_size=100, # Batch size for queries
    num_queries=100,       # Number of test queries
    enable_recall_eval=True,
    recall_k_values=[1, 5, 10, 50, 100, 500, 1000, 2000]
)
```

### GPU Configuration
```python
GPUConfig(
    device_id=0,
    memory_limit_gb=40.0,  # A100 has 40GB
    reserved_memory_gb=2.0  # Reserve for safety
)
```

## Recall Evaluation

The implementation now properly evaluates recall at multiple K values:
- Recall@1, @5, @10, @50, @100, @500, @1000, @2000
- Synthetic ground truth generation for testing
- Comparison with exact search results

## Error Handling

### CUDA Out of Memory
- Automatic cache clearing and garbage collection
- Graceful degradation to smaller batch sizes
- Memory monitoring before operations

### Failed GPU Operations
- Retry logic with exponential backoff
- Fallback to remaining GPUs if one fails
- Detailed error logging and reporting

## Memory Management

### Before Operation
```python
with CUDAMemoryManager.managed_allocation(gpu_config, "building index"):
    # Operation code here
    # Automatic cleanup on exit
```

### Memory Monitoring
```python
stats = get_memory_stats()
# Returns RAM usage, GPU memory per device
```

## Troubleshooting

### CUDA Out of Memory
1. Reduce batch size in SearchConfig
2. Use IVF-PQ instead of IVF-Flat for compression
3. Reduce embedding dimensions if possible
4. Clear cache: `torch.cuda.empty_cache()`

### Slow Performance
1. Ensure parallel execution is working (check logs)
2. Verify both GPUs are being utilized
3. Adjust index parameters (n_lists, pq_dim)
4. Profile with nvprof or nsys

### Import Errors
1. Ensure CUDA 12.x is installed
2. Install correct versions: `pip install pylibraft-cu12 cuvs-cu12`
3. For Narval: Load required modules first

## Future Improvements
1. Implement CUDA streams for overlapping compute/transfer
2. Add distributed multi-node support
3. Implement dynamic batch sizing based on memory
4. Add quantization for further memory reduction
5. Integrate with production serving framework

## Citation
If you use this implementation, please cite:
```
Improved Multi-GPU RAG Implementation
2024, University of Victoria
```

## Support
For issues or questions:
- Check error logs in `rag_parallel_*.err`
- Monitor GPU usage with `nvidia-smi`
- Review memory stats in output files