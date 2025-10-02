#!/bin/bash
#SBATCH --job-name=rag_parallel_gpu
#SBATCH --account=def-schester  # Fixed: removed _gpu suffix
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2  # Request 2 A100 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=rag_parallel_%j.out
#SBATCH --error=rag_parallel_%j.err
#SBATCH --mail-user=tanujd@uvic.ca
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# CRITICAL: Set working directory to where your files are
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la

# Load required modules for Narval
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudacore/12.2
module load python/3.11
module load openmpi/4.1.5
module load cmake/3.27.7
module load imkl/2023.2.0

# Show loaded modules
echo "Loaded modules:"
module list

# Set CUDA paths
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Show GPU information (should work now with GPUs allocated)
echo "GPU Information:"
nvidia-smi

# Set up Python environment
echo "Setting up Python environment..."

# Use scratch for virtual environment (more space)
VENV_DIR="$HOME/scratch/rag_env"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in scratch..."
    python -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
echo "Installing Python packages..."

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install sentence-transformers numpy pandas matplotlib seaborn psutil

# Try installing cuVS without NVIDIA index issues
pip install --no-deps cuvs || echo "cuVS installation failed, will use fallback"

# Build FAISS from source if needed
if ! python -c "import faiss" 2>/dev/null; then
    echo "Building FAISS from source..."

    # Check if FAISS is already built
    if [ ! -d "$HOME/scratch/faiss" ]; then
        cd $HOME/scratch
        git clone https://github.com/facebookresearch/faiss.git
        cd faiss

        cmake -B build . \
            -DFAISS_ENABLE_GPU=ON \
            -DFAISS_ENABLE_PYTHON=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="80" \
            -DPython_EXECUTABLE=$(which python)

        make -C build -j8
        cd build/faiss/python
        pip install .
    else
        echo "FAISS directory exists, attempting to install..."
        cd $HOME/scratch/faiss/build/faiss/python
        pip install .
    fi

    # Return to job directory
    cd $SLURM_SUBMIT_DIR
fi

# Show Python and package versions
echo "Python version: $(python --version)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
python -c "import torch; print(torch.cuda.device_count())"
echo "GPU Names:"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "No GPUs detected in Python"

# Check if main script exists
if [ -f "improved_multi_gpu_rag.py" ]; then
    echo "Running improved multi-GPU RAG implementation..."
    python improved_multi_gpu_rag.py
else
    echo "ERROR: improved_multi_gpu_rag.py not found in $(pwd)"
    echo "Please ensure all files are copied to the job directory"
fi

# Run inline benchmark test
echo "Running simplified benchmark test..."
python << 'EOF'
import torch
import numpy as np
import pandas as pd
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability
if not torch.cuda.is_available():
    print("WARNING: No GPUs available to Python. Running CPU-only test.")
    device = 'cpu'
    num_gpus = 1
else:
    num_gpus = torch.cuda.device_count()
    device = 'cuda'
    print(f"Found {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Test configuration
NUM_VECTORS = 100000
TOP_K = 2000
EMBEDDING_DIM = 768
NUM_QUERIES = 10

print(f'\nTest Configuration:')
print(f'  Vectors: {NUM_VECTORS:,}')
print(f'  Top-K: {TOP_K}')
print(f'  Queries: {NUM_QUERIES}')
print(f'  Device: {device}')

# Generate synthetic embeddings
print('\nGenerating synthetic data...')
embeddings = torch.randn(NUM_VECTORS, EMBEDDING_DIM, dtype=torch.float32, device=device)
print(f'Embeddings shape: {embeddings.shape}')

# Test parallel processing simulation
if num_gpus > 1:
    print(f'\nSimulating parallel processing across {num_gpus} GPUs...')
    chunks = torch.chunk(embeddings, num_gpus, dim=0)

    def process_chunk(gpu_id, chunk):
        torch.cuda.set_device(gpu_id)
        start = time.time()
        # Simulate index building
        result = torch.matmul(chunk, chunk.T)
        torch.cuda.synchronize(gpu_id)
        return time.time() - start

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_chunk, i, chunk.to(f'cuda:{i}'))
                  for i, chunk in enumerate(chunks)]
        times = [f.result() for f in futures]

    print(f'Processing times per GPU: {times}')
    print(f'Average time: {np.mean(times):.3f}s')
else:
    print('\nSingle device processing...')
    start = time.time()
    result = torch.matmul(embeddings[:1000], embeddings[:1000].T)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'Processing time: {elapsed:.3f}s')

# Test search with top-2K
print(f'\nTesting search with top-{TOP_K} retrieval...')
queries = torch.randn(NUM_QUERIES, EMBEDDING_DIM, dtype=torch.float32, device=device)

start = time.time()
for i, query in enumerate(queries):
    # Compute distances
    distances = torch.cdist(query.unsqueeze(0), embeddings)
    # Get top-K
    top_k_distances, top_k_indices = torch.topk(distances,
                                                 min(TOP_K, NUM_VECTORS),
                                                 largest=False)
    if i == 0:
        print(f'Query 1 retrieved {len(top_k_indices[0])} results')

total_time = time.time() - start
avg_time = (total_time / NUM_QUERIES) * 1000
print(f'Average search time: {avg_time:.2f}ms')

# Memory stats
if torch.cuda.is_available():
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f'GPU {i} memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB')

print('\nBenchmark test completed successfully!')

# Save results
results = {
    'device': device,
    'num_gpus': num_gpus if device == 'cuda' else 0,
    'num_vectors': NUM_VECTORS,
    'top_k': TOP_K,
    'avg_search_ms': avg_time
}

df = pd.DataFrame([results])
df.to_csv('narval_test_results.csv', index=False)
print('Results saved to narval_test_results.csv')
EOF

echo "Job completed at $(date)"