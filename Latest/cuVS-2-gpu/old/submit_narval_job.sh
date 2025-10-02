#!/bin/bash
#SBATCH --job-name=rag_parallel_gpu
#SBATCH --account=def-schester_gpu  # GPU allocation account
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2  # Request 2 A100 GPUs
#SBATCH --partition=gpu  # Ensure we're on GPU partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=rag_parallel_%j.out
#SBATCH --error=rag_parallel_%j.err
#SBATCH --mail-user=tanujd@uvic.ca  # CHANGE THIS
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set working directory (IMPORTANT: Change this to your actual path)
cd $SLURM_SUBMIT_DIR  # This will be where you run sbatch from
echo "Working directory: $(pwd)"

# Load required modules for Narval
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load openmpi/4.1.5
module load cmake/3.27.7
module load imkl/2023.2.0

# Verify GPU access
echo "Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU access available. Check your allocation."
    exit 1
fi

# Show loaded modules
echo "Loaded modules:"
module list

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pylibraft-cu12 cuvs-cu12 --extra-index-url https://pypi.nvidia.com
pip install sentence-transformers numpy pandas matplotlib seaborn psutil
pip install faiss-gpu

# For FAISS from source (if pre-built doesn't work)
# Follow the build instructions in build_instructions_for_FAISS.md

# Show GPU information
echo "GPU Information:"
nvidia-smi

# Show Python and package versions
echo "Python version: $(python --version)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "Number of GPUs:"
python -c "import torch; print(torch.cuda.device_count())"

# Run the improved implementation
echo "Starting parallel GPU RAG implementation..."
python improved_multi_gpu_rag.py

# Run the comparison benchmark
echo "Running comprehensive benchmark..."
python -c "
import torch
import numpy as np
import pandas as pd
import time
import gc
from sentence_transformers import SentenceTransformer
import pylibraft
from cuvs.neighbors import ivf_flat, ivf_pq, cagra
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pylibraft.config.set_output_as(lambda device_ndarray: device_ndarray.copy_to_host())

# Configuration
NUM_VECTORS = 1000000  # 1M vectors
TOP_K = 2000  # Top-2K retrieval
EMBEDDING_DIM = 768
NUM_QUERIES = 100

print(f'Configuration:')
print(f'  Vectors: {NUM_VECTORS:,}')
print(f'  Top-K: {TOP_K}')
print(f'  Queries: {NUM_QUERIES}')
print(f'  GPUs: {torch.cuda.device_count()}')

# Generate synthetic embeddings
print('Generating synthetic data...')
embeddings = torch.randn(NUM_VECTORS, EMBEDDING_DIM, dtype=torch.float32, device='cuda:0')

# Split across GPUs
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    chunks = torch.chunk(embeddings, num_gpus, dim=0)
    embedding_parts = [chunk.to(f'cuda:{i}') for i, chunk in enumerate(chunks)]
else:
    embedding_parts = [embeddings]

print(f'Data split across {len(embedding_parts)} GPU(s)')

# Test parallel index building
def build_index_on_gpu(gpu_id, embeddings, index_type):
    torch.cuda.set_device(gpu_id)
    start_time = time.time()

    if index_type == 'ivf_flat':
        params = ivf_flat.IndexParams(n_lists=256)
        index = ivf_flat.build(params, embeddings)
    elif index_type == 'ivf_pq':
        params = ivf_pq.IndexParams(n_lists=512, pq_dim=96, pq_bits=8)
        index = ivf_pq.build(params, embeddings)
    elif index_type == 'cagra':
        params = cagra.IndexParams(intermediate_graph_degree=64, graph_degree=32)
        index = cagra.build(params, embeddings)
    else:
        raise ValueError(f'Unknown index type: {index_type}')

    build_time = time.time() - start_time
    return index, build_time

# Test each index type
results = []

for index_type in ['ivf_flat', 'ivf_pq', 'cagra']:
    print(f'\\nTesting {index_type.upper()}...')

    try:
        # Parallel index building
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, part in enumerate(embedding_parts):
                future = executor.submit(build_index_on_gpu, i, part, index_type)
                futures.append((i, future))

            indexes = {}
            build_times = []
            for gpu_id, future in futures:
                index, build_time = future.result(timeout=300)
                indexes[gpu_id] = index
                build_times.append(build_time)

        avg_build_time = np.mean(build_times)
        print(f'  Build time: {avg_build_time:.2f}s')

        # Test search with top-2K
        query = torch.randn(1, EMBEDDING_DIM, dtype=torch.float32, device='cuda:0')

        start_time = time.time()
        all_distances = []
        all_indices = []

        for gpu_id, index in indexes.items():
            query_gpu = query.to(f'cuda:{gpu_id}')

            if index_type == 'ivf_flat':
                search_params = ivf_flat.SearchParams()
                distances, indices = ivf_flat.search(search_params, index, query_gpu, TOP_K)
            elif index_type == 'ivf_pq':
                search_params = ivf_pq.SearchParams()
                distances, indices = ivf_pq.search(search_params, index, query_gpu, TOP_K)
            elif index_type == 'cagra':
                search_params = cagra.SearchParams()
                distances, indices = cagra.search(search_params, index, query_gpu, TOP_K)

            all_distances.extend(distances.flatten())
            all_indices.extend(indices.flatten())

        # Merge results
        all_distances = np.array(all_distances)
        all_indices = np.array(all_indices)
        sorted_idx = np.argsort(all_distances)[:TOP_K]

        search_time = (time.time() - start_time) * 1000
        print(f'  Search time: {search_time:.2f}ms')
        print(f'  Retrieved: {len(sorted_idx)} results')

        results.append({
            'index_type': index_type,
            'build_time_s': avg_build_time,
            'search_time_ms': search_time,
            'num_results': len(sorted_idx),
            'success': True
        })

        # Cleanup
        del indexes
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  Error: {e}')
        results.append({
            'index_type': index_type,
            'build_time_s': None,
            'search_time_ms': None,
            'num_results': 0,
            'success': False,
            'error': str(e)
        })

# Display results
import pandas as pd
results_df = pd.DataFrame(results)
print('\\n' + '='*60)
print('BENCHMARK RESULTS')
print('='*60)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('narval_benchmark_results.csv', index=False)
print('\\nResults saved to narval_benchmark_results.csv')
"

echo "Job completed at $(date)"
echo "Results saved to narval_benchmark_results.csv"
