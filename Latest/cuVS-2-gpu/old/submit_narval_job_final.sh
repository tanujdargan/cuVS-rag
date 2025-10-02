#!/bin/bash
#SBATCH --job-name=rag_parallel_gpu
#SBATCH --account=def-schester
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
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

# Set working directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la *.py *.sh 2>/dev/null

# Load required modules for Narval
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11

echo "Loaded modules:"
module list

# Set CUDA paths
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Show GPU information
echo ""
echo "GPU Information:"
nvidia-smi

# Set up Python environment
echo ""
echo "Setting up Python environment..."

# Use scratch for virtual environment
VENV_DIR="$HOME/scratch/rag_env"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in scratch..."
    python -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install required packages
echo "Installing/Updating Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install sentence-transformers numpy pandas matplotlib psutil --quiet

# Try to install FAISS
pip install faiss-cpu faiss-gpu 2>/dev/null || echo "FAISS installation needs building from source"

# Verify CUDA availability
echo ""
echo "Python environment check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null

# Check if main script exists and run it
if [ -f "improved_multi_gpu_rag.py" ]; then
    echo ""
    echo "Running improved multi-GPU RAG implementation..."
    python improved_multi_gpu_rag.py
else
    echo "WARNING: improved_multi_gpu_rag.py not found"
fi

# Run a simple GPU test
echo ""
echo "Running GPU memory and performance test..."
python << 'EOF'
import torch
import time
import numpy as np

print("=== GPU Performance Test ===")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPU(s)")

for i in range(num_gpus):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")

    # Memory info
    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    print(f"  Total Memory: {total_memory:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB")

    # Simple performance test
    torch.cuda.set_device(i)
    size = 10000
    a = torch.randn(size, size, device=f'cuda:{i}')
    b = torch.randn(size, size, device=f'cuda:{i}')

    torch.cuda.synchronize(i)
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize(i)
    elapsed = time.time() - start

    gflops = (2 * size**3) / (elapsed * 1e9)
    print(f"  Matrix multiply (10K x 10K): {elapsed:.3f}s ({gflops:.1f} GFLOPS)")

    # Cleanup
    del a, b, c
    torch.cuda.empty_cache()

print("\nGPU test completed successfully!")
EOF

echo ""
echo "Job completed at $(date)"