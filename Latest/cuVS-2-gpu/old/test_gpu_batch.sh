#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=def-schester_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

echo "GPU Test Batch Job"
echo "=================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load modules
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

# Test 1: nvidia-smi
echo "1. NVIDIA-SMI Output:"
echo "---------------------"
nvidia-smi
echo ""

# Test 2: CUDA environment variables
echo "2. CUDA Environment:"
echo "--------------------"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Test 3: PyTorch CUDA test
echo "3. PyTorch CUDA Test:"
echo "---------------------"
python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Test GPU computation
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print(f"GPU computation test: SUCCESS")
        print(f"Result shape: {z.shape}")

        # Memory info
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    else:
        print("ERROR: CUDA not available!")

except ImportError as e:
    print(f"ERROR: {e}")
    print("PyTorch not installed. Install with:")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu121")
except Exception as e:
    print(f"ERROR: {e}")
EOF

echo ""
echo "=================="
echo "Test complete at $(date)"