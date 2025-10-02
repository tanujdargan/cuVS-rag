#!/bin/bash
# Quick GPU test script for Narval cluster
# Run this to verify GPU access and functionality

echo "==================================="
echo "Quick GPU Test on Narval"
echo "==================================="
echo ""

# Test 1: Quick nvidia-smi check (10 minutes max)
echo "Test 1: Quick GPU check with srun"
echo "Running: srun --gres=gpu:1 --time=0:10:00 --account=def-schester_gpu nvidia-smi"
echo ""
srun --gres=gpu:1 --time=0:10:00 --account=def-schester_gpu nvidia-smi

echo ""
echo "==================================="
echo ""

# Test 2: Python CUDA availability check
echo "Test 2: Python CUDA availability"
echo "Running Python to check CUDA..."
echo ""

srun --gres=gpu:1 --time=0:10:00 --account=def-schester_gpu bash -c "
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

python -c '
import sys
print(f\"Python version: {sys.version}\")

try:
    import torch
    print(f\"PyTorch version: {torch.__version__}\")
    print(f\"CUDA available: {torch.cuda.is_available()}\")
    if torch.cuda.is_available():
        print(f\"CUDA version: {torch.version.cuda}\")
        print(f\"Number of GPUs: {torch.cuda.device_count()}\")
        for i in range(torch.cuda.device_count()):
            print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")
            props = torch.cuda.get_device_properties(i)
            print(f\"  Memory: {props.total_memory / 1024**3:.1f} GB\")
            print(f\"  Compute capability: {props.major}.{props.minor}\")
except ImportError:
    print(\"PyTorch not installed. Install with:\")
    print(\"pip install torch --index-url https://download.pytorch.org/whl/cu121\")
'
"

echo ""
echo "==================================="
echo "Test complete!"
echo ""
echo "If you see GPU information above, your GPU access is working."
echo "If not, check:"
echo "  1. Your account has GPU allocation: sshare -U $USER -A def-schester_gpu"
echo "  2. GPUs are available: sinfo -p gpu --Format=NodeList,Gres:30,GresUsed:30"
echo "  3. Your job is in the queue: squeue -u $USER"