#!/bin/bash
# Setup script for Narval cluster
# Run this after logging into Narval

echo "Setting up RAG Multi-GPU project on Narval..."

# Create directory structure
echo "Creating project directories..."
mkdir -p ~/projects/def-schester/tanujd/rag_multi_gpu
mkdir -p ~/scratch/tanujd/rag_datasets
mkdir -p ~/scratch/tanujd/rag_outputs

# Navigate to project directory
cd ~/projects/def-schester/tanujd/rag_multi_gpu

# Create symbolic links to scratch for large data
ln -sfn ~/scratch/tanujd/rag_datasets data
ln -sfn ~/scratch/tanujd/rag_outputs outputs

echo "Directory structure created:"
echo "  Project: ~/projects/def-schester/tanujd/rag_multi_gpu"
echo "  Data:    ~/scratch/tanujd/rag_datasets (linked as ./data)"
echo "  Outputs: ~/scratch/tanujd/rag_outputs (linked as ./outputs)"

# Check GPU allocation
echo ""
echo "Checking your GPU allocation..."
echo "Account info:"
sshare -U $USER -A def-schester_gpu

echo ""
echo "GPU availability on cluster:"
sinfo -p gpu --Format=NodeList,Gres:30,GresUsed:30,StateCompact

echo ""
echo "Note: GPUs are NOT available on login nodes!"
echo "To access GPUs, you must:"
echo ""
echo "Option 1 - Interactive session (for testing):"
echo "  salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 --account=def-schester_gpu"
echo "  Then run: nvidia-smi"
echo ""
echo "Option 2 - Submit batch job:"
echo "  1. Copy your files to: ~/projects/def-schester/tanujd/rag_multi_gpu/"
echo "  2. cd ~/projects/def-schester/tanujd/rag_multi_gpu/"
echo "  3. sbatch submit_narval_job.sh"
echo ""
echo "Option 3 - Quick GPU test:"
echo "  srun --gres=gpu:1 --time=0:10:00 --account=def-schester_gpu nvidia-smi"
echo ""
echo "To check job status: squeue -u $USER"
echo "To cancel a job: scancel <job_id>"