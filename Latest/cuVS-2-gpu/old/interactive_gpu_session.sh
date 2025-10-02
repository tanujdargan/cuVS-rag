#!/bin/bash
# Interactive GPU session script for Narval cluster
# This script starts an interactive GPU session for testing and development

echo "Starting interactive GPU session on Narval..."
echo ""
echo "This will request:"
echo "  - 1 A100 GPU"
echo "  - 8 CPU cores"
echo "  - 32GB RAM"
echo "  - 2 hours time limit"
echo ""
echo "Once allocated, you can:"
echo "  - Run nvidia-smi to check GPU"
echo "  - Test your code interactively"
echo "  - Debug GPU-related issues"
echo ""

# Request interactive session with GPU
salloc \
    --account=def-schester_gpu \
    --gres=gpu:a100:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=2:00:00 \
    --job-name=interactive_gpu

# Note: After allocation, you'll be placed in an interactive shell
# on the compute node with GPU access