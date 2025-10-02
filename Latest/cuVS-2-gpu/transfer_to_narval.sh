#!/bin/bash
# Script to transfer files to Narval cluster

echo "Transferring files to Narval cluster..."

# Set your username
USERNAME="tanujd"
REMOTE_HOST="narval.alliancecan.ca"
PROJECT_DIR="~/projects/def-schester/tanujd/rag_multi_gpu"

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh ${USERNAME}@${REMOTE_HOST} "mkdir -p ${PROJECT_DIR}"

# Transfer key files
echo "Transferring implementation files..."
scp improved_multi_gpu_rag.py ${USERNAME}@${REMOTE_HOST}:${PROJECT_DIR}/
scp submit_narval_job_final.sh ${USERNAME}@${REMOTE_HOST}:${PROJECT_DIR}/submit_job.sh
scp generate_embeddings.py ${USERNAME}@${REMOTE_HOST}:${PROJECT_DIR}/

# Check if additional files exist before copying
if [ -f "README_improved.md" ]; then
    scp README_improved.md ${USERNAME}@${REMOTE_HOST}:${PROJECT_DIR}/
fi

if [ -f "colab_a100_test.ipynb" ]; then
    scp colab_a100_test.ipynb ${USERNAME}@${REMOTE_HOST}:${PROJECT_DIR}/
fi

echo "Files transferred successfully!"
echo ""
echo "Next steps on Narval:"
echo "1. SSH to Narval: ssh ${USERNAME}@${REMOTE_HOST}"
echo "2. Navigate to: cd ${PROJECT_DIR}"
echo "3. Submit job: sbatch submit_job.sh"
echo "4. Monitor: squeue -u ${USERNAME}"