#!/bin/bash
# Transfer medical Q&A data to Narval

echo "Transferring medical Q&A dataset to Narval..."

# Set paths
LOCAL_DIR="medical_qa_data"
REMOTE_HOST="tanujd@narval.computecanada.ca"
REMOTE_DIR="~/scratch/rag_datasets/"

# Create remote directory
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Transfer files
echo "Uploading files..."
scp -r $LOCAL_DIR/* $REMOTE_HOST:$REMOTE_DIR/

echo "Transfer complete!"
echo "Files are now at: $REMOTE_DIR on Narval"
