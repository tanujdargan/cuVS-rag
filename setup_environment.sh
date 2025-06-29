#!/bin/bash

# Setup script for cuVS testing environment

echo "Creating conda environment 'cuVStesting'..."

# Create the conda environment with basic packages
conda create --name cuVStesting --file conda-requirements.txt -y

if [ $? -eq 0 ]; then
    echo "Conda environment created successfully!"
    
    # Activate the environment
    echo "Activating environment and installing pip packages..."
    conda activate cuVStesting
    
    # Install pip packages
    pip install -r pip-requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "Environment setup completed successfully!"
        echo "To activate the environment, run: conda activate cuVStesting"
    else
        echo "Error installing pip packages. Please check the pip-requirements.txt file."
        exit 1
    fi
else
    echo "Error creating conda environment. Please check the conda-requirements.txt file."
    exit 1
fi 