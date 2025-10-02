#!/usr/bin/env python3
"""
Prepare medical Q&A dataset for Narval cluster
Downloads locally and saves in format suitable for offline use
"""

import os
import json
import numpy as np
import pickle
from datasets import load_dataset
from tqdm import tqdm
import torch

def main():
    print("="*60)
    print("DATASET PREPARATION FOR NARVAL")
    print("="*60)

    # Configuration
    OUTPUT_DIR = "medical_qa_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Your HuggingFace token (from original code)
    HF_TOKEN = "HFTOKENHERE"

    print("\n1. Loading dataset from HuggingFace...")
    try:
        # Load the medical Q&A dataset
        ds = load_dataset(
            "Malikeh1375/medical-question-answering-datasets",
            "all-processed",
            token=HF_TOKEN
        )
        print(f"   Dataset loaded successfully!")
        print(f"   Splits available: {list(ds.keys())}")

        # Focus on train split for now
        if 'train' in ds:
            dataset = ds['train']
        else:
            dataset = ds[list(ds.keys())[0]]  # Use first available split

        print(f"   Number of samples: {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")

        # Show sample
        print("\n   Sample entry:")
        sample = dataset[0]
        for key, value in sample.items():
            print(f"     {key}: {str(value)[:100]}...")

    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("\n   Creating synthetic medical Q&A data instead...")

        # Create synthetic medical data for testing
        medical_questions = [
            "What are the symptoms of diabetes?",
            "How is hypertension diagnosed?",
            "What causes migraine headaches?",
            "What is the treatment for asthma?",
            "How to prevent heart disease?",
            "What are the side effects of antibiotics?",
            "How is cancer detected early?",
            "What is the difference between Type 1 and Type 2 diabetes?",
            "What are the symptoms of COVID-19?",
            "How to manage chronic pain?",
        ]

        medical_answers = [
            "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
            "Diagnosis involves blood pressure measurements over multiple visits, typically readings above 130/80 mmHg.",
            "Triggers include stress, certain foods, hormonal changes, and environmental factors.",
            "Treatment includes bronchodilators, corticosteroids, and avoiding triggers.",
            "Prevention includes healthy diet, regular exercise, not smoking, and managing stress.",
            "Common side effects include nausea, diarrhea, and allergic reactions in some patients.",
            "Early detection methods include regular screening, blood tests, and imaging studies.",
            "Type 1 is autoimmune and requires insulin; Type 2 is often lifestyle-related and may be managed with diet.",
            "Symptoms include fever, cough, fatigue, loss of taste or smell, and shortness of breath.",
            "Management includes medication, physical therapy, lifestyle changes, and stress reduction.",
        ]

        # Expand dataset
        dataset_entries = []
        for i in range(1000):  # Create 1000 samples
            q_idx = i % len(medical_questions)
            dataset_entries.append({
                'instruction': medical_questions[q_idx],
                'input': f"Patient case {i}: Additional context for the medical question.",
                'output': medical_answers[q_idx]
            })

        dataset = dataset_entries

    print("\n2. Preparing data for offline use...")

    # Save as JSON for easy loading
    if isinstance(dataset, list):
        data_to_save = dataset
    else:
        # Convert HuggingFace dataset to list of dicts
        data_to_save = []
        for i in tqdm(range(min(len(dataset), 100000)), desc="Converting"):  # Limit to 100k for testing
            entry = dataset[i]
            data_to_save.append({
                'instruction': entry.get('instruction', ''),
                'input': entry.get('input', ''),
                'output': entry.get('output', '')
            })

    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, "medical_qa.json")
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f)
    print(f"   Saved {len(data_to_save)} entries to {json_path}")

    # Save as pickle for faster loading
    pickle_path = os.path.join(OUTPUT_DIR, "medical_qa.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"   Saved pickle version to {pickle_path}")

    # Create a smaller test set
    test_data = data_to_save[:100]
    test_path = os.path.join(OUTPUT_DIR, "medical_qa_test.json")
    with open(test_path, 'w') as f:
        json.dump(test_data, f)
    print(f"   Saved test set (100 samples) to {test_path}")

    print("\n3. Generating sample embeddings (using CPU)...")

    try:
        from sentence_transformers import SentenceTransformer

        # Use a smaller model for testing
        model_name = 'all-MiniLM-L6-v2'  # Smaller, faster model
        print(f"   Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Prepare texts
        texts = []
        for entry in test_data[:10]:  # Just 10 for demo
            combined = f"Instruction: {entry['instruction']}\nInput: {entry['input']}"
            texts.append(combined)

        # Generate embeddings
        print(f"   Generating embeddings for {len(texts)} samples...")
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

        # Save embeddings
        embeddings_path = os.path.join(OUTPUT_DIR, "sample_embeddings.pt")
        torch.save(embeddings.cpu(), embeddings_path)
        print(f"   Saved sample embeddings to {embeddings_path}")
        print(f"   Embedding shape: {embeddings.shape}")

    except ImportError:
        print("   Sentence-transformers not installed. Skipping embedding generation.")
        print("   Install with: pip install sentence-transformers")

    print("\n4. Creating upload script...")

    # Create script to transfer data to Narval
    script_content = f"""#!/bin/bash
# Transfer medical Q&A data to Narval

echo "Transferring medical Q&A dataset to Narval..."

# Set paths
LOCAL_DIR="{OUTPUT_DIR}"
REMOTE_HOST="tanujd@narval.computecanada.ca"
REMOTE_DIR="~/scratch/rag_datasets/"

# Create remote directory
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Transfer files
echo "Uploading files..."
scp -r $LOCAL_DIR/* $REMOTE_HOST:$REMOTE_DIR/

echo "Transfer complete!"
echo "Files are now at: $REMOTE_DIR on Narval"
"""

    with open("upload_to_narval.sh", "w") as f:
        f.write(script_content)
    os.chmod("upload_to_narval.sh", 0o755)
    print("   Created upload_to_narval.sh")

    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the data in ./medical_qa_data/")
    print("2. Run: ./upload_to_narval.sh")
    print("3. On Narval, data will be in: ~/scratch/rag_datasets/")

    # Summary
    print(f"\nFiles created:")
    for file in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()