
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Helper function for pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(instruction: str, query: str) -> str:
    """
    Formats the instruction and query into a single string.
    """
    return f'Instruct: {instruction}\nQuery: {query}'

def main():
    """
    Main function to run the embedding generation pipeline.
    """
    # 1. Load Hugging Face dataset
    print("Loading dataset from Hugging Face...")
    # Make sure to set HUGGING_FACE_TOKEN environment variable
    # In a SLURM script, you would do: export HUGGING_FACE_TOKEN='your_token'
    hf_token = "HFTOKENHERE"
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable.")

    ds = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed", token=hf_token)
    print("Dataset loaded successfully.")

    # 2. Set up model and tokenizer for GPU
    print("Setting up model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU. This will be very slow.")
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left', trust_remote_code=True)
    
    # Use Flash Attention 2 for acceleration and memory saving if available
    try:
        model = AutoModel.from_pretrained(
            'Qwen/Qwen3-Embedding-8B', 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
        print("Model loaded with Flash Attention 2.")
    except ImportError:
        print("Flash Attention 2 not available. Loading model without it.")
        model = AutoModel.from_pretrained(
            'Qwen/Qwen3-Embedding-8B',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)

    model.eval() # Set model to evaluation mode

    # 3. Generate embeddings
    print("Generating embeddings...")
    
    # Combine instruction and input columns
    def prepare_input_texts(examples):
        # The dataset has 'instruction', 'input', and 'output'
        # We will create embeddings for the combination of instruction and input
        texts = [get_detailed_instruct(instruction, inp) for instruction, inp in zip(examples['instruction'], examples['input'])]
        return {'prepared_text': texts}

    ds = ds.map(prepare_input_texts, batched=True, num_proc=4)

    max_length = 8192 # As recommended for the model
    batch_size = 16 # Adjust based on your GPU memory

    def generate_embeddings_batch(examples):
        input_texts = examples['prepared_text']
        
        with torch.no_grad():
            batch_dict = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return {'embedding': embeddings.cpu().numpy()}

    # Use map to generate embeddings in batches
    ds_with_embeddings = ds.map(
        generate_embeddings_batch, 
        batched=True, 
        batch_size=batch_size,
        desc="Generating embeddings"
    )

    print("Embeddings generated successfully.")

    # 4. Save embeddings for reuse
    print("Saving dataset with embeddings...")
    output_dir = "medical-ds-with-embeddings"
    ds_with_embeddings.save_to_disk(output_dir)
    print(f"Dataset with embeddings saved to '{output_dir}'")

if __name__ == "__main__":
    main()
