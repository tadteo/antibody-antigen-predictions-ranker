#!/usr/bin/env python3
"""
Add ESM-2 embeddings to existing H5 files.

This script reads the residue_one_letter sequence from each H5 file,
computes ESM-2 embeddings using a protein language model, and creates
new H5 files with the embeddings added while preserving all original data.

The ESM embeddings provide learned representations for each residue that capture:
- Amino acid biochemical properties
- Evolutionary conservation patterns
- Structural propensities
- Context-dependent behavior

to run:
python 07_add_esm_embeddings.py --input_dir input_path_with_h5 --output_dir output_path_with_esm_h5 --model_name facebook/esm2_t6_8M_UR50D

Example with different models:
python 07_add_esm_embeddings.py --input_dir /path/to/input --output_dir /path/to/output --model_name facebook/esm2_t6_8M_UR50D
python 07_add_esm_embeddings.py --input_dir /path/to/input --output_dir /path/to/output --model_name facebook/esm2_t12_35M_UR50D
"""

import argparse
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

def load_esm_model(model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Load ESM-2 model and tokenizer.
    
    Available models:
    - facebook/esm2_t6_8M_UR50D: 6 layers, 320-dim embeddings (smallest, fastest)
    - facebook/esm2_t12_35M_UR50D: 12 layers, 480-dim embeddings
    - facebook/esm2_t30_150M_UR50D: 30 layers, 640-dim embeddings
    - facebook/esm2_t33_650M_UR50D: 33 layers, 1280-dim embeddings (largest, slowest)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    print(f"Loading ESM model: {model_name}")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def compute_esm_embeddings(sequence, model, tokenizer, device):
    """
    Compute per-residue ESM embeddings for a sequence.
    
    Args:
        sequence: List or array of single-letter amino acid codes
        model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        embeddings: numpy array of shape [L, d_esm] where L is sequence length
    """
    # Convert sequence array to string
    if isinstance(sequence, np.ndarray):
        if isinstance(sequence[0], bytes):
            sequence = [s.decode('utf-8') for s in sequence]
        seq_str = ''.join(sequence)
    else:
        seq_str = ''.join(sequence)
    
    # Tokenize
    inputs = tokenizer(seq_str, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Compute embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract per-residue embeddings (exclude BOS/EOS tokens)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # [L, d_esm]
    
    return embeddings.cpu().numpy()

def process_h5_file(input_path, output_path, model, tokenizer, device):
    """
    Process a single H5 file to add ESM embeddings.
    
    This function:
    1. Reads an input H5 file containing antibody-antigen complex predictions
    2. For each sample, extracts the residue_one_letter sequence
    3. Computes ESM embeddings for the sequence
    4. Saves all original data plus the ESM embeddings to a new H5 file
    
    Args:
        input_path (str): Path to the input H5 file
        output_path (str): Path where the processed H5 file will be saved
        model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        str: Status message indicating success or failure
    """
    try:
        with h5py.File(input_path, 'r') as in_hf, h5py.File(output_path, 'w') as out_hf:
            # Iterate through each antibody-antigen complex in the file
            for complex_id in in_hf.keys():
                in_complex_group = in_hf[complex_id]
                out_complex_group = out_hf.create_group(complex_id)
                
                # Handle complex-level datasets (like pae_col_mean, pae_col_median, etc.)
                for key in in_complex_group.keys():
                    item = in_complex_group[key]
                    
                    # If it's a dataset at complex level, copy it
                    if isinstance(item, h5py.Dataset):
                        in_complex_group.copy(key, out_complex_group)
                        continue
                    
                    # If it's a group (sample), process it
                    if isinstance(item, h5py.Group):
                        sample_name = key
                        in_sample_group = item
                        out_sample_group = out_complex_group.create_group(sample_name)
                        
                        # Copy all existing datasets from the input sample
                        for dataset_key in in_sample_group.keys():
                            in_sample_group.copy(dataset_key, out_sample_group)
                        
                        # Compute and add ESM embeddings
                        if 'residue_one_letter' in in_sample_group:
                            residue_one_letter = in_sample_group['residue_one_letter'][()]
                            
                            try:
                                embeddings = compute_esm_embeddings(residue_one_letter, model, tokenizer, device)
                                
                                # Verify dimensions match
                                if len(embeddings) != len(residue_one_letter):
                                    print(f"  WARNING: Length mismatch for {complex_id}/{sample_name}: "
                                          f"sequence={len(residue_one_letter)}, embeddings={len(embeddings)}")
                                else:
                                    # Add ESM embeddings to output file
                                    out_sample_group.create_dataset('esm_embeddings', data=embeddings, compression='gzip')
                            
                            except Exception as e:
                                print(f"  ERROR computing embeddings for {complex_id}/{sample_name}: {e}")
                        else:
                            print(f"  WARNING: {complex_id}/{sample_name} missing residue_one_letter")
        
        return f"Successfully processed {os.path.basename(input_path)}"
    
    except Exception as e:
        return f"ERROR processing {os.path.basename(input_path)}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description='Add ESM-2 embeddings to H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available ESM-2 models:
  facebook/esm2_t6_8M_UR50D      - 320-dim embeddings (fastest, recommended)
  facebook/esm2_t12_35M_UR50D    - 480-dim embeddings
  facebook/esm2_t30_150M_UR50D   - 640-dim embeddings
  facebook/esm2_t33_650M_UR50D   - 1280-dim embeddings (slowest)

Example:
  python 07_add_esm_embeddings.py \\
      --input_dir /path/to/input_h5_files \\
      --output_dir /path/to/output_h5_files \\
      --model_name facebook/esm2_t6_8M_UR50D
        """
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input H5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed H5 files with ESM embeddings')
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help='ESM model name (default: esm2_t6_8M_UR50D, 320-dim)')
    parser.add_argument('--pattern', type=str, default='*.h5',
                        help='H5 file pattern (default: *.h5)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load ESM model
    model, tokenizer, device = load_esm_model(args.model_name)
    
    # Get embedding dimension
    with torch.no_grad():
        test_emb = compute_esm_embeddings("ACDEF", model, tokenizer, device)
        esm_dim = test_emb.shape[1]
    print(f"ESM embedding dimension: {esm_dim}")
    print()
    
    # Find all H5 files in input directory
    input_dir = Path(args.input_dir)
    h5_files = sorted(input_dir.glob(args.pattern))
    
    if not h5_files:
        print(f"ERROR: No H5 files found matching pattern '{args.pattern}' in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files to process")
    print()
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for input_path in tqdm(h5_files, desc="Processing H5 files"):
        # Create output path with same filename
        output_path = output_dir / input_path.name
        
        try:
            status_msg = process_h5_file(input_path, output_path, model, tokenizer, device)
            if "Successfully" in status_msg:
                success_count += 1
            else:
                error_count += 1
                print(f"\n{status_msg}")
        except Exception as e:
            error_count += 1
            print(f"\nERROR processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total files:      {len(h5_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors:           {error_count}")
    print(f"\nESM model:        {args.model_name}")
    print(f"Embedding dimension: {esm_dim}")
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("Update your config file with:")
    print("  data:")
    print(f"    use_esm_embeddings: true")
    print(f"    esm_embedding_dim: {esm_dim}")
    print("\nThe model input_dim will be automatically calculated as:")
    print(f"  5 (base features) + 2*{esm_dim} (ESM i,j) = {5 + 2*esm_dim} features")

if __name__ == '__main__':
    main()

