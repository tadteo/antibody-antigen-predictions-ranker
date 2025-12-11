#!/usr/bin/env python3
"""
Simple preprocessing script for inference on AF3 predictions.

Creates a single HDF5 file containing all features needed for model inference.

Usage:
    python preprocess_inference.py \
        --input_dir /path/to/af3_predictions \
        --output_h5 inference_data.h5 \
        --antibody_chains H,L \
        --antigen_chains A \
        --add_esm

Example for CAPRI target:
    python preprocess_inference.py \
        --input_dir /path/to/CAPRI_target/RUN01 \
        --output_h5 capri_target.h5 \
        --antibody_chains H,L \
        --antigen_chains A \
        --add_esm
"""

import argparse
import os
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# For structure parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
except ImportError:
    print("ERROR: BioPython not installed. Run: pip install biopython")
    exit(1)

# For ESM embeddings (optional)
try:
    from transformers import EsmModel, EsmTokenizer
    import torch
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


def load_ranking_scores(input_dir, run_dirs=None):
    """
    Load AF3 ranking scores from ranking_scores.csv in each RUN directory.
    
    Returns:
        dict: {sample_name: ranking_score}
    """
    import csv
    ranking_scores = {}
    input_path = Path(input_dir)
    
    # Determine search directories
    if run_dirs:
        search_dirs = [input_path / run_dir for run_dir in run_dirs]
        search_dirs = [d for d in search_dirs if d.exists()]
    else:
        search_dirs = [input_path]
    
    for search_dir in search_dirs:
        ranking_csv = search_dir / 'ranking_scores.csv'
        if not ranking_csv.exists():
            print(f"  WARNING: No ranking_scores.csv in {search_dir}")
            continue
        
        # Read ranking scores
        with open(ranking_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seed = row['seed']
                sample = row['sample']
                score = float(row['ranking_score'])
                sample_name = f"seed-{seed}_sample-{sample}"
                ranking_scores[sample_name] = score
    
    print(f"  Loaded {len(ranking_scores)} ranking scores from CSV files")
    return ranking_scores


def find_prediction_samples(input_dir, run_dirs=None):
    """
    Find all AF3 prediction samples across multiple RUN directories.
    
    Expected structure:
        input_dir/
            RUN01/
                ranking_scores.csv
                seed-X_sample-Y/
                    model.cif (or model.pdb)
                    confidences.json
            RUN02/
                ranking_scores.csv
                seed-X_sample-Y/
                    ...
    
    Or flat structure:
        input_dir/
            seed-X_sample-Y/
                model.cif (or model.pdb)
                confidences.json
    
    Args:
        input_dir: Base directory to search
        run_dirs: List of RUN subdirectories to search (e.g., ['RUN01', 'RUN02'])
                  If None, searches directly in input_dir
    
    Returns:
        list of dict: [{'path': path_to_sample_dir, 'name': sample_name}, ...]
    """
    samples = []
    input_path = Path(input_dir)
    
    # Determine search directories
    if run_dirs:
        search_dirs = [input_path / run_dir for run_dir in run_dirs]
        # Filter out non-existent directories
        search_dirs = [d for d in search_dirs if d.exists()]
        if not search_dirs:
            print(f"  WARNING: None of the specified RUN directories exist!")
            return []
    else:
        search_dirs = [input_path]
    
    # Search each directory
    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"  WARNING: Directory does not exist: {search_dir}")
            continue
        
        print(f"  Scanning: {search_dir}")
        found_in_dir = 0
        
        # Look for seed-*_sample-* directories
        for item in sorted(search_dir.iterdir()):
            if item.is_dir() and 'sample' in item.name.lower():
                # Check if it has required files
                has_model = any((item / f"model.{ext}").exists() for ext in ['cif', 'pdb'])
                has_conf = (item / 'confidences.json').exists()
                
                if has_model and has_conf:
                    samples.append({
                        'path': str(item),
                        'name': item.name
                    })
                    found_in_dir += 1
                else:
                    print(f"    WARNING: Skipping {item.name} - missing required files")
        
        print(f"    Found {found_in_dir} valid samples")
    
    return samples


def load_structure(structure_file):
    """Load PDB or CIF structure file."""
    path = Path(structure_file)
    
    if path.suffix.lower() in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == '.pdb':
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unknown structure format: {path.suffix}")
    
    return parser.get_structure('structure', str(path))


def merge_chains(structure, chain_ids, new_chain_id='X'):
    """
    Merge multiple chains into a single chain.
    
    Args:
        structure: BioPython Structure object
        chain_ids: list of chain IDs to merge
        new_chain_id: ID for the merged chain
    
    Returns:
        BioPython Structure with merged chains
    """
    merged_structure = Structure('merged')
    merged_model = Model(0)
    merged_chain = Chain(new_chain_id)
    
    residue_counter = 1
    for model in structure:
        for chain_id in chain_ids:
            if chain_id in [c.id for c in model]:
                chain = model[chain_id]
                for residue in chain:
                    # Create new residue with sequential numbering
                    new_res = residue.copy()
                    # Update residue numbering
                    res_id = list(new_res.id)
                    res_id[1] = residue_counter
                    new_res.id = tuple(res_id)
                    merged_chain.add(new_res)
                    residue_counter += 1
    
    merged_model.add(merged_chain)
    merged_structure.add(merged_model)
    return merged_structure


def extract_interchain_features(structure, ab_chains, ag_chains):
    """
    Extract interchain features between antibody and antigen.
    
    Returns:
        dict with:
            - inter_idx_i: antibody residue indices
            - inter_idx_j: antigen residue indices  
            - ca_distances: CA-CA distances
            - residue_one_letter: full sequence (all residues)
    """
    # Get all CA atoms for each chain set
    ab_residues = []
    ag_residues = []
    all_residues = []
    
    for model in structure:
        # Collect antibody residues
        for chain_id in ab_chains:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if 'CA' in residue:
                        ab_residues.append(residue)
                        all_residues.append(residue)
        
        # Collect antigen residues
        for chain_id in ag_chains:
            if chain_id in [c.id for c in model]:
                for residue in model[chain_id]:
                    if 'CA' in residue:
                        ag_residues.append(residue)
                        all_residues.append(residue)
    
    # Build full sequence
    residue_one_letter = []
    for res in all_residues:
        resname = res.get_resname()
        # Convert 3-letter to 1-letter
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        residue_one_letter.append(aa_map.get(resname, 'X'))
    
    # Create mapping from global index to position in ab/ag lists
    ab_global_indices = list(range(len(ab_residues)))
    ag_global_indices = list(range(len(ab_residues), len(ab_residues) + len(ag_residues)))
    
    # Compute all interchain pairs and distances
    inter_idx_i = []
    inter_idx_j = []
    ca_distances = []
    
    for i, ab_res in enumerate(ab_residues):
        ab_ca = ab_res['CA']
        for j, ag_res in enumerate(ag_residues):
            ag_ca = ag_res['CA']
            distance = ab_ca - ag_ca  # BioPython computes Euclidean distance
            
            # Store all pairs (we'll filter by distance later if needed)
            inter_idx_i.append(ab_global_indices[i])
            inter_idx_j.append(ag_global_indices[j])
            ca_distances.append(distance)
    
    return {
        'inter_idx': np.array(inter_idx_i, dtype=np.int32),
        'inter_jdx': np.array(inter_idx_j, dtype=np.int32),
        'ca_distances': np.array(ca_distances, dtype=np.float32),
        'residue_one_letter': residue_one_letter
    }


def load_confidences(confidences_file):
    """
    Load AF3 confidence scores and PAE matrix.
    
    Returns:
        dict with:
            - pae_matrix: full PAE matrix
            - ranking_score: AF3 ranking score
            - ptm: predicted TM score
            - iptm: interface predicted TM score
    """
    with open(confidences_file, 'r') as f:
        data = json.load(f)
    
    # Extract PAE matrix
    pae_matrix = np.array(data.get('pae', []), dtype=np.float32)
    
    # Extract scores
    ranking_score = data.get('ranking_score', 0.0)
    
    # Handle different AF3 output formats
    if 'ptm' in data:
        ptm = data['ptm']
    else:
        ptm = data.get('overall_confidence', {}).get('ptm', 0.0)
    
    if 'iptm' in data:
        iptm = data['iptm']
    else:
        iptm = data.get('overall_confidence', {}).get('iptm', 0.0)
    
    return {
        'pae_matrix': pae_matrix,
        'ranking_score': ranking_score,
        'ptm': ptm,
        'iptm': iptm
    }


def extract_interchain_pae(pae_matrix, inter_idx, inter_jdx):
    """Extract PAE values for interchain residue pairs."""
    interchain_pae = []
    
    for i, j in zip(inter_idx, inter_jdx):
        if i < pae_matrix.shape[0] and j < pae_matrix.shape[1]:
            interchain_pae.append(pae_matrix[i, j])
        else:
            interchain_pae.append(np.nan)
    
    return np.array(interchain_pae, dtype=np.float32)


def compute_esm_embeddings(sequence, model, tokenizer, device):
    """
    Compute per-residue ESM embeddings.
    
    Args:
        sequence: list of single-letter amino acids
        model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        numpy array [L, d_esm]
    """
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


def process_sample(sample_info, ab_chains, ag_chains, esm_embeddings=None, ranking_scores_dict=None):
    """
    Process a single AF3 prediction sample.
    
    IMPORTANT: NO distance filtering here! Distance filtering happens during inference
    to match training pipeline where filtering is done in dataloader, not preprocessing.
    
    Args:
        sample_info: dict with sample path and name
        ab_chains: antibody chain IDs
        ag_chains: antigen chain IDs
        esm_embeddings: Pre-computed ESM embeddings to reuse (optional)
        ranking_scores_dict: dict mapping sample_name to ranking_score from CSV
    
    Returns:
        dict with all extracted features (NO distance filtering)
    """
    sample_dir = Path(sample_info['path'])
    sample_name = sample_info['name']
    
    # Find structure file
    structure_file = None
    for ext in ['cif', 'pdb']:
        candidate = sample_dir / f'model.{ext}'
        if candidate.exists():
            structure_file = candidate
            break
    
    if structure_file is None:
        raise FileNotFoundError(f"No model file found in {sample_dir}")
    
    confidences_file = sample_dir / 'confidences.json'
    
    # Load structure
    structure = load_structure(structure_file)
    
    # Extract features (ALL pairs, no distance filtering yet)
    features = extract_interchain_features(structure, ab_chains, ag_chains)
    
    # Load confidences
    conf_data = load_confidences(confidences_file)
    
    # Extract interchain PAE values
    interchain_pae = extract_interchain_pae(
        conf_data['pae_matrix'],
        features['inter_idx'],
        features['inter_jdx']
    )
    
    # Get ranking score from CSV if available, otherwise fall back to confidences.json
    if ranking_scores_dict and sample_name in ranking_scores_dict:
        ranking_score = ranking_scores_dict[sample_name]
    else:
        ranking_score = conf_data['ranking_score']
    
    # Extract full matrices and chain info
    # Get all residues with CA atoms
    all_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    all_residues.append(residue)
    
    # Get chain IDs for all residues
    chain_ids = []
    for res in all_residues:
        chain_ids.append(res.get_parent().get_id())
    
    # Build full CA distance matrix
    n_residues = len(all_residues)
    ca_distance_matrix = np.zeros((n_residues, n_residues), dtype=np.float32)
    for i, res_i in enumerate(all_residues):
        ca_i = res_i['CA'].get_coord()
        for j, res_j in enumerate(all_residues):
            ca_j = res_j['CA'].get_coord()
            ca_distance_matrix[i, j] = np.linalg.norm(ca_i - ca_j)
    
    # Prepare output (NO distance filtering - done at inference time)
    output = {
        'sample_name': sample_name,
        'inter_idx': features['inter_idx'],
        'inter_jdx': features['inter_jdx'],
        'interchain_pae_vals': interchain_pae,
        'interchain_ca_distances': features['ca_distances'],
        'residue_one_letter': features['residue_one_letter'],
        'ranking_score': ranking_score,
        'ptm': conf_data['ptm'],
        'iptm': conf_data['iptm'],
        # Add full matrices for proper centering at inference
        'pae_matrix': conf_data['pae_matrix'],
        'ca_distance_matrix': ca_distance_matrix,
        'chain_ids': chain_ids
    }
    
    # Add pre-computed ESM embeddings if provided
    if esm_embeddings is not None:
        output['esm_embeddings'] = esm_embeddings
    
    return output


def compute_pae_statistics(all_samples):
    """
    Compute mean, median, std of PAE across all samples (no distance filtering yet).
    This matches the training data preprocessing where stats are computed on ALL pairs.
    """
    pae_arrays = [s['interchain_pae_vals'] for s in all_samples]
    
    # Only compute stats if all samples have same length
    lengths = [len(p) for p in pae_arrays]
    if len(set(lengths)) == 1:
        # All same length - can compute stats
        pae_matrix = np.stack(pae_arrays, axis=0)
        stats = {
            'pae_col_mean': np.mean(pae_matrix, axis=0),
            'pae_col_median': np.median(pae_matrix, axis=0),
            'pae_col_std': np.std(pae_matrix, axis=0)
        }
        print(f"  Computed PAE statistics: {len(pae_arrays)} samples, {len(stats['pae_col_mean'])} pairs each")
        return stats
    else:
        print(f"  WARNING: Cannot compute PAE statistics - samples have different pair counts: {set(lengths)}")
        return None


def save_to_h5(output_file, all_samples, target_id='target'):
    """
    Save all samples to a single HDF5 file.
    
    Structure:
        target_id/
            pae_col_mean (if available)
            pae_col_median (if available)
            pae_col_std (if available)
            sample_0/
                inter_idx
                inter_jdx
                interchain_pae_vals
                interchain_ca_distances
                residue_one_letter
                ranking_score
                ptm
                iptm
                esm_embeddings (if computed)
            sample_1/
                ...
    """
    with h5py.File(output_file, 'w') as hf:
        # Create target group
        target_group = hf.create_group(target_id)
        
        # Compute and save PAE statistics if possible
        pae_stats = compute_pae_statistics(all_samples)
        if pae_stats is not None:
            for key, value in pae_stats.items():
                target_group.create_dataset(key, data=value, compression='gzip')
        
        # Save each sample
        for i, sample_data in enumerate(all_samples):
            sample_name = sample_data['sample_name']
            sample_group = target_group.create_group(sample_name)
            
            # Save all datasets
            for key, value in sample_data.items():
                if key == 'sample_name':  # Skip non-data keys
                    continue
                
                # Handle string arrays
                if key == 'residue_one_letter':
                    value = np.array(value, dtype='S1')
                elif key == 'chain_ids':
                    value = np.array([s.encode('utf-8') for s in value], dtype='S10')
                
                # Save with compression for large arrays
                if isinstance(value, np.ndarray) and value.size > 100:
                    sample_group.create_dataset(key, data=value, compression='gzip')
                else:
                    sample_group.create_dataset(key, data=value)
    
    print(f"\nSaved {len(all_samples)} samples to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess AF3 predictions for inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing AF3 prediction samples')
    parser.add_argument('--output_h5', type=str, required=True,
                        help='Output HDF5 file path')
    
    # Chain specification
    parser.add_argument('--antibody_chains', type=str, required=True,
                        help='Antibody chain IDs (comma-separated, e.g., "H,L")')
    parser.add_argument('--antigen_chains', type=str, required=True,
                        help='Antigen chain IDs (comma-separated, e.g., "A")')
    
    # Optional
    parser.add_argument('--target_id', type=str, default='target',
                        help='Target ID to use in HDF5 file (default: "target")')
    parser.add_argument('--add_esm', action='store_true',
                        help='Compute and add ESM embeddings')
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help='ESM model name (default: esm2_t6_8M_UR50D, 320-dim)')
    parser.add_argument('--run_dirs', type=str, default='RUN01,RUN02,RUN03,RUN04,RUN05',
                        help='Comma-separated list of RUN directories to process (default: RUN01,RUN02,RUN03,RUN04,RUN05). Use empty string for flat structure.')
    
    args = parser.parse_args()
    
    # Parse chain IDs
    ab_chains = [c.strip() for c in args.antibody_chains.split(',')]
    ag_chains = [c.strip() for c in args.antigen_chains.split(',')]
    
    # Parse RUN directories
    run_dirs = None
    if args.run_dirs.strip():
        run_dirs = [r.strip() for r in args.run_dirs.split(',') if r.strip()]
    
    print("="*80)
    print("PREPROCESSING AF3 PREDICTIONS FOR INFERENCE")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output HDF5:      {args.output_h5}")
    print(f"Target ID:        {args.target_id}")
    print(f"Antibody chains:  {ab_chains}")
    print(f"Antigen chains:   {ag_chains}")
    print(f"RUN directories:  {run_dirs if run_dirs else 'Flat structure (no RUN subdirs)'}")
    print(f"Add ESM:          {args.add_esm}")
    print()
    
    # Load ranking scores from CSV files
    print("Loading AF3 ranking scores from CSV files...")
    ranking_scores_dict = load_ranking_scores(args.input_dir, run_dirs=run_dirs)
    
    # Find all samples
    print("\nDiscovering prediction samples...")
    samples = find_prediction_samples(args.input_dir, run_dirs=run_dirs)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print("ERROR: No valid samples found!")
        return
    
    # Compute ESM embeddings once if requested (same sequence for all samples)
    esm_embeddings = None
    
    if args.add_esm:
        if not ESM_AVAILABLE:
            print("ERROR: ESM requested but transformers not installed!")
            print("Install with: pip install transformers")
            return
        
        print(f"\nLoading ESM model: {args.esm_model}")
        esm_tokenizer = EsmTokenizer.from_pretrained(args.esm_model)
        esm_model = EsmModel.from_pretrained(args.esm_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esm_model = esm_model.to(device)
        esm_model.eval()
        print(f"ESM model loaded on {device}")
        
        # Compute ESM embeddings once from first sample (same sequence for all)
        print("\nComputing ESM embeddings (once, reused for all samples)...")
        first_sample = samples[0]
        first_dir = Path(first_sample['path'])
        
        # Load first structure to get sequence
        for ext in ['cif', 'pdb']:
            structure_file = first_dir / f'model.{ext}'
            if structure_file.exists():
                structure = load_structure(structure_file)
                features = extract_interchain_features(structure, ab_chains, ag_chains)
                esm_embeddings = compute_esm_embeddings(
                    features['residue_one_letter'],
                    esm_model, esm_tokenizer, device
                )
                print(f"  ESM embeddings computed: shape {esm_embeddings.shape}")
                break
        
        # Free up GPU memory
        del esm_model
        del esm_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process all samples (NO distance filtering - done at inference time)
    print("\nProcessing samples...")
    all_samples = []
    failed = 0
    
    for sample_info in tqdm(samples):
        try:
            sample_data = process_sample(
                sample_info, ab_chains, ag_chains,
                esm_embeddings=esm_embeddings,
                ranking_scores_dict=ranking_scores_dict
            )
            all_samples.append(sample_data)
        except Exception as e:
            print(f"\nERROR processing {sample_info['name']}: {e}")
            failed += 1
            continue
    
    print(f"\nSuccessfully processed: {len(all_samples)}/{len(samples)}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    # Save to HDF5
    if len(all_samples) > 0:
        print("\nSaving to HDF5...")
        save_to_h5(args.output_h5, all_samples, args.target_id)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Output file: {args.output_h5}")
        print(f"Total samples: {len(all_samples)}")
        print("\nNext step: Run inference with run_inference.py")
    else:
        print("\nERROR: No samples were successfully processed!")


if __name__ == '__main__':
    main()

