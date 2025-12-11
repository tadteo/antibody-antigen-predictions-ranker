#!/usr/bin/env python3
"""
Simple inference script for antibody-antigen ranking model.

Loads preprocessed HDF5 file directly into memory and runs predictions.

Usage:
    python run_inference.py \
        --h5_file inference_data.h5 \
        --checkpoint path/to/model_checkpoint.pt \
        --config path/to/config.yaml \
        --output_dir results/

This will generate 3 CSV files with different ranking strategies:
    - ranked_by_af3.csv: Ordered by original AF3 ranking score
    - ranked_by_model.csv: Ordered by model predicted DockQ
    - ranked_by_combined.csv: Ordered by combined score (max of both normalized)
"""

import argparse
import os
import yaml
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Import model and data utilities
from src.models.deep_set import DeepSet
from src.data.triangular_positional_encoding import triangular_encode_features


def load_config(config_path):
    """Load training configuration."""
    return OmegaConf.load(config_path)


def load_model(checkpoint_path, config):
    """
    Load trained model from checkpoint.
    
    Returns:
        model: loaded model in eval mode
        device: torch device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate input dimension (must match training)
    input_dim_base = int(config.model.input_dim)
    use_ca = config.data.get('use_interchain_ca_distances', True)
    use_pae = config.data.get('use_interchain_pae', True)
    use_esm = config.data.get('use_esm_embeddings', True)
    esm_dim = config.data.get('esm_embedding_dim', 320)
    
    input_dim = input_dim_base
    input_dim += (1 if use_ca else 0)
    input_dim -= (0 if use_pae else 2)
    if use_esm:
        input_dim += 2 * esm_dim
    
    print(f"Model input dimension: {input_dim}")
    print(f"  Base: {input_dim_base}")
    print(f"  CA distances: {use_ca}")
    print(f"  PAE: {use_pae}")
    print(f"  ESM embeddings: {use_esm} (dim={esm_dim})")
    
    # Initialize model
    model = DeepSet(
        input_dim=input_dim,
        phi_hidden_dims=config.model.phi_hidden_dims,
        rho_hidden_dims=config.model.rho_hidden_dims,
        aggregator=config.model.aggregator
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Device: {device}")
    
    return model, device


def load_h5_into_memory(h5_file, target_id=None, antibody_chains=None, antigen_chains=None):
    """
    Load entire HDF5 file into memory.
    IMPORTANT: Computes PAE statistics ACROSS ALL SAMPLES for proper centering.
    
    Returns:
        list of dict: each dict contains all features for one sample
    """
    samples = []
    
    with h5py.File(h5_file, 'r') as hf:
        # Get target ID
        if target_id is None:
            target_id = list(hf.keys())[0]
        
        target_group = hf[target_id]
        
        # Load all samples first
        for sample_name in target_group.keys():
            item = target_group[sample_name]
            
            # Skip non-group items
            if not isinstance(item, h5py.Group):
                continue
            
            # Load all data for this sample
            sample_data = {
                'target_id': target_id,
                'sample_name': sample_name
            }
            
            # Load all datasets
            for key in item.keys():
                data = item[key][()]
                
                # Convert bytes to strings if needed
                if key == 'residue_one_letter' and isinstance(data[0], bytes):
                    data = [d.decode('utf-8') for d in data]
                elif key == 'chain_ids' and isinstance(data[0], bytes):
                    data = [d.decode('utf-8') for d in data]
                
                sample_data[key] = data
            
            samples.append(sample_data)
    
    print(f"Loaded {len(samples)} samples into memory")
    
    # Compute PAE statistics ACROSS ALL SAMPLES (critical for proper centering!)
    if len(samples) > 0 and 'pae_matrix' in samples[0]:
        print("Computing PAE statistics across all samples...")
        
        # Extract interchain PAE values for all samples
        all_interchain_pae = []
        for sample in samples:
            # Get interchain pairs (before any filtering)
            inter_idx = sample['inter_idx']
            inter_jdx = sample['inter_jdx']
            pae_matrix = sample['pae_matrix']
            
            # Extract interchain PAE values
            interchain_pae_vals = pae_matrix[inter_idx, inter_jdx]
            all_interchain_pae.append(interchain_pae_vals)
        
        # Check if all samples have same number of pairs
        pair_counts = [len(p) for p in all_interchain_pae]
        if len(set(pair_counts)) == 1:
            # All same - can compute statistics
            pae_matrix_stacked = np.stack(all_interchain_pae, axis=0)  # [num_samples, num_pairs]
            pae_col_mean = np.mean(pae_matrix_stacked, axis=0)  # [num_pairs]
            pae_col_std = np.std(pae_matrix_stacked, axis=0)    # [num_pairs]
            
            print(f"  Computed PAE statistics: {len(samples)} samples, {len(pae_col_mean)} pairs")
            print(f"  Mean PAE range: [{pae_col_mean.min():.2f}, {pae_col_mean.max():.2f}]")
            
            # Add to all samples
            for sample in samples:
                sample['pae_col_mean'] = pae_col_mean
                sample['pae_col_std'] = pae_col_std
        else:
            print(f"  WARNING: Samples have different pair counts: {set(pair_counts)}")
            print(f"  Cannot compute PAE statistics")
    
    return samples


def prepare_features(sample, config):
    """
    Prepare features EXACTLY like training dataloader.
    CRITICAL: Must apply X-residue filtering AND distance filtering in same order as training!
    
    Returns:
        torch.Tensor: features [F, N]
    """
    use_pae = config.data.get('use_interchain_pae', True)
    use_ca = config.data.get('use_interchain_ca_distances', True)
    use_esm = config.data.get('use_esm_embeddings', True)
    use_centering = config.data.get('feature_centering', True)
    use_distance_cutoff = config.data.get('use_distance_cutoff', True)
    distance_cutoff = config.data.get('distance_cutoff', 12.0)
    
    # Load raw data
    inter_idx_i = sample['inter_idx'].copy()
    inter_idx_j = sample['inter_jdx'].copy()
    interchain_pae = sample['interchain_pae_vals'].copy() if 'interchain_pae_vals' in sample else None
    ca_dist = sample['interchain_ca_distances'].copy() if 'interchain_ca_distances' in sample else None
    residue_one_letter = sample.get('residue_one_letter', None)
    
    # STEP 1: Filter X residues (EXACTLY like training dataloader)
    n_pairs_initial = len(inter_idx_i)
    n_pairs_after_x_filter = n_pairs_initial
    n_pairs_after_distance_filter = n_pairs_initial
    
    if residue_one_letter is not None:
        # Decode if bytes
        if isinstance(residue_one_letter[0], bytes):
            residue_one_letter = [r.decode('utf-8') for r in residue_one_letter]
        
        # Create mask for non-X residues
        non_x_mask = np.array([res != 'X' for res in residue_one_letter])
        
        # Filter pairs: keep only if BOTH residues are not X
        valid_pair_mask = non_x_mask[inter_idx_i] & non_x_mask[inter_idx_j]
        n_pairs_after_x_filter = valid_pair_mask.sum()
        
        # Apply mask to all data
        inter_idx_i = inter_idx_i[valid_pair_mask]
        inter_idx_j = inter_idx_j[valid_pair_mask]
        if interchain_pae is not None:
            interchain_pae = interchain_pae[valid_pair_mask]
        if ca_dist is not None:
            ca_dist = ca_dist[valid_pair_mask]
        
        # Remap indices (because X residues removed from sequence)
        non_x_indices = np.where(non_x_mask)[0]
        old_to_new_idx = np.full(len(residue_one_letter), -1, dtype=int)
        old_to_new_idx[non_x_indices] = np.arange(len(non_x_indices))
        inter_idx_i = old_to_new_idx[inter_idx_i]
        inter_idx_j = old_to_new_idx[inter_idx_j]
    else:
        valid_pair_mask = None
    
    # STEP 2: Apply distance cutoff filtering (EXACTLY like training dataloader)
    distance_mask = None
    if use_distance_cutoff and ca_dist is not None:
        distance_mask = ca_dist <= distance_cutoff
        n_pairs_after_distance_filter = distance_mask.sum()
        
        # If ALL pairs filtered, keep closest (matches training)
        if distance_mask.sum() == 0:
            closest_idx = np.argmin(ca_dist)
            distance_mask = np.zeros(len(ca_dist), dtype=bool)
            distance_mask[closest_idx] = True
            n_pairs_after_distance_filter = 1
        
        # Apply mask to all arrays
        inter_idx_i = inter_idx_i[distance_mask]
        inter_idx_j = inter_idx_j[distance_mask]
        if interchain_pae is not None:
            interchain_pae = interchain_pae[distance_mask]
        ca_dist = ca_dist[distance_mask]
    
    # DEBUG: Show filtering statistics for first sample
    if not hasattr(prepare_features, '_filtering_shown'):
        print(f"\n[DEBUG] Pair Filtering Statistics:")
        print(f"  Initial pairs: {n_pairs_initial}")
        print(f"  After X-residue filter: {n_pairs_after_x_filter} (removed {n_pairs_initial - n_pairs_after_x_filter})")
        print(f"  After distance cutoff ({distance_cutoff}Å): {n_pairs_after_distance_filter} (removed {n_pairs_after_x_filter - n_pairs_after_distance_filter})")
        print(f"  Final pairs: {len(inter_idx_i)}")
        print(f"  Total removed: {n_pairs_initial - len(inter_idx_i)} ({100 * (n_pairs_initial - len(inter_idx_i)) / n_pairs_initial:.1f}%)")
        if ca_dist is not None and len(ca_dist) > 0:
            print(f"  Distance range of kept pairs: [{ca_dist.min():.2f}, {ca_dist.max():.2f}] Å")
        prepare_features._filtering_shown = True
    
    # STEP 3: Filter pae_col_mean with BOTH masks (EXACTLY like training)
    if use_centering and 'pae_col_mean' in sample:
        pae_col_mean_filtered = sample['pae_col_mean'].copy()
        # Apply valid_pair_mask first
        if valid_pair_mask is not None:
            pae_col_mean_filtered = pae_col_mean_filtered[valid_pair_mask]
        # Then apply distance_mask
        if distance_mask is not None:
            pae_col_mean_filtered = pae_col_mean_filtered[distance_mask]
    else:
        pae_col_mean_filtered = None
    
    # Build base features
    if use_pae:
        if use_centering and pae_col_mean_filtered is not None:
            # Centered PAE features
            pae_centered = interchain_pae - pae_col_mean_filtered
            
            # DEBUG: Check if centering produces all zeros (bug indicator!)
            if not hasattr(prepare_features, '_centering_checked'):
                print(f"\n[DEBUG] PAE Centering Check:")
                print(f"  interchain_pae: min={interchain_pae.min():.4f}, max={interchain_pae.max():.4f}, mean={interchain_pae.mean():.4f}")
                print(f"  pae_col_mean: min={pae_col_mean_filtered.min():.4f}, max={pae_col_mean_filtered.max():.4f}, mean={pae_col_mean_filtered.mean():.4f}")
                print(f"  pae_centered: min={pae_centered.min():.4f}, max={pae_centered.max():.4f}, mean={pae_centered.mean():.4f}")
                if np.abs(pae_centered).max() < 1e-6:
                    print(f"  WARNING: pae_centered is all zeros! Centering is broken!")
                prepare_features._centering_checked = True
            
            if use_ca:
                feats = np.array([
                    inter_idx_i,
                    inter_idx_j,
                    pae_col_mean_filtered,
                    pae_centered,
                    ca_dist
                ], dtype=np.float32)
            else:
                feats = np.array([
                    inter_idx_i,
                    inter_idx_j,
                    pae_col_mean_filtered,
                    pae_centered
                ], dtype=np.float32)
        else:
            # Non-centered PAE features
            if use_ca:
                feats = np.array([
                    inter_idx_i,
                    inter_idx_j,
                    interchain_pae,
                    ca_dist
                ], dtype=np.float32)
            else:
                feats = np.array([
                    inter_idx_i,
                    inter_idx_j,
                    interchain_pae
                ], dtype=np.float32)
    else:
        # No PAE - only CA distances
        feats = np.array([
            inter_idx_i,
            inter_idx_j,
            ca_dist
        ], dtype=np.float32)
    
    # STEP 4: Add ESM embeddings (filter by non_x_mask first, EXACTLY like training)
    if use_esm and 'esm_embeddings' in sample:
        esm_emb_raw = sample['esm_embeddings']  # [L, d_esm]
        
        # Filter ESM embeddings by non_x_mask if X residues were filtered
        if residue_one_letter is not None:
            non_x_mask = np.array([res != 'X' for res in residue_one_letter])
            esm_emb = esm_emb_raw[non_x_mask]  # [L', d_esm]
        else:
            esm_emb = esm_emb_raw
        
        # Extract embeddings for interchain pairs (using already-filtered indices)
        esm_i = esm_emb[inter_idx_i].T  # [d_esm, N]
        esm_j = esm_emb[inter_idx_j].T  # [d_esm, N]
        feats = np.vstack([feats, esm_i, esm_j])
    
    # Apply triangular encoding if configured
    if config.data.get('feature_transform', False):
        feats_before = feats.copy()
        feats = triangular_encode_features(feats)
        # DEBUG: Only print for first sample
        if not hasattr(prepare_features, '_debug_printed'):
            print(f"\n[DEBUG] Feature preparation:")
            print(f"  Features before triangular encoding: shape {feats_before.shape}")
            print(f"    Min: {feats_before.min():.4f}, Max: {feats_before.max():.4f}")
            print(f"  Features after triangular encoding: shape {feats.shape}")
            print(f"    Min: {feats.min():.4f}, Max: {feats.max():.4f}")
            print(f"  Number of pairs after filtering: {feats.shape[1]}")
            prepare_features._debug_printed = True
    
    return torch.from_numpy(feats).float()


def run_inference_batch(model, batch_features, batch_lengths, device, samples_per_batch):
    """
    Run model inference on a batch.
    Groups samples together to match training setup (K samples per batch item).
    
    Args:
        model: trained model
        batch_features: list of feature tensors (length = B*K)
        batch_lengths: list of sequence lengths (length = B*K)
        device: torch device
        samples_per_batch: K - number of samples per batch item (matches training)
    
    Returns:
        predictions: numpy array of predicted DockQ scores
    """
    # Group samples into batches of K
    num_samples = len(batch_features)
    B = (num_samples + samples_per_batch - 1) // samples_per_batch  # ceiling division
    
    max_len = max(batch_lengths)
    feature_dim = batch_features[0].shape[0]
    
    # Create padded tensor [B, K, N, F] - matches training
    padded = torch.zeros(B, samples_per_batch, max_len, feature_dim)
    lengths_tensor = torch.zeros(B, samples_per_batch, dtype=torch.long)
    
    for i in range(num_samples):
        batch_idx = i // samples_per_batch
        sample_idx = i % samples_per_batch
        feat = batch_features[i]
        N = feat.shape[1]
        padded[batch_idx, sample_idx, :N, :] = feat.T  # Transpose to [N, F]
        lengths_tensor[batch_idx, sample_idx] = batch_lengths[i]
    
    # Move to device
    padded = padded.to(device)
    lengths_tensor = lengths_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(padded, lengths_tensor)  # [B, K]
        
        # DEBUG: Print logit statistics
        print(f"\n[DEBUG] Logits stats:")
        print(f"  Shape: {logits.shape}")
        print(f"  Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
        print(f"  Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
        print(f"  Sample logits: {logits.flatten()[:10].cpu().numpy()}")
        
        predictions = torch.sigmoid(logits).cpu().numpy().flatten()
        
        print(f"[DEBUG] Predictions after sigmoid:")
        print(f"  Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
    
    # Only return predictions for actual samples (not padding)
    return predictions[:num_samples]


def normalize_scores(scores):
    """Normalize scores to [0, 1] range."""
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score < 1e-8:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)


def save_rankings(samples, predictions, output_dir, target_id):
    """
    Save three different ranking strategies to CSV files.
    
    1. Ranked by AF3 original ranking score
    2. Ranked by model predicted DockQ
    3. Ranked by combined score (sum of raw scores)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data (KISS - only essential columns)
    data = []
    for i, sample in enumerate(samples):
        data.append({
            'target_id': sample['target_id'],
            'sample_name': sample['sample_name'],
            'af3_ranking_score': sample['ranking_score'],
            'model_predicted_dockq': predictions[i]
        })
    
    df = pd.DataFrame(data)
    
    # Combined score: simple sum (KISS)
    df['combined_score'] = df['af3_ranking_score'] + df['model_predicted_dockq']
    
    # Strategy 1: Rank by AF3 score (higher is better)
    df_af3 = df.sort_values('af3_ranking_score', ascending=False).reset_index(drop=True)
    df_af3['rank'] = df_af3.index + 1
    output_file = Path(output_dir) / f'{target_id}_ranked_by_af3.csv'
    df_af3.to_csv(output_file, index=False)
    print(f"\nSaved AF3-ranked results: {output_file}")
    
    # Strategy 2: Rank by model prediction (higher is better)
    df_model = df.sort_values('model_predicted_dockq', ascending=False).reset_index(drop=True)
    df_model['rank'] = df_model.index + 1
    output_file = Path(output_dir) / f'{target_id}_ranked_by_model.csv'
    df_model.to_csv(output_file, index=False)
    print(f"Saved model-ranked results: {output_file}")
    
    # Strategy 3: Rank by combined score (higher is better)
    df_combined = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    df_combined['rank'] = df_combined.index + 1
    output_file = Path(output_dir) / f'{target_id}_ranked_by_combined.csv'
    df_combined.to_csv(output_file, index=False)
    print(f"Saved combined-ranked results: {output_file}")
    
    # Print top 5 from each strategy
    print("\n" + "="*80)
    print("TOP 5 PREDICTIONS BY DIFFERENT RANKING STRATEGIES")
    print("="*80)
    
    print("\n1. Ranked by AF3 Score:")
    print(df_af3[['rank', 'sample_name', 'af3_ranking_score', 'model_predicted_dockq']].head())
    
    print("\n2. Ranked by Model Prediction:")
    print(df_model[['rank', 'sample_name', 'model_predicted_dockq', 'af3_ranking_score']].head())
    
    print("\n3. Ranked by Combined Score (sum of raw scores):")
    print(df_combined[['rank', 'sample_name', 'combined_score', 'af3_ranking_score', 'model_predicted_dockq']].head())
    
    # Correlation between AF3 and model scores
    corr = df['af3_ranking_score'].corr(df['model_predicted_dockq'])
    print(f"\nCorrelation between AF3 and model scores: {corr:.3f}")
    
    return df_af3, df_model, df_combined


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on preprocessed AF3 predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--h5_file', type=str, required=True,
                        help='Preprocessed HDF5 file from preprocess_inference.py')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml used during training')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save ranking results')
    parser.add_argument('--target_id', type=str, default=None,
                        help='Target ID in HDF5 file (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle samples before inference (for better batch variety)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RUNNING INFERENCE ON AF3 PREDICTIONS")
    print("="*80)
    print(f"HDF5 file:    {args.h5_file}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Config:       {args.config}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Shuffle:      {args.shuffle}")
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load model
    print("\nLoading model...")
    model, device = load_model(args.checkpoint, config)
    
    # Load H5 file into memory
    print(f"\nLoading HDF5 file: {args.h5_file}")
    samples = load_h5_into_memory(args.h5_file, args.target_id)
    
    # Get target ID
    target_id = samples[0]['target_id']
    print(f"Target ID: {target_id}")
    
    # Shuffle if requested (optional - all samples are from same complex)
    if args.shuffle:
        print("Shuffling samples...")
        print("  Note: All samples are from the SAME complex/target")
        print("  Shuffling affects which samples end up in each K-group")
        np.random.shuffle(samples)
    
    # Prepare features
    print("\nPreparing features...")
    all_features = []
    all_lengths = []
    
    for sample in tqdm(samples, desc="Preparing features"):
        features = prepare_features(sample, config)
        all_features.append(features)
        all_lengths.append(features.shape[1])
    
    # Run inference
    # CRITICAL: All samples are from the SAME complex!
    # For CAPRI: 500 samples of one target should be grouped together
    # Use larger K to match training semantics (K samples from same complex)
    
    # Option 1: Use all samples as one batch (best for accuracy)
    # samples_per_batch = len(samples)  # All 500 as K=500
    
    # Option 2: Use larger groups (better for memory)
    samples_per_batch = min(100, len(samples))  # K=100 or all if fewer
    
    print(f"\nRunning inference ({len(samples)} samples from ONE complex, grouped as K={samples_per_batch})...")
    all_predictions = []
    
    # Process in chunks - each chunk represents samples from the SAME complex
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Inference"):
        batch_end = min(i + args.batch_size, len(samples))
        batch_features = all_features[i:batch_end]
        batch_lengths = all_lengths[i:batch_end]
        
        predictions = run_inference_batch(model, batch_features, batch_lengths, device, samples_per_batch)
        all_predictions.extend(predictions)
    
    all_predictions = np.array(all_predictions)
    
    print(f"\nInference complete!")
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
    print(f"Mean prediction: {all_predictions.mean():.4f}")
    
    # Save rankings
    print("\nGenerating rankings...")
    save_rankings(samples, all_predictions, args.output_dir, target_id)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - {target_id}_ranked_by_af3.csv")
    print(f"  - {target_id}_ranked_by_model.csv")
    print(f"  - {target_id}_ranked_by_combined.csv")


if __name__ == '__main__':
    main()

