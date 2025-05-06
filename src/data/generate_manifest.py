#!/usr/bin/env python3
"""
generate_manifest.py

Scans a directory of antibody–antigen H5 files, extracts DockQ labels,
assigns global DockQ buckets, computes per-sample weights, and splits into
train/val/test with uniform DockQ distribution in validation/test.

Usage:
    python generate_manifest.py --h5_dir /path/to/h5_dir \
        [--val_frac 0.1] [--test_frac 0.1] [--seed 42] \
        --out_csv data/manifest.csv

Default splits:  train: 1 - val_frac - test_frac,
                 val: val_frac,
                 test: test_frac.
Uniformity across buckets ensured by stratified sampling per bucket.
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import yaml

# Global DockQ bin edges
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.00]
NUM_BUCKETS = len(BIN_EDGES) - 1


def build_manifest(h5_dir, val_frac, test_frac, seed, out_csv, config):
    rows = []
    num_complexes = 0
    # 1. iterate complexes

    # scan h5 files
    for fname in sorted(os.listdir(h5_dir)):
        if not fname.endswith('.h5'):
            continue
        h5_path = os.path.join(h5_dir, fname)
        # print(f"H5 path: {h5_path}")
        with h5py.File(h5_path, 'r') as hf:
            
            # get complex_id
            for complex_id in hf.keys():
                id = complex_id
                is_complex_valid = True
                for sample in hf[complex_id].keys():
                    dockq = float(hf[f"{complex_id}/{sample}/abag_dockq"][()])
                    ptm = float(hf[f"{complex_id}/{sample}/ptm"][()])
                    iptm = float(hf[f"{complex_id}/{sample}/iptm"][()])
                    ranking_confidence = float(hf[f"{complex_id}/{sample}/ranking_confidence"][()])
                    tm_normalized = float(hf[f"{complex_id}/{sample}/tm_normalized_reference"][()])
                    bucket = np.digitize(dockq, BIN_EDGES, right=False) - 1
                    # Skip if DockQ is NaN
                    if np.isnan(dockq):
                        is_complex_valid = False
                        break
                    else:
                        rows.append({
                            'complex_id': id,
                            'h5_file': h5_path,
                            'sample': sample,
                            'len_sample': len(hf[f"{complex_id}/{sample}/interchain_pae_vals"][()]),
                            'label': dockq,
                            'ptm': ptm,
                            'iptm': iptm,
                            'ranking_confidence': ranking_confidence,
                            'tm_normalized': tm_normalized,
                            'bucket': int(bucket),
                            'weight_bucket': None,
                            'weight': None,
                            'split': None
                        })

                if is_complex_valid:
                    num_complexes += 1


    print(f"Found {num_complexes} valid complexes with a total of {len(rows)} samples")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No samples found in {h5_dir}")

    # 2. Compute each complex's mean and std DockQ and create split train/val/test to ensure uniform distribution
    # Compute mean and std DockQ for each complex and create a dictionary with complex_id as key and mean and std as values
    
    n_bins_for_strata = 5
    # 1) Per‐complex stats
    complex_stats = (
        df
        .groupby('complex_id')['label']
        .agg(['mean','std'])
        .reset_index()
        .rename(columns={'mean':'mean_dockq','std':'std_dockq'})
    )

    #check the means and stds of all the complexes means and stds
    print(f"Mean of means: {complex_stats.mean_dockq.mean()}")
    print(f"Std of means: {complex_stats.mean_dockq.std()}")
    print(f"Mean of stds: {complex_stats.std_dockq.mean()}")
    print(f"Std of stds: {complex_stats.std_dockq.std()}")

    # 2) Quantile‐bin mean & std
    complex_stats['mean_bin'] = pd.qcut(
        complex_stats['mean_dockq'],
        q=n_bins_for_strata, labels=False, duplicates='drop'
    )
    # fill NaN std (if any constant‐score complexes) with 0 before binning
    complex_stats['std_bin'] = pd.qcut(
        complex_stats['std_dockq'].fillna(0),
        q=n_bins_for_strata, labels=False, duplicates='drop'
    )
    
    # 3) Joint strata
    complex_stats['strata'] = (
        complex_stats['mean_bin'].astype(str) + '_' + complex_stats['std_bin'].astype(str)
    )
    
    # 4) Merge any strata with <2 complexes into a single 'other' bucket
    counts = complex_stats['strata'].value_counts()
    small = counts[counts < 2].index
    if len(small) > 0:
        complex_stats.loc[complex_stats['strata'].isin(small), 'strata'] = 'other'
    
    # 5) Perform splits, with fallback to mean‐only if needed
    try:
        # first test split
        train_val, test = train_test_split(
            complex_stats,
            test_size=test_frac,
            stratify=complex_stats['strata'],
            random_state=seed
        )
        # then train/val split
        train, val = train_test_split(
            train_val,
            test_size=val_frac/(1-test_frac),
            stratify=train_val['strata'],
            random_state=seed
        )
    except ValueError:
        # fallback: stratify only on mean_bin
        print("WARNING: joint mean+std stratification failed; falling back to mean‐only stratification")
        train_val, test = train_test_split(
            complex_stats,
            test_size=test_frac,
            stratify=complex_stats['mean_bin'],
            random_state=seed
        )
        train, val = train_test_split(
            train_val,
            test_size=val_frac/(1-test_frac),
            stratify=train_val['mean_bin'],
            random_state=seed
        )
    
    # 6) Map splits back onto the full df
    split_map = (
        pd.concat([
            train.assign(split='train'),
            val.assign(split='val'),
            test.assign(split='test')
        ])
        .set_index('complex_id')['split']
    )
    df['split'] = df['complex_id'].map(split_map)
    # Also add split to the complex_stats table
    complex_stats['split'] = complex_stats['complex_id'].map(split_map)
    
    # Sanity check
    print(f"Splitting {len(complex_stats)} complexes → "
        f"{len(train)} train, {len(val)} val, {len(test)} test")

    for split in ['train','val','test']:
        # std/mean over all labels in the split
        grp = df[df['split']==split]['label']
        print(f"  {split:5s} | complexes={df[df['split']==split]['complex_id'].nunique():3d} | "
            f"mean={grp.mean():.3f}, std={grp.std():.3f}")

        # per‐complex stds and means in that split
        sub_std  = complex_stats[complex_stats['split']==split]['std_dockq']
        sub_mean = complex_stats[complex_stats['split']==split]['mean_dockq']
        print(f"    per‐complex std:  mean={sub_std.mean():.3f}, std={sub_std.std():.3f}")
        print(f"    per‐complex mean: mean={sub_mean.mean():.3f}, std={sub_mean.std():.3f}")


    # Add the split and weight column to the df
    df['split'] = df['complex_id'].map(split_map)

    #for each split, compute the weight of the complexes
    for split in ['train','val','test']:
        sub_df = df[df['split']==split]
        
        # compute weight for each bucket    
        bucket_counts = sub_df['bucket'].value_counts().to_dict()
        sub_df['weight_bucket'] = sub_df['bucket'].map(lambda b: 1.0 / bucket_counts[b])

        #divide the weight_bucket by the number of buckets
        sub_df['weight_bucket'] = sub_df['weight_bucket'] / NUM_BUCKETS

        # final weight is product of the two:
        # df['weight'] = df['weight_complex']
        sub_df['weight'] = sub_df['weight_bucket']

        #add the weight to the df
        df.loc[df['split']==split, 'weight_bucket'] = sub_df['weight_bucket']
        df.loc[df['split']==split, 'weight'] = sub_df['weight']


    #check for each split that the weight bucket is correct
    for split in ['train','val','test']:
        sub_df = df[df['split']==split]
        #count the number of elements in each bucket
        bucket_counts = sub_df['bucket'].value_counts().to_dict()
        # calculate the weight of each bucket
        bucket_weights = []
        for b in bucket_counts:
            bucket_weights.append(bucket_counts[b]/len(sub_df))

        for bucket in bucket_counts:
            print(f"The weight bucket for the {split} split is {bucket_weights[bucket]}")
        print(f"The sum of the weights for the {split} split is {sum(bucket_weights)}")
        
    #check that the sum of the weights is 1 for each split
    for split in ['train','val','test']:
        sub_df = df[df['split']==split]
        print(f"The sum of the weights for the {split} split is {sub_df['weight'].sum()}")

    # Write to CSV the manifest for each sample and apply the correct split to each sample
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate stratified manifest.csv with uniform dockq splits'
    )
    p.add_argument('--h5_dir', required=True,
                   help='Directory of complex subfolders containing .h5 files')
    p.add_argument('--val_frac', type=float, default=0.1,
                   help='Fraction for validation split')
    p.add_argument('--test_frac', type=float, default=0.1,
                   help='Fraction for test split')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--out_csv', default='data/manifest_with_ptm_no_normalization.csv',
                   help='Output path for manifest CSV')
    p.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML config file')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.val_frac + args.test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    build_manifest(
        h5_dir=args.h5_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        out_csv=args.out_csv,
        config=config
    )


if __name__ == '__main__':
    main()
