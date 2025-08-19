#!/usr/bin/env python3

"""
This script is designed to filter redundant antibody-antigen complexes from a dataset.
It uses a greedy algorithm to prune redundant samples based on their DockQ and PAE values.

to run:
python 04_filter_dataset.py --input_dir input_path_with_original_h5 --output_dir output_path_with_filtered_h5 --n_workers 8

Example with 4 workers:
python 04_filter_dataset.py --input_dir input_path_with_original_h5 --output_dir output_path_with_filtered_h5 --n_workers 4
"""

import argparse
import h5py
import numpy as np
import os
from scipy.stats import wasserstein_distance
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def is_redundant(new_sample, existing_sample, dockq_threshold=0.01, pae_diff_threshold=0.5):
    dockq_diff = abs(new_sample['dockq'] - existing_sample['dockq'])
    if dockq_diff > dockq_threshold:
        return False, None, None  # Not redundant, keep it

    pae_diff = wasserstein_distance(new_sample['pae'], existing_sample['pae'])
    # print(f"  PAE Wasserstein diff = {pae_diff}")
    if pae_diff < pae_diff_threshold:
        return True, dockq_diff, pae_diff  # Redundant
    else:
        # print(f"  Not redundant: PAE diff {pae_diff} >= {pae_diff_threshold}")
        return False, None, None  # Not redundant

def greedy_prune(samples, dockq_threshold=0.01, pae_diff_threshold=0.5, output_dir=None, complex_name=None):
    kept_samples = []
    pruned_samples = []
    # Dictionary to track which samples are clustered with each kept sample
    clusters = {}
    
    #add first sample
    kept_samples.append(samples[0])
    clusters[samples[0]['name']] = []

    start_time = time.time()
    for sample in samples[1:]:
        redundant = False
        for kept in kept_samples:
            is_red, dockq_diff, pae_diff = is_redundant(sample, kept, dockq_threshold, pae_diff_threshold)
            if is_red:
                redundant = True
                pruned_samples.append(sample)
                clusters[kept['name']].append({
                    'name': sample['name'],
                    'dockq_diff': dockq_diff,
                    'pae_diff': pae_diff
                })
                # print(f"Sample {sample['name']} is redundant with {kept['name']}, skipping.")
                break
        if not redundant:
            kept_samples.append(sample)
            clusters[sample['name']] = []

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Create the JSON report structure
    json_report = {
        "summary": {
            "total_samples": len(samples),
            "kept_samples": len(kept_samples),
            "removed_samples": len(pruned_samples),
            "dockq_threshold": dockq_threshold,
            "pae_diff_threshold": pae_diff_threshold
        },
        "clusters": {}
    }
    
    # Add each kept sample and its similar samples to the JSON report
    for kept_name, similar_samples in clusters.items():
        kept_sample = next(s for s in kept_samples if s['name'] == kept_name)
        json_report["clusters"][kept_name] = {
            "kept_sample": {
                "name": kept_name,
                "dockq": kept_sample['dockq']
            },
            "similar_samples": [
                {
                    "name": similar['name'],
                    "dockq_diff": similar['dockq_diff'],
                    "pae_diff": similar['pae_diff']
                }
                for similar in similar_samples
            ]
        }
    
    # Save JSON report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{complex_name}_clustering_report.json")
        import json
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        print(f"\nJSON clustering report saved to: {json_path}")
    
    return list(kept_samples), pruned_samples

def compute_distance_matrix(samples):
    n = len(samples)
    dist_matrix = np.zeros((n, n))
    total = n * (n - 1) // 2
    with tqdm(total=total, desc="Computing distance matrix") as pbar:
        for i in range(n):
            for j in range(i+1, n):
                d = wasserstein_distance(samples[i]['pae'], samples[j]['pae'])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # symmetric
                pbar.update(1)
    return dist_matrix

def estimate_threshold(samples, num_samples=500):
    #pick 500 random different samples
    distances = []
    for i in range(num_samples):
        random_sample = random.choice(samples)
        random_sample2 = random.choice(samples)
        while random_sample == random_sample2:
            random_sample2 = random.choice(samples)
        distance = wasserstein_distance(random_sample['pae'], random_sample2['pae'])
        distances.append(distance)
    return {
                'min': np.min(distances),
                '10%': np.percentile(distances, 10),
                '25%': np.percentile(distances, 25),
                'median': np.median(distances),
                '75%': np.percentile(distances, 75),
                'max': np.max(distances)
            }

def plot_distance_distribution(dist_matrix, output_path=None):
    # Get upper triangle distances, excluding diagonal
    dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    plt.figure(figsize=(8, 5))
    plt.hist(dists, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Wasserstein Distance')
    plt.ylabel('Count')
    plt.title('Distribution of Pairwise Wasserstein Distances')
    if output_path:
        plt.savefig(output_path)
        print(f"Distance distribution plot saved to {output_path}")
    else:
        plt.show()

def process_single_h5_file(args):
    """
    Process a single H5 file for filtering redundant samples.
    
    Args:
        args (tuple): Tuple containing (h5_file, input_dir, output_dir)
    
    Returns:
        dict: Processing results with status and statistics
    """
    h5_file, input_dir, output_dir = args
    
    try:
        full_path = os.path.join(input_dir, h5_file)
        print(f"\nProcessing {full_path}")
        samples = []
        
        # Load samples from the H5 file
        with h5py.File(full_path, 'r') as hf:
            for complex_name in hf.keys():
                complex_group = hf[complex_name]
                # Iterate only over sample groups, skip complex-level datasets (e.g., pae_col_mean)
                sample_names = [name for name, obj in complex_group.items() if isinstance(obj, h5py.Group)]
                for sample_name in sample_names:
                    sample_obj = complex_group[sample_name]
                    try:
                        dockq = sample_obj['abag_dockq'][()]
                        pae = sample_obj['interchain_pae_vals'][()]
                        samples.append({
                            'name': f"{complex_name}/{sample_name}",
                            'dockq': float(dockq),
                            'pae': np.array(pae),
                            'raw_group': sample_obj
                        })
                    except KeyError as e:
                        print(f"Skipping {complex_name}/{sample_name} due to missing key: {e}")
                        continue

        if not samples:
            return {
                'file': h5_file,
                'status': 'error',
                'message': 'No valid samples found',
                'original_samples': 0,
                'kept_samples': 0,
                'removed_samples': 0
            }

        # Estimate threshold
        time_start = time.time()
        est_threshold = estimate_threshold(samples)
        time_end = time.time()
        
        print(f"Time taken to estimate threshold: {time_end - time_start:.2f} seconds")
        print(f"Estimated threshold: {est_threshold}")

        print(f"\nLoaded {len(samples)} samples.")
        
        # Get the first complex name for the report
        with h5py.File(full_path, 'r') as hf:
            complex_name = list(hf.keys())[0]
        
        kept_samples, pruned_samples = greedy_prune(
            samples, 
            dockq_threshold=0.01, 
            pae_diff_threshold=est_threshold['25%'], 
            output_dir=output_dir, 
            complex_name=complex_name
        )

        print(f"Filtered down to {len(kept_samples)} samples.")
        print("DockQ values of retained samples:", [round(s['dockq'], 3) for s in kept_samples])

        # Save filtered dataset
        with h5py.File(full_path, 'r') as hf:
            # Create a set of pruned sample names for faster lookup
            pruned_sample_names = {s['name'].split('/')[-1] for s in pruned_samples}
            
            # Save the filtered dataset
            new_h5_file = h5_file.replace('.h5', '_filtered.h5')
            output_path = os.path.join(output_dir, new_h5_file)
            with h5py.File(output_path, 'w') as out_hf:
                for complex_name in hf.keys():
                    complex_group = hf[complex_name]
                    out_complex_group = out_hf.create_group(complex_name)  # Create complex group

                    # First, copy complex-level datasets (e.g., pae_col_mean/median/std)
                    for key, obj in complex_group.items():
                        if isinstance(obj, h5py.Dataset):
                            if obj.shape == ():  # Scalar dataset
                                out_complex_group.create_dataset(key, data=obj[()], dtype=obj.dtype)
                            else:
                                out_complex_group.create_dataset(key, data=obj[:], dtype=obj.dtype)

                    # Then, copy only non-pruned sample groups
                    sample_names_to_copy = [
                        name for name, obj in complex_group.items()
                        if isinstance(obj, h5py.Group) and name not in pruned_sample_names
                    ]
                    for sample_name in sample_names_to_copy:
                        sample_obj = complex_group[sample_name]
                        out_complex_group.create_group(sample_name)
                        # Copy all datasets from the original sample group
                        for key in sample_obj.keys():
                            if isinstance(sample_obj[key], h5py.Dataset):
                                if sample_obj[key].shape == ():  # Scalar dataset
                                    out_complex_group[sample_name].create_dataset(
                                        key,
                                        data=sample_obj[key][()],
                                        dtype=sample_obj[key].dtype
                                    )
                                else:  # Array dataset
                                    out_complex_group[sample_name].create_dataset(
                                        key,
                                        data=sample_obj[key][:],
                                        dtype=sample_obj[key].dtype
                                    )

        # Verify the contents of the filtered file
        with h5py.File(output_path, 'r') as filtered_hf:
            for complex_name in filtered_hf.keys():
                complex_group = filtered_hf[complex_name]
                # Only count sample groups, skip complex-level datasets
                filtered_samples = [name for name, obj in complex_group.items() if isinstance(obj, h5py.Group)]
                
                if len(filtered_samples) != len(kept_samples):
                    raise ValueError(f"Number of samples in filtered file ({len(filtered_samples)}) does not match the expected number of samples ({len(kept_samples)})")
                
                # Verify that all kept samples are present in the filtered file
                kept_sample_names = {s['name'].split('/')[-1] for s in kept_samples}
                filtered_sample_names = set(filtered_samples)
                
                if kept_sample_names != filtered_sample_names:
                    raise ValueError("Mismatch in samples between kept_samples and filtered file")

        return {
            'file': h5_file,
            'status': 'success',
            'message': f'Successfully processed {h5_file}',
            'original_samples': len(samples),
            'kept_samples': len(kept_samples),
            'removed_samples': len(pruned_samples),
            'output_file': new_h5_file
        }

    except Exception as e:
        return {
            'file': h5_file,
            'status': 'error',
            'message': f'Error processing {h5_file}: {str(e)}',
            'original_samples': 0,
            'kept_samples': 0,
            'removed_samples': 0
        }

def main():
    parser = argparse.ArgumentParser(description="Filter redundant AF3 samples in HDF5 files using parallel processing")
    parser.add_argument('--input_dir', default='/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm/', help="Path to the folder containing the input HDF5 files")
    parser.add_argument('--output_dir', default='/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered/', help="Path to output filtered HDF5 files")
    parser.add_argument('--n_workers', type=int, default=8, help='Number of parallel workers (default: 8, recommended: 4-8)')
    args = parser.parse_args()

    # Validate number of workers
    if args.n_workers < 1:
        print("Error: Number of workers must be at least 1")
        return
    if args.n_workers > mp.cpu_count():
        print(f"Warning: Requested {args.n_workers} workers but only {mp.cpu_count()} CPUs available")
        print(f"Using {mp.cpu_count()} workers instead")
        args.n_workers = mp.cpu_count()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all H5 files in the input directory
    h5_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and f.endswith('.h5')]
    
    if not h5_files:
        print(f"No H5 files found in {args.input_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files to process")
    print(f"Using {args.n_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    file_args = [(h5_file, args.input_dir, args.output_dir) for h5_file in h5_files]
    
    # Process files in parallel
    if args.n_workers == 1:
        # Single-threaded processing with tqdm
        results = []
        for file_arg in tqdm(file_args, desc='Processing H5 files'):
            result = process_single_h5_file(file_arg)
            results.append(result)
    else:
        # Multi-threaded processing
        with mp.Pool(processes=args.n_workers) as pool:
            # Use imap for better progress tracking
            results = list(tqdm(
                pool.imap(process_single_h5_file, file_args),
                total=len(file_args),
                desc=f'Processing H5 files ({args.n_workers} workers)'
            ))
    
    # Print summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    total_original_samples = sum(r['original_samples'] for r in results if r['status'] == 'success')
    total_kept_samples = sum(r['kept_samples'] for r in results if r['status'] == 'success')
    total_removed_samples = sum(r['removed_samples'] for r in results if r['status'] == 'success')
    
    print(f"\n" + "="*60)
    print(f"PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Files processed successfully: {successful}")
    if failed > 0:
        print(f"Files failed: {failed}")
    
    print(f"\nSample Statistics (successful files only):")
    print(f"  Total original samples: {total_original_samples:,}")
    print(f"  Total kept samples: {total_kept_samples:,}")
    print(f"  Total removed samples: {total_removed_samples:,}")
    if total_original_samples > 0:
        removal_rate = (total_removed_samples / total_original_samples) * 100
        print(f"  Removal rate: {removal_rate:.1f}%")
    
    if failed > 0:
        print(f"\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['message']}")
    
    print(f"\nFiltered HDF5 files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
