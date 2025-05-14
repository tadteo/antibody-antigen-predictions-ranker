#!/usr/bin/env python3
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

def main():
    parser = argparse.ArgumentParser(description="Filter redundant AF3 samples in an HDF5 file")
    parser.add_argument('--h5_file', default='/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm/', help="Path to the folder containing the input HDF5 files")
    parser.add_argument('--output', default='/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered/', help="Path to output filtered HDF5 file")
    args = parser.parse_args()

    #create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    complexes = []
    
    complexes_lengths = []
    complexes_was_distance_time = []
    #load all complexes and samples
    for h5_file in os.listdir(args.h5_file):
        full_path = os.path.join(args.h5_file, h5_file)
        if not os.path.isfile(full_path):
            continue  # skip directories and non-files
        if not h5_file.endswith('.h5'):
            continue  # skip non-h5 files
        print(f"Processing {full_path}")
        samples = []
        
        with h5py.File(full_path, 'r') as hf:
            for complex_name in hf.keys():
                complex = complex_name
                print(f"Processing {complex}")
                complex_group = hf[complex_name]
                complexes.append(complex_name)
                for sample_name in complex_group.keys():
                    group = complex_group[sample_name]
                    try:
                        dockq = group['abag_dockq'][()]
                        pae = group['interchain_pae_vals'][()]
                        samples.append({
                            'name': f"{complex_name}/{sample_name}",
                            'dockq': float(dockq),
                            'pae': np.array(pae),
                            'raw_group': group
                        })
                    except KeyError as e:
                        print(f"Skipping {complex_name}/{sample_name} due to missing key: {e}")
                        continue


        #first estimate wasserstein distance threshold
        time_start = time.time()
        est_threshold = estimate_threshold(samples)
        time_end = time.time()
        print(f"Time taken to estimate threshold: {time_end - time_start:.2f} seconds")
        print(f"Estimated threshold: {est_threshold}")

        print(f"\nLoaded {len(samples)} samples.")
        kept_samples, pruned_samples = greedy_prune(samples, dockq_threshold=0.01, pae_diff_threshold=est_threshold['25%'], output_dir=args.output, complex_name=complex)

        print(f"Filtered down to {len(kept_samples)} samples.")
        print("DockQ values of retained samples:", [round(s['dockq'], 3) for s in kept_samples])


        # Save filtered dataset
        # this check the similar samples and remove those lines from the dataset
        #create a copy of the original dataset
        with h5py.File(full_path, 'r') as hf:
            # Create a set of pruned sample names for faster lookup
            pruned_sample_names = {s['name'].split('/')[-1] for s in pruned_samples}
            
            #save the filtered dataset
            new_h5_file = h5_file.replace('.h5', '_filtered.h5')
            output_path = os.path.join(args.output, new_h5_file)
            with h5py.File(output_path, 'w') as out_hf:
                for complex_name in hf.keys():
                    complex_group = hf[complex_name]
                    out_hf.create_group(complex_name)  # Create complex group
                    for sample_name in complex_group.keys():
                        # Only copy if this sample is not in pruned_samples
                        if sample_name not in pruned_sample_names:
                            group = complex_group[sample_name]
                            out_hf[complex_name].create_group(sample_name)
                            # Copy all datasets from the original group
                            for key in group.keys():
                                if isinstance(group[key], h5py.Dataset):
                                    if group[key].shape == ():  # Scalar dataset
                                        out_hf[complex_name][sample_name].create_dataset(
                                            key,
                                            data=group[key][()],
                                            dtype=group[key].dtype
                                        )
                                    else:  # Array dataset
                                        out_hf[complex_name][sample_name].create_dataset(
                                            key,
                                            data=group[key][:],
                                            dtype=group[key].dtype
                                        )

        # Verify the contents of the filtered file and that the number of samples is correct
        print("\nVerifying filtered file contents:")
        with h5py.File(output_path, 'r') as filtered_hf:
            for complex_name in filtered_hf.keys():
                print(f"\nComplex: {complex_name}")
                complex_group = filtered_hf[complex_name]
                filtered_samples = list(complex_group.keys())
                print(f"Number of samples in filtered file: {len(filtered_samples)}")
                print(f"Expected number of samples (from greedy_prune): {len(kept_samples)}")
                if len(filtered_samples) != len(kept_samples):
                    raise ValueError(f"Number of samples in filtered file ({len(filtered_samples)}) does not match the expected number of samples ({len(kept_samples)})")
                # Verify that all kept samples are present in the filtered file
                kept_sample_names = {s['name'].split('/')[-1] for s in kept_samples}
                filtered_sample_names = set(filtered_samples)
                
                if kept_sample_names == filtered_sample_names:
                    print("✓ All kept samples are present in the filtered file")
                else:
                    print("✗ Mismatch in samples!")
                    print("Samples in kept_samples but not in filtered file:", kept_sample_names - filtered_sample_names)
                    print("Samples in filtered file but not in kept_samples:", filtered_sample_names - kept_sample_names)
                
                print("\nSample names in filtered file:", filtered_samples)
                
                # Print first sample's data structure as example
                if complex_group.keys():
                    first_sample = list(complex_group.keys())[0]
                    print(f"\nStructure of first sample ({first_sample}):")
                    for key in complex_group[first_sample].keys():
                        dataset = complex_group[first_sample][key]
                        print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

    print(f"\nFiltered HDF5 written to: {args.output}")

if __name__ == "__main__":
    main()


# Step 1: Estimate Wasserstein calculation time for each complex
    # ws_times = []
    # ws_lengths = []
    # print("Estimating Wasserstein calculation time for each complex...")
    # for h5_file in os.listdir(args.h5_file):
    #     full_path = os.path.join(args.h5_file, h5_file)
    #     if not os.path.isfile(full_path):
    #         continue  # skip directories and non-files
    #     if not h5_file.endswith('.h5'):
    #         continue  # skip non-h5 files

    #     with h5py.File(full_path, 'r') as hf:
    #         for complex_name in hf.keys():
    #             complex_group = hf[complex_name]
    #             sample_names = list(complex_group.keys())
    #             if len(sample_names) < 2:
    #                 continue  # need at least 2 samples
    #             try:
    #                 pae1 = complex_group[sample_names[0]]['interchain_pae_vals'][()]
    #                 pae2 = complex_group[sample_names[1]]['interchain_pae_vals'][()]
    #                 start = time.time()
    #                 _ = wasserstein_distance(pae1, pae2)
    #                 elapsed = time.time() - start
    #                 ws_times.append(elapsed)
    #                 ws_lengths.append(len(pae1))
    #             except KeyError:
    #                 continue
    #             break  # only do this for the first complex in the file

    # # Plot time vs. length
    # if ws_times:
    #     plt.figure(figsize=(8, 5))
    #     plt.scatter(ws_lengths, ws_times, alpha=0.7)
    #     plt.xlabel('Length of PAE vector')
    #     plt.ylabel('Time to calculate Wasserstein distance (s)')
    #     plt.title('Wasserstein Distance Calculation Time vs. PAE Length')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig("wasserstein_time_vs_length.png")
    #     plt.show()
    #     print("Wasserstein time vs. length plot saved as wasserstein_time_vs_length.png")
    # else:
    #     print("No Wasserstein timings collected.")
    
    # #assuming there are 200 samples per complex, the time to calculate the upper distance matrix is:
    # distance_upper_matrix_dimension= (200 * (200 - 1) / 2)
    # distance_upper_matrix_time = []
    # for i in range(len(ws_times)):
    #     distance_upper_matrix_time.append(ws_times[i] * distance_upper_matrix_dimension)
    # print(f"Time to calculate distance matrix: {sum(distance_upper_matrix_time)} seconds")

    # #take the shortest time complex to calculate the distance matrix
    # shortest_time_complex = ws_times.index(min(ws_times))
    # print(f"Shortest time complex: {shortest_time_complex}")
    # print(f"Shortest time complex time upper matrix: {distance_upper_matrix_time[shortest_time_complex]} seconds")
    
    # #open the shortest time complex
    # full_path = os.path.join(args.h5_file, os.listdir(args.h5_file)[shortest_time_complex])
    # print(f"Full path: {full_path}, length of pae: {len(os.listdir(args.h5_file)[shortest_time_complex])}")
    # samples = []
    # with h5py.File(full_path, 'r') as hf:
    #         for complex_name in hf.keys():
    #             complex_group = hf[complex_name]
    #             for sample_name in complex_group.keys():
    #                 group = complex_group[sample_name]
    #                 try:
    #                     dockq = group['abag_dockq'][()]
    #                     pae = group['interchain_pae_vals'][()]
    #                     samples.append({
    #                         'name': f"{complex_name}/{sample_name}",
    #                         'dockq': float(dockq),
    #                         'pae': np.array(pae),
    #                         'raw_group': group
    #                     })
    #                 except KeyError as e:
    #                     print(f"Skipping {complex_name}/{sample_name} due to missing key: {e}")
    #                     continue

    # print("Computing distance matrix...")
    # time_start = time.time()
    # dist_matrix = compute_distance_matrix(samples)
    # time_end = time.time()
    # #save the distance matrix as a numpy array with complex name
    # print(f"Time taken to compute distance matrix: {time_end - time_start:.2f} seconds for a pae length of {len(samples[0]['pae'])}")
    # plot_distance_distribution(dist_matrix, output_path="distance_distribution.png")
