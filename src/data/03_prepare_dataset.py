"""
This script is designed to post-process AlphaFold prediction results for antibody-antigen complexes.
It computes statistical measures (mean, median, and standard deviation) of the interchain PAE values
across multiple prediction samples for each complex, which can be used to assess prediction
confidence and reliability.

to run:
python 03_prepare_dataset.py --input_dir input_path_with_original_h5 --output_dir output_path_with_pae_stats_h5 --n_workers 4

Example with 8 workers:
python 03_prepare_dataset.py --input_dir input_path_with_original_h5 --output_dir output_path_with_pae_stats_h5 --n_workers 8
"""
import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_h5_file(input_path, output_path):
    """
    Process a single H5 file to compute per-column statistics for interchain PAE values.
    
    This function:
    1. Reads an input H5 file containing antibody-antigen complex predictions
    2. For each complex, processes multiple prediction samples
    3. Computes mean, median, and standard deviation across samples for each position in the PAE array
    4. Saves the original data plus the computed statistics to a new H5 file
    
    Args:
        input_path (str): Path to the input H5 file
        output_path (str): Path where the processed H5 file will be saved
    
    Returns:
        str: Status message indicating success or failure
    """
    try:
        with h5py.File(input_path, 'r') as in_hf, h5py.File(output_path, 'w') as out_hf:
            # Iterate through each antibody-antigen complex in the file
            for complex_id in in_hf.keys():
                in_complex_group = in_hf[complex_id]
                out_complex_group = out_hf.create_group(complex_id)
                pae_list = []  # Store PAE values from all samples for this complex
                ca_distances_list = []  # Store CA distances values from all samples for this complex
                sample_names = list(in_complex_group.keys())
                
                # Process each prediction sample for the current complex
                for sample_name in sample_names:
                    in_sample_group = in_complex_group[sample_name]
                    out_sample_group = out_complex_group.create_group(sample_name)
                    
                    # Copy all datasets from the input sample to the output (preserve original data)
                    for key in in_sample_group.keys():
                        in_sample_group.copy(key, out_sample_group)
                    
                    # Extract the interchain PAE (Predicted Aligned Error) values for this sample
                    # PAE measures the confidence in the predicted structure alignment
                    pae = in_sample_group['interchain_pae_vals'][()]
                    pae_list.append(pae)

                    # Extract the interchain CA distances for this sample
                    if 'interchain_ca_distances' in in_sample_group.keys():
                        ca_distances = in_sample_group['interchain_ca_distances'][()]
                        ca_distances_list.append(ca_distances)

                # Compute statistics across all samples for this complex
                if pae_list:
                    # Stack all PAE arrays into a matrix: (num_samples, pae_length)
                    # Each row represents one prediction sample, each column represents one residue position
                    pae_matrix = np.stack(pae_list, axis=0)
                    
                    # Compute mean, median, and standard deviation across samples for each residue position
                    # This gives us confidence metrics for each position in the structure
                    pae_col_mean = np.mean(pae_matrix, axis=0)    # Average PAE across samples for each position
                    pae_col_median = np.median(pae_matrix, axis=0) # Median PAE across samples for each position
                    pae_col_std = np.std(pae_matrix, axis=0)      # Standard deviation of PAE across samples for each position
                    
                    # Store the computed statistics in the output file
                    out_complex_group.create_dataset('pae_col_mean', data=pae_col_mean)
                    out_complex_group.create_dataset('pae_col_median', data=pae_col_median)
                    out_complex_group.create_dataset('pae_col_std', data=pae_col_std)

                if ca_distances_list:
                    # Stack all CA distances arrays into a matrix: (num_samples, ca_distances_length)
                    # Each row represents one prediction sample, each column represents one residue position
                    ca_distances_matrix = np.stack(ca_distances_list, axis=0)
                    
                    # Compute mean, median, and standard deviation across samples for each residue position
                    # This gives us confidence metrics for each position in the structure
                    ca_distances_col_mean = np.nanmean(ca_distances_matrix, axis=0)    # Average CA distances across samples for each position
                    ca_distances_col_median = np.nanmedian(ca_distances_matrix, axis=0) # Median CA distances across samples for each position
                    ca_distances_col_std = np.nanstd(ca_distances_matrix, axis=0)      # Standard deviation of CA distances across samples for each position
                    
                    # Store the computed statistics in the output file
                    out_complex_group.create_dataset('ca_distances_col_mean', data=ca_distances_col_mean)
                    out_complex_group.create_dataset('ca_distances_col_median', data=ca_distances_col_median)
                    out_complex_group.create_dataset('ca_distances_col_std', data=ca_distances_col_std)

        return f"Successfully processed {os.path.basename(input_path)}"
    
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {str(e)}"


def process_file_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments.
    
    Args:
        args (tuple): Tuple containing (input_path, output_path)
    
    Returns:
        str: Status message from process_h5_file
    """
    input_path, output_path = args
    return process_h5_file(input_path, output_path)


def main():
    """
    Main function that processes all H5 files in a directory using parallel processing.
    
    This script is designed to post-process AlphaFold prediction results for antibody-antigen complexes.
    It computes statistical measures (mean, median, and standard deviation) of the interchain PAE values
    across multiple prediction samples for each complex, which can be used to assess prediction
    confidence and reliability.
    """
    parser = argparse.ArgumentParser(description='Compute per-column mean/median/std of interchain_pae_vals for each complex in H5 files.')
    parser.add_argument('--input_dir', default="/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered/", help='Input directory containing H5 files')
    parser.add_argument('--output_dir', default="/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered_with_pae_stats/", help='Output directory for processed H5 files')
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
    h5_files = [f for f in os.listdir(args.input_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No H5 files found in {args.input_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files to process")
    print(f"Using {args.n_workers} parallel workers")
    
    # Prepare file paths for processing
    file_pairs = []
    for fname in h5_files:
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        file_pairs.append((input_path, output_path))
    
    # Process files in parallel with progress tracking
    if args.n_workers == 1:
        # Single-threaded processing with tqdm
        results = []
        for file_pair in tqdm(file_pairs, desc='Processing H5 files'):
            result = process_file_wrapper(file_pair)
            results.append(result)
    else:
        # Multi-threaded processing
        with mp.Pool(processes=args.n_workers) as pool:
            # Use imap for better progress tracking
            results = list(tqdm(
                pool.imap(process_file_wrapper, file_pairs),
                total=len(file_pairs),
                desc=f'Processing H5 files ({args.n_workers} workers)'
            ))
    
    # Print summary of results
    successful = sum(1 for r in results if r.startswith("Successfully"))
    failed = len(results) - successful
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {successful} files")
    if failed > 0:
        print(f"  Failed: {failed} files")
        print("\nErrors:")
        for result in results:
            if not result.startswith("Successfully"):
                print(f"  {result}")

if __name__ == '__main__':
    main()
