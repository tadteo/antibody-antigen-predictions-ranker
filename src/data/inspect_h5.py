#!/usr/bin/env python3
import argparse
import h5py
import os

def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 file contents")
    parser.add_argument('--h5_file', default='/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered/7wvd_interchain_data_filtered.h5', help="Path to HDF5 file to inspect")
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_file):
        print(f"Error: File {args.h5_file} does not exist")
        return

    with h5py.File(args.h5_file, 'r') as hf:
        print(f"\nInspecting file: {args.h5_file}")
        print(f"Total complexes: {len(hf.keys())}")
        
        for complex_name in hf.keys():
            print(f"\nComplex: {complex_name}")
            complex_group = hf[complex_name]
            samples = list(complex_group.keys())
            print(f"Number of samples: {len(samples)}")
            print("Sample names:", samples)
            
            # Print detailed information about the first sample
            if samples:
                first_sample = samples[0]
                print(f"\nStructure of first sample ({first_sample}):")
                sample_group = complex_group[first_sample]
                for key in sample_group.keys():
                    dataset = sample_group[key]
                    print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                
                # Print some example values for scalar datasets
                print("\nExample values from first sample:")
                for key in sample_group.keys():
                    dataset = sample_group[key]
                    if dataset.shape == ():  # Scalar dataset
                        print(f"  {key}: {dataset[()]}")
                    elif len(dataset.shape) == 1 and dataset.shape[0] <= 5:  # Small 1D array
                        print(f"  {key}: {dataset[:]}")
                    elif len(dataset.shape) == 2 and dataset.shape[0] <= 3 and dataset.shape[1] <= 3:  # Small 2D array
                        print(f"  {key}: {dataset[:]}")
                    else:
                        print(f"  {key}: [Array of shape {dataset.shape}]")

if __name__ == "__main__":
    main() 
