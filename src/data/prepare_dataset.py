import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm

def process_h5_file(input_path, output_path):
    with h5py.File(input_path, 'r') as in_hf, h5py.File(output_path, 'w') as out_hf:
        for complex_id in in_hf.keys():
            in_complex_group = in_hf[complex_id]
            out_complex_group = out_hf.create_group(complex_id)
            pae_list = []
            sample_names = list(in_complex_group.keys())
            # Copy all sample groups and collect PAE
            for sample_name in sample_names:
                in_sample_group = in_complex_group[sample_name]
                out_sample_group = out_complex_group.create_group(sample_name)
                # Copy all datasets in the sample group
                for key in in_sample_group.keys():
                    in_sample_group.copy(key, out_sample_group)
                # Collect interchain_pae_vals
                pae = in_sample_group['interchain_pae_vals'][()]
                pae_list.append(pae)
            # Stack and compute per-column mean/std if any samples
            if pae_list:
                pae_matrix = np.stack(pae_list, axis=0)  # shape: (num_samples, pae_length)
                pae_col_mean = np.mean(pae_matrix, axis=0)
                pae_col_std = np.std(pae_matrix, axis=0)
                out_complex_group.create_dataset('pae_col_mean', data=pae_col_mean)
                out_complex_group.create_dataset('pae_col_std', data=pae_col_std)


def main():
    parser = argparse.ArgumentParser(description='Compute per-column mean/std of interchain_pae_vals for each complex in H5 files.')
    parser.add_argument('--input_dir', default= "/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered/", help='Input directory containing H5 files')
    parser.add_argument('--output_dir', default= "/proj/berzelius-2021-29/users/x_matta/abag_dataset_processed_with_ptm_filtered_with_pae_stats/", help='Output directory for processed H5 files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    h5_files = [f for f in os.listdir(args.input_dir) if f.endswith('.h5')]
    for fname in tqdm(h5_files, desc='Processing H5 files'):
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)
        process_h5_file(input_path, output_path)

if __name__ == '__main__':
    main()
