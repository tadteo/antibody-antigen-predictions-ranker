#!/usr/bin/env python3
"""
rename_ranking_confidence.py

A simple script to rename 'ranking_confidence' fields to 'ranking_score' in H5 files.
This script processes all H5 files in a given directory and performs the renaming operation
in-place while preserving all other data and structure.

Usage:
    python rename_ranking_confidence.py <input_directory>

Example:
    python rename_ranking_confidence.py /path/to/h5/files/

Requirements:
    - h5py library
    - Python 3.6+

Author: Generated for antibody-antigen predictions dataset processing
"""

import os
import sys
import argparse
import h5py
from pathlib import Path


def rename_ranking_confidence_in_file(h5_file_path):
    """
    Rename 'ranking_confidence' to 'ranking_score' in a single H5 file.
    
    Args:
        h5_file_path (str): Path to the H5 file to process
        
    Returns:
        bool: True if any renaming was performed, False otherwise
    """
    renamed_count = 0
    datasets_to_rename = []
    
    try:
        # First pass: collect all paths that need renaming
        with h5py.File(h5_file_path, 'r') as f:
            def find_ranking_confidence(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith('ranking_confidence'):
                    datasets_to_rename.append(name)
            
            f.visititems(find_ranking_confidence)
        
        # Second pass: perform the renaming
        if datasets_to_rename:
            with h5py.File(h5_file_path, 'r+') as f:
                for dataset_path in datasets_to_rename:
                    # Get the parent group path
                    parent_path = '/'.join(dataset_path.split('/')[:-1])
                    parent_group = f[parent_path] if parent_path else f
                    
                    # Read the data and attributes
                    data = f[dataset_path][()]
                    attrs = dict(f[dataset_path].attrs)
                    
                    # Delete the old dataset
                    del parent_group['ranking_confidence']
                    
                    # Create new dataset with the same data and attributes
                    new_dataset = parent_group.create_dataset('ranking_score', data=data)
                    
                    # Copy attributes
                    for key, value in attrs.items():
                        new_dataset.attrs[key] = value
                    
                    renamed_count += 1
                    new_path = f"{parent_path}/ranking_score" if parent_path else "ranking_score"
                    print(f"  Renamed: {dataset_path} -> {new_path}")
            
    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")
        return False
    
    return renamed_count > 0


def process_directory(input_dir):
    """
    Process all H5 files in the given directory.
    
    Args:
        input_dir (str): Directory containing H5 files to process
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return
    
    # Find all H5 files
    h5_files = list(input_path.glob('*.h5'))
    
    if not h5_files:
        print(f"No H5 files found in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files to process")
    
    processed_count = 0
    renamed_files = 0
    
    for h5_file in h5_files:
        print(f"\nProcessing: {h5_file.name}")
        
        try:
            if rename_ranking_confidence_in_file(str(h5_file)):
                renamed_files += 1
                print(f"  ✓ Successfully renamed fields in {h5_file.name}")
            else:
                print(f"  - No 'ranking_confidence' fields found in {h5_file.name}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed to process {h5_file.name}: {e}")
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"Files processed: {processed_count}/{len(h5_files)}")
    print(f"Files with renamed fields: {renamed_files}")
    print(f"="*50)


def main():
    
    input_directory = '/proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new/abag_dataset_new_with_stats_filtered'
    
    process_directory(input_directory)


if __name__ == '__main__':
    main()
