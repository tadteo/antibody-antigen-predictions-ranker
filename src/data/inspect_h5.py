#!/usr/bin/env python3
import argparse
import h5py

def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 file contents")
    parser.add_argument('h5_file', help="Path to HDF5 file to inspect")
    args = parser.parse_args()

    with h5py.File(args.h5_file, 'r') as hf:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        hf.visititems(print_item)

if __name__ == "__main__":
    main() 
