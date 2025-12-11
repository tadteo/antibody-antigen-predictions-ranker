#!/usr/bin/env python3
"""
Migrate ESM embeddings from sample level to complex level in H5 files.

This script updates existing H5 files that have ESM embeddings stored in each
sample group to the new format where ESM embeddings are stored once at the
complex level (since all samples in a complex share the same sequence).

This reduces file size and improves efficiency.

Usage:
    # Migrate a single file (in-place)
    python 08_migrate_esm_to_complex_level.py --input /path/to/file.h5

    # Migrate all H5 files in a directory (in-place)
    python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/

    # Create backups before migration
    python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/ --backup

    # Dry run (show what would be done without making changes)
    python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/ --dry-run
"""

import argparse
import os
import shutil
import h5py
from pathlib import Path
from tqdm import tqdm


def check_file_format(h5_path):
    """
    Check the current format of ESM embeddings in an H5 file.
    
    Returns:
        str: 'complex_level' if already migrated,
             'sample_level' if needs migration,
             'none' if no ESM embeddings found,
             'mixed' if inconsistent format
    """
    with h5py.File(h5_path, 'r') as hf:
        has_complex_level = False
        has_sample_level = False
        
        for complex_id in hf.keys():
            complex_group = hf[complex_id]
            
            # Check if esm_embeddings exists at complex level
            if 'esm_embeddings' in complex_group and isinstance(complex_group['esm_embeddings'], h5py.Dataset):
                has_complex_level = True
            
            # Check sample groups for esm_embeddings
            for key in complex_group.keys():
                item = complex_group[key]
                if isinstance(item, h5py.Group):
                    if 'esm_embeddings' in item:
                        has_sample_level = True
        
        if has_complex_level and has_sample_level:
            return 'mixed'
        elif has_complex_level:
            return 'complex_level'
        elif has_sample_level:
            return 'sample_level'
        else:
            return 'none'


def migrate_h5_file(h5_path, dry_run=False):
    """
    Migrate ESM embeddings from sample level to complex level in an H5 file.
    
    This function:
    1. For each complex, reads esm_embeddings from the first sample
    2. Copies it to the complex level
    3. Deletes esm_embeddings from all sample groups
    
    Args:
        h5_path (str): Path to the H5 file to migrate
        dry_run (bool): If True, only report what would be done
    
    Returns:
        dict: Migration statistics
    """
    stats = {
        'complexes_migrated': 0,
        'samples_cleaned': 0,
        'embeddings_shape': None,
        'skipped': False,
        'reason': None
    }
    
    # First check the format
    current_format = check_file_format(h5_path)
    
    if current_format == 'complex_level':
        stats['skipped'] = True
        stats['reason'] = 'Already at complex level'
        return stats
    elif current_format == 'none':
        stats['skipped'] = True
        stats['reason'] = 'No ESM embeddings found'
        return stats
    elif current_format == 'mixed':
        # Handle mixed format - just clean up sample level
        pass
    
    if dry_run:
        # Just count what would be done
        with h5py.File(h5_path, 'r') as hf:
            for complex_id in hf.keys():
                complex_group = hf[complex_id]
                
                # Check if already has complex-level embeddings
                has_complex_level = 'esm_embeddings' in complex_group and isinstance(complex_group['esm_embeddings'], h5py.Dataset)
                
                sample_groups = [k for k in complex_group.keys() 
                               if isinstance(complex_group[k], h5py.Group)]
                
                samples_with_esm = sum(1 for s in sample_groups 
                                      if 'esm_embeddings' in complex_group[s])
                
                if samples_with_esm > 0:
                    if not has_complex_level:
                        stats['complexes_migrated'] += 1
                        # Get shape from first sample with embeddings
                        for s in sample_groups:
                            if 'esm_embeddings' in complex_group[s]:
                                stats['embeddings_shape'] = complex_group[s]['esm_embeddings'].shape
                                break
                    stats['samples_cleaned'] += samples_with_esm
        return stats
    
    # Actual migration
    with h5py.File(h5_path, 'r+') as hf:
        for complex_id in hf.keys():
            complex_group = hf[complex_id]
            
            # Get sample groups
            sample_groups = [k for k in complex_group.keys() 
                           if isinstance(complex_group[k], h5py.Group)]
            
            # Check if already has complex-level embeddings
            has_complex_level = 'esm_embeddings' in complex_group and isinstance(complex_group['esm_embeddings'], h5py.Dataset)
            
            # Find first sample with esm_embeddings
            esm_source_sample = None
            for sample_name in sample_groups:
                if 'esm_embeddings' in complex_group[sample_name]:
                    esm_source_sample = sample_name
                    break
            
            if esm_source_sample is None:
                continue  # No ESM embeddings in this complex
            
            # Copy to complex level if not already there
            if not has_complex_level:
                esm_data = complex_group[esm_source_sample]['esm_embeddings'][()]
                stats['embeddings_shape'] = esm_data.shape
                complex_group.create_dataset('esm_embeddings', data=esm_data, compression='gzip')
                stats['complexes_migrated'] += 1
            
            # Delete from all samples
            for sample_name in sample_groups:
                sample_group = complex_group[sample_name]
                if 'esm_embeddings' in sample_group:
                    del sample_group['esm_embeddings']
                    stats['samples_cleaned'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Migrate ESM embeddings from sample level to complex level in H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single file
  python 08_migrate_esm_to_complex_level.py --input /path/to/file.h5

  # Migrate all H5 files in a directory
  python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/

  # Create backups before migration
  python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/ --backup

  # Dry run to see what would be done
  python 08_migrate_esm_to_complex_level.py --input /path/to/h5_files/ --dry-run
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to H5 file or directory containing H5 files')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup of files before migration (adds .bak extension)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--pattern', type=str, default='*.h5',
                        help='H5 file pattern when input is a directory (default: *.h5)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine files to process
    if input_path.is_file():
        h5_files = [input_path]
    elif input_path.is_dir():
        h5_files = sorted(input_path.glob(args.pattern))
    else:
        print(f"ERROR: {input_path} does not exist")
        return
    
    if not h5_files:
        print(f"ERROR: No H5 files found matching pattern '{args.pattern}' in {input_path}")
        return
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Found {len(h5_files)} H5 file(s) to process")
    if args.backup and not args.dry_run:
        print("Backups will be created with .bak extension")
    print()
    
    # Process files
    total_complexes = 0
    total_samples = 0
    skipped_files = 0
    migrated_files = 0
    
    for h5_path in tqdm(h5_files, desc="Processing files"):
        # Create backup if requested
        if args.backup and not args.dry_run:
            backup_path = h5_path.with_suffix(h5_path.suffix + '.bak')
            if not backup_path.exists():
                shutil.copy2(h5_path, backup_path)
        
        try:
            stats = migrate_h5_file(h5_path, dry_run=args.dry_run)
            
            if stats['skipped']:
                skipped_files += 1
                tqdm.write(f"  Skipped {h5_path.name}: {stats['reason']}")
            else:
                migrated_files += 1
                total_complexes += stats['complexes_migrated']
                total_samples += stats['samples_cleaned']
                
                if stats['complexes_migrated'] > 0 or stats['samples_cleaned'] > 0:
                    tqdm.write(f"  {h5_path.name}: migrated {stats['complexes_migrated']} complex(es), "
                              f"cleaned {stats['samples_cleaned']} sample(s)")
        
        except Exception as e:
            tqdm.write(f"  ERROR processing {h5_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}MIGRATION COMPLETE")
    print("="*80)
    print(f"Total files processed: {len(h5_files)}")
    print(f"Files migrated:        {migrated_files}")
    print(f"Files skipped:         {skipped_files}")
    print(f"Complexes migrated:    {total_complexes}")
    print(f"Samples cleaned:       {total_samples}")
    
    if args.dry_run:
        print("\nThis was a dry run. No changes were made.")
        print("Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
