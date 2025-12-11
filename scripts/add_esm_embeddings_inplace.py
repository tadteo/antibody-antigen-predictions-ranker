#!/usr/bin/env python3
"""
Add ESM embeddings in-place to HDF5 files from a filtered-ESM source folder.

Behavior:
- For each `.h5` file in the target directory (--target-dir), find a corresponding 
  `.h5` in the ESM directory (--esm-dir) by matching the basename (ignores an 
  optional `_filtered` suffix).
- For each top-level complex in the target file, read the first `esm_embeddings`
  dataset found under the corresponding complex in the ESM file and write that
  same embedding into every sample subgroup for that complex in the target file.
- Updates files IN PLACE - be careful!

Default: do not overwrite existing `esm_embeddings` in the destination; pass
`--overwrite` to replace them. Use `--dry-run` to preview actions.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def build_esm_map(esm_dir: Path):
    """Map base_id -> esm h5 path. base_id strips optional `_filtered` suffix."""
    esm_map = {}
    for p in sorted(esm_dir.glob('*.h5')):
        base = p.stem
        if base.endswith('_filtered'):
            base_id = base[: -len('_filtered')]
        else:
            base_id = base
        if base_id not in esm_map:
            esm_map[base_id] = p
    return esm_map


def find_first_esm_for_complex(esm_hf: h5py.File, complex_key: str):
    """Return numpy array of the first `esm_embeddings` found under complex_key.
    Returns None if not found."""
    if complex_key not in esm_hf:
        return None
    grp = esm_hf[complex_key]
    # check direct dataset
    if 'esm_embeddings' in grp:
        try:
            return grp['esm_embeddings'][:]
        except Exception:
            return None
    # iterate child groups
    for child_name in grp:
        try:
            child = grp[child_name]
        except Exception:
            continue
        if isinstance(child, h5py.Dataset):
            # unlikely, but check
            if child_name == 'esm_embeddings':
                return child[:]
            continue
        # child is a group
        if 'esm_embeddings' in child:
            try:
                return child['esm_embeddings'][:]
            except Exception:
                continue
    return None


def inject_embeddings_inplace(target_path: Path, esm_path: Path, overwrite=False, dry_run=False):
    """
    Open target file in read-write mode and inject ESM embeddings from esm_path.
    """
    logger = logging.getLogger('inject')
    logger.info('Processing %s (esm=%s)', target_path.name, esm_path.name if esm_path else 'MISSING')

    if dry_run:
        logger.info('Dry-run enabled: would inject embeddings from %s into %s', esm_path, target_path)
        return {'status': 'dryrun'}

    if esm_path is None or not esm_path.exists():
        logger.warning('No ESM file for %s; skipping embedding injection', target_path.name)
        return {'status': 'no_esm'}

    # Open target file in read-write mode
    with h5py.File(target_path, 'r+') as target_hf, h5py.File(esm_path, 'r') as esm_hf:
        injected = 0
        skipped = 0
        missing_complex = 0
        
        for complex_key in target_hf:
            emb = find_first_esm_for_complex(esm_hf, complex_key)
            if emb is None:
                logger.debug('No esm embedding found for complex %s in esm file %s', complex_key, esm_path.name)
                missing_complex += 1
                continue

            # ensure emb is a numpy array (force dtype float32)
            emb = np.asarray(emb)
            if emb.dtype != np.float32:
                try:
                    emb = emb.astype(np.float32)
                except Exception:
                    pass

            grp = target_hf[complex_key]
            # iterate sample subgroups and set dataset
            for sample_name in grp:
                sample = grp[sample_name]
                # only act on groups
                if not isinstance(sample, h5py.Group):
                    continue
                if 'esm_embeddings' in sample and not overwrite:
                    logger.debug('esm_embeddings already exists in %s/%s; skipping (use --overwrite to replace)', 
                                complex_key, sample_name)
                    skipped += 1
                    continue
                # warn if sequence length differs (if seq dataset exists)
                seq_len = None
                if 'seq' in sample:
                    try:
                        seq_data = sample['seq'][:]
                        seq_len = len(seq_data)
                    except Exception:
                        seq_len = None
                if seq_len is not None and emb.shape[0] != seq_len:
                    logger.debug('Embedding length %d vs seq length %s in %s/%s', 
                                emb.shape[0], seq_len, target_path.name, f'{complex_key}/{sample_name}')
                # delete existing if overwrite
                if 'esm_embeddings' in sample and overwrite:
                    try:
                        del sample['esm_embeddings']
                        logger.debug('Deleted existing esm_embeddings in %s/%s', complex_key, sample_name)
                    except Exception as e:
                        logger.warning('Failed to delete existing esm_embeddings in %s/%s: %s', 
                                     complex_key, sample_name, e)
                        continue
                # create dataset
                try:
                    sample.create_dataset('esm_embeddings', data=emb, compression='gzip')
                    injected += 1
                    logger.debug('Injected esm_embeddings into %s/%s', complex_key, sample_name)
                except Exception as e:
                    logger.warning('Failed to write esm_embeddings into %s/%s: %s', complex_key, sample_name, e)

        return {'status': 'done', 'injected': injected, 'skipped': skipped, 'missing_complex': missing_complex}


def main():
    parser = argparse.ArgumentParser(description='Add ESM embeddings in-place to h5 files.')
    parser.add_argument('--target-dir', required=True, type=Path, 
                       help='Target folder with h5 files to update IN PLACE')
    parser.add_argument('--esm-dir', required=True, type=Path, 
                       help='Folder with filtered ESM h5 files')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing esm_embeddings in target files')
    parser.add_argument('--dry-run', action='store_true', 
                       help="Don't modify files; just show what would happen")
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logger = logging.getLogger('inject')

    target_dir = args.target_dir
    esm_dir = args.esm_dir

    if not target_dir.exists() or not target_dir.is_dir():
        logger.error('Target directory %s does not exist or is not a directory', target_dir)
        sys.exit(2)
    if not esm_dir.exists() or not esm_dir.is_dir():
        logger.error('ESM directory %s does not exist or is not a directory', esm_dir)
        sys.exit(2)

    esm_map = build_esm_map(esm_dir)
    logger.info('Found %d ESM files in %s', len(esm_map), esm_dir)

    summary = {'processed': 0, 'injected': 0, 'skipped': 0, 'missing_esm': 0, 'missing_complex': 0}

    target_files = sorted(target_dir.glob('*.h5'))
    logger.info('Found %d h5 files to process in %s', len(target_files), target_dir)

    for target_path in target_files:
        base = target_path.stem
        base_id = base[:-len('_filtered')] if base.endswith('_filtered') else base
        
        esm_path = esm_map.get(base_id)
        if esm_path is None:
            # maybe the esm file uses the same base but with _filtered suffix
            alt = f'{base_id}_filtered'
            esm_path = esm_map.get(alt)

        if esm_path is None:
            logger.warning('No matching ESM file found for %s (base_id=%s)', target_path.name, base_id)
            summary['missing_esm'] += 1
            continue

        result = inject_embeddings_inplace(target_path, esm_path, overwrite=args.overwrite, dry_run=args.dry_run)
        summary['processed'] += 1
        if result and result.get('status') == 'done':
            summary['injected'] += result.get('injected', 0)
            summary['skipped'] += result.get('skipped', 0)
            summary['missing_complex'] += result.get('missing_complex', 0)
        elif result and result.get('status') == 'no_esm':
            summary['missing_esm'] += 1

    logger.info('='*70)
    logger.info('Summary:')
    logger.info('  Files processed: %d', summary['processed'])
    logger.info('  Embeddings injected: %d', summary['injected'])
    logger.info('  Embeddings skipped (already exist): %d', summary['skipped'])
    logger.info('  Complexes missing from ESM files: %d', summary['missing_complex'])
    logger.info('  Target files missing ESM match: %d', summary['missing_esm'])
    logger.info('='*70)


if __name__ == '__main__':
    main()
