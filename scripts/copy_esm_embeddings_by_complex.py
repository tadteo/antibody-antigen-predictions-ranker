#!/usr/bin/env python3
"""
Copy first ESM embedding per complex from a filtered-ESM folder into all samples
of matching HDF5 files in a source folder, writing results to an output folder.

Behavior:
- For each `.h5` file in `--src-dir`, find a corresponding `.h5` in `--esm-dir` by
  matching the basename (ignores an optional `_filtered` suffix).
- For each top-level complex in the source file, read the first `esm_embeddings`
  dataset found under the corresponding complex in the ESM file and write that
  same embedding into every sample subgroup for that complex in the output file.

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


def copy_and_inject(src_path: Path, esm_path: Path, out_path: Path, overwrite=False, dry_run=False):
    logger = logging.getLogger('copy')
    logger.info('Processing %s -> %s (esm=%s)', src_path.name, out_path.name, esm_path.name if esm_path else 'MISSING')

    if out_path.exists() and not overwrite:
        logger.info('Output %s exists; skipping (use --overwrite to replace)', out_path)
        return {'status': 'skipped_exists'}

    if dry_run:
        logger.info('Dry-run enabled: would copy input file and inject embeddings from %s', esm_path)
        return {'status': 'dryrun'}

    # copy whole file structure from src to out
    with h5py.File(src_path, 'r') as src_hf, h5py.File(out_path, 'w') as out_hf:
        # copy top-level objects
        for key in src_hf:
            try:
                src_hf.copy(key, out_hf)
            except Exception as e:
                logger.warning('Failed to copy key %s: %s', key, e)

        # open esm file if present
        if esm_path is None:
            logger.warning('No ESM file for %s; skipping embedding injection', src_path.name)
            return {'status': 'copied_no_esm'}

        with h5py.File(esm_path, 'r') as esm_hf:
            injected = 0
            skipped = 0
            missing_complex = 0
            for complex_key in out_hf:
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

                grp = out_hf[complex_key]
                # iterate sample subgroups and set dataset
                for sample_name in grp:
                    sample = grp[sample_name]
                    # only act on groups
                    if not isinstance(sample, h5py.Group):
                        continue
                    if 'esm_embeddings' in sample and not overwrite:
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
                        logger.debug('Embedding length %d vs seq length %s in %s/%s', emb.shape[0], seq_len, src_path.name, f'{complex_key}/{sample_name}')
                    # delete existing if overwrite
                    if 'esm_embeddings' in sample and overwrite:
                        try:
                            del sample['esm_embeddings']
                        except Exception:
                            pass
                    # create dataset
                    try:
                        sample.create_dataset('esm_embeddings', data=emb, compression='gzip')
                        injected += 1
                    except Exception as e:
                        logger.warning('Failed to write esm_embeddings into %s/%s: %s', complex_key, sample_name, e)

            return {'status': 'done', 'injected': injected, 'skipped': skipped, 'missing_complex': missing_complex}


def main():
    parser = argparse.ArgumentParser(description='Copy first ESM per complex to all samples and write outputs.')
    parser.add_argument('--src-dir', required=True, type=Path, help='Source folder with h5 files to augment')
    parser.add_argument('--esm-dir', required=True, type=Path, help='Folder with filtered ESM h5 files')
    parser.add_argument('--out-dir', required=True, type=Path, help='Output folder for new h5 files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing esm_embeddings in target')
    parser.add_argument('--dry-run', action='store_true', help="Don't write files; just show what would happen")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('copy')

    src_dir = args.src_dir
    esm_dir = args.esm_dir
    out_dir = args.out_dir

    if not src_dir.exists() or not src_dir.is_dir():
        logger.error('Source directory %s does not exist or is not a directory', src_dir)
        sys.exit(2)
    if not esm_dir.exists() or not esm_dir.is_dir():
        logger.error('ESM directory %s does not exist or is not a directory', esm_dir)
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    esm_map = build_esm_map(esm_dir)

    summary = {'processed': 0, 'injected': 0, 'skipped': 0, 'missing_esm': 0}

    for src_path in sorted(src_dir.glob('*.h5')):
        base = src_path.stem
        base_id = base[:-len('_filtered')] if base.endswith('_filtered') else base
        esm_path = esm_map.get(base_id)
        if esm_path is None:
            # maybe the esm file uses the same base but with _filtered suffix
            alt = f'{base_id}_filtered'
            esm_path = esm_map.get(alt)

        out_path = out_dir / src_path.name

        result = copy_and_inject(src_path, esm_path, out_path, overwrite=args.overwrite, dry_run=args.dry_run)
        summary['processed'] += 1
        if result is None:
            summary['missing_esm'] += 1
        else:
            if result.get('injected'):
                summary['injected'] += result.get('injected', 0)
            if result.get('skipped'):
                summary['skipped'] += result.get('skipped', 0)
            if result.get('status') == 'copied_no_esm':
                summary['missing_esm'] += 1

    logger.info('Done. Files processed: %d, embeddings injected: %d, skipped: %d, missing esm files for %d',
                summary['processed'], summary['injected'], summary['skipped'], summary['missing_esm'])


if __name__ == '__main__':
    main()
