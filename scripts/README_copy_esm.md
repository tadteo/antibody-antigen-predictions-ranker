Copy first ESM embedding per complex into all samples
==================================================

This script copies the first `esm_embeddings` found for each complex in the
filtered-ESM HDF5 files and injects that same embedding into every sample of
the corresponding complex in the source HDF5 files. Output HDF5 files are
written into an output folder (the source files are not modified).

Usage
-----

Example:

```bash
python3 scripts/copy_esm_embeddings_by_complex.py \
  --src-dir /proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new_test_with_stats \
  --esm-dir /proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new_test_with_stats_filtered_esm \
  --out-dir /proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/new_test_with_stats_with_esm_first_per_complex \
  --dry-run
```

Flags
-----
- `--src-dir`: folder with source `.h5` files to augment.
- `--esm-dir`: folder containing filtered esm `.h5` files that include `esm_embeddings`.
- `--out-dir`: folder where new augmented `.h5` files will be written.
- `--overwrite`: if provided, existing `esm_embeddings` in destination files will be replaced.
- `--dry-run`: only log planned actions; do not write files.
- `--verbose` / `-v`: enable debug logging.

Notes
-----
- The script matches files by basename (it tolerates `_filtered` suffix in esm filenames).
- It uses the first `esm_embeddings` dataset found under each complex in the esm file
  and copies that exact array into every sample group of the corresponding complex
  in the source file. If the embedding length differs from a sample's sequence
  length, the script will still copy the embedding but will log a debug message.
- The output file keeps the structure/content of the source file and only adds
  (or replaces when `--overwrite`) the `esm_embeddings` datasets.

Dependencies
------------
- `h5py`, `numpy`
