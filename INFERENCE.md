# Inference Pipeline

Simple two-script inference pipeline for antibody-antigen ranking.

## Quick Start

```bash
# Edit chain IDs if needed in run_capri_inference.sh
bash run_capri_inference.sh
```

This processes all RUN directories (RUN01-05), runs the model, and generates 3 ranking CSVs.

## Files

- `preprocess_inference.py` - Extract features from AF3 predictions to HDF5
- `run_inference.py` - Load HDF5, run model, generate rankings
- `run_capri_inference.sh` - Run both steps for CAPRI target

## Output

Three CSV files with different ranking strategies:
1. `*_ranked_by_af3.csv` - AF3's original ranking
2. `*_ranked_by_model.csv` - Model's predicted DockQ
3. `*_ranked_by_combined.csv` - Max of normalized scores (recommended)

## Manual Usage

```bash
# Preprocess
python preprocess_inference.py \
    --input_dir /path/to/AF3_MODELS \
    --output_h5 target.h5 \
    --antibody_chains H,L \
    --antigen_chains A \
    --target_id target_name \
    --run_dirs RUN01,RUN02,RUN03,RUN04,RUN05 \
    --add_esm

# Run inference
python run_inference.py \
    --h5_file target.h5 \
    --checkpoint path/to/checkpoint.pt \
    --config path/to/config.yaml \
    --output_dir results \
    --shuffle
```

## Model Requirements

Preprocessing automatically matches training:
- ESM embeddings (320-dim, esm2_t6_8M_UR50D) - **computed once, reused for all samples**
- Interchain CA distances
- Interchain PAE values  
- Feature centering
- **Distance cutoff filtering (12.0 Å)** - removes residue pairs >12 Å apart

## Key Implementation Details

1. **ESM Efficiency**: ESM embeddings computed ONCE from sequence (reused for all 500 samples!)
2. **Distance Filtering**: Applied during INFERENCE, not preprocessing (matches training pipeline)
   - Preprocessing: extracts ALL pairs, computes statistics
   - Inference: applies 12.0 Å cutoff when loading features
   - If ALL pairs filtered, keeps closest pair
3. **Feature Centering**: PAE statistics computed on unfiltered data, then filtered during inference
4. **Exact Training Match**: All preprocessing steps match training pipeline exactly

