#!/bin/bash
# CAPRI inference: preprocess RUN01-05 and generate rankings
set -e

# Configuration
CAPRI_DIR="/proj/berzelius-2021-29/users/x_matta/CAPRI_2025_October/T314_61/INIT_MODELS/AF3_MODELS"
TARGET_ID="capri314"
ANTIBODY_CHAINS="H,L"
ANTIGEN_CHAINS="A"
OUTPUT_DIR="inference_results/${TARGET_ID}"

# Model from models_to_test.txt
MODEL_DIR="/proj/berzelius-2021-29/users/x_matta/abag/output_new_multigpu/DeepSet_2025-10-06_21-55-13_aggregator_concat_stats_by_set_size_lr_scheduler_OneCycleLR_maxlr_0.005_epochs_250_steps_per_epoch_22"
CHECKPOINT="${MODEL_DIR}/checkpoint_epoch20.pt"
CONFIG="${MODEL_DIR}/config.yaml"

# Verify files exist
[ -d "$CAPRI_DIR" ] || { echo "ERROR: CAPRI directory not found"; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "ERROR: Checkpoint not found"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: Config not found"; exit 1; }

mkdir -p "$OUTPUT_DIR"

echo "=== Preprocessing RUN01-05 ==="
python preprocess_inference.py \
    --input_dir "$CAPRI_DIR" \
    --output_h5 "${OUTPUT_DIR}/${TARGET_ID}.h5" \
    --antibody_chains "$ANTIBODY_CHAINS" \
    --antigen_chains "$ANTIGEN_CHAINS" \
    --target_id "$TARGET_ID" \
    --run_dirs "RUN01,RUN02,RUN03,RUN04,RUN05" \
    --add_esm \
    --esm_model "facebook/esm2_t6_8M_UR50D"

echo -e "\n=== Running inference ==="
python run_inference.py \
    --h5_file "${OUTPUT_DIR}/${TARGET_ID}.h5" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --target_id "$TARGET_ID" \
    --batch_size 50 \
    --shuffle

echo -e "\n=== Top 10 predictions (combined ranking) ==="
head -11 "${OUTPUT_DIR}/${TARGET_ID}_ranked_by_combined.csv" | column -t -s,

echo -e "\nResults in: $OUTPUT_DIR"
echo "  - ${TARGET_ID}_ranked_by_af3.csv (AF3 ranking)"
echo "  - ${TARGET_ID}_ranked_by_model.csv (model ranking)"
echo "  - ${TARGET_ID}_ranked_by_combined.csv (recommended)"

