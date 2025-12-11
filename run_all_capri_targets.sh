#!/bin/bash
# Run inference for all CAPRI targets
# Strategy: Preprocess all targets first, then run all inferences
set -e

CAPRI_BASE="/proj/berzelius-2021-29/users/x_matta/CAPRI_2025_October"
ANTIBODY_CHAINS="H,L"
ANTIGEN_CHAINS="A"

MODEL_DIR="/proj/berzelius-2021-29/users/x_matta/abag/output_new_multigpu/DeepSet_2025-10-06_21-55-13_aggregator_concat_stats_by_set_size_lr_scheduler_OneCycleLR_maxlr_0.005_epochs_250_steps_per_epoch_22"
CHECKPOINT="${MODEL_DIR}/checkpoint_epoch20.pt"
CONFIG="${MODEL_DIR}/config.yaml"

# CAPRI targets
TARGETS=("T312_61" "T314_61" "T315_61" "T316_61")

echo "=========================================="
echo "STEP 1: PREPROCESSING ALL TARGETS"
echo "=========================================="

for TARGET in "${TARGETS[@]}"; do
    CAPRI_DIR="${CAPRI_BASE}/${TARGET}/INIT_MODELS/AF3_MODELS"
    OUTPUT_DIR="inference_results/${TARGET}"
    TARGET_ID=$(echo $TARGET | tr '[:upper:]' '[:lower:]')
    H5_FILE="${OUTPUT_DIR}/${TARGET_ID}.h5"
    
    echo ""
    echo "--- Preprocessing $TARGET ---"
    
    if [ ! -d "$CAPRI_DIR" ]; then
        echo "WARNING: Directory not found, skipping: $CAPRI_DIR"
        continue
    fi
    
    if [ -f "$H5_FILE" ]; then
        echo "H5 file already exists, skipping: $H5_FILE"
        continue
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    python preprocess_inference.py \
        --input_dir "$CAPRI_DIR" \
        --output_h5 "$H5_FILE" \
        --antibody_chains "$ANTIBODY_CHAINS" \
        --antigen_chains "$ANTIGEN_CHAINS" \
        --target_id "$TARGET_ID" \
        --run_dirs "RUN01,RUN02,RUN03,RUN04,RUN05" \
        --add_esm \
        --esm_model "facebook/esm2_t6_8M_UR50D"
done

echo ""
echo "=========================================="
echo "STEP 2: RUNNING INFERENCE ON ALL TARGETS"
echo "=========================================="

for TARGET in "${TARGETS[@]}"; do
    CAPRI_DIR="${CAPRI_BASE}/${TARGET}/INIT_MODELS/AF3_MODELS"
    OUTPUT_DIR="inference_results/${TARGET}"
    TARGET_ID=$(echo $TARGET | tr '[:upper:]' '[:lower:]')
    H5_FILE="${OUTPUT_DIR}/${TARGET_ID}.h5"
    
    echo ""
    echo "--- Inference for $TARGET ---"
    
    if [ ! -d "$CAPRI_DIR" ]; then
        echo "WARNING: CAPRI directory not found, skipping: $CAPRI_DIR"
        continue
    fi
    
    if [ ! -f "$H5_FILE" ]; then
        echo "ERROR: H5 file not found: $H5_FILE"
        echo "Run preprocessing first or remove this target"
        continue
    fi
    
    python run_inference.py \
        --h5_file "$H5_FILE" \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --target_id "$TARGET_ID" \
        --batch_size 50 \
        --shuffle
    
    echo -e "\nTop 5 predictions for $TARGET (combined ranking):"
    head -6 "${OUTPUT_DIR}/${TARGET_ID}_ranked_by_combined.csv"
done

echo ""
echo "=========================================="
echo "ALL TARGETS COMPLETE!"
echo "=========================================="
echo "Results in: inference_results/"
for TARGET in "${TARGETS[@]}"; do
    TARGET_ID=$(echo $TARGET | tr '[:upper:]' '[:lower:]')
    if [ -d "inference_results/${TARGET}" ]; then
        echo "  ✓ ${TARGET_ID}/"
    fi
done

