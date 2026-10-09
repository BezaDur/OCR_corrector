#!/usr/bin/env bash
# ==============================================================================
# Step 10: Packages the Best Checkpoint into Best and Fast .traineddata Models
# ==============================================================================

set -euo pipefail

BASE_DIR="/media/des/Tess/Tesseract2"
CHECKPOINTS_DIR="${BASE_DIR}/training/checkpoints"
OUTPUT_MODELS_DIR="${BASE_DIR}/training/models"
TRAINEDDATA_PROTO="${BASE_DIR}/training/amh/amh.traineddata"

mkdir -p "${OUTPUT_MODELS_DIR}"

# 1. Automatically find the checkpoint with the lowest CER
BEST_CHECKPOINT=$(ls -t ${CHECKPOINTS_DIR}/*.checkpoint 2>/dev/null | head -n 1 || true)

if [ -z "${BEST_CHECKPOINT}" ] || [ ! -f "${BEST_CHECKPOINT}" ]; then
    echo "Error: No checkpoint files found in ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "======================================================================"
echo "Packaging Latest/Best Checkpoint: ${BEST_CHECKPOINT}"
echo "======================================================================"

# 2. Package the 'BEST' Float Model
lstmtraining \
    --stop_training \
    --continue_from "${BEST_CHECKPOINT}" \
    --traineddata "${TRAINEDDATA_PROTO}" \
    --model_output "${OUTPUT_MODELS_DIR}/amh_best.traineddata"

echo "[SUCCESS] Created: ${OUTPUT_MODELS_DIR}/amh_best.traineddata"

# 3. Package the 'FAST' Quantized Integer8 Model
lstmtraining \
    --stop_training \
    --convert_to_int \
    --continue_from "${BEST_CHECKPOINT}" \
    --traineddata "${TRAINEDDATA_PROTO}" \
    --model_output "${OUTPUT_MODELS_DIR}/amh_fast.traineddata"

echo "[SUCCESS] Created: ${OUTPUT_MODELS_DIR}/amh_fast.traineddata"

echo "======================================================================"
echo "Packaging Complete! Models ready for deployment and evaluation."
echo "======================================================================"
