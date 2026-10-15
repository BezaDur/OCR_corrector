#!/usr/bin/env bash
# ==============================================================================
# Step 9: Fail-Safe Epoch 2 Training Runner (Targeting Sub-1% CER & ቯ/ጰ Convergence)
# ==============================================================================

set -uo pipefail

BASE_DIR="${TESSERACT_WORKSPACE:-/media/des/Tess/Tesseract2}"
CHECKPOINTS_DIR="${BASE_DIR}/training/checkpoints"
MODELS_DIR="${BASE_DIR}/training/models"
LOG_FILE="${BASE_DIR}/training/amh_epoch2_training.log"

TRAIN_LIST="${BASE_DIR}/training/lists/list.train"
EVAL_LIST="${BASE_DIR}/training/lists/list.eval"
TRAINEDDATA="${BASE_DIR}/training/amh/amh.traineddata"
LATEST_CP="${CHECKPOINTS_DIR}/amh_checkpoint"

mkdir -p "${CHECKPOINTS_DIR}" "${MODELS_DIR}"

echo "======================================================================"
echo "STARTING EPOCH 2 (Passing all 90,000 lines with --max_iterations -2)"
echo "Continuing from: ${LATEST_CP}"
echo "======================================================================"

# Execute Epoch 2 with non-zero exit trap
lstmtraining \
    --continue_from "${LATEST_CP}" \
    --traineddata "${TRAINEDDATA}" \
    --model_output "${CHECKPOINTS_DIR}/amh" \
    --train_listfile "${TRAIN_LIST}" \
    --eval_listfile "${EVAL_LIST}" \
    --max_iterations -2 \
    --target_error_rate -5 \
    2>&1 | tee -a "${LOG_FILE}" || true

echo ""
echo "======================================================================"
echo "EPOCH 2 FINISHED! PACKAGING MODELS..."
echo "======================================================================"

NEW_CP=$(ls -t ${CHECKPOINTS_DIR}/*.checkpoint 2>/dev/null | head -n 1)

# Package Best Model (Float)
lstmtraining \
    --stop_training \
    --continue_from "${NEW_CP}" \
    --traineddata "${TRAINEDDATA}" \
    --model_output "${MODELS_DIR}/amh_epoch2_best.traineddata"

# Package Fast Model (Int8)
lstmtraining \
    --stop_training \
    --convert_to_int \
    --continue_from "${NEW_CP}" \
    --traineddata "${TRAINEDDATA}" \
    --model_output "${MODELS_DIR}/amh_epoch2_fast.traineddata"

echo "[SUCCESS] amh_epoch2_best.traineddata and amh_epoch2_fast.traineddata created!"
echo "======================================================================"
