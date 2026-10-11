#!/usr/bin/env bash
# ==============================================================================
# Step 9: Stefan Weil Sequential Epoch Runner for Tesseract LSTM
# ==============================================================================

set -euo pipefail

BASE_DIR="/media/des/Tess/Tesseract2"
CHECKPOINTS_DIR="${BASE_DIR}/training/checkpoints"
MODELS_DIR="${BASE_DIR}/training/models"
LOG_FILE="${BASE_DIR}/training/amh_epoch_training.log"

TRAIN_LIST="${BASE_DIR}/training/lists/list.train"
EVAL_LIST="${BASE_DIR}/training/lists/list.eval"
TRAINEDDATA="${BASE_DIR}/training/amh/amh.traineddata"
BASE_LSTM="${BASE_DIR}/true_google_best/amh.lstm"

TOTAL_EPOCHS=5

mkdir -p "${CHECKPOINTS_DIR}" "${MODELS_DIR}"

echo "======================================================================"
echo "STEFAN WEIL SEQUENTIAL EPOCH TRAINING (Total Epochs: ${TOTAL_EPOCHS})"
echo "======================================================================"

for (( epoch=1; epoch<=TOTAL_EPOCHS; epoch++ ))
do
    EPOCH_ITER="-${epoch}"
    echo ""
    echo "----------------------------------------------------------------------"
    echo ">> RUNNING EPOCH ${epoch} OF ${TOTAL_EPOCHS} (Flag: --max_iterations ${EPOCH_ITER})"
    echo "----------------------------------------------------------------------"

    # Find latest checkpoint from previous epoch (or start fresh on Epoch 1)
    LATEST_CHECKPOINT=$(ls -t ${CHECKPOINTS_DIR}/*.checkpoint 2>/dev/null | head -n 1 || true)

    if [ "${epoch}" -eq 1 ] && [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "Initializing Epoch 1 with Layer Surgery (APPEND_INDEX=5, O1c391)..."
        lstmtraining \
            --continue_from "${BASE_LSTM}" \
            --append_index 5 \
            --net_spec '[Lfx256 O1c391]' \
            --traineddata "${TRAINEDDATA}" \
            --model_output "${CHECKPOINTS_DIR}/amh" \
            --train_listfile "${TRAIN_LIST}" \
            --eval_listfile "${EVAL_LIST}" \
            --max_iterations "${EPOCH_ITER}" \
            --target_error_rate -5 \
            2>&1 | tee -a "${LOG_FILE}"
    else
        echo "Continuing Epoch ${epoch} from latest checkpoint: ${LATEST_CHECKPOINT}"
        lstmtraining \
            --continue_from "${LATEST_CHECKPOINT}" \
            --traineddata "${TRAINEDDATA}" \
            --model_output "${CHECKPOINTS_DIR}/amh" \
            --train_listfile "${TRAIN_LIST}" \
            --eval_listfile "${EVAL_LIST}" \
            --max_iterations "${EPOCH_ITER}" \
            --target_error_rate -5 \
            2>&1 | tee -a "${LOG_FILE}"
    fi

    # Snapshot and package after each completed epoch
    BEST_EPOCH_CP=$(ls -t ${CHECKPOINTS_DIR}/*.checkpoint 2>/dev/null | head -n 1 || true)
    if [ -n "${BEST_EPOCH_CP}" ]; then
        echo "[SNAPSHOT] Epoch ${epoch} completed. Packaging amh_epoch_${epoch}.traineddata..."
        lstmtraining \
            --stop_training \
            --continue_from "${BEST_EPOCH_CP}" \
            --traineddata "${TRAINEDDATA}" \
            --model_output "${MODELS_DIR}/amh_epoch_${epoch}.traineddata"
    fi

done

echo ""
echo "======================================================================"
echo "ALL ${TOTAL_EPOCHS} EPOCHS COMPLETED SUCCESSFULLY!"
echo "Final models saved in: ${MODELS_DIR}"
echo "======================================================================"