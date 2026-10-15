#!/usr/bin/env bash
# Fast smoke run of 01_evaluate_full_corrector.py on the first N pages.
# Builds a small subset by copying files, so it works with any evaluator version.
# Usage: ./smoke_eval.sh [N]        (default N=10)
# Override paths with environment variables, e.g. DATA=/other/path ./smoke_eval.sh 5
set -euo pipefail

N="${1:-10}"
SUB="${SUB:-/tmp/ocr_smoke$N}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Defaults: the synthetic evaluation set is outside the repo (point
# TESSERACT_WORKSPACE at it); the confusion matrix and dictionary have
# in-repo copies, and KenLM is pointed at explicitly.
DATA="${DATA:-${TESSERACT_WORKSPACE:-/media/des/Tess/Tesseract2}/synthetic_eval}"
KENLM="${KENLM:-${AMHARIC_WORD_KENLM:-/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary}}"
CONF="${CONF:-$REPO/4_ocr_correction/error_detection/Model/confusion_matrix.json}"
DICT="${DICT:-$REPO/4_ocr_correction/error_detection/Model/main_i7_frq10.txt}"
MIN_FREQ="${MIN_FREQ:-2}"
CHAR_LM="${CHAR_LM:-}"
CHAR_ARGS=()
[ -n "$CHAR_LM" ] && CHAR_ARGS=(--char-lm "$CHAR_LM")
TSV_DIR="${TSV_DIR:-}"
[ -n "$TSV_DIR" ] && CHAR_ARGS+=(--tsv-dir "$TSV_DIR")
VERDICT_DIR="${VERDICT_DIR:-}"
[ -n "$VERDICT_DIR" ] && CHAR_ARGS+=(--verdict-dir "$VERDICT_DIR")
HERE="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$SUB" && mkdir -p "$SUB/gt" "$SUB/raw"
count=0
for gt in $(ls "$DATA/ground_truth/"*.gt.txt | sort | head -n "$N"); do
  base=$(basename "$gt" .gt.txt)
  if [ ! -f "$DATA/ocr_new_model/$base.ocr.txt" ]; then
    echo "missing raw OCR for $base" >&2; exit 1
  fi
  cp "$gt" "$SUB/gt/"
  cp "$DATA/ocr_new_model/$base.ocr.txt" "$SUB/raw/"
  count=$((count + 1))
done
echo "Subset: $count pages in $SUB"

cd "$REPO"
nohup nice -n 19 python3 "$HERE/01_evaluate_full_corrector.py" \
  --gt-dir "$SUB/gt" --raw-dir "$SUB/raw" --output-dir "$SUB/out" \
  --kenlm "$KENLM" --confusion "$CONF" --dictionary "$DICT" \
  --dict-min-freq "$MIN_FREQ" "${CHAR_ARGS[@]}" \
  --json-report "$SUB/report.json" > "$SUB/run.log" 2>&1 &
echo "Running in background, PID $!. Log: $SUB/run.log"
echo "Watch progress with: tail -f $SUB/run.log"
