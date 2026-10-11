#!/usr/bin/env bash
# Fast smoke run of 04_evaluate_full_corrector.py on the first N pages.
# Builds a small subset by copying files, so it works with any evaluator version.
# Usage: ./smoke_eval.sh [N]        (default N=10)
# Override paths with environment variables, e.g. DATA=/other/path ./smoke_eval.sh 5
set -euo pipefail

N="${1:-10}"
SUB="${SUB:-/tmp/ocr_smoke$N}"
DATA="${DATA:-/media/des/Tess/Tesseract2/synthetic_eval}"
KENLM="${KENLM:-/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary}"
CONF="${CONF:-/media/des/Tess/Tesseract2/confusion_matrix.json}"
DICT="${DICT:-/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt}"
MIN_FREQ="${MIN_FREQ:-2}"
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

cd "$HERE"
nohup nice -n 19 python3 04_evaluate_full_corrector.py \
  --gt-dir "$SUB/gt" --raw-dir "$SUB/raw" --output-dir "$SUB/out" \
  --kenlm "$KENLM" --confusion "$CONF" --dictionary "$DICT" \
  --dict-min-freq "$MIN_FREQ" \
  --json-report "$SUB/report.json" > "$SUB/run.log" 2>&1 &
echo "Running in background, PID $!. Log: $SUB/run.log"
echo "Watch progress with: tail -f $SUB/run.log"
