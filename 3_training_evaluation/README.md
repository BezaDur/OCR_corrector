# 3_training_evaluation — Evaluate OCR model

**Parent project, step 3: evaluate OCR training.**

## What it does
- Run OCR eval (01_run_ocr_eval.py)
- Diagnose OCR engine (07_diagnose_ocr_engine.py)
- Analyze error composition (05_analyze_error_composition.py)
- Benchmark new vs existing models (02_benchmark_new_model.py, 03_benchmark_existing_ocr.py)
- Audit raw OCR sentences (06_audit_raw_ocr_sentences.py)

## Metrics (moved from root)
- ocr_metrics.py — canonical CER/WER, edit distance
- tsv_gt.py — ground-truth alignment for labels
- tsv_confidence.py — confidence handling
- tsv_gt_stats.py — stats

## How to run
```bash
python 01_run_ocr_eval.py
python 05_analyze_error_composition.py
```

## Output
- CER, WER, confusion analysis
- Input to 4_ocr_correction (build confusion matrix)
