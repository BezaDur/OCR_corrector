# 3_training_evaluation — measure the OCR model

**Parent project, step 3: score what step 2 produced.**

## What it does

| Script | What it does |
|---|---|
| `02_run_ocr_eval.py` | run the `tesseract` binary over the evaluation pages (multiprocessing, PSM 3) |
| `ocr_metrics.py` | **the metric core**: edit distance, co-optimal word alignment (with tie ranges), CER/WER, clean-token damage (`score_texts`) |
| `tsv_gt.py` | annotate Tesseract TSV rows with the ground-truth word |
| `tsv_confidence.py` | read `<page>.conf.tsv`, align word confidences (0–100) |
| `tsv_gt_stats.py` | aggregate per-TSV GT/confidence statistics |
| `06_diagnose_ocr_engine.py` | per-page GT vs OCR diagnosis |
| `09_analyze_error_composition.py` | split errors into channels, compare with a synthetic corruption model |
| `10_compare_ocr_outputs.py` | page-level CER/WER diff between two OCR directories |
| `11_benchmark_new_model.py`, `11b_benchmark_existing_ocr.py` | benchmark a model / an existing OCR engine |
| `12_audit_raw_ocr_sentences.py` | sentence-level audit of raw OCR |
| `query_raw_characters.py` | count rare characters across a corpus |

## Inputs and outputs

* Needs `TESSERACT_WORKSPACE` (`synthetic_eval/` with `images/`, `ground_truth/`),
  the `tesseract` binary and a `tessdata` directory. `02_run_ocr_eval.py`
  hard-codes `/usr/share/tesseract-ocr/5/tessdata` for `TESSDATA_DIR` — change
  it or set it there before running.
* Output: `ocr_pred/<page>.ocr.txt` plus `<page>.conf.tsv`, then CER/WER and the
  error-composition report.

The checked-in 100-page ground truth + OCR output (with `.conf.tsv`) that the
detector uses lives in `OCR_error_detection/Evaluation/`; its `analysis.tsv`
was scored by an older detector — see `docs/current_state.md` §5.2.

`ocr_metrics.py` is also imported by `5_correction_evaluation/` (through the root
shim `ocr_metrics.py`), so a change here moves the correction benchmark too.

## How to run

From the repository root:

```bash
python repo_paths.py                              # confirm the workspace root
python 3_training_evaluation/02_run_ocr_eval.py
python 3_training_evaluation/09_analyze_error_composition.py
python 3_training_evaluation/10_compare_ocr_outputs.py --gt-dir ... --ocr-dir ...
```

## Reported metrics

The 100-page run stored in `OCR_error_detection/Evaluation/` has 4,267 wrong
tokens out of 62,385 scored tokens — **93.2 % word accuracy / 6.8 % WER** (the older README
rounded this to 93 % / 7 %, with 98.07 % character accuracy). The corrector
benchmark's strict vs canonical CER/WER and its clean-token-damage definition are
documented in `docs/two_phase_benchmark.md`.
