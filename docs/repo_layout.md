# Repository layout

The pipeline runs in five stages, from raw data to the final correction score.
Each stage has its own folder. File names keep their original numbers, so the
numbers show the order inside a stage and match the existing documentation.

```
1_data_preparation/        create and clean the training and evaluation data
2_training/                train the OCR model and package it
3_training_evaluation/     measure the trained OCR model
4_ocr_correction/          build and run the post-OCR corrector
5_correction_evaluation/   measure what the corrector changes
```

Shared code stays at the repo root, because every stage imports it:

| Module | Used for |
| --- | --- |
| `amharic_ocr_corrector.py` | the corrector (Stages 1–3 of the framework) |
| `suspicion.py` | Stage 1 error detection |
| `normalizer.py`, `amharic_canonical_normalizer.py` | text normalisation |
| `ocr_metrics.py` | CER / WER and token alignment |
| `ocr_decision_audit.py` | per-decision audit of corrections |

Also at the root: `tests/` (unit tests), `docs/` (design notes and this file),
`vendor/` (the morphological segmenter), and `archive/` (superseded files kept
for reference).

## Stage contents

### 1. Data preparation
| File | Purpose |
| --- | --- |
| `07_generate_eval_pages.py` | render synthetic evaluation pages with several fonts and degradations |
| `05_generate_training_boxes.py` | create Tesseract training box files |
| `06_compile_lstmf_and_split.py` | compile `.lstmf` files and split train / eval |
| `04_assemble_balanced_training_data.py`, `03_build_final_training_data.py` | assemble the training text |
| `audit_training_text.py`, `audit_rare_glyphs.py` | check the training text and glyph coverage |
| `02_rare_character_miner.py`, `01_discover_character_inventory.py` | find rare characters and the character inventory |
| `check_pango_names.py`, `clean_font_directories.py`, `inspect_and_install_fonts.py`, `diagnose_font_accuracy.py`, `diagnose_font_cmaps.py` | font checks and setup |

### 2. Training
| File | Purpose |
| --- | --- |
| `01_train_epochs.sh`, `02_run_epoch2.sh` | run the training epochs |
| `03_package_best_and_fast.sh` | package the best and the fast model |
| `diagnose_job_failure.py` | explain a failed training job |

### 3. Training evaluation
| File | Purpose |
| --- | --- |
| `01_run_ocr_eval.py` | run the OCR engine on the evaluation pages |
| `02_benchmark_new_model.py`, `03_benchmark_existing_ocr.py` | benchmark the new model and the existing engine |
| `04_compare_ocr_outputs.py` | compare two OCR outputs |
| `07_diagnose_ocr_engine.py`, `05_analyze_error_composition.py` | find where the OCR engine fails |
| `query_raw_characters.py` | count rare characters across the raw corpus |
| `06_audit_raw_ocr_sentences.py` | audit raw OCR sentences |

### 4. OCR correction
| File | Purpose |
| --- | --- |
| `01_build_confusion_matrix.py` | build the optical confusion matrix from aligned OCR output |
| `confusion_matrix_coverage_audit.py` | check which character pairs the matrix covers |
| `trace_candidate.py` | trace how one token's candidates are scored |
| `test_corrector.py`, `test_page1_fixes.py` | corrector smoke runs (need kenlm and the lexicon) |

### 5. Correction evaluation
| File | Purpose |
| --- | --- |
| `01_evaluate_full_corrector.py` | end-to-end benchmark: CER / WER and clean-token damage |
| `smoke_eval.sh` | fast run on the first N pages (see below) |
| `02_collect_error_cases.py`, `03_audit_correction_decisions.py` | classify every correction decision |

## How to run

Run every command from the repo root and pass the script's path:

```bash
python3 5_correction_evaluation/01_evaluate_full_corrector.py --help
./5_correction_evaluation/smoke_eval.sh 10
```

Scripts that import a shared module add the repo root to `sys.path` themselves,
so they run from any working directory.

## Adding a new script

Put it in the stage it belongs to, use the next free number for that stage,
and start the file with a docstring that says what it reads and writes.
