# Repository layout

The pipeline runs in five stages, from raw data to the final correction score.
Each stage has its own folder; file prefixes keep their original numbers so the
ordering inside a stage survives.

```
1_data_preparation/        create and clean the training and evaluation data
2_training/                train the OCR model and package it
3_training_evaluation/     measure the trained OCR model
4_ocr_correction/          build and run the post-OCR corrector
5_correction_evaluation/   measure what the corrector changes
OCR_error_detection/       child project: word-level error detector (self-contained)
vendor/segmenter/          frozen copy of the Amharic morphological suite
archive/                   superseded code, models and reports
docs/                      design notes, reviews, verified facts
fonts/                     Ethiopic fonts used by the page generator
models/                    packaged Tesseract models (amh_best / amh_fast)
```

Everything imports from the repository root, and every script that needs the
root adds it to `sys.path` itself, so commands work from any directory.

---

## The root is a compatibility surface, not the code

The modules that used to sit at the root were moved into their stage folders.
What is left at the root are **thin shims** that load the real module and install
it under the old name, so old imports keep working:

| Root shim | Real module |
|---|---|
| `amharic_ocr_corrector.py` | `4_ocr_correction/amharic_ocr_corrector.py` |
| `normalizer.py` | `4_ocr_correction/normalizer.py` |
| `amharic_canonical_normalizer.py` | `4_ocr_correction/amharic_canonical_normalizer.py` |
| `suspicion.py` | `4_ocr_correction/suspicion.py` |
| `ocr_metrics.py` | `3_training_evaluation/ocr_metrics.py` |
| `tsv_gt.py`, `tsv_confidence.py`, `tsv_gt_stats.py` | `3_training_evaluation/` |
| `stage1_fit.py`, `stage1_labels.py` | `5_correction_evaluation/` |
| `generate_eval_pages.py`, `generate_eval_dataset.py` | `1_data_preparation/` |
| `ocr_decision_audit.py` | `OCR_error_detection/scripts/audit_decision.py` |

Two files at the root are real code, not shims:

* **`repo_paths.py`** — the single place that knows where external trees live
  (KenLM models, dictionaries, Tesseract workspace/training, raw corpora). Every
  root is an environment variable with the author's machine path as the default;
  `python repo_paths.py` prints what resolves where. It also prefers in-repo
  copies (`OCR_error_detection/Model/main_i7_frq10.txt`,
  `.../confusion_matrix.json`, `vendor/segmenter/data/morphemes.json`) when
  they exist, so the corrector and detector work offline.
* **`requirements.txt`** — pip dependencies for the pipeline and the child.

`telegram_c4.txt` (42 MB) is a source corpus dump kept at the root; it is used
only by `docs/lgbm_review/build_synthetic_source.py`.

---

## Stage contents

### 1. Data preparation
| File | Purpose |
|---|---|
| `01_generate_eval_pages.py` | render synthetic evaluation pages with several fonts and degradations |
| `07_generate_training_boxes.py` | create Tesseract training box files |
| `08_compile_lstmf_and_split.py` | compile `.lstmf` files, split train/eval |
| `assemble_balanced_training_data.py`, `build_final_training_data.py` | assemble the training text |
| `audit_training_text.py`, `audit_rare_glyphs.py` | check the training text and glyph coverage |
| `rare_character_miner.py`, `discover_character_inventory.py` | find rare characters, build the inventory |
| `check_pango_names.py`, `clean_font_directories.py`, `inspect_and_install_fonts.py`, `diagnose_font_accuracy.py`, `diagnose_font_cmaps.py` | font checks and setup |
| `data/amh_final_training_text.txt` | 4.42 M tokens, 638 k types, 359,776 lines, 58 MB |
| `data/amh.unicharset`, `data/amh.training_text`, … | Tesseract inputs |

### 2. Training
| File | Purpose |
|---|---|
| `09_train_epochs.sh`, `09_run_epoch2.sh` | run the training epochs |
| `10_package_best_and_fast.sh` | package the best and fast models |
| `diagnose_job_failure.py` | explain a failed training job |
| `models/best/amh.traineddata` (8 MB), `models/best/amh.lstm`, `models/google_amh.lstm` | packaged models |
| `config/` | `amh.unicharset`, training config; **`confusion_matrix.json` here is stale and read by no code path** — the `OCR_error_detection/Model/` copy is authoritative |

### 3. Training evaluation
| File | Purpose |
|---|---|
| `02_run_ocr_eval.py` | run Tesseract over the evaluation pages |
| `ocr_metrics.py` | CER/WER, edit distance, co-optimal alignment (used by stage 5 too) |
| `tsv_gt.py`, `tsv_confidence.py`, `tsv_gt_stats.py` | ground-truth alignment and confidence handling for TSV output |
| `11_benchmark_new_model.py`, `11b_benchmark_existing_ocr.py` | benchmark models |
| `10_compare_ocr_outputs.py` | diff two OCR outputs |
| `06_diagnose_ocr_engine.py`, `09_analyze_error_composition.py` | find where the engine fails |
| `12_audit_raw_ocr_sentences.py`, `query_raw_characters.py` | corpus audits |

### 4. OCR correction
| File | Purpose |
|---|---|
| `amharic_ocr_corrector.py` | the corrector (1,817 lines) |
| `normalizer.py`, `amharic_canonical_normalizer.py`, `suspicion.py` | normalisation and the rule-based Stage 1 fallback |
| `03_build_confusion_matrix.py`, `confusion_matrix_patch.json`, `confusion_matrix_coverage_audit.py` | build and audit the optical confusion matrix |
| `trace_candidate.py` | score trace for one token |
| `test_corrector.py`, `test_page1_fixes.py` | manual smoke runs (need KenLM) |

### 5. Correction evaluation
| File | Purpose |
|---|---|
| `04_evaluate_full_corrector.py` | end-to-end benchmark: CER/WER, clean-token damage, JSON report |
| `05_collect_error_cases.py`, `13_audit_correction_decisions.py` | alignment-derived per-token decision audit |
| `14_stage1_precision.py`, `15_fit_stage1_scores.py`, `stage1_fit.py`, `stage1_labels.py` | Stage 1 precision and score fitting |
| `smoke_eval.sh` | fast run on the first N pages |
| `data/correction_decisions.tsv`, `data/failed_cases.tsv` | last audit outputs |
| `benchmarks/benchmark_full_symspell.json` | last committed benchmark — from an older corrector revision, see `docs/current_state.md` §5.1 |

### Child project
`OCR_error_detection/` is self-contained: `detector.py`, `features.py`,
`adapter.py`, `Model/` (model, dictionaries, confusion matrix, training corpus),
`Evaluation/` (100 pages + `analysis.tsv`), `tests/` (56 tests), `scripts/`,
`archive/`. Its README documents the model and the lexicon filter, and
`scripts/evaluate_out_of_sample.py` recomputes its held-out performance (see `docs/detector_out_of_sample.md`).

---

## Where data lives

Big inputs and outputs are **not** in git. `repo_paths.py` maps them:

| Variable | Default (author's machine) | Holds |
|---|---|---|
| `AMHARIC_DATA_ROOT` | `/mnt/mintdata/Users/Des/Mint` | KenLM models, `main_i7.txt`, raw corpora |
| `TESSERACT_WORKSPACE` | `/media/des/Tess/Tesseract2` | `synthetic_eval/`, `ocr_pred/`, training output |
| `TESSERACT_TRAINING` | `/media/des/Tess/Tesseract/TessTraining` | `Fonts/`, `langdata/` |
| `AMHARIC_WORD_KENLM` | `<data root>/Models/KenLM/word_model/amharic_5gram.binary` | word 5-gram (required by the corrector) |
| `AMHARIC_CHAR_KENLM` | `<data root>/Models/KenLM/character_model/amharic_char_model.klm` | char LM (optional) |
| `AMHARIC_MORPH_PATHS` | vendored copy, then `…/p16_morpheme_analysis` | live segmenter checkouts (higher priority than vendored) |

---

## Size of the repository

`git ls-files`: **1,143 files, ~960 MB tracked**. The heavy items are
`OCR_error_detection/Evaluation/images/` (570 MB, 100 page PNGs),
`harmonized_corpus_jiwer.tsv` (50 MB), `amh_final_training_text.txt` and
`mined_rare_sentences.txt` (58 MB each), `telegram_c4.txt` (42 MB),
`amh.training_text` (25 MB), and the 15 MB dictionary twice.

Still tracked despite `.gitignore` (ignore rules do not apply to tracked files):
`__pycache__/*.pyc`, `augraphy_cache/`, `font_diag_temp/`.

---

## How to run

From the repository root:

```bash
pip install -r requirements.txt pytest
python -m pytest OCR_error_detection/tests -q    # child: 56 tests
python -m pytest tests -q                        # repo: 142 passed, 52 skipped
python repo_paths.py                             # what data resolves where
python 5_correction_evaluation/04_evaluate_full_corrector.py --help
```

## Adding a new script

Put it in the stage it belongs to, use the next free number for that stage, and
start the file with a docstring that says what it reads and writes. Add a
root shim only if old code imports it by its old root-level name.
