# Repository layout

Verified against the working tree on 2026-10-25 (branch
`arena/c4c81d74-ocr-corrector`, after the 2026-10-25 reorganisation merged in
`e8cb916`).

The pipeline runs in five stages, from raw data to the final correction score.
Each stage has its own folder, and **script numbers are per stage**: they show the
order things run *inside* that stage and start at 01 in each folder. The old
global numbering (01…15 scattered across five folders, with two different files
both named `09_`) recorded the order things were written, not the order they run.

```
1_data_preparation/        create and clean the training and evaluation data
2_training/                train the OCR model and package it
3_training_evaluation/     measure the trained OCR model
4_ocr_correction/          detect word errors, build the confusion matrix, correct
5_correction_evaluation/   measure what the corrector changes
Evaluation/                the 100-page corpus, shared by stage 3 and the detector
tools/                     cross-stage helpers, not a pipeline stage
vendor/segmenter/          frozen copy of the Amharic morphological suite
archive/                   superseded code, models and reports
docs/                      design notes, reviews, verified facts
tests/                     the pipeline test suite
```

The repository root itself holds only `README.md`, `conftest.py` and
`requirements.txt`. There are no root-level module shims any more — the stage
folders are put on `sys.path` by `tools/stage_paths.py` (see below).

---

## How imports work now

Stage folder names start with a digit, so they can never be Python packages. Every
shared module is imported **by bare name**, and the folders are registered once:

```python
import os, sys
_r = os.path.dirname(os.path.abspath(__file__))
while _r != os.path.dirname(_r) and not os.path.exists(os.path.join(_r, "tools", "stage_paths.py")):
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "tools"))
import stage_paths
stage_paths.register()      # `import repo_paths`, `from error_detection import ...` now work
```

`conftest.py` does this for the whole test suite, which is why `tests/` can import
stage modules with no path setup of its own. `tests/test_repo_layout.py` checks
that the layout and those imports keep working.

| File | Purpose |
|---|---|
| `tools/stage_paths.py` | puts every stage folder on `sys.path`; replaces the thirteen root shims that used to forward imports one by one |
| `tools/repo_paths.py` | the one place that knows where external data lives (KenLM models, dictionaries, Tesseract workspace/training, raw corpora) |
| `tools/analysis/` | the 14 exploratory scripts behind `docs/lgbm_detector_review.md` (moved here from `docs/lgbm_review/`) |

---

## Stage contents

### 1. Data preparation
| File | Purpose |
|---|---|
| `01_discover_character_inventory.py` … `04_assemble_balanced_training_data.py` | build the training text and character inventory |
| `05_generate_training_boxes.py` | create Tesseract training box files |
| `06_compile_lstmf_and_split.py` | compile `.lstmf` files, split train/eval |
| `07_generate_eval_pages.py` | render synthetic evaluation pages with several fonts and degradations |
| `generate_eval_pages_simple.py`, `generate_eval_dataset_simulated.py` | two alternative generators (real Tesseract vs. confusion-matrix simulation) |
| `audit_training_text.py`, `audit_rare_glyphs.py` | check the training text and glyph coverage |
| `check_pango_names.py`, `clean_font_directories.py`, `inspect_and_install_fonts.py`, `diagnose_font_accuracy.py`, `diagnose_font_cmaps.py` | font checks and setup |
| `data/amh_final_training_text.txt` | 4.42 M tokens, 638 k types, 359,776 lines, 58 MB |
| `data/amh.unicharset`, `data/amh.training_text`, … | Tesseract inputs |
| `data/telegram_c4.txt` | 42 MB corpus dump, read only by `tools/analysis/build_synthetic_source.py` |
| `fonts/` | the Ethiopic `.ttf` set |

### 2. Training
| File | Purpose |
|---|---|
| `01_train_epochs.sh`, `02_run_epoch2.sh` | run the training epochs |
| `03_package_best_and_fast.sh` | package the best and fast models |
| `diagnose_job_failure.py` | explain a failed training job |
| `models/best/amh.traineddata` (8 MB), `models/best/amh.lstm`, `models/google_amh.lstm` | trained models |
| `models/packaged/amh_best.traineddata`, `amh_fast.traineddata` | deployable models |
| `config/` | `amh.unicharset` and training config; **`config/confusion_matrix.json` is stale and read by no code path** — the `4_ocr_correction/error_detection/Model/` copy is authoritative |

### 3. Training evaluation
| File | Purpose |
|---|---|
| `01_run_ocr_eval.py` | run Tesseract over the evaluation pages |
| `ocr_metrics.py` | CER/WER, edit distance, co-optimal alignment (used by stage 5 too) |
| `tsv_gt.py`, `tsv_confidence.py`, `tsv_gt_stats.py` | ground-truth alignment and confidence handling for TSV output |
| `02_benchmark_new_model.py`, `03_benchmark_existing_ocr.py` | benchmark models |
| `04_compare_ocr_outputs.py` | diff two OCR outputs |
| `07_diagnose_ocr_engine.py`, `05_analyze_error_composition.py` | find where the engine fails |
| `06_audit_raw_ocr_sentences.py`, `query_raw_characters.py` | corpus audits |

### 4. OCR correction
| File | Purpose |
|---|---|
| `amharic_ocr_corrector.py` | the corrector (1,817 lines) |
| `normalizer.py`, `amharic_canonical_normalizer.py`, `suspicion.py` | normalisation and the rule-based Stage 1 fallback |
| `01_build_confusion_matrix.py`, `confusion_matrix_patch.json`, `confusion_matrix_coverage_audit.py` | build and audit the optical confusion matrix |
| `trace_candidate.py` | score trace for one token |
| `tests/` | corrector smoke tests (KenLM-dependent ones skip without it) |
| `error_detection/` | the detector — see below |

### 5. Correction evaluation
| File | Purpose |
|---|---|
| `01_evaluate_full_corrector.py` | end-to-end benchmark: CER/WER, clean-token damage, JSON report |
| `02_collect_error_cases.py`, `03_audit_correction_decisions.py` | alignment-derived per-token decision audit |
| `04_stage1_precision.py`, `05_fit_stage1_scores.py`, `stage1_fit.py`, `stage1_labels.py` | Stage 1 precision and score fitting |
| `smoke_eval.sh` | fast run on the first N pages |
| `data/correction_decisions.tsv`, `data/failed_cases.tsv` | last audit outputs |
| `benchmarks/benchmark_full_symspell.json` | last committed benchmark — from an older corrector revision, see `docs/current_state.md` §5.1 |

### Child project — `4_ocr_correction/error_detection/`
Self-contained: `detector.py`, `features.py`, `adapter.py`, `Model/` (model,
dictionaries, confusion matrix, training corpus), `tests/`, `scripts/`,
`archive/`. Its README documents the model and the lexicon filter.

| Script | Purpose |
|---|---|
| `scripts/evaluate_child.py` | score the child on the corpus with the production path |
| `scripts/tune_gates.py` | choose PASS/FLAG gates with page-level CV instead of on the pages you report |
| `scripts/evaluate_out_of_sample.py` | the held-out measurement behind `docs/detector_out_of_sample.md` |
| `scripts/verify_claims.py` | re-derive every published number and try to break it |
| `scripts/make_eval_corpus.py` | turn a new `Evaluation/`-style folder into a scorable corpus |
| `scripts/compare_lexicon_configs.py`, `probe_late_fusion.py`, `audit_decision.py`, `test_child.py` | lexicon/feature/decision experiments |

The page corpus they score lives at **`Evaluation/`** in the repository root
(`GroundTruth/` 100 `.gt.txt`, `OCRed/` 200 `.txt` + `.conf.tsv`, `images/` 100
PNGs, `analysis.tsv` 62 k rows), because stage 3 and the detector both read it.

---

## Where data lives

Big inputs and outputs are **not** in git. `tools/repo_paths.py` maps them:

| Variable | Default (author's machine) | Holds |
|---|---|---|
| `AMHARIC_DATA_ROOT` | `/mnt/mintdata/Users/Des/Mint` | KenLM models, `main_i7.txt`, raw corpora |
| `TESSERACT_WORKSPACE` | `/media/des/Tess/Tesseract2` | `synthetic_eval/`, `ocr_pred/`, training output |
| `TESSERACT_TRAINING` | `/media/des/Tess/Tesseract/TessTraining` | `Fonts/`, `langdata/` |
| `AMHARIC_WORD_KENLM` | `<data root>/Models/KenLM/word_model/amharic_5gram.binary` | word 5-gram (required by the corrector) |
| `AMHARIC_CHAR_KENLM` | `<data root>/Models/KenLM/character_model/amharic_char_model.klm` | char LM (optional) |
| `AMHARIC_MORPH_PATHS` | vendored copy, then `…/p16_morpheme_analysis` | live segmenter checkouts (higher priority than vendored) |

`repo_paths.py` also prefers in-repo copies
(`4_ocr_correction/error_detection/Model/main_i7_frq10.txt`,
`…/confusion_matrix.json`, `vendor/segmenter/data/morphemes.json`) when they
exist, so the corrector and the detector work offline.

---

## Size of the repository

`git ls-files`: **1,089 files, ~950 MB tracked**. The heavy items are
`Evaluation/images/` (570 MB, 100 page PNGs), `harmonized_corpus_jiwer.tsv`
(50 MB), `amh_final_training_text.txt` and `mined_rare_sentences.txt` (58 MB
each), `telegram_c4.txt` (42 MB), `amh.training_text` (25 MB), and the 15 MB
dictionary twice.

Still tracked despite `.gitignore` (ignore rules do not apply to tracked files):
`__pycache__/*.pyc`, `augraphy_cache/`, `font_diag_temp/`.

---

## How to run

From the repository root:

```bash
pip install -r requirements.txt pytest
python -m pytest 4_ocr_correction/error_detection/tests -q   # child: 97 tests
python -m pytest tests -q                        # repo: 158 passed, 52 skipped
python tools/repo_paths.py                       # what data resolves where
python 5_correction_evaluation/01_evaluate_full_corrector.py --help
```

## Adding a new script

Put it in the stage it belongs to, use the next free number **for that stage**,
and start the file with a docstring that says what it reads and writes. Cross-stage
helpers go in `tools/`, not at the root, and are imported through
`stage_paths.register()`. If a script is not a step in its stage's flow (an audit,
a diagnostic, a shared library), give it no number.
