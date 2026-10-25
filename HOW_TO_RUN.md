# How to run everything — no command-line arguments to remember

**The rule for every script in this repo:**

```bash
python <path/to/the_script.py>        # that's it. No flags.
```

Every script that used to need `--this /path --that 3.5` now carries its
settings **inside the file**, in one clearly marked block near the top:

```python
# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 5_correction_evaluation/01_evaluate_full_corrector.py
# ============================================================================
CONFIG = {
    "gt_dir": GT_DIR,
    "limit_pages": None,
    ...
}
```

To change what a script does: **open the file, edit `CONFIG`, run it.**
The old CLI flags still exist and still override `CONFIG` (same names in
`--kebab-case`), which keeps `5_correction_evaluation/smoke_eval.sh` and the
tests working — but you never *need* them. `tests/test_no_required_cli.py`
fails if any script grows a required argument again.

Scripts that never had CLI arguments (most of stage 1, the diagnostics) keep
their settings as the `Configuration` constants at the top of the file — same
idea, edit-and-run. Shell scripts (`2_training/*.sh`, `smoke_eval.sh`) keep
theirs as the variables at the top, overridable from the environment.

One thing no in-file setting can fix: **big data that is not in git** (KenLM
models, the Tesseract workspace). Those locations come from
`tools/repo_paths.py` — run `python tools/repo_paths.py` to see where
everything resolves on your machine, and set `AMHARIC_DATA_ROOT` /
`TESSERACT_WORKSPACE` / `TESSERACT_TRAINING` if your layout differs from the
defaults (`/mnt/mintdata/Users/Des/Mint`, `/media/des/Tess/Tesseract2`,
`/media/des/Tess/Tesseract/TessTraining`). Scripts that need missing data say
which variable to set instead of dying with a bare `FileNotFoundError`.

---

## How the pieces connect (the 2-minute version)

There are **no git submodules**. There is one *parent* pipeline, one *child*
package inside it, one *vendored* snapshot, and two shared helpers:

```
1_data_preparation ──► 2_training ──► 3_training_evaluation ──► 4_ocr_correction ──► 5_correction_evaluation
   training text,        Tesseract      CER/WER of the            correct the OCR       did correction help?
   boxes, .lstmf,        LSTM model     trained model             text
   synthetic pages                                                     │
                                                                       ▼
                                                  4_ocr_correction/error_detection/   ← the "child"
                                                  LightGBM word-error detector
                                                  (PASS / GRAY / FLAG), own Model/,
                                                  own tests/, own scripts/
```

- **Parent stages 1–5** are folders; script numbers are *per stage*
  (`04_` in stage 3 is unrelated to `04_` in stage 5). Unnumbered files are
  audits, diagnostics, or shared libraries (`ocr_metrics.py`, …).
- **The child**, `4_ocr_correction/error_detection/`, is a real Python
  package (it has `__init__.py`). The parent corrector uses it through
  `error_detection/adapter.py` → `CorrectorDetectionAdapter`, which reuses
  the corrector's own vocab/SymSpell/confusion map instead of loading a
  second copy; if LightGBM can't load, the corrector falls back to the
  rule-based `4_ocr_correction/suspicion.py`. The child never imports the
  parent, so there is no cycle.
- Stage folders start with a digit, so they **cannot** be Python packages.
  Everything cross-stage is imported by bare name (`from ocr_metrics import
  ...`) after `tools/stage_paths.py` puts each stage folder on `sys.path` —
  every stage script starts with that bootstrap, and the root `conftest.py`
  does it for the tests. `tools/repo_paths.py` is the single place that
  knows where out-of-repo data lives.
- **`vendor/segmenter/`** is a frozen copy (not a submodule) of the Amharic
  morphological suite; the corrector imports it as a fallback when no live
  checkout is on `AMHARIC_MORPH_PATHS`. See `vendor/segmenter/VENDORING.md`.
- **`Evaluation/`** (100 pages: `GroundTruth/`, `OCRed/`, `images/`) sits at
  the root because both stage 3 and the child detector read it.
- **`archive/`** is frozen, superseded code. It is deliberately *not* covered
  by the no-argument rule. Details: `docs/repo_layout.md`,
  `docs/architecture.md`.

How it got this way, briefly: shared modules used to live at the repo root
(13 hand-written import shims) with one global script numbering across all
folders; the 2026-10-25 reorganisation moved everything into its stage,
replaced the shims with `stage_paths.py`, renumbered per stage, vendored the
segmenter so a fresh clone can import it, and centralised machine paths in
`repo_paths.py`. Verified facts about the current state:
`docs/current_state.md`.

---

## Run book, stage by stage

All commands are run from the repo root with no arguments. "Edit" = the
`CONFIG` block (or top constants) in that file.

### Stage 1 — `1_data_preparation/` (needs the Tesseract workspace + fonts)

| Script | Does | Edit |
|---|---|---|
| `01_discover_character_inventory.py` | scan corpora → character inventory | top constants |
| `02_rare_character_miner.py` | mine sentences containing rare glyphs | top constants |
| `03_build_final_training_data.py` | assemble the final training text | top constants |
| `04_assemble_balanced_training_data.py` | balance/downsample the training text | top constants |
| `05_generate_training_boxes.py` | render `.tif`/`.box` training lines | `CONFIG`: fonts_dir, training_text, out_dir, lines |
| `06_compile_lstmf_and_split.py` | compile `.lstmf`, write train/eval lists | `CONFIG`: gt_dir, lists_dir |
| `07_generate_eval_pages.py` | render synthetic evaluation pages | `CONFIG`: fonts_dir, gt_source, out_dir, pages |
| `generate_eval_pages_simple.py`, `generate_eval_dataset_simulated.py` | alternative eval-page generators | top constants |
| `audit_*.py`, `diagnose_*.py`, `check_pango_names.py`, `clean_font_directories.py`, `inspect_and_install_fonts.py` | checks and font setup | top constants |

### Stage 2 — `2_training/` (needs Tesseract training tools)

| Script | Does | Edit |
|---|---|---|
| `bash 2_training/01_train_epochs.sh` | sequential epoch training | variables at the top (`BASE_DIR`/`TESSERACT_WORKSPACE`, `TOTAL_EPOCHS`) |
| `bash 2_training/02_run_epoch2.sh` | continue into epoch 2 | variables at the top |
| `bash 2_training/03_package_best_and_fast.sh` | package best/fast `.traineddata` | variables at the top |
| `diagnose_job_failure.py` | explain a failed training run | top constants |

### Stage 3 — `3_training_evaluation/` (measure the OCR model)

| Script | Does | Edit |
|---|---|---|
| `01_run_ocr_eval.py` | batch-OCR the synthetic eval pages | `CONFIG`: dataset_dir, images_dir, out_dir |
| `02_benchmark_new_model.py` | OCR + score a new model | top constants |
| `03_benchmark_existing_ocr.py` | score OCR text already on disk | `CONFIG`: compare_all (default on), gt_dir, limit, workers |
| `04_compare_ocr_outputs.py` | CER/WER table for several OCR outputs | `CONFIG`: gt, ocr (`name=dir` list), limit |
| `05_analyze_error_composition.py` | non-word vs real-word errors, correctability ceiling | `CONFIG`: pairs_gt/pairs_ocr (None = synthetic), targets, seed |
| `06_audit_raw_ocr_sentences.py`, `07_diagnose_ocr_engine.py`, `query_raw_characters.py` | corpus audits / engine diagnosis | top constants |
| `tsv_gt.py` | add GT columns to one Tesseract TSV | `CONFIG`: tsv_in, gt_txt, tsv_out |
| `tsv_gt_stats.py` | confidence-vs-GT summary over a TSV folder | `CONFIG`: tsv_dir, gt_dir, low, high |

Libraries (import, don't run): `ocr_metrics.py`, `tsv_confidence.py`.

### Stage 4 — `4_ocr_correction/` (correct the OCR text)

| Script | Does | Edit |
|---|---|---|
| `01_build_confusion_matrix.py` | learn letter confusions → `confusion_matrix.json` | `CONFIG`: gt_dir, ocr_dir, out |
| `amharic_ocr_corrector.py` | the corrector itself — reads stdin, or set `text`/`file` | `CONFIG` above `main()` (kenlm/confusion/vocab default via `repo_paths`) |
| `confusion_matrix_coverage_audit.py` | same-family order-confusion coverage of the matrix | `CONFIG`: confusion_matrix, generate_missing |
| `trace_candidate.py` | forensic trace for one OCR token → expected word | `CONFIG`: ocr_token, expected_word, context_sentence |

Libraries: `normalizer.py`, `amharic_canonical_normalizer.py`, `suspicion.py`.

### The child — `4_ocr_correction/error_detection/`

| Script | Does | Edit |
|---|---|---|
| `run_ocr.py` | OCR `Evaluation/images/` → `Evaluation/OCRed/` | `CONFIG`: images, out |
| `Model/train_final.py` | train the shipped-recipe detector (writes `detector_lgbm_final.pkl`) | `CONFIG`: out, no_save, sweep, quick |
| `Model/train_LightGBM.py` | the full training implementation | `CONFIG`: lexicon, corpus, min_freq, gates, … |
| `scripts/evaluate_child.py` | score the detector on the 100-page corpus | `CONFIG`: max_pages, model, lexicon, gates |
| `scripts/evaluate_out_of_sample.py` | held-out measurement (`docs/detector_out_of_sample.md`) | `CONFIG`: corpus, model, dev_pages, limit |
| `scripts/verify_claims.py` | re-derive every published number | `CONFIG`: corpus, model, claims, fast |
| `scripts/tune_gates.py` | choose PASS/FLAG gates with page-level CV | `CONFIG`: grids, budgets, folds |
| `scripts/make_eval_corpus.py` | build a scorable corpus from an `Evaluation/`-style folder | `CONFIG`: eval_dir (default: root `Evaluation/`), out, report |
| `scripts/compare_lexicon_configs.py` | compare model/lexicon combinations | `CONFIG`: config, model_dir, max_pages |
| `scripts/probe_late_fusion.py` | late-fusion feature probe | `CONFIG`: model, lexicon, dump/load |
| `scripts/test_child.py` | run the child's test suite (optional `-v`) | — |
| `scripts/audit_decision.py` | decision-audit library used by stage 5 (import, don't run) | — |

### Stage 5 — `5_correction_evaluation/` (measure the correction)

| Script | Does | Edit |
|---|---|---|
| `01_evaluate_full_corrector.py` | end-to-end benchmark: CER/WER, clean-token damage, JSON report | `CONFIG`: gt/raw/output dirs, kenlm, all lambdas, limit_pages for a smoke run |
| `bash 5_correction_evaluation/smoke_eval.sh` | the same, on the first 10 pages (pass a number to change) | env vars at the top |
| `02_collect_error_cases.py` | three-way alignment → `correction_decisions.tsv` | `CONFIG`: gt/raw/corrected dirs, outputs |
| `03_audit_correction_decisions.py` | audit of the decisions above | top constants |
| `04_stage1_precision.py` | Stage-1 flag precision from verdict files | `CONFIG`: verdict_dir, target |
| `05_fit_stage1_scores.py` | which Stage-1 signals actually separate errors | `CONFIG`: verdict_dir, target, l2 |

Libraries: `stage1_fit.py`, `stage1_labels.py`.
`verdict_dir` input is produced by running `01_evaluate_full_corrector.py`
with `CONFIG["verdict_dir"]` set.

### `tools/`

| Script | Does |
|---|---|
| `tools/repo_paths.py` | print where every external data root resolves |
| `tools/stage_paths.py` | print which folders get registered on `sys.path` |
| `tools/analysis/build_synthetic_source.py` | pick de-overlapped source text (`CONFIG`: source, out, pages) |
| `tools/analysis/make_synthetic_set.py` | render it into page images + GT (`CONFIG`: text, font, out) |
| `tools/analysis/ocr_synthetic_api.py` | OCR via tesserocr where no `tesseract` binary exists (`CONFIG`: images, out, models) |
| `tools/analysis/check_dictionary_coverage.py` | dictionary coverage of GT words (`CONFIG`: dict, set) |
| `tools/analysis/*.py` (rest) | one-off probes behind `docs/lgbm_detector_review.md`; they already ran with constants only |

---

## Tests

```bash
pip install -r requirements.txt pytest     # kenlm is separate, see requirements.txt
python -m pytest tests -q                                   # pipeline suite
python -m pytest 4_ocr_correction/error_detection/tests -q  # child suite
python -m pytest tests/test_no_required_cli.py -q           # the no-arguments rule
```

KenLM-dependent tests skip automatically when the model/module is absent.
