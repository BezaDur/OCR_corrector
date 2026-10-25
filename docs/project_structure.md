# Project Structure — Sensible Organization

> **Status (2026-10-25): this document is the plan that was executed; it is kept
> as the historical rationale.** For where files actually live today, use
> **[`docs/repo_layout.md`](repo_layout.md)**. The "Current Layout (before reorg)"
> tree below is the before-picture. The after-tree is what you now see, plus the
> 2026-10-25 reorganisation: per-stage script numbering, `tools/` for cross-stage
> helpers, and the detector at `4_ocr_correction/error_detection/`.
>
> Numbers in this file are stale where they differ from `docs/current_state.md`
> ("80K lines" for the corrector — it is 1,817 lines / 88 KB; "default 50k types
> freq>=10" — the default is freq ≥ 500 → 54,057 types; "recall 94.2 % FF 0.01 %"
> — that was selection on 67 pages, held-out is 92.4 % at 0.102 %).

**You asked: Organizing it in a sensible manner, not just data but also scripts and other stuff is good practice. Also, directory structures, I want files relevant to detection should be inside the directory.**

## Layout

The full annotated tree is in the root `README.md`. This file covers the rules
behind it and why the layout looks the way it does.

```
.
├── 1_data_preparation/    prepare data for Tesseract OCR training
├── 2_training/            train the Tesseract LSTM
├── 3_training_evaluation/ evaluate the OCR engine (CER/WER)
├── 4_ocr_correction/      detect errors, build the confusion matrix, correct
│   └── error_detection/   the detector (child project)
├── 5_correction_evaluation/  evaluate the corrections
├── Evaluation/            the shared 100-page corpus
├── tools/                 cross-stage utilities (not a pipeline stage)
│   ├── stage_paths.py     puts every stage folder on sys.path
│   ├── repo_paths.py      one place for the external data roots
│   └── analysis/          exploratory scripts behind docs/lgbm_detector_review.md
├── tests/                 the cross-stage test suite
├── docs/                  architecture, components, this file, the plain-language guide
├── vendor/segmenter/      vendored morphological suite
├── archive/               superseded code, kept for reference, not on sys.path
├── conftest.py            calls stage_paths.register() for pytest
└── README.md, requirements.txt, .gitignore
```

### Numbering: the folder says what it is for, the number says when it runs

Numbers are **per stage** and start at `01` in each one. They used to be a single
global counter spread across the five folders, which recorded the order scripts
were *written* rather than the order they *run*: `02_run_ocr_eval.py` sat in stage
3, `03_build_confusion_matrix.py` in stage 4, `04_evaluate_full_corrector.py` in
stage 5, and two different files were both named `09_`. Reading a filename told
you nothing about where it belonged.

Now `3_training_evaluation/01_run_ocr_eval.py` is unambiguously the first thing
stage 3 does. Scripts that are not steps in their stage's flow — audits,
diagnostics, shared libraries — carry **no** number, so a number always means
"pipeline step, in order".

### Why stage folders are on sys.path instead of being packages

`4_ocr_correction` starts with a digit, so it can never be a Python package:
`import 4_ocr_correction.x` is a syntax error. But the stages genuinely import
each other — stage 5 pulls `ocr_metrics` from stage 3 and
`amharic_canonical_normalizer` from stage 4, and the detector's scripts pull
`ocr_metrics` from stage 3. The only way those bare imports resolve is for each
stage folder itself to be on `sys.path`.

That used to be done with thirteen root-level shim modules, one per shared file,
each an importlib forward to the real module inside a stage. Every new shared
module needed another shim written by hand. `tools/stage_paths.py` replaces all
thirteen by registering the *folders*, so a new shared module needs no
root-level change at all. The root `conftest.py` calls it for pytest; scripts
call it from the bootstrap block documented in that module.

### Why Evaluation/ sits at the root

The 100-page corpus (GroundTruth, OCRed, and 570 MB of source page images) is
read by stage 3's `01_run_ocr_eval.py` and `02_benchmark_new_model.py` *and* by
the detector's `run_ocr.py` and `scripts/evaluate_child.py`. It belongs to
neither, so it lives above both. The images are the irreplaceable input — the
OCR output is derived from them, not the other way round.

## What Goes Where — Rules

1. **Files relevant to detection inside 4_ocr_correction/error_detection/**:
   - detector.py, adapter.py, normalizer.py (child), signals.py, confusion.py, scoring.py, alignment.py, utils.py, run_ocr.py
   - Model/* (pkl, txt dicts, json confusion, tsv corpus)
   - tests/* (child tests)
   - scripts/* (evaluate_child.py, tune_gates.py, compare_lexicon_configs.py,
     probe_late_fusion.py, audit_decision.py, test_child.py)
   - Model/train_LightGBM.py, Model/train_final.py (training entry points)

2. **Parent corrector files inside 4_ocr_correction/**:
   - amharic_ocr_corrector.py (main corrector)
   - normalizer.py (parent, bigger)
   - suspicion.py (fallback)
   - amharic_canonical_normalizer.py
   - 01_build_confusion_matrix.py

3. **Evaluation metrics inside 3_training_evaluation/**:
   - ocr_metrics.py, tsv_gt.py, tsv_confidence.py, tsv_gt_stats.py

4. **Data preparation files inside 1_data_preparation/**:
   - amh_final_training_text.txt moved to 1_data_preparation/data/

5. **Old detection/ archived**:
   - detection/ → archive/detection_old/, README says deprecated, use error_detection

6. **Shared corpus at the root, not inside a consumer**:
   - Evaluation/ (GroundTruth, OCRed, images, analysis.tsv) at the repo root,
     because stage 3 and the detector both read it

7. **Two different `normalizer.py` modules — always import them qualified**:
   - `4_ocr_correction/normalizer.py` (311 lines, the parent's)
   - `4_ocr_correction/error_detection/normalizer.py` (90 lines, the detector's)
   - A bare `from normalizer import AmharicNormalizer` resolves by sys.path
     order, so it silently picks one. Write
     `from error_detection.normalizer import AmharicNormalizer` for the child's.

8. **Three distinct confusion files, not one** (this corrects the earlier
   "single source, no duplicates" claim; see docs/readme_claims_audit.md):
   - `4_ocr_correction/error_detection/Model/confusion_matrix.json` — 47,847 B,
     304 reverse substitutions, the one the detector loads
   - `2_training/config/confusion_matrix.json` — 65,839 B
   - `4_ocr_correction/confusion_matrix_patch.json` — 33,843 B
   - Three different checksums and three different contents. Know which one a
     number came from before comparing results.

9. **Cross-stage utilities live in tools/, not at the root**:
   - `tools/repo_paths.py` (external data roots), `tools/stage_paths.py`
     (sys.path registrar), `tools/analysis/` (exploratory scripts)
   - Nothing importable sits at the repo root except `conftest.py`

10. **Parent imports child via adapter**:
   - 4_ocr_correction/amharic_ocr_corrector.py does `from error_detection.adapter import CorrectorDetectionAdapter`
   - Adapter reuses parent's vocab, no double load
   - Child doesn't import parent (no circular)

## How to Import Child to Parent — Code

```python
# In 4_ocr_correction/amharic_ocr_corrector.py
from error_detection import DetectionEngine
from error_detection.adapter import CorrectorDetectionAdapter

# Option 1: Direct use (self-contained)
det = DetectionEngine(
    dict_vocab_path="4_ocr_correction/error_detection/Model/main_i7_frq10.txt",
    dict_min_freq=500,  # 54k types, best FF 0.01%
    max_lexicon_types=None,
    enable_symspell=False
)
verdict = det.verdict("በሸታ", doc_freq=1)

# Option 2: Via adapter (reuses parent's vocab)
class AmharicOCRCorrector:
    def __init__(self):
        # load vocab, symspell, confusion_map, normalizer
        self.suspicion = CorrectorDetectionAdapter(self)  # reuses vocab

    def correct(self, word, context, conf, doc_freq):
        verdict = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=doc_freq)
        if verdict.label == "FLAG":
            # correct
            ...
```

## Benefits

- **Sensible**: Parent pipeline 1→2→3→4→5 clear, child self-contained
- **No crash**: default lexicon 54,057 types, no SymSpell (3.7 s init, 0.48 ms/word measured 2026-10-24)
- **Single source**: Dictionary main_i7_frq10.txt default for all needs, confusion_matrix.json single source
- **Good practice**: Scripts call dictionary, don't train it; child doesn't train dict, parent doesn't duplicate
- **Easy import**: Parent calls child via adapter, no circular, no double memory
- **Clean tests**: Child tests in 4_ocr_correction/error_detection/tests/, parent tests in tests/ or 5_correction_evaluation/

## Migration Steps (implemented)

The first reorganization moved the loose root modules into the stage folders and
left thirteen shims behind at the root so the old import paths kept working.

The second one (this layout) removed that indirection:

1. `OCR_error_detection/` → `4_ocr_correction/error_detection/` — the detector is
   stage 4's first half, so it sits beside the corrector it feeds
2. `OCR_error_detection/Evaluation/` → `Evaluation/` at the root — shared corpus
3. Deleted the thirteen root shims; added `tools/stage_paths.py` and
   `conftest.py` to replace them
4. `repo_paths.py` → `tools/repo_paths.py`
5. Renumbered every stage's scripts from `01` within that stage
6. Root data into the stages that own it: `fonts/` → `1_data_preparation/fonts/`,
   `telegram_c4.txt` → `1_data_preparation/data/`, `models/` →
   `2_training/models/packaged/`
7. `docs/lgbm_review/*.py` → `tools/analysis/` — they are scripts, not docs
8. Stage 4's tests into `4_ocr_correction/tests/`
9. Deleted dead tracked artifacts: `font_diag_temp/`, `augraphy_cache/` (both
   already in `.gitignore` yet still tracked, referenced by no code) and three
   committed `.pyc` files
10. Fixed `_vendored_segmenter_path()`, which resolved `vendor/segmenter`
    relative to `4_ocr_correction/` instead of the repo root, so the vendored
    morphology suite was never found and the corrector ran with `lambda_morph`
    silently cancelled out
11. Added the missing `pydantic` dependency to `requirements.txt`
12. Full suite green after every step
