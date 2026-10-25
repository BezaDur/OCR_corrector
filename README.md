# OCR Corrector — Parent + Child Project

**Whole pipeline: Prepare data → Train OCR → Evaluate OCR → Correct OCR → Evaluate Correction**
**Child project: error_detection (detect errors PASS/GRAY/FLAG) used inside correction**

## Project Structure — Sensible Organization

**The folder tells you what a script is for; the number tells you when it runs
inside that folder.** Numbers are per-stage and start at 01 in each one, so
`3_training_evaluation/01_run_ocr_eval.py` is the first thing stage 3 does. They
are not a global pipeline counter — the old global numbering (01…15 scattered
across five folders, with two different files both named `09_`) recorded the
order things were written, not the order they run. Scripts that are not steps in
their stage's flow (audits, diagnostics, shared libraries) carry no number.

```
Repository root
├── 1_data_preparation/ — prepare data for Tesseract OCR training
│   ├── 01_discover_character_inventory.py … 06_compile_lstmf_and_split.py — the
│   │   training-data pipeline, in the order it runs
│   ├── 07_generate_eval_pages.py — production page-synthesis engine
│   ├── generate_eval_pages_simple.py, generate_eval_dataset_simulated.py —
│   │   two alternative generators (real Tesseract vs. confusion-matrix simulation)
│   ├── audit_*.py, diagnose_*.py, check_pango_names.py, clean_font_directories.py,
│   │   inspect_and_install_fonts.py — checks and setup, not pipeline steps
│   ├── data/ — amh_final_training_text.txt (4.4M tokens, 56MB),
│   │   mined_rare_sentences.txt, telegram_c4.txt
│   ├── fonts/ — the Ethiopic .ttf set
│   └── README.md
├── 2_training/ — train Tesseract LSTM
│   ├── 01_train_epochs.sh, 02_run_epoch2.sh, 03_package_best_and_fast.sh
│   ├── diagnose_job_failure.py
│   ├── models/best/ — amh.lstm, amh.traineddata
│   ├── models/packaged/ — amh_best.traineddata, amh_fast.traineddata (deployable)
│   └── README.md
├── 3_training_evaluation/ — evaluate OCR (CER/WER)
│   ├── 01_run_ocr_eval.py … 07_diagnose_ocr_engine.py
│   ├── ocr_metrics.py, tsv_gt.py, tsv_gt_stats.py, tsv_confidence.py — shared
│   │   libraries; stages 4 and 5 and the detector all import them
│   ├── query_raw_characters.py
│   └── README.md
├── 4_ocr_correction/ — detect errors, build the confusion matrix, then correct
│   ├── error_detection/ — the detector; see "Detector" below
│   ├── 01_build_confusion_matrix.py → confusion_matrix.json
│   ├── amharic_ocr_corrector.py — the main corrector
│   ├── normalizer.py, amharic_canonical_normalizer.py, suspicion.py
│   ├── trace_candidate.py, confusion_matrix_coverage_audit.py
│   ├── tests/
│   └── README.md (how parent imports child via adapter)
├── 5_correction_evaluation/ — evaluate the full corrector
│   ├── 01_evaluate_full_corrector.py … 05_fit_stage1_scores.py
│   ├── stage1_fit.py, stage1_labels.py — shared libraries
│   ├── benchmarks/, data/, smoke_eval.sh
│   └── README.md
│
├── Evaluation/ — the 100-page corpus, shared by stage 3 and the detector, so it
│   │   lives at the root rather than inside either consumer
│   ├── GroundTruth/ (100 .txt), OCRed/ (200 .txt + .conf.tsv)
│   ├── images/ (100 page PNGs, 570MB) — the source input run_ocr.py reads
│   └── analysis.tsv — 62k rows
│
├── tools/ — cross-stage utilities; not a pipeline stage
│   ├── stage_paths.py — puts every stage folder on sys.path so the stages can
│   │   import each other by bare module name (stage folders start with a digit,
│   │   so they can never be Python packages themselves)
│   ├── repo_paths.py — one place for the data roots every stage needs
│   └── analysis/ — 14 exploratory scripts behind docs/lgbm_detector_review.md
│
├── tests/ — the cross-stage test suite
├── conftest.py — calls stage_paths.register() so tests/ needs no path setup
├── docs/
│   ├── components_guide.md — plain explanation of KenLM, SymSpell, LightGBM, dict, model
│   ├── architecture.md — parent vs child, data flow, how parent imports child
│   ├── project_structure.md — directory layout, what goes where
│   └── plain_language_guide.md — top-to-bottom guide for non-technical readers
├── vendor/segmenter/ — vendored morphological suite
├── archive/ — superseded code, kept for reference (not on sys.path)
├── README.md, requirements.txt, .gitignore
```

### Detector (child project) — `4_ocr_correction/error_detection/`

Stage 4 is detect-then-correct, so the detector lives inside it beside the
corrector it feeds, imported through `adapter.py`.

```
4_ocr_correction/error_detection/
├── README.md — the child overview; held-out numbers and their limits
├── detector.py — main LightGBM, default 50k types freq>=10 no SymSpell (2.9s init, 0.56ms/word, no crash)
├── adapter.py — for parent to import child (reuses vocab, no double load)
├── normalizer.py — child normalizer (lexicon keying, ሐ→ሀ etc.); NOT the same
│   module as 4_ocr_correction/normalizer.py, so always import it qualified
├── features.py — the single feature-extraction path shared by training and serving
├── signals.py, confusion.py, scoring.py, alignment.py, utils.py, run_ocr.py
├── Model/
│   ├── detector_lgbm_conffree.pkl — FINAL 3.5MB, main_i7 freq>=500 54k types, 1267 pages, 14 feats conf-free
│   ├── main_i7_frq10.txt — 737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, default dict for all needs
│   ├── local_lexicon.txt — 750k types fallback
│   ├── confusion_matrix.json — single source 47K
│   ├── harmonized_corpus_jiwer.tsv — 1367 pages 946k words
│   └── train_LightGBM.py, train_final.py, train_sweep_report.json
├── scripts/ — evaluate_child.py, tune_gates.py, compare_lexicon_configs.py,
│   probe_late_fusion.py, audit_decision.py, test_child.py,
│   evaluate_out_of_sample.py, verify_claims.py, make_eval_corpus.py
└── tests/
```

## Parent vs Child — How they work together

**Parent pipeline:**
1. **1_data_preparation**: amh_final_training_text.txt (4.4M tokens) + fonts → training boxes, lstmf
2. **2_training**: train Tesseract LSTM → amh.traineddata
3. **3_training_evaluation**: run OCR on 100 eval pages, CER/WER via ocr_metrics.py, error composition
4. **4_ocr_correction**: build confusion_matrix.json from errors, main corrector amharic_ocr_corrector.py uses SymSpell + confusion + morphology + KenLM + **DetectionEngine (child)** for suspicion
5. **5_correction_evaluation**: evaluate corrector, does correction improve CER/WER?

**Child detection:**
- Input: OCR word "በሸታ" + doc_freq
- Normalizer (child): ሐ→ሀ etc.
- Dictionary: main_i7_frq10.txt 737k types (default, from bigger data, freq>=10, max 5.6M) → in_lexicon, log_freq (gain 985k biggest)
- Confusion Matrix: confusion_matrix.json → one-flip test "በሸታ"→"በሽታ"
- Surface: token_len, rare_glyph_ratio, has_digit, mixed_script
- SymSpell OFF (gain 0, no quality loss, tested OFF vs ON same recall 94.5%, for speed, was crashing PC)
- KenLM OFF (gain 0)
- LightGBM: 14 feats → P(error) → PASS <0.10, GRAY 0.10-0.65, FLAG ≥0.65 — **does major work 90%**

**How parent imports child:**

```python
# In 4_ocr_correction/amharic_ocr_corrector.py
from error_detection.adapter import CorrectorDetectionAdapter

class AmharicOCRCorrector:
    def __init__(self):
        # ... load vocab, symspell, confusion_map ...
        try:
            self.suspicion = CorrectorDetectionAdapter(self)  # reuses vocab, no double load
            print("Using DetectionEngine (LightGBM) authoritative")
        except:
            from suspicion import SuspicionScorer  # fallback
            self.suspicion = SuspicionScorer(...)

    def correct_word(self, word, context, conf, doc_freq):
        verdict = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=doc_freq)
        if verdict.label == "FLAG":
            # generate candidates and correct
            ...
```

**Child self-contained, parent calls it, no circular dependency. Files relevant to detection inside 4_ocr_correction/error_detection/.**

## Final Model — main_i7

**Default dictionary: Model/main_i7_frq10.txt — 737k types, 15MB, from bigger data (5M dict main_i7.txt 4.9M unique), freq 9 and lower removed (min freq 10), max freq 5.6M vs 60k local, cleaner.**

Tested freq filters on 1267 pages (1200 train / 67 val):

| Dictionary | Filter | Types | Val AUC | Recall | False Flag | Score |
|------------|--------|-------|---------|--------|------------|-------|
| local_lexicon | freq>=1 (full 750k) | 719k | 0.9593 | 92.1% | 0.10% | 0.919 |
| local_lexicon | top 150k (freq>=3) | 150k | 0.9485 | 94.2% | 0.04% | 0.941 |
| local_lexicon | **freq>=10 — prev best** | 52k | 0.9346 | **94.8%** | 0.07% | **0.946** |
| main_i7 | freq>=10 (all) | 735k | 0.9512 | 90.7% | 0.13% | 0.904 |
| main_i7 | freq>=100 | 163k | 0.9423 | 93.5% | 0.11% | 0.913 |
| **main_i7 — FINAL** | **freq>=500 — 54k types** | **54,057** | 0.9242 | 94.2% | **0.01%** | 0.942 |
| main_i7 | freq>=1000 | 32k | 0.9116 | 94.3% | 0.02% | 0.939 |

**These are selection numbers.** The 67 validation pages are the same pages
that chose the dictionary filter, the round count and the gates, so "94.2 %
recall at 0.01 % false flags" says what the search found, not what the model
does. Scoring the shipped pickle on the 100 pages held out of the fit:

| held-out measurement | value |
|---|---:|
| recall (flag + gray) | **92.39 %** |
| flag-only recall | 2.71 % |
| false flag on clean words | **0.102 %** |
| gray on clean words | 15.77 % |
| AUC / average precision | 0.9232 / 0.4117 |
| calibration error (ECE) | 0.0050 |

A 5-fold CV grouped by document over the 1,267 training pages agrees (91.3 % at
0.030 % calibrated). So expect **≈ 91–92 % recall at ≈ 0.1 % false flags** — about
10× the advertised false-flag rate. The model is also essentially an
out-of-vocabulary detector: 98.1 % recall on OOV errors but 43.3 % on real-word
errors, which are 10.5 % of all errors and 78 % of the misses.

Full tables, the document structure of the corpus (~55 documents, not 1,367
independent pages) and the limits: `docs/detector_out_of_sample.md`. To check
every number yourself: `docs/verify_claims.md`.

**Performance fix:** Default 50k types freq>=10 no SymSpell → 2.9s init, 0.56 ms/word, won't crash PC (before full 750k + SymSpell 10-15s 1-2GB crash)

## How to use

```python
# Child alone — uses main_i7 default, fast, no crash
from error_detection import DetectionEngine
det = DetectionEngine()  # auto-finds Model/main_i7_frq10.txt, 50k types, no SymSpell
result = det.verdict("በሸታ", doc_freq=2)
print(result.label, result.suspicion)  # FLAG 0.85

# Parent corrector (imports child via adapter)
from importlib.util import spec_from_file_location, module_from_spec
import os
_path = os.path.join("4_ocr_correction", "amharic_ocr_corrector.py")
_spec = spec_from_file_location("amharic_ocr_corrector", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
corrector = _mod.AmharicOCRCorrector()
print(corrector.correct_text("ሰላም በሸታ"))
```

## Docs

- `docs/current_state.md` — verified facts: defaults, counts, timings, what is measured and what is not
- `docs/detector_out_of_sample.md` — the detector's held-out numbers, calibration, lexicon split, limits
- `docs/verify_claims.md` — how to re-derive every number (and test on new data) before believing it
- `docs/components_guide.md` — plain explanation of each component (dictionary, normalizer, confusion, surface, SymSpell, KenLM, LightGBM, DetectionEngine, Adapter) with diagrams & many Amharic examples, why SymSpell OFF no quality loss (tested OFF vs ON same recall 94.5%)
- `docs/architecture.md` — parent vs child, data flow, how parent imports child
- `docs/project_structure.md` — directory layout, what goes where, migration steps
- `4_ocr_correction/error_detection/README.md` — child overview, final model
- `4_ocr_correction/README.md` — parent corrector, how it calls child
- `1_data_preparation/README.md`, `2_training/README.md`, etc.

## Tests

```bash
# Child tests
python -m pytest 4_ocr_correction/error_detection/tests/ -v
# Or old
python -m unittest discover -s 4_ocr_correction/error_detection/tests -p 'test_*.py'

# Parent tests (via wrappers at root for backward compat)
python -m unittest discover -s tests -p 'test_*.py'
```

Counts measured on 2026-10-25: root **158 passed, 52 skipped, 2,984 subtests**;
`4_ocr_correction/error_detection/tests/` **97 passed**. The roots of the suite are
`tests/` for the pipeline stages and the child package's own `tests/` for the
detector; both must pass before a change to the detector is merged.

## One-line

**Parent: 1_data_preparation (4.4M tokens) → 2_training (Tesseract) → 3_training_evaluation (CER/WER) → 4_ocr_correction (main corrector + confusion + child detection via adapter) → 5_correction_evaluation. Child: error_detection self-contained (Model/main_i7_frq10.txt 737k types default, detector_lgbm_conffree.pkl freq>=500 = 54,057 types, LightGBM does the work, SymSpell and KenLM gain 0, 0.48 ms/word). Held-out: recall (flag + gray) 92.4 % at 0.102 % false flags on 100 pages excluded from the fit — the old "94.2 % / 0.01 %" was selection on 67 pages. Files relevant to detection inside 4_ocr_correction/error_detection/, corpus at Evaluation/, tools in tools/.**
