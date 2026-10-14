# OCR Corrector — Parent + Child Project

**Whole pipeline: Prepare data → Train OCR → Evaluate OCR → Correct OCR → Evaluate Correction**
**Child project: OCR_error_detection (detect errors PASS/GRAY/FLAG) used inside correction**

## Project Structure — Sensible Organization

```
Parent Project (OCR Training + Correction)
├── 1_data_preparation/ — prepare data for Tesseract OCR training
│   ├── data/amh_final_training_text.txt (4.4M tokens, 56MB, moved from root)
│   ├── 01_generate_eval_pages.py, 07_generate_training_boxes.py, etc.
│   └── README.md
├── 2_training/ — train Tesseract LSTM
│   ├── 09_train_epochs.sh, 10_package_best_and_fast.sh
│   └── README.md
├── 3_training_evaluation/ — evaluate OCR (CER/WER)
│   ├── ocr_metrics.py, tsv_gt.py, tsv_confidence.py (moved from root)
│   ├── 02_run_ocr_eval.py, 09_analyze_error_composition.py, etc.
│   └── README.md
├── 4_ocr_correction/ — build confusion matrix + main corrector
│   ├── amharic_ocr_corrector.py (main corrector, 80K, moved from root)
│   ├── normalizer.py (parent normalizer, bigger, moved from root)
│   ├── amharic_canonical_normalizer.py, suspicion.py (fallback, moved from root)
│   ├── 03_build_confusion_matrix.py → confusion_matrix.json
│   └── README.md (how parent imports child via adapter)
└── 5_correction_evaluation/ — evaluate full corrector
    ├── stage1_fit.py, stage1_labels.py (moved from root)
    ├── 04_evaluate_full_corrector.py, etc.
    └── README.md

Child Project (Error Detection, self-contained)
└── OCR_error_detection/
    ├── README.md — final model main_i7 freq>=500 54k types recall 94.2% FF 0.01%
    ├── detector.py — main LightGBM, default 50k types freq>=10 no SymSpell (2.9s init, 0.56ms/word, no crash)
    ├── adapter.py — for parent to import child (reuses vocab, no double load)
    ├── normalizer.py — child normalizer (lexicon keying, ሐ→ሀ etc.)
    ├── signals.py, confusion.py, scoring.py, alignment.py, utils.py, run_ocr.py
    ├── Model/
    │   ├── detector_lgbm_conffree.pkl — FINAL 3.5MB, main_i7 freq>=500 54k types, 1267 pages, 14 feats conf-free
    │   ├── main_i7_frq10.txt — 737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, default dict for all needs
    │   ├── local_lexicon.txt — 750k types fallback
    │   ├── confusion_matrix.json — single source 47K
    │   └── harmonized_corpus_jiwer.tsv — 1367 pages 946k words
    ├── Evaluation/ — 100 pages GroundTruth + OCRed + analysis.tsv 62k rows
    ├── tests/
    ├── scripts/audit_decision.py (moved from root ocr_decision_audit.py)
    └── docs/

Old (archived)
├── archive/detection_old/ — old logistic regression detection/ (deprecated, use OCR_error_detection)
└── detection -> archive/detection_old (symlink for backward compat)

Docs
└── docs/
    ├── components_guide.md — plain explanation of KenLM, SymSpell, LightGBM, dict, model with diagrams & examples
    ├── architecture.md — parent vs child, data flow, how parent imports child
    └── project_structure.md — directory layout, what goes where

Vendor
└── vendor/segmenter/ — morphological suite
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
from OCR_error_detection.adapter import CorrectorDetectionAdapter

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

**Child self-contained, parent calls it, no circular dependency. Files relevant to detection inside OCR_error_detection/.**

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

**FINAL authoritative: main_i7 freq>=500 — 54,057 clean types, recall 94.2%, false_flag 0.01% (7x better than local 0.07%, 85x better than WITH conf 0.85%), score 0.942, model 3.5MB, 1267 pages, 14 feats conf-free, pass 0.10 flag 0.65**

**Performance fix:** Default 50k types freq>=10 no SymSpell → 2.9s init, 0.56 ms/word, won't crash PC (before full 750k + SymSpell 10-15s 1-2GB crash)

## How to use

```python
# Child alone — uses main_i7 default, fast, no crash
from OCR_error_detection import DetectionEngine
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

- `docs/components_guide.md` — plain explanation of each component (dictionary, normalizer, confusion, surface, SymSpell, KenLM, LightGBM, DetectionEngine, Adapter) with diagrams & many Amharic examples, why SymSpell OFF no quality loss (tested OFF vs ON same recall 94.5%)
- `docs/architecture.md` — parent vs child, data flow, how parent imports child
- `docs/project_structure.md` — directory layout, what goes where, migration steps
- `OCR_error_detection/README.md` — child overview, final model
- `4_ocr_correction/README.md` — parent corrector, how it calls child
- `1_data_preparation/README.md`, `2_training/README.md`, etc.

## Tests

```bash
# Child tests
python -m pytest OCR_error_detection/tests/ -v
# Or old
python -m unittest discover -s OCR_error_detection/tests -p 'test_*.py'

# Parent tests (via wrappers at root for backward compat)
python -m unittest discover -s tests -p 'test_*.py'
```

Tests: 40 OK for child, detection OFF vs ON same quality

## One-line

**Parent: 1_data_preparation (4.4M tokens) → 2_training (Tesseract) → 3_training_evaluation (CER/WER) → 4_ocr_correction (main corrector + confusion + child detection via adapter) → 5_correction_evaluation. Child: OCR_error_detection self-contained (Model/main_i7_frq10.txt 737k types default, detector_lgbm_conffree.pkl FINAL 54k types freq>=500 recall 94.2% FF 0.01% BEST, LightGBM does major work 90%, SymSpell OFF gain 0 no quality loss, default 50k types no SymSpell 2.9s init no crash). Files relevant to detection inside OCR_error_detection/, parent files inside 4_ocr_correction/ and 3_training_evaluation/, no duplicates, single source confusion_matrix.json in Model/, parent imports child via adapter, old detection/ archived.**
