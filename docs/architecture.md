# Architecture — Parent Project vs Child Project

**You asked: Organize whole project sensibly, not just data but scripts, and think about how parent imports child.**

## Overview

```
Parent Project: OCR Training + Correction
  ├── 1_data_preparation: prepare data for Tesseract OCR training
  ├── 2_training: train Tesseract LSTM
  ├── 3_training_evaluation: evaluate OCR (CER, WER)
  ├── 4_ocr_correction: build confusion matrix, main corrector (amharic_ocr_corrector.py)
  └── 5_correction_evaluation: evaluate full corrector

Child Project: OCR_error_detection (error detection)
  ├── Detects errors: PASS / GRAY / FLAG
  ├── Self-contained: detector.py, Model/, Evaluation/, tests/
  └── Used by parent via adapter: 4_ocr_correction calls OCR_error_detection
```

**Parent starts from data preparation, then training, then evaluation. Child is number of processes for detection, used inside parent's correction step.**

---

## Parent Project — Data Flow

```
[1_data_preparation]
  amh_final_training_text.txt (4.4M tokens, 56MB) + fonts
    ↓ generate boxes, lstmf, splits
  amh.training_text, amh.lstm-unicharset, etc.
    ↓
[2_training]
  train Tesseract LSTM: amh.traineddata (OCR engine)
    ↓
[3_training_evaluation]
  Run OCR on 100 eval pages (GroundTruth vs OCRed)
  Metrics: CER, WER via ocr_metrics.py, tsv_gt.py
  Output: confusion analysis, error composition
    ↓
[4_ocr_correction]
  03_build_confusion_matrix.py: learn which letters Tesseract confuses → confusion_matrix.json
  amharic_ocr_corrector.py: main corrector
    - Uses vendor/segmenter (morphology)
    - Uses SymSpell + confusion_matrix.json for candidate generation
    - Uses KenLM for context scoring
    - Uses DetectionEngine (child) for suspicion (PASS/GRAY/FLAG)
  Input: OCR text, Output: corrected text
    ↓
[5_correction_evaluation]
  Evaluate corrector: does correction improve CER/WER?
  Audit decisions: correction_decisions.tsv
```

---

## Child Project — OCR_error_detection — Data Flow

```
[OCR_error_detection]
  Input: OCR word "በሸታ" + doc_freq + optional conf
    ↓
  Normalizer (child): ሐ→ሀ, ሠ→ሰ etc. → canonical form
    ↓
  Dictionary (Model/main_i7_frq10.txt, 737k types, 15MB, from bigger data)
    → in_lexicon, log_corpus_freq (biggest feature, gain 985k)
    ↓
  Confusion Matrix (Model/confusion_matrix.json, 47K)
    → one-flip test: "በሸታ" → "በሽታ" hit?
    ↓
  Surface: token_len, rare_glyph_ratio, has_digit, mixed_script
    ↓
  SymSpell (OFF by default, gain 0) → n_neighbors
  KenLM (OFF, gain 0) → char_lm_score
    ↓
  LightGBM (Model/detector_lgbm_conffree.pkl, 3.5MB, FINAL)
    → P(error) → PASS (P<0.10), GRAY (0.10≤P<0.65), FLAG (P≥0.65)
    ↓
  Output: Verdict(label, suspicion, signals, reason)

  Model training:
    harmonized_corpus_jiwer.tsv (1367 pages, 946k words, 49MB)
    → 1200 train / 67 val, 14 feats conf-free, freq>=500 filter 54k types
    → Val recall 94.2% FF 0.01% BEST
```

**LightGBM does major work (90%), others are feature providers.**

---

## How Parent Imports Child

**Before (messy):**
- Root had confusion_matrix.json duplicate, normalizer.py duplicate, suspicion.py fallback, etc.
- Parent corrector had its own SuspicionScorer, not using child DetectionEngine
- No clear import path

**After (clean):**

```python
# In 4_ocr_correction/amharic_ocr_corrector.py (parent corrector)
# Primary: use child DetectionEngine via adapter (authoritative)
try:
    from OCR_error_detection.adapter import CorrectorDetectionAdapter
    self.suspicion = CorrectorDetectionAdapter(
        self,  # reuses parent's vocab, symspell, confusion map, normalizer
        lgbm_model_path="OCR_error_detection/Model/detector_lgbm_conffree.pkl",
        dict_vocab_path="OCR_error_detection/Model/main_i7_frq10.txt"
    )
    if self.suspicion.lgbm_model is not None:
        print("Using DetectionEngine (LightGBM) as authoritative")
except Exception as e:
    # Fallback: old SuspicionScorer (rule-based)
    from suspicion import SuspicionScorer
    self.suspicion = SuspicionScorer(...)
```

**Adapter reuses parent's vocab to avoid double memory:**
- `CorrectorDetectionAdapter.__init__(self, corrector)` takes parent corrector instance
- Reuses `corrector.vocab`, `corrector.symspell`, `corrector.confusion_map`, `corrector.normalizer`
- Implements same interface `verdict(tok, context, ocr_conf, doc_freq)`

**Why this is good:**
- Single source of truth: LightGBM authoritative, SuspicionScorer fallback
- No duplicate dictionary loading (saves 1-2GB RAM)
- Child self-contained, parent calls it, no circular dependency
- Default 50k types freq>=10 no SymSpell → 2.9s init, 0.56 ms/word, no PC crash

---

## Directory Organization — Current vs Proposed

### Current (messy)

```
. (root)
├── 1_data_preparation/
├── 2_training/
├── 3_training_evaluation/
├── 4_ocr_correction/ (only has 03_build_confusion_matrix.py, etc., but main corrector at root)
├── 5_correction_evaluation/
├── detection/ (old logistic regression, deprecated, confusing with OCR_error_detection)
├── OCR_error_detection/ (new LightGBM, child)
│   ├── Model/
│   ├── Evaluation/
│   └── ...
├── amharic_ocr_corrector.py (main corrector, should be in 4_ocr_correction/)
├── normalizer.py (parent normalizer, should be in 4_ocr_correction/, duplicate with child)
├── suspicion.py (fallback, should be in 4_ocr_correction/)
├── amharic_canonical_normalizer.py (eval util, should be in 3_ or 4_ or common)
├── ocr_metrics.py (eval metric, should be in 3_ or common)
├── tsv_gt.py, tsv_confidence.py (eval, should be in 3_)
├── confusion_matrix.json (duplicate, should be single in OCR_error_detection/Model/)
├── amh_final_training_text.txt (training text, should be in 1_data_preparation/data or data/)
├── tests/ (mixed: test_detection.py tests old detection/, test_ocr_corrector.py tests parent)
└── docs/
```

**Problems:**
- Root files mixed: some parent, some child, some old
- Duplicate files: confusion_matrix.json at root and in Model/, normalizer.py at root and in child
- detection/ vs OCR_error_detection/ confusing
- No clear import path parent → child
- Evaluation data in multiple places: eval_data/, evaluation_data/, OCR_error_detection/Evaluation/

### Proposed (clean)

```
.
├── README.md (overall project, links to docs)
├── docs/
│   ├── components_guide.md (plain explanation of KenLM, SymSpell, LightGBM, dict, model)
│   ├── architecture.md (this file, parent vs child)
│   └── project_structure.md (directory layout, what goes where)
├── 1_data_preparation/
│   ├── README.md
│   ├── 01_generate_eval_pages.py
│   ├── 07_generate_training_boxes.py
│   ├── 08_compile_lstmf_and_split.py
│   ├── data/
│   │   └── amh_final_training_text.txt (4.4M tokens, 56MB)
│   └── ...
├── 2_training/
│   ├── README.md
│   ├── 09_train_epochs.sh
│   └── ...
├── 3_training_evaluation/
│   ├── README.md
│   ├── 02_run_ocr_eval.py
│   ├── ocr_metrics.py (moved from root, canonical metrics)
│   ├── tsv_gt.py, tsv_confidence.py (moved from root)
│   └── ...
├── 4_ocr_correction/ (parent corrector)
│   ├── README.md (explains corrector, how it calls child)
│   ├── amharic_ocr_corrector.py (main corrector, moved from root)
│   ├── normalizer.py (parent normalizer, bigger, punctuation etc., moved from root)
│   ├── amharic_canonical_normalizer.py (moved from root)
│   ├── suspicion.py (fallback, moved from root)
│   ├── 03_build_confusion_matrix.py (builds confusion_matrix.json)
│   └── ...
├── 5_correction_evaluation/
│   ├── README.md
│   ├── 04_evaluate_full_corrector.py
│   └── ...
├── OCR_error_detection/ (child, self-contained)
│   ├── README.md (child overview)
│   ├── __init__.py (exports DetectionEngine, Adapter)
│   ├── detector.py (main, LightGBM, 50k default no SymSpell)
│   ├── adapter.py (for parent to import child)
│   ├── normalizer.py (child, lexicon keying, smaller)
│   ├── signals.py
│   ├── confusion.py
│   ├── scoring.py
│   ├── alignment.py
│   ├── utils.py
│   ├── run_ocr.py
│   ├── Model/
│   │   ├── detector_lgbm_conffree.pkl (FINAL, 3.5MB, main_i7 freq>=500 54k types, recall 94.2% FF 0.01%)
│   │   ├── main_i7_frq10.txt (737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, default dict)
│   │   ├── local_lexicon.txt (750k types, fallback)
│   │   ├── confusion_matrix.json (single source, 47K)
│   │   └── harmonized_corpus_jiwer.tsv (1367 pages, 946k words)
│   ├── Evaluation/
│   │   ├── analysis.tsv (62k rows)
│   │   ├── GroundTruth/ (100 pages)
│   │   └── OCRed/ (100 pages)
│   ├── tests/ (child tests)
│   ├── scripts/
│   │   ├── train_final.py (train final model)
│   │   └── audit_decision.py (moved from root ocr_decision_audit.py)
│   └── docs/ (child docs)
├── detection/ (old logistic regression, deprecated, move to archive/detection_old/)
├── archive/
│   ├── detection_old/
│   └── ...
├── vendor/
│   └── segmenter/ (morphology)
├── eval_data/ (deprecated, use OCR_error_detection/Evaluation/)
├── requirements.txt
└── .gitignore
```

**Benefits:**
- Files relevant to detection inside OCR_error_detection/ directory (child self-contained)
- Parent files inside 4_ocr_correction/ (corrector) and 3_training_evaluation/ (metrics)
- No duplicates: confusion_matrix.json single source in Model/, parent loads via path
- Clear import: parent → child via adapter, child doesn't import parent (no circular)
- Old detection/ archived, not confusing
- Data: amh_final_training_text.txt in 1_data_preparation/data/, not root
- Tests: child tests in OCR_error_detection/tests/, parent tests in tests/ or 5_correction_evaluation/

---

## Implementation Steps

1. **Move parent corrector files from root to 4_ocr_correction/**:
   - amharic_ocr_corrector.py, normalizer.py (parent), suspicion.py, amharic_canonical_normalizer.py
   - Update imports in amharic_ocr_corrector.py to use OCR_error_detection via adapter (already does)

2. **Move evaluation utils from root to 3_training_evaluation/**:
   - ocr_metrics.py, tsv_gt.py, tsv_confidence.py, tsv_gt_stats.py

3. **Move detection-relevant files from root to OCR_error_detection/scripts/**:
   - ocr_decision_audit.py → OCR_error_detection/scripts/audit_decision.py
   - stage1_fit.py, stage1_labels.py → OCR_error_detection/archive/ or detection/

4. **Single source for confusion_matrix.json**:
   - Keep in OCR_error_detection/Model/confusion_matrix.json (authoritative)
   - Root confusion_matrix.json becomes symlink or removed, parent loads via path

5. **Archive old detection/**:
   - Move detection/ to archive/detection_old/, keep README explaining deprecated

6. **Update docs**:
   - docs/project_structure.md explains layout
   - 4_ocr_correction/README.md explains corrector and how it calls child
   - OCR_error_detection/README.md already exists, update to reflect default dict main_i7

7. **Ensure tests pass**:
   - After moves, run tests/test_detection.py (old) and OCR_error_detection/tests/

8. **Import child to parent**:
   - Already done via CorrectorDetectionAdapter
   - Document in 4_ocr_correction/README.md

---

## How to import child to parent — Code Example

```python
# Parent corrector (4_ocr_correction/amharic_ocr_corrector.py)
import sys
sys.path.insert(0, "..")  # if needed to find OCR_error_detection at repo root

from OCR_error_detection import DetectionEngine
from OCR_error_detection.adapter import CorrectorDetectionAdapter

# Option 1: Use DetectionEngine directly (self-contained, loads its own dict)
det = DetectionEngine(
    dict_vocab_path="OCR_error_detection/Model/main_i7_frq10.txt",
    dict_min_freq=500,  # 54k types, best FF 0.01%
    max_lexicon_types=None,
    enable_symspell=False  # default, fast, no crash
)
verdict = det.verdict("በሸታ", doc_freq=1)
print(verdict.label, verdict.suspicion)  # FLAG 0.85

# Option 2: Use via adapter (reuses parent's vocab, no double load)
class AmharicOCRCorrector:
    def __init__(self):
        # ... load vocab, symspell, confusion_map, normalizer ...
        self.suspicion = CorrectorDetectionAdapter(self)
    
    def correct_word(self, word, context, conf, doc_freq):
        verdict = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=doc_freq)
        if verdict.label == "FLAG":
            # generate candidates and correct
            ...
```

**Parent calls child, child doesn't call parent — no circular dependency.**

---

## One-line Summary

- **Parent**: 1_data_preparation → 2_training → 3_training_evaluation → 4_ocr_correction (main corrector) → 5_correction_evaluation
- **Child**: OCR_error_detection (detection, self-contained, Model/, Evaluation/, tests/)
- **Parent imports child** via adapter (CorrectorDetectionAdapter reuses vocab, no double load)
- **Files relevant to detection inside OCR_error_detection/**, parent files inside 4_ocr_correction/ and 3_training_evaluation/
- **No duplicates**: confusion_matrix.json single source in Model/, normalizer separate (parent bigger, child smaller for lexicon keying)
- **Old detection/** archived to archive/detection_old/
