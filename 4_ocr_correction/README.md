# 4_ocr_correction — Build confusion matrix and main corrector

**Parent project, step 4: OCR correction.**

## What it does
- Build confusion matrix (01_build_confusion_matrix.py) from OCR errors → confusion_matrix.json
- Main corrector: amharic_ocr_corrector.py (80K lines, production CPU-based)
  - EthiopicPunctuationManager, AmharicNormalizer, SymSpell + reverse optical mapping, MorphologicalSegmenter, KenLM 5-gram Viterbi
- Uses child project error_detection via adapter for suspicion (PASS/GRAY/FLAG)

## Files (moved from root)
- amharic_ocr_corrector.py — main corrector (moved from root)
- normalizer.py — parent normalizer, bigger (punctuation, numerals, abbreviations)
- amharic_canonical_normalizer.py — canonicalize for metrics
- suspicion.py — fallback SuspicionScorer (old, deprecated, LightGBM authoritative now)
- ocr_metrics.py, tsv_gt.py etc. now in 3_training_evaluation, but also used here

## How parent imports child

```python
# In amharic_ocr_corrector.py
try:
    from error_detection.adapter import CorrectorDetectionAdapter
    self.suspicion = CorrectorDetectionAdapter(self)  # reuses vocab, no double load
    # LightGBM authoritative
except Exception as e:
    from suspicion import SuspicionScorer  # fallback
    self.suspicion = SuspicionScorer(...)
```

**Child self-contained, parent calls it, no circular dependency.**

## How to run
```bash
python 01_build_confusion_matrix.py --ocr-dir .../ocr_pred --out confusion_matrix.json
python -m 4_ocr_correction.test_corrector
```

## Output
- confusion_matrix.json (single source now in 4_ocr_correction/error_detection/Model/, no symlinks)
- Corrected text
- Input to 5_correction_evaluation
