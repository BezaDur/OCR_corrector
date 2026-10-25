# 4_ocr_correction — Build confusion matrix and main corrector

**Parent project, step 4: OCR correction.**

## What it does
- Build confusion matrix (01_build_confusion_matrix.py) from OCR errors → confusion_matrix.json
- Main corrector: amharic_ocr_corrector.py (80K lines, production CPU-based)
  - EthiopicPunctuationManager, AmharicNormalizer, SymSpell + reverse optical mapping, MorphologicalSegmenter, KenLM 5-gram Viterbi
- Uses child project error_detection via adapter for suspicion (PASS/GRAY/FLAG)

## Files
- error_detection/ — the detector (child project); see its own README
- tests/ — test_corrector.py, test_page1_fixes.py
- amharic_ocr_corrector.py — main corrector
- normalizer.py — parent normalizer, bigger (punctuation, numerals, abbreviations).
  NOT the same module as error_detection/normalizer.py (311 lines vs 90); always
  import the child's one qualified
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

# this folder starts with a digit, so it can never be a Python package and
# `python -m 4_ocr_correction...` is a syntax error; run the tests by path
python -m pytest tests/ -v
```

## Output
- confusion_matrix.json (the copy the detector loads is error_detection/Model/confusion_matrix.json;
  note there are three distinct confusion files in this repo — see
  docs/readme_claims_audit.md before comparing numbers between them)
- Corrected text
- Input to 5_correction_evaluation
