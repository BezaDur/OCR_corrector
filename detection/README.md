# detection/

OCR error detection. Scores each word for the chance it is an OCR error and flags
the likely ones. Correction is a separate step; it takes these flags as input.

## Contents
- `detector.py`: `ErrorDetector`, a logistic model over per-word features. Saved as plain JSON.
- `train.py`: fits the model from labelled verdict files and reports leave-one-page-out
  AUC and precision/recall per threshold. Writes the model trained on all pages.

## Use
1. Produce labelled verdicts on evaluation pages:
   `5_correction_evaluation/04_evaluate_full_corrector.py ... --verdict-dir DIR`
2. Train and check:
   `python3 -m detection.train DIR --out detection/model.json --target 0.85`
3. Apply to new words:
   ```python
   from detection import ErrorDetector
   det = ErrorDetector.load("detection/model.json")
   flags = det.flag(feature_dicts)   # one dict per word, same keys as det.names
   ```

## Features
The keys come from the corrector's `TokenSignals.as_vector()`. The model stores the
names it was trained on. Extra keys are ignored; a missing key raises KeyError.

## Status
- Trained on the 10-page evaluation set only. Out-of-sample numbers come from
  leave-one-page-out, not from a held-out test set.
- Tesseract confidence is not in the feature set yet (it needs a TSV for every page).
