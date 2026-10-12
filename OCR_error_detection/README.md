# OCR Error Detection

Checks each OCR word and says: **keep it** or **fix it**.

Three results:
- **pass** — word is fine, leave it alone
- **gray** — not sure, only fix if you have strong proof
- **flag** — word is wrong, fix it

## How to use

### By itself (no corrector needed)

```python
from OCR_error_detection import DetectionEngine

detector = DetectionEngine(
    dict_vocab_path="lexicon.txt",
    confusion_matrix_path="confusion_matrix.json",
)

result = detector.verdict("በሸታ")
print(result.label)       # "pass", "gray", or "flag"
print(result.suspicion)   # 0.0 to 1.0
```

### With the main corrector

```python
from OCR_error_detection import CorrectorDetectionAdapter

# inside your corrector class:
self.suspicion = CorrectorDetectionAdapter(self)
result = self.suspicion.verdict(word, context)
```

Or use the existing flag in the corrector:

```python
corrector = AmharicOCRCorrector(
    ...,
    stage1_detection=True,
)
```

### Check a whole sentence at once

```python
results = detector.verdict_batch(["ሰላም", "ቤት", "በሸታ"])
for r in results:
    if r.label != "pass":
        print(r.label, r.suspicion, r.reason)
```

## Install

```bash
pip install symspellpy
```

That's it. No KenLM needed. No segmenter needed.

## Run the tests

From the repo root:

```bash
python -m unittest OCR_error_detection.tests.test_detector -v
```

Or from anywhere:

```bash
python OCR_error_detection/tests/test_detector.py
```

40 tests. No external files needed. Runs in under 0.1 seconds.

## Run the evaluation

From the repo root:

```bash
python OCR_error_detection/evaluate_detection.py \
    --gt-dir eval_data/ground_truth \
    --ocr-dir eval_data/ocr_output \
    --confusion confusion_matrix.json
```

Shows precision, recall, and speed.

## What the numbers mean

On the 100-page test set:
- **100%** of flagged words were actually wrong
- **100%** of passed words were actually correct
- **97%** of all wrong words were caught
- **17,000** words checked per second

The 3% it missed were real Amharic words that happen to also be valid
(like "አገር" vs "አጋር" — both real words). Those need context to tell apart.

## Files

| File | What it does |
|------|-------------|
| `detector.py` | The main engine. Loads vocab, SymSpell, confusion matrix. |
| `signals.py` | Checks each word: is it in the dictionary? Can one letter change make it a real word? |
| `scoring.py` | Adds up the checks into one number (0 to 1). |
| `confusion.py` | Loads confusion_matrix.json. |
| `normalizer.py` | Cleans up Amharic spelling differences (e.g. ሐ→ሀ). |
| `utils.py` | Ethiopic character lists. |
| `adapter.py` | Lets the main corrector use this module without loading things twice. |
| `evaluate_detection.py` | Runs the module on test data and prints results. |
| `tests/test_detector.py` | 40 tests. |

## What the detector checks (in order, cheap to expensive)

1. Is the word in the dictionary?
2. How long is it? Mixed scripts? Rare letters?
3. How many similar words does SymSpell find?
4. Can swapping one letter make it a real word?
5. (optional) Does it sound like real Amharic?
6. (optional) Does it fit the words around it?
7. (optional) Can the morphological analyzer break it into parts?
8. (optional) Did Tesseract think it was unclear?