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

detector = DetectionEngine(dict_vocab_path="lexicon.txt")

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

From inside the folder:
```bash
cd OCR_error_detection
python tests/test_detector.py
```

40 tests. No external files needed. Runs in under 0.1 seconds.

## Evaluation with real images

### Step 1 — Put your images and ground truth in the Evaluation folder

```
OCR_error_detection/Evaluation/
├── images/           ← put 100 .png/.jpg images here
├── GroundTruth/      ← put matching .gt.txt files here (one per image)
└── OCRed/            ← Tesseract output goes here (created by step 2)
```

Ground truth files must be named to match the images:
- `page_0001.png` → `page_0001.gt.txt`
- Each `.gt.txt` file has the clean text, one word per line or space-separated.

### Step 2 — Run Tesseract OCR

```bash
python OCR_error_detection/run_ocr.py
```

That's it. No flags. Reads from `Evaluation/images/`, writes to `Evaluation/OCRed/`.

Requires Tesseract with Amharic:
```bash
sudo apt install tesseract-ocr tesseract-ocr-amh
```

### Step 3 — Run the evaluation

```bash
python OCR_error_detection/evaluate_detection.py
```

That's it. No flags. The paths are in the script.

Output looks like:

```
============================================================
OCR Error Detection Evaluation Results
============================================================
Pages processed:     100
Pages w/ confidence: 100
Aligned tokens:      62,278
  Error tokens:      10,132 (16.27%)
  Correct tokens:    52,146
Insertions:          1,857 (OCR words with no GT match)

Verdict     Correct      Error      Total    Prec%  Recall%
pass         51,998      1,423     53,421      2.7     14.0
gray            148      8,002      8,150     98.2     79.0
flag              0        707        707    100.0      7.0

PASS precision:  97.3%
Recall:          86.0%
Flag precision:  98.3%
F1:              91.7
```

## Detection methods

The detector checks each word with cheap checks first, expensive checks only if unsure:

| # | Check | What it does | Needs |
|---|-------|-------------|-------|
| 1 | **In-lexicon** | Is the word in the dictionary? | lexicon |
| 2 | **Surface** | Word length, mixed scripts, rare letters (ሠ, ጰ, ፀ, ዐ) | nothing |
| 3 | **SymSpell neighbours** | How many similar real words exist within 1 edit | lexicon + symspellpy |
| 4 | **One-flip test** | Can changing ONE letter (using known confusions) make a real word? | confusion matrix |
| 5 | Character LM | Does the word "sound like" real Amharic? | kenlm (optional) |
| 6 | Word LM fit | Does the word fit the words around it? | kenlm (optional) |
| 7 | Morphology | Can the segmenter break it into parts? | segmenter (optional) |
| 8 | **OCR confidence** | Did Tesseract think it was unclear? | Tesseract TSV |

Checks 1-4 always run. Checks 5-8 are optional — the detector works without them.

## Files

| File | What it does |
|------|-------------|
| `detector.py` | The main engine. Loads vocab, SymSpell, confusion matrix. |
| `signals.py` | Checks each word: in dictionary? Can one flip make it real? |
| `scoring.py` | Adds up the checks into one number (0 to 1). |
| `confusion.py` | Loads confusion_matrix.json. |
| `normalizer.py` | Fixes spelling differences (e.g. ሐ→ሀ). |
| `utils.py` | Ethiopic character lists. |
| `adapter.py` | Lets the main corrector use this without loading things twice. |
| `run_ocr.py` | Runs Tesseract on images with confidence output. |
| `evaluate_detection.py` | Runs the detector on test data and prints results. |
| `tests/test_detector.py` | 40 tests. |