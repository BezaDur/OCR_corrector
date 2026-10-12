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

### Step 2 — Run Tesseract OCR with confidence scores

Make sure Tesseract is installed with Amharic data:

```bash
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-amh

# macOS:
brew install tesseract tesseract-lang
```

Run the OCR script:

```bash
python OCR_error_detection/run_ocr.py
```

This produces two files per image in `Evaluation/OCRed/`:
- `page_NNNN.ocr.txt` — the OCR text
- `page_NNNN.conf.tsv` — per-word confidence (0-100) from Tesseract

To override paths or language:

```bash
python OCR_error_detection/run_ocr.py \
    --images /path/to/images \
    --output /path/to/output \
    --lang amh \
    --psm 6
```

### Step 3 — Run the evaluation

```bash
python OCR_error_detection/evaluate_detection.py \
    --gt-dir OCR_error_detection/Evaluation/GroundTruth \
    --ocr-dir OCR_error_detection/Evaluation/OCRed \
    --confusion confusion_matrix.json \
    --conf-dir OCR_error_detection/Evaluation/OCRed
```

The `--conf-dir` flag tells it to use Tesseract's confidence scores.
Without it, the detector runs on text alone (checks 1-4 only).
With it, Tesseract's own doubt (check 8) is also used.

Output looks like:

```
============================================================
OCR Error Detection Evaluation Results
============================================================
Pages processed:    100
Pages w/ confidence: 100
Total tokens:       26,592
Error tokens:       115 (0.43%)
Correct tokens:     26,477

--- Verdict breakdown ---
Verdict     Correct      Error      Total    Prec%  Recall%
pass         24,590          3     24,593      0.0      2.6
gray          1,887          6      1,893      0.3      5.2
flag              0        106        106    100.0     92.2

--- Summary ---
PASS precision (correct among passed): 100.0%
Flag+Gray recall (errors caught):      97.4%
Flag+Gray precision (errors among flagged): 5.6%
F1 (flag+gray):                        10.6

Throughput: 17,673 tokens/sec (1.50s total)
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