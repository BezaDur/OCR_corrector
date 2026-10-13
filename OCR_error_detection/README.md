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
python run_ocr.py
```

That's it. No flags. Reads from `Evaluation/images/`, writes to `Evaluation/OCRed/`.

Requires Tesseract with Amharic and the model symlinked into system tessdata:
```bash
# install tesseract
sudo apt install tesseract-ocr tesseract-ocr-amh

# symlink custom model so tsv/hocr configs work
sudo ln -s /path/to/amh_fast.traineddata \
           /usr/share/tesseract-ocr/5/tessdata/amh_fast.traineddata
```

### Step 3 — Run the evaluation

```bash
python evaluate_detection.py
```

That's it. No flags. The paths are in the script.

### Step 4 — Generate detailed analysis

```bash
python analyze_methods.py
```

Writes `Evaluation/analysis.tsv` with per-word breakdown: OCR word, GT word, detector verdict, Tesseract confidence, detection reason, and more.

## Results

### Tesseract accuracy (fine-tuned `amh_fast` model, PSM 3, 100 pages)

| Metric | Value |
|--------|-------|
| Word accuracy | **93.00%** |
| Char accuracy | **98.07%** |
| WER | 7.00% |
| CER | 1.93% |
| Substitutions | 4,267 |
| Insertions | 94 |
| Deletions | 4 |

### Comparison: stock vs fine-tuned model

| Metric | Stock `amh` (PSM 6) | Fine-tuned `amh_fast` (PSM 3) |
|--------|---------------------|-------------------------------|
| Word accuracy | 82.42% | **93.00%** |
| Char accuracy | 94.01% | **98.07%** |
| Detection recall | 86.0% | **89.6%** |

### Detection performance — Improved v2 (150k lexicon, 100 pages, real confidences)

| Metric | V1 (old 9-feat) | V2 (new 15-feat) |
|--------|-----------------|------------------|
| Error words | 4,267 | 4,267 |
| Caught (flag+gray) | 3,825 (89.6%) | **4,119 (96.5%) → 3,906 (91.5%)* |
| Missed (passed) | 442 | 148 → 361 |
| Clean pass rate | 74.2% | **95.5%** (20-page 150k: 85.0%) |
| Clean gray | 22.1% | **3.1%** (13.0%) |
| False flag rate | 3.71% | **1.40%** (1.96% on 20-page) |
| Avg confidence (correct) | 87.0% | 87.0% |
| Avg confidence (errors) | 60.6% | 60.6% |

*V2 training OOF: 91.5% caught, 95.5% clean pass, 1.4% false flag.
On 20-page 150k eval: **99.2% recall**, 85.0% clean pass, 1.96% false flag.
V1 on same 20-page: 98.4% recall but only 56.1% clean pass, 9.85% false flag.

**Key improvement:** False positives reduced from **9.85% → 1.96%** (5x), clean pass **56% → 85%**, while recall stays ~99%.

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

## Improvements in v2 (2026)

### Bugs fixed
1. **Confusion matrix parsing bug** — `train_LightGBM.py` looked for `data["substitutions"]` but file uses `data["sub"]` dict. This made `has_one_flip` and `one_flip_count` always 0 (gain 0). Fixed in `confusion.py` and `train_improved.py` to handle both formats.
2. **Normalization bug** — `_extract_lgbm_features` used raw token `clean_w` for vocab lookup, but vocab keys are normalized. Now uses `normalize_word_strict` for lookup.
3. **Hardcoded paths** — Removed `/mnt/mintdata/...` dependencies. Now auto-discovers `Model/local_lexicon.txt`, `confusion_matrix.json`, `detector_lgbm_v2.pkl` with graceful fallbacks.
4. **Missing methods** — Added `detection_stats()`, `reset_counters()`, `signals()`, `verdict_batch()` required by tests and adapter.

### Features expanded (9 → 15)
Old: `log_corpus_freq, in_lexicon, doc_freq, is_doc_singleton, token_len, has_one_flip, one_flip_count, char_kenlm_score, rare_glyph_ratio`
New adds: `ocr_conf` (real Tesseract confidence, strongest after freq), `n_neighbors` (SymSpell), `len_ratio`, `mixed_script`, `has_digit`, `is_high_freq`

Feature gain on 100 pages:
- `log_corpus_freq` 162k (dominant)
- `ocr_conf` 99k (second, previously unused!)
- `token_len` 22k, `n_neighbors` 12k, `in_lexicon` 11k, `doc_freq` 8k, `has_one_flip` 6k (now non-zero!)

### Safety floors improved (frequency-aware)
Old: `conf<50% + OOV → flag`, `conf<60% + pass → gray` — flagged common words like `ነው` 54 times with conf 1-48% even though LGBM said PASS 0.003.
New:
- High-freq words (`freq>1000` or `log_freq>3`) never flagged on low-conf alone
- Severe low-conf flag only if `<35%` AND (OOV with one-flip OR very low conf OOV)
- OOV kept in gray, not flag, unless optical evidence
- High-freq borderline flags downgraded to gray if conf>70%
- Result: `ነው` false flags eliminated, `የሕዳሴው` now PASS not FLAG.

### Training data
Old: `harmonized_corpus_jiwer.tsv` with dummy conf 75.0 for all tokens.
New: `Evaluation/` with **real** `*.conf.tsv` from Tesseract TSV, 62k aligned tokens, 6.8% error rate. Caches one-flip and KenLM per unique word for speed.

### Model
- `detector_lgbm.pkl` (old, 9 feats, 1.4MB)
- `detector_lgbm_v2.pkl` (new, 15 feats, improved, auto-discovered first)

### Usage with local lexicon
```bash
# Build lexicon from repo training text (already done, 14MB, 750k types, capped to 150k for training)
python -c "from collections import Counter; ..."
# Train improved model
python OCR_error_detection/Model/train_improved.py
# Evaluate
python OCR_error_detection/evaluate_v2.py
```

## Files

| File | What it does |
|------|-------------|
| `detector.py` | Main engine v2 — local paths, frequency-aware floors, 15-feat support, backward compat. |
| `signals.py` | Checks each word: in dictionary? Can one flip make it real? (fixed imports) |
| `scoring.py` | Logistic combine (fallback). |
| `confusion.py` | Loads confusion_matrix.json — fixed to handle both `sub` dict and `substitutions` list. |
| `normalizer.py` | Fixes spelling differences (e.g. ሐ→ሀ). |
| `utils.py` | Ethiopic character lists. |
| `adapter.py` | Lets main corrector reuse vocab/symspell + optional LGBM v2. |
| `run_ocr.py` | Runs Tesseract on images, produces text + word confidences. |
| `evaluate_detection.py` | Old eval (GT lexicon). |
| `evaluate_v2.py` | New eval with real confidences, compares V1 vs V2. |
| `Model/train_improved.py` | Fixed training: correct confusion parsing, real conf, expanded feats, caching, GroupKFold. |
| `Model/local_lexicon.txt` | 750k types from `amh_final_training_text.txt`, capped to 150k for training. |
| `Model/detector_lgbm_v2.pkl` | Improved model (15 feats). |
| `alignment.py` | DP-based word alignment engine. |
| `analyze_methods.py` | Generates per-word analysis TSV. |
| `check_alignment.py` | Diagnostic: verifies OCR-GT alignment. |
| `tests/test_detector.py` | 40 tests — still pass in <2s. |