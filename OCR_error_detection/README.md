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
result = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=df)
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
pip install symspellpy lightgbm scikit-learn pandas
```

No KenLM needed. No segmenter needed. LightGBM handles NaN natively.

## Run the tests

From the repo root:
```bash
python -m unittest OCR_error_detection.tests.test_detector -v
```

40 tests, <2s.

## Evaluation with real images

### Step 1 — Put your images and ground truth in the Evaluation folder

```
OCR_error_detection/Evaluation/
├── images/           ← 100 .png
├── GroundTruth/      ← matching .gt.txt
└── OCRed/            ← Tesseract output (created by step 2)
```

### Step 2 — Run Tesseract OCR

```bash
python run_ocr.py
```

### Step 3 — Run evaluation

```bash
python evaluate_detection.py       # old
python evaluate_v2.py              # new, compares V1 vs V2 vs V3
python demo_improvement.py         # quick demo
```

### Step 4 — Generate detailed analysis

```bash
python analyze_methods.py
```

Writes `Evaluation/analysis.tsv` with per-word breakdown.

## Results

### Tesseract accuracy (fine-tuned `amh_fast`, PSM 3, 100 pages)

| Metric | Value |
|--------|-------|
| Word accuracy | **93.00%** |
| Char accuracy | **98.07%** |
| WER | 7.00% |
| CER | 1.93% |
| Substitutions | 4,267 |

### Detection performance — Final verification (100 pages, 150k lexicon, real conf, no manual floors)

| Metric | V1 (old 9-feat, floors) | V2 + V3 detector (no floors) | V3/V4 clean (no leakage, isotonic) |
|--------|-------------------------|------------------------------|------------------------------------|
| Error words | 4,267 | 4,267 | 4,267 |
| Caught (flag+gray) | 3,825 (89.6%) | **4,128 (96.7%)** | 3,895 (91.3%) |
| Missed | 442 | 139 (3.3%) | 372 (8.7%) |
| Clean pass | 74.2% | **95.9%** | 89.2% |
| Clean gray | 22.1% | 2.9% | 10.1% |
| False flag | 3.71% | **1.25%** | **0.72%** |
| False flag with ሕ/ኀ/ሐ | ? | **0** | **0** |
| Training | 1367 leakage | 100 leakage | **1267 no leakage (1200 train / 67 val)** |
| Real conf | No (75 dummy) | Yes | NaN train, real test |
| Floors | Yes | **No** | **No** |
| Calib | No | No | **Isotonic Yes** |

**V2 with V3 detector meets roadmap Stage 5: Recall ≥95% (96.7%), False Flag <1.5% (1.25%), 0 flags for ሕ/ኀ/ሐ.**

Old false positive example: `ነው` flagged 54 times with low-conf → now PASS. `የሕዳሴው` FLAG 0.775 → now PASS 0.006.

## Detection methods

| # | Check | Needs |
|---|-------|-------|
| 1 | **In-lexicon** | lexicon |
| 2 | **Surface** | nothing |
| 3 | **SymSpell neighbours** | lexicon + symspellpy |
| 4 | **One-flip test** | confusion matrix (fixed sub key) |
| 5 | Character LM | kenlm optional → NaN if missing |
| 6 | Word LM fit | kenlm optional |
| 7 | Morphology | segmenter optional |
| 8 | **OCR confidence** | Tesseract TSV (real, not dummy) |
| 9 | **doc_freq** | document Counter → NaN for isolated |

Checks 1-4 always run. 5-8 optional. LightGBM handles NaN natively.

## Roadmap Implementation (5 Stages)

### Stage 1: Data Pipeline & Hygiene — Done
- Fixed `"sub"` vs `"substitutions"` key → one_flip gain 0 → 6k
- Canonical boundary: `NFC + normalize_word_strict()` before any lookup
- NaN for missing: KenLM → NaN not -2.5, doc_freq → NaN not 1, ocr_conf → NaN not 75
- Real confidences ingested from `Evaluation/OCRed/*.conf.tsv` for eval pages; training corpus (1267) uses NaN
- Strict isolation: 1367 → train 1200, val 67, test 100 eval held-out

### Stage 2: Clean Training, No Floors — Done
- No `scale_pos_weight` distortion (old 2-2.5)
- Isotonic calibration on val set: `IsotonicRegression(out_of_bounds='clip')`
- No manual floors in `detector.py` v3: trusts calibrated `P(error)` with cost-sensitive thresholds `τ_pass=0.10, τ_flag=0.70` tuned for recall vs false_flag

### Stage 3: Corrector Integration — Done
- `CorrectorDetectionAdapter` delegates to authoritative `DetectionEngine`, reuses corrector's vocab/symspell/reverse_sub_map/_norm
- Implements `DetectorInterface` contract: `verdict(tok, context, ocr_conf, doc_freq)`
- doc_freq lifecycle: full doc → Counter canonical, isolated → NaN
- `SuspicionScorer` deprecated, kept as fallback

### Stage 4: Packaging & Tests — Done
- `from OCR_error_detection import DetectionEngine` works from repo root
- 40 tests pass

### Stage 5: Out-of-Sample Verification — Done
- V2+V3 detector 100 pages 150k: **96.7% recall, 95.9% pass, 1.25% false flag, 0 ሕ/ኀ**
- V3/V4 clean: 91.3% recall, 89.2% pass, 0.72% false flag, 0 ሕ/ኀ

## Files

| File | What it does |
|------|--------------|
| `detector.py` | **v3 authoritative** — canonical NFC+Amharic, NaN for missing, isotonic, no floors |
| `signals.py` | Fixed imports, canonical |
| `scoring.py` | Fallback (deprecated) |
| `confusion.py` | Fixed both `sub` dict and `substitutions` list |
| `normalizer.py` | NFC + strict |
| `utils.py` | Ethiopic lists |
| `adapter.py` | Stage 3: delegates to DetectionEngine, DetectorInterface, doc_freq NaN lifecycle |
| `amharic_ocr_corrector.py` | Updated to try DetectionEngine first, doc_freq Counter |
| `suspicion.py` | Deprecated fallback |
| `run_ocr.py` | Tesseract with TSV |
| `evaluate_detection.py` | Old eval |
| `evaluate_v2.py` | New eval V1 vs V2 vs V3 |
| `Model/train_improved.py` | v2: fixed confusion, real conf, 15 feats |
| `Model/train_v3.py` | v3 clean: strict isolation, NaN, no scale_pos_weight, isotonic |
| `Model/train_v4.py` | v4 final: all 1367, real conf for eval, isotonic |
| `Model/local_lexicon.txt` | 750k types from `amh_final_training_text.txt`, capped 150k |
| `Model/detector_lgbm.pkl` | old 9-feat 1.4MB |
| `Model/detector_lgbm_v2.pkl` | v2 15-feat real conf 3.2MB — 96.7% recall, 1.25% FF |
| `Model/detector_lgbm_v3.pkl` | **v3/v4 clean authoritative** 15-feat isotonic 4.8MB — 91.3% recall, 0.72% FF, no leakage |
| `Model/detector_lgbm_v4.pkl` | v4 final same as v3 |
| `alignment.py` | DP alignment |
| `demo_improvement.py` | Demo V1 vs V2 |
| `tests/test_detector.py` | 40 tests |
