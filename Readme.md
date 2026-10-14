# OCR Error Detection — Plain Guide

**One job:** Look at each OCR word and decide: keep it, or fix it.

- **pass** = word is fine, do nothing
- **gray** = not sure, fix only if very confident
- **flag** = word is wrong, fix it

---

## How to use (3 lines)

```python
from OCR_error_detection import DetectionEngine

detector = DetectionEngine()  # auto-finds model and dictionary
result = detector.verdict("በሸታ", ocr_conf=85, doc_freq=2)

print(result.label)      # pass, gray, or flag
print(result.suspicion)  # 0.0 to 1.0, higher = more likely error
```

Check many words at once:

```python
results = detector.verdict_batch(
    ["ሰላም", "ቤት", "በሸታ"],
    ocr_confs=[95, 90, 60],      # optional, conf-free model ignores it
    doc_freqs=[5, 3, 1]
)
```

---

## How it works (plain)

LightGBM model (small, fast, CPU). For each word it looks at:

1. **Is it in dictionary?** How common? (`log_corpus_freq`, `in_lexicon`)
2. **Does it look weird?** Mixed scripts, digits, rare letters ሠ ጰ ፀ ዐ (`mixed_script`, `has_digit`, `rare_glyph_ratio`)
3. **Can one letter change make real word?** `በሸታ` → `በሽታ` (`has_one_flip`, `one_flip_count`)
4. **How often on this page?** (`doc_freq`, `is_doc_singleton`)

No Tesseract confidence — unreliable (19% errors have conf>85, 7.5% correct have conf<70). Conf-free gives same verdict for conf=95 vs 20.

- `P < 0.10` → **pass**
- `0.10 ≤ P < 0.70` → **gray**
- `P ≥ 0.70` → **flag**

---

## Dictionary — how we filter

**Full files:**
- `amh_final_training_text.txt` — 4.4M tokens (359k lines, 56MB) — raw training text, ~5M words
- `Model/local_lexicon.txt` — 750,870 unique types (15MB) — built from 4.4M tokens, after NFC + Amharic normalization (ሐ→ሀ, ዐ→አ, ሠ→ሰ), **no stemming**, keep surface forms

**For training we filter by frequency (cleaner than top-N):**

| Filter | Types kept | Why |
|--------|------------|-----|
| freq>=1 (full 750k) | 719k | includes typos, noise |
| top 150k (freq>=3) | 150k | includes freq 3-4 noise |
| **freq>=10 — FINAL** | **52,494** | **clean, words appear at least 10 times → real, not typos** |
| freq>=100 | 6,763 | too small, many legit words become OOV |

Tested on 1267 pages (883k rows, 1200 train / 67 val):

| Filter | Val AUC | Recall | Pass | False Flag | Score |
|--------|---------|--------|------|------------|-------|
| freq>=1 (750k) | 0.9593 | 92.1% | 92.0% | 0.10% | 0.919 |
| top 150k (freq>=3) | 0.9485 | 94.2% | 89% | 0.04% | 0.941 |
| **freq>=10 (52k) — FINAL** | 0.9346 | **94.8%** | 82.9% | 0.07% | **0.946** |
| freq>=100 (6.7k) | 0.8634 | 90.4% | 69.9% | 0.04% | 0.903 |

**Best is freq>=10** — highest recall 94.8%, best score 0.946, clean (filters 700k noisy rare types). Pass lower (82.9% vs 89%) means more gray (safe, checked by corrector) not flag.

For your 5M dictionary (main_i7.txt, 4.9M unique, no duplicates, already clean), do **freq>=100** on it — would keep ~50-100k clean types from 5M, filtering web noise. GitHub limit 100MB, so use Gdrive if you have it.

Scripts **call** dictionary (load with freq filter), they **don't train** dictionary itself — dictionary static, model uses `in_lexicon` and `log_freq` features.

---

## Files

**Core:**
- `detector.py` — main engine, loads `detector_lgbm_conffree.pkl` (freq>=10, 52k types, 2.1MB)
- `adapter.py` — corrector adapter, reuses vocab
- `confusion.py` — confusion matrix
- `normalizer.py` — ሐ→ሀ etc.
- `signals.py`, `scoring.py`, `utils.py`, `alignment.py`, `run_ocr.py`

**Model and data:**
- `Model/detector_lgbm_conffree.pkl` — **FINAL authoritative, conf-free, freq>=10** (52k types, 1267 pages, 14 feats, 2.1MB, val recall 94.8% FF 0.07% score 0.946, pass 0.10 flag 0.70)
- `Model/local_lexicon.txt` — 750k types (15MB), built from 4.4M tokens, no stemming, only normalization
- `Model/confusion_matrix.json` — 47K (also at package root for tests)
- `Model/harmonized_corpus_jiwer.tsv` — 1367 pages, 946k words (49MB, dummy conf ignored)
- `Model/train_final.py` — final training note
- `Evaluation/` — 100 pages images + GT + OCR + `analysis.tsv` (62k rows)

**Tests:** `tests/test_detector.py` — 40 tests <2s

**Archive:** old training scripts, old models v1/v2/v3/v4, 750k test model

---

## Results

**Tesseract (amh_fast, PSM 3, 100 pages):** Word acc 93%, Char acc 98.07%, WER 7%, 4267 errors, conf: correct mean 87.0, error mean 60.2, but 19% errors conf>85, 7.5% correct conf<70

**Detection (conf-free, 1267 pages train, 150k vs freq filters):**

- WITH conf (old): AUC 0.9606, recall 87.9%, FF 0.85%, high-conf errors caught 56% (misses 44%), low-conf correct FF 11.7%
- WITHOUT conf 150k: recall 94.2% FF 0.04% score 0.941
- **WITHOUT conf freq>=10 (52k) FINAL:** recall **94.8%** FF 0.07% score **0.946** (best), high-conf caught 100% in sample, low-conf FF 0%

**One-line:** Old 9.85% FF, WITH conf 0.85% FF but misses 44% high-conf errors, **FINAL conf-free freq>=10: 0.07% FF, 94.8% recall, 100% high-conf caught, 0% low-conf FF, 0 ሕ/ኀ errors, 2.1MB, linguistic only.**

---

## Future

1. Use full 750k lexicon with freq>=10 filter already best — for 5M dict, use freq>=100
2. Add word LM (KenLM word 5-gram) for real-word errors
3. Add morphology for gray zone
4. More diverse training pages (not just more quantity)
5. End-to-end detector+corrector training
