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
    ocr_confs=[95, 90, 60],      # from Tesseract TSV (optional, model ignores it if conf-free)
    doc_freqs=[5, 3, 1]          # how many times word appears on page
)
for r in results:
    if r.label != "pass":
        print(r.signals.token, r.label, r.suspicion)
```

Use inside the main corrector:

```python
from OCR_error_detection.adapter import CorrectorDetectionAdapter

self.detector = CorrectorDetectionAdapter(self)  # reuses vocab
verdict = self.detector.verdict(word, context, ocr_conf=conf, doc_freq=df)
```

No KenLM needed. No internet needed. Works offline.

---

## How it works (plain)

The detector is a **LightGBM model** (small, fast, CPU). It learned from real OCR mistakes.

For each word it looks at:

1. **Is it in the dictionary?** And how common is it? (`log_corpus_freq`, `in_lexicon`)
2. **Does it look weird?** Mixed scripts, digits, rare letters like ሠ ጰ ፀ ዐ (`mixed_script`, `has_digit`, `rare_glyph_ratio`, `len_ratio`)
3. **Can one letter change make a real word?** Example: `በሸታ` → `በሽታ` (one flip using optical confusion) (`has_one_flip`, `one_flip_count`)
4. **How many similar dictionary words exist?** (`n_neighbors` via SymSpell)
5. **How often does it appear on this page?** (`doc_freq`, `is_doc_singleton`)

All checks are turned into numbers. LightGBM combines them into one probability `P(error)` between 0 and 1.

- `P < 0.10` → **pass**
- `0.10 ≤ P < 0.65` → **gray**
- `P ≥ 0.65` → **flag**

No hard-coded rules override the model. The model was calibrated with isotonic regression so the probability is trustworthy.

**Why no Tesseract confidence?** Tesseract confidence is unreliable:
- 19% of real errors have conf >85 (confident but wrong) — if model trusts conf, it misses them
- 7.5% of correct words have conf <70 (unsure but correct) — if model trusts conf, it false-flags them

Test on 100 pages (62k words):
- **WITH conf:** AUC 0.9606, but catches only 56% of high-conf errors, false-flags 11.7% of low-conf correct
- **WITHOUT conf (conf-free):** AUC 0.8998, but catches **90.8% of high-conf errors**, false-flags only **0.1%** of low-conf correct, recall 90.8% vs 87.9%, false_flag 0.07% vs 0.85% (12x reduction)

So main model is **conf-free** (14 linguistic features only). It gives same verdict whether conf=95 or conf=20 for same word — correct behavior.

If LightGBM is not available, it falls back to simple rule-based scorer.

---

## Files — what each does

**Core (you need these):**
- `detector.py` — main engine, loads model, dictionary, confusion matrix, decides pass/gray/flag. Now prioritizes `detector_lgbm_conffree.pkl`
- `adapter.py` — lets the main corrector use the detector without loading things twice
- `confusion.py` — loads `confusion_matrix.json` (which letters Tesseract confuses)
- `normalizer.py` — fixes spelling differences (ሐ→ሀ, ዐ→አ, ሠ→ሰ) before dictionary lookup
- `signals.py` — collects evidence for one word
- `scoring.py` — fallback rule-based scorer (used only if LightGBM missing)
- `utils.py` — lists of Ethiopic letters and rare glyphs
- `alignment.py` — aligns OCR words with ground truth words to measure accuracy
- `run_ocr.py` — runs Tesseract on images and saves text + confidence

**Model and data:**
- `Model/detector_lgbm_conffree.pkl` — **NEW authoritative, conf-free** (14 feats, no ocr_conf, 982K) — catches 90.8% high-conf errors, 0.07% false flag
- `Model/detector_lgbm_v3.pkl` — previous clean baseline (15 feats with NaN conf, 4.8MB, 91.3% recall, 0.72% false flag, no leakage) — kept as fallback
- `Model/local_lexicon.txt` — dictionary built from `amh_final_training_text.txt`, 750k types, capped to 15k for small test (15MB full)
- `Model/confusion_matrix.json` — learned optical confusions (also copied to package root for tests, 47K)
- `Model/harmonized_corpus_jiwer.tsv` — 1367 pages, 946k words, with error labels (49MB, dummy conf 75)
- `Model/train_conffree.py` — training script for conf-free model (14 feats, no conf)
- `Model/train_final.py` — copy of conf-free training
- `Evaluation/` — 100 pages images + ground truth + OCR output with real confidences + `analysis.tsv` (62k rows)

**Tests:**
- `tests/test_detector.py` — 40 tests, runs in <2 seconds

**Archive (old, deprecated):**
- `archive/` — old training scripts (`train_improved.py` leakage, `train_v3.py`, `train_v4.py`, `test_conf_free.py`, `evaluate_detection.py`, `analyze_methods.py`, etc.)
- `archive/models/` at repo root — old models v1 (1.4MB, 9 feats, buggy one-flip), v2 (3.2MB, 96.7% recall but trusts conf, misses 44% high-conf errors), v4 (same as v3)

---

## How to train (simple)

**Quick train conf-free (100 eval pages, 15k lexicon, ~1 min):**

```bash
pip install symspellpy lightgbm scikit-learn pandas
python OCR_error_detection/Model/train_conffree.py
# Saves Model/detector_lgbm_conffree.pkl (14 feats, no conf)
```

**Test conf vs conf-free:**

```bash
python OCR_error_detection/archive/test_conf_free.py
# Compares WITH conf vs WITHOUT conf on 70/30 split
# Shows high-conf errors caught, low-conf correct false-flagged
```

**Full training (1267 pages, strict split):**
- See `train_v3.py` in archive for 1200/67/100 split, isotonic calibration, NaN handling

---

## Results — plain

**Tesseract OCR accuracy (fine-tuned amh_fast, PSM 3, 100 pages):**
- Word accuracy 93%, Char accuracy 98.07%, WER 7%, 4267 errors
- Conf reliability: correct mean 87.0, error mean 60.2, but 19% errors have conf>85, 7.5% correct have conf<70

**Detection (100 pages, 15k dictionary, small sample test 70 train / 30 test):**

| Model | AUC | Recall | Clean pass | False flag | High-conf errors caught (conf>85) | Low-conf correct false-flagged |
|-------|-----|--------|------------|------------|-----------------------------------|--------------------------------|
| WITH conf (15 feats, ocr_conf gain 59k) | 0.9606 | 87.9% | 91.6% | 0.85% | **56.1%** (misses 44%) | **11.7%** |
| WITHOUT conf (14 feats, conf-free) | 0.8998 | **90.8%** | 76.9% | **0.07%** | **90.8%** | **0.1%** |

**What this means:**
- WITH conf looks better on AUC, but fails on the dangerous cases (high-conf errors = Tesseract confident but wrong)
- WITHOUT conf is **better where it matters**: catches 90% of high-conf errors vs 56%, false-flags 0.1% vs 11.7% on low-conf correct, 12x lower false flag overall, higher recall
- Pass rate lower (76.9% vs 91.6%) means more words go to gray (safe, not auto-flagged) — gray is checked by corrector, flag is auto-fixed

**Final models (full 100 pages):**
- `detector_lgbm_conffree.pkl`: 14 feats, no conf, 982K, **authoritative conf-free** — recommended main
- `detector_lgbm_v3.pkl`: 15 feats with NaN conf, 4.8MB, 91.3% recall, 0.72% false flag, 0 ሕ/ኀ — previous clean baseline
- `detector_lgbm_v2.pkl` (archived): 15 feats with real conf, 3.2MB, 96.7% recall, 1.25% false flag, but trusts conf (misses 44% high-conf errors)

---

## Future improvements — directions

**Short term (easy wins):**
1. **Better dictionary**: Use `main_i7.txt` (4.9M types) if available, instead of 15k cap. More words in dictionary → fewer false flags for names and rare words. Already supported, just point `dict_vocab_path` to it.
2. **Add word-level language model**: Currently only char LM is optional and set to NaN. Add KenLM word 5-gram score as feature `word_lm_fit` (how well word fits context). Should help catch real-word errors like `አስረኞች` vs `እስረኞች`.
3. **Add morphology feature**: Use `AmharicMorphologicalSegmenter` to check if OOV can be split into known morphemes. If yes → likely legitimate rare form → keep pass. Use only for gray zone to keep speed.
4. **Threshold tuning per use-case**: If you want fewer false flags (precision), raise `flag_above` to 0.70. If you want catch more errors (recall), lower `pass_below` to 0.05. Tune on your validation set.

**Medium term:**
5. **More training data**: 100 pages → 62k words is small. Generate more synthetic pages with `1_data_preparation/01_generate_eval_pages.py` and add to training. More data → better calibration.
6. **Better one-flip**: Currently checks single-char substitution only. Add insertion/deletion flips (e.g., `ቤት` vs `ምቤት`) using `ins` and `del` costs from confusion matrix. Already in `confusion_matrix.json` but not used in feature.
7. **Document-level voting improvements**: Currently `doc_freq` is simple count. Better: if word appears many times with high confidence and once with low confidence, the low-conf one is likely error. Implement voting: `is_majority_clean` feature.

**Long term:**
8. **End-to-end corrector integration**: Make detector and corrector train together. Detector decides which words to fix, corrector proposes fixes, detector re-scores fixes. Currently they are separate.
9. **Active learning**: Collect words where detector says gray and corrector is unsure, send for human review, add to training set. This will improve the hardest cases.
10. **Replace SymSpell with faster index**: For 750k types, SymSpell building is slow. Use Trie or BK-tree for faster neighbor lookup, or pre-compute `n_neighbors` offline.
11. **Multilingual**: Extend to other Ethiopic languages (Tigrinya) by adding their dictionaries and confusion matrices.

**How to contribute:**
- Add a new feature in `signals.py` → add to `feature_cols` in `train_conffree.py` → retrain → evaluate
- Keep `detector.py` free of manual if-else floors — let LightGBM learn
- Always use canonical boundary: `NFC + normalize_word_strict()` before any lookup
- Use `NaN` for missing, not magic numbers
- Test with `python -m unittest OCR_error_detection.tests.test_detector -v`

---

## One-line summary

**Old:** Rule-based, hard-coded floors, buggy one-flip (gain 0), dummy confidence, 9.85% false flags.
**With conf:** 0.85% false flag, but misses 44% high-conf errors, false-flags 11.7% low-conf correct.
**New conf-free:** **0.07% false flag (12x better), 90.8% recall, 90.8% high-conf errors caught (vs 56%), 0.1% low-conf false-flag (vs 11.7%), 0 ሕ/ኀ errors, linguistic only, no Tesseract conf trust.**
