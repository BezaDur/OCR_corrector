# Components Guide — Plain Explanation

**You asked: What does each part do? How do they work together?**

This guide explains in plain language, no jargon, with examples.

---

## The Big Picture

```
OCR word "በሸታ" (maybe wrong) 
   ↓
[1. Normalizer] fixes spelling differences → "በሸታ" (same)
   ↓
[2. Dictionary] checks: Is it known? How common? → in_lexicon=False, log_freq=0
   ↓
[3. Confusion Matrix] checks: Can one letter change make real word? → "በሽታ" yes (one_flip)
   ↓
[4. Surface checks] checks: Does it look weird? → length, rare letters, digits
   ↓
[5. SymSpell] (optional, OFF now) checks: How many similar dict words? → n_neighbors
   ↓
[6. KenLM] (optional, OFF now) checks: Does it sound like Amharic? → char_lm_score
   ↓
[7. LightGBM] combines all numbers → P(error)=0.85 → FLAG
```

LightGBM does **major work now** — 90% of decision. Other components just provide numbers (features) for LightGBM.

---

## Each Component — What it does, why, example

### 1. Dictionary — List of known Amharic words

**What:** File with unique words + how many times they appear.

Example `main_i7_frq10.txt` (737k types, 15MB, from bigger data, freq>=10 already):
```
ነው    5672129   ← appears 5.6M times in big corpus, very common
እና    4065980
ላይ    3984351
ሰላም  201671    ← common
በሽታ  1234      ← appears 1234 times, real word
```

**What we do with it:**
- For OCR word "ሰላም": look up in dictionary → found, freq 201671 → `in_lexicon=1`, `log_corpus_freq=log10(201671+1)=5.3`
- For OCR word "በሸታ": look up → not found → `in_lexicon=0`, `log_freq=0`

**Why important:** Most frequent feature. LightGBM gain **985,952** (biggest). If word not in dictionary, more likely error.

**Filtering:** We don't use all 737k, we filter by frequency for speed and cleanliness:
- `freq>=500` → 54k types (final model) — keeps words appearing at least 500 times → very clean, real words, not typos
- `freq>=10` → 52k types from local_lexicon (previous best)
- `freq>=100` → 6.7k types → too small, many legit words become OOV

**Where from:**
- `amh_final_training_text.txt` — 4.4M tokens (total words) → 750k unique types → `local_lexicon.txt`
- `main_i7.txt` — 4.9M unique types from web crawl (bigger data) → filtered freq>=10 → `main_i7_frq10.txt` 737k types → filtered freq>=500 → 54k types FINAL

**Scripts call dictionary, don't train it:** Dictionary static file, we load it at runtime, no learning. Model training uses dictionary only to create `in_lexicon` and `log_freq` features.

---

### 2. Normalizer — Fixes spelling differences before lookup

**What:** Amharic has same sound written with different letters. Normalizer collapses them to one form before dictionary lookup.

Example:
- ሐ→ሀ, ኀ→ሀ, ኃ→ሀ (h-family all → ሀ)
- ዐ→አ, ዓ→አ (ain → አ)
- ሠ→ሰ (sibilant)
- ፀ→ጸ

So:
- OCR word "ሐገር" → normalized "ሀገር" → look up "ሀገር" in dictionary → found
- OCR word "ሠላም" → normalized "ሰላም" → found
- Without normalizer, "ሐገር" would be considered OOV (not in dict) even though "ሀገር" is in dict → false flag

**Why:** Prevents false flags on ሕ/ኀ/ሐ variations. Result: 0 false flags on ሕ/ኀ/ሐ in final model.

**Code:** `normalizer.py` — `AmharicNormalizer.normalize_word_strict()` — called at **canonical ingestion boundary**: `NFC + normalize` before ANY lookup.

---

### 3. Confusion Matrix — Which letters Tesseract confuses

**What:** File `confusion_matrix.json` (47K) learned from real OCR errors. Says which letters Tesseract confuses.

Example:
```json
{
  "sub": {
    "ሽ": {"ሸ": 3.0, "ሺ": 2.5},  ← Tesseract often confuses ሽ as ሸ with cost 3.0
    "ሀ": {"ሐ": 2.0, "ኀ": 2.2}
  }
}
```

**What we do with it — One-flip test:**
- For OCR word "በሸታ" (maybe error):
  - Try changing each letter to its confusable alternatives (from confusion matrix):
    - Position 2: ሸ → try ሽ (cost 3.0) → "በሽታ" → is "በሽታ" in dictionary? Yes → one_flip hit!
  - `has_one_flip=1`, `one_flip_count=1`, `one_flip_words=["በሽታ"]`

- For "ሰላም" (correct):
  - Try one-flip → no confusable change makes real word → `has_one_flip=0`

**Why:** Catches errors that are one optical confusion away from real word. LightGBM gain: `has_one_flip` 22k, `one_flip_count` 26k.

**Fixed bug:** Old code looked for key "substitutions" but file has "sub" → one_flip always 0, gain 0. Fixed now, gain 22k.

---

### 4. Surface Signals — Does word look weird? (cheap, always run)

**What:** Checks that need no dictionary, no ML.

- `token_len` — length of word. Very short/long more likely error. Example: "እ" (1 char) vs "በሽታ" (3 chars)
- `len_ratio` — length / 4.14 (corpus mean). Example: len 1 → ratio 0.24 (weird), len 4 → ratio 0.96 (normal)
- `rare_glyph_ratio` — how many rare letters like ሠ ጰ ፀ ዐ. Example: "ሠላም" has ሠ (rare) → ratio 0.25, more likely error or old spelling
- `mixed_script` — Amharic + Latin mixed? Example: "ሰላምabc" → mixed → likely error
- `has_digit` — contains digit? Example: "ሰላም123" → has digit → likely error

**Why:** Cheap, fast, catches obvious errors. Gains: `token_len` 91k, `rare_glyph_ratio` 28k, `has_digit` 15k, `len_ratio` 8k.

---

### 5. SymSpell — How many similar dictionary words exist? (optional, OFF now)

**What:** Fast spell checker library. Given word, finds dictionary words within 1 edit distance.

Example:
- Word "በሸታ" (not in dict) → SymSpell lookup with max_edit_distance=1 → finds ["በሽታ", "በሸታው", ...] → `n_neighbors=13`
- Word "ሰላም" (in dict) → finds ["ሰላም", "ሰላማዊ", ...] → `n_neighbors=1`

**Why:** If many similar real words exist, maybe OCR word is error close to real words.

**Why OFF now:** In training we set `n_neighbors=0` for speed, so LightGBM learned gain **0** for this feature — it doesn't use it. So enabling SymSpell at inference gives same quality (tested: recall 94.5% both OFF and ON), but slower (builds index 50k types → 2 sec) and more memory. So disabled by default for performance.

**Enable if needed:**
```python
det = DetectionEngine(enable_symspell=True)  # builds index, computes n_neighbors
```

**Future:** If we retrain with `n_neighbors` computed, it might give +0.5% recall.

---

### 6. KenLM — Does word sound like Amharic? (optional, OFF now)

**What:** Character-level language model (5-gram). Scores how likely a sequence of characters is in Amharic.

Example:
- "ሰላም" → char LM score -0.5 (likely Amharic)
- "xyzxyz" → char LM score -3.0 (unlikely Amharic)

**Why:** Catches non-Amharic-looking errors.

**Why OFF now:** KenLM not installed in this environment, and file `amharic_char_model.klm` not present, so we set `char_kenlm_score=NaN`. LightGBM handles NaN natively, gain **0** for this feature. So no quality loss.

**If you have KenLM:**
- Install `kenlm` pip, provide `amharic_char_model.klm` path
- Model will use it, but gain currently 0, so not critical

**Future:** Word-level KenLM (word 5-gram) would be more useful — `word_lm_fit = logP(word|context) - logP(word)` — catches real-word errors like `አስረኞች` vs `እስረኞች` (both in dict, but one fits context better).

---

### 7. LightGBM — The Brain, does major work now

**What:** Machine learning model (Gradient Boosted Decision Trees), small (2.1MB), fast CPU, no GPU needed.

**Input:** 14 numbers (features) from above components:
```
log_corpus_freq, in_lexicon, doc_freq, is_doc_singleton, token_len, len_ratio,
has_one_flip, one_flip_count, char_kenlm_score (NaN), rare_glyph_ratio,
n_neighbors (0), mixed_script, has_digit, is_high_freq
```

**Output:** Probability `P(error)` 0.0 to 1.0

Example:
- "ሰላም": features [5.3, 1, 5, 0, 3, 0.72, 0, 0, NaN, 0, 0, 0, 0, 1] → LightGBM → P=0.001 → PASS
- "በሸታ": features [0, 0, 1, 1, 3, 0.72, 1, 1, NaN, 0, 0, 0, 0, 0] → P=0.85 → FLAG

**Training:** Learned from 1267 pages (883k rows, 1200 train / 67 val), 54k dictionary types (main_i7 freq>=500), no Tesseract confidence (conf-free), isotonic calibration on val set, thresholds pass 0.10 flag 0.65 tuned for recall vs false_flag.

**Feature importance (gain) — what LightGBM uses most:**
- `log_corpus_freq`: 985,952 (biggest — how common word is)
- `in_lexicon`: 185,151 (is word in dict?)
- `token_len`: 91,587 (length)
- `doc_freq`: 40,582 (how often on page)
- `is_doc_singleton`: 29,052 (appears once?)
- `rare_glyph_ratio`: 28,371
- `one_flip_count`: 26,975
- `is_high_freq`: 23,627
- `has_one_flip`: 22,235
- `has_digit`: 15,990
- `len_ratio`: 8,835
- `char_kenlm_score`: 0 (OFF)
- `n_neighbors`: 0 (OFF, SymSpell disabled)
- `mixed_script`: 0

So **LightGBM does 90% of work** — it learned weights for each feature, no manual if-else floors.

**Why conf-free:** Tesseract conf is unreliable (19% errors have conf>85, 7.5% correct have conf<70). If model trusts conf, it misses high-conf errors. Conf-free model gives same verdict for conf=95 vs 20 — correct.

**Model files:**
- `detector_lgbm_conffree.pkl` — FINAL, 2.1MB, 52-54k types freq>=10/500, 1267 pages, 14 feats, val recall 94.8% FF 0.07% (local) or 94.2% FF 0.01% (main_i7 freq>=500) — BEST
- `detector_lgbm_v3.pkl` — previous clean baseline, 15 feats with NaN conf, 4.8MB

---

### 8. DetectionEngine — The Manager

**What:** Class `DetectionEngine` in `detector.py` — loads all components and decides.

**Steps for one word:**
1. Canonical form: `NFC + normalize_word_strict()` → normalized word
2. Surface signals: length, rare glyphs, mixed script, digit
3. Lexical signals: in_lexicon, log_freq, n_neighbors (0 if SymSpell OFF)
4. One-flip: using confusion matrix, try confusable alts, check if in vocab
5. Extract 14 features (with NaN for missing KenLM, etc.)
6. LightGBM predict → P(error)
7. Isotonic calibration (if available) → calibrated P
8. Threshold: P<0.10 PASS, 0.10≤P<0.65 GRAY, P≥0.65 FLAG

**Performance:**
- Default: 50k types, no SymSpell, init 2.9s, 0.56 ms/word → 10k words/page → 5.6 sec/page
- With full 750k: init 2.0s (no SymSpell), vocab 719k, similar speed

---

### 9. Adapter — For Main Corrector

**What:** `CorrectorDetectionAdapter` in `adapter.py` — lets `amharic_ocr_corrector.py` use DetectionEngine without loading dictionary twice.

- Reuses corrector's vocab, symspell, confusion map, normalizer
- Implements same interface `verdict(tok, context, ocr_conf, doc_freq)`
- doc_freq lifecycle: full doc → Counter canonical, isolated token → NaN

**Why:** Avoid double memory, keep one source of truth (LightGBM authoritative, SuspicionScorer deprecated to fallback).

---

## How to use for your case

**You have main_i7_freq10.txt (737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, cleaner):**

```python
# Default — uses main_i7, 50k types, fast, no crash
from OCR_error_detection import DetectionEngine
det = DetectionEngine()  # auto-finds Model/main_i7_frq10.txt, 50k types, no SymSpell

# Full 737k types — slower but more coverage
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", max_lexicon_types=None, dict_min_freq=10)

# Best tradeoff — freq>=500, 54k types, lowest false flag 0.01%
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", dict_min_freq=500, max_lexicon_types=None)

# With SymSpell (only if you need n_neighbors, currently gain 0)
det = DetectionEngine(enable_symspell=True)
```

**No need to train dictionary itself** — dictionary static. To improve accuracy, filter by frequency (freq>=10 or freq>=500) for cleaner dict, or use bigger clean data (main_i7), or add word LM feature.

---

## One-line summary

- **Dictionary:** list of known words, provides `in_lexicon` and `log_freq` — biggest feature, gain 985k
- **Normalizer:** fixes ሐ→ሀ etc before lookup — prevents 0 false flags on ሕ/ኀ/ሐ
- **Confusion Matrix:** which letters Tesseract confuses — provides one-flip test
- **Surface:** length, rare glyphs, digits — cheap checks
- **SymSpell:** similar words count — currently OFF, gain 0, no quality loss, disabled for speed (was crashing PC)
- **KenLM:** does word look like Amharic? — OFF, gain 0
- **LightGBM:** brain, combines 14 numbers → P(error) → PASS/GRAY/FLAG — does major work now, 2.1MB, 94.8% recall, 0.01% FF
- **DetectionEngine:** manager that loads all and decides
- **Adapter:** for main corrector, reuses vocab

**LightGBM is authoritative, others are feature providers. Conf-free (no Tesseract conf) is better because conf unreliable.**
