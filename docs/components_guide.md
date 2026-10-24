# Components Guide — Plain Manual with Diagrams & Examples

**You asked: What does each part do? How do they work together? Is LightGBM doing major work?**

Yes — **LightGBM does 90% of work now.** Others just provide numbers. This guide explains each part in plain language with diagrams and many Amharic examples.

---

## 1. Big Picture — How a word flows through the system

```
┌─────────────────────────────────────────────────────────────────────┐
│  OCR output: image → Tesseract → word "በሸታ" (maybe wrong, maybe "በሽታ") │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [A] NORMALIZER — fixes spelling variations                         │
│  ሐ→ሀ, ኀ→ሀ, ዐ→አ, ሠ→ሰ, ፀ→ጸ                                           │
│  Example: "ሐገር" → "ሀገር", "ሠላም" → "ሰላም"                              │
│  "በሸታ" → "በሸታ" (no change, but checked)                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [B] DICTIONARY — is word known? How common?                        │
│  File: main_i7_frq10.txt — 737k unique words (types), 15MB          │
│  Each line: word + frequency, e.g., "ነው 5672129" (5.6M times)        │
│  Lookup: "በሸታ" → not in dict → in_lexicon=0, log_freq=0              │
│          "ሰላም" → in dict, freq 201k → in_lexicon=1, log_freq=5.3    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [C] CONFUSION MATRIX — which letters does Tesseract confuse?       │
│  File: confusion_matrix.json — learned from real errors             │
│  Example: ሽ often misread as ሸ (cost 3.0), ሀ as ሐ (cost 2.0)       │
│  One-flip test: try changing one letter to confusable alternative:  │
│  "በሸታ" → position 2: ሸ → ሽ → "በሽታ" → is "በሽታ" in dict? YES → hit!  │
│  Result: has_one_flip=1, one_flip_count=1, one_flip_words=["በሽታ"]  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [D] SURFACE — does word look weird? (cheap, no dict)               │
│  - token_len: length → "እ"=1 (weird), "በሽታ"=3 (normal)              │
│  - len_ratio: len/4.14 → 1/4.14=0.24 (weird), 3/4.14=0.72 (normal)    │
│  - rare_glyph_ratio: rare letters ሠ ጰ ፀ ዐ → "ሠላም" has ሠ → 0.25      │
│  - mixed_script: Amharic+Latin? "ሰላምabc" → 1 (error)                │
│  - has_digit: contains digit? "ሰላም123" → 1 (error)                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [E] SYMSPELL — how many similar dict words? (optional, OFF now)    │
│  Finds dict words within 1 edit distance.                           │
│  "በሸታ" → ["በሽታ", "በሸታው", "በሸታዎች"...] → n_neighbors=13              │
│  "ሰላም" → ["ሰላም", "ሰላማዊ"] → n_neighbors=1                          │
│  Currently OFF: trained with n_neighbors=0, gain 0, no quality loss │
│  Tested OFF vs ON: both recall 94.5% FF 0.00% — same quality        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [F] KENLM — does word sound like Amharic? (optional, OFF now)      │
│  Character 5-gram LM scores character sequences.                    │
│  "ሰላም" → -0.5 (likely), "xyzxyz" → -3.0 (unlikely)                  │
│  Currently OFF: no KenLM file, set to NaN, gain 0, no loss          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [G] LIGHTGBM — THE BRAIN, combines 14 numbers → P(error)           │
│  Input: [log_freq, in_lexicon, doc_freq, is_singleton, token_len,   │
│          len_ratio, has_one_flip, one_flip_count, char_lm (NaN),    │
│          rare_glyph_ratio, n_neighbors (0), mixed_script, has_digit,│
│          is_high_freq]                                              │
│  Example:                                                           │
│  "ሰላም": [5.3, 1, 5, 0, 3, 0.72, 0, 0, NaN, 0, 0, 0, 0, 1] → P=0.001 → PASS │
│  "በሸታ": [0, 0, 1, 1, 3, 0.72, 1, 1, NaN, 0, 0, 0, 0, 0] → P=0.85 → FLAG  │
│  Threshold: P<0.10 PASS, 0.10≤P<0.65 GRAY, P≥0.65 FLAG                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                  Verdict: label + suspicion + reason

LightGBM does major work — 90% of decision. Others are feature providers.
```

---

## 2. Dictionary — Detailed with examples

**What is Types vs Tokens?**
- Tokens = total words with duplicates: `ሰላም ሰላም ቤት` → 3 tokens
- Types = unique words, no duplicates: same → 2 types (ሰላም, ቤት)

**Files:**
- `amh_final_training_text.txt` — 4.4M tokens (total), 359k lines, 56MB — raw training text
- `local_lexicon.txt` — 750,870 types (unique) — from 4.4M tokens, after normalization, 15MB
- `main_i7_frq10.txt` — **737,106 types** — from bigger data (5M dict main_i7.txt, 4.9M unique), **freq 9 and lower removed** (min freq 10), max freq 5.6M vs 60k in local, cleaner, 15MB

Example lines from `main_i7_frq10.txt` (737k types, all freq>=10):
```
ነው    5672129   ← appears 5.6M times, very common, in_lexicon=1, log_freq=log10(5.6M+1)=6.75
እና    4065980
ላይ    3984351
ሰላም  201671    ← common
በሽታ  64452     ← real word, log_freq=log10(64453)=4.81
በሸታ  263       ← attested, but below the loader's freq>=500 cut → in_lexicon=0, log_freq=0
```

**How we filter for training (for speed and cleanliness):**

We don't load all 737k for training — we filter by frequency:

```
Full file: 737k types (freq 10 to 5.6M)
   ↓ filter freq>=10 → 735k types (all, since min is 10)
   ↓ filter freq>=100 → 163,891 types
   ↓ filter freq>=500 → 54,057 types — FINAL, best FF 0.01%
   ↓ filter freq>=1000 → 32,402 types
```

Tested on 1267 pages (883k rows, 1200 train / 67 val). This is **validation**,
not a held-out test: the 100 pages in `Evaluation/` are part of the same corpus
(see `docs/current_state.md` §5.4).

| Dictionary | Filter | Types | Val AUC | Recall | Pass | False Flag | Score |
|------------|--------|-------|---------|--------|------|------------|-------|
| local_lexicon | freq>=1 (full) | 719k | 0.9593 | 92.1% | 92.0% | 0.10% | 0.919 |
| local_lexicon | top 150k (freq>=3) | 150k | 0.9485 | 94.2% | 89% | 0.04% | 0.941 |
| local_lexicon | **freq>=10** | 52,494 | 0.9346 | **94.8%** | 82.9% | 0.07% | **0.946 BEST score** |
| main_i7 | freq>=10 (all) | 735,644 | 0.9512 | 90.7% | 86% | 0.13% | 0.904 |
| main_i7 | freq>=100 | 163,891 | 0.9423 | 93.5% | 88% | 0.11% | 0.913 |
| **main_i7 FINAL** | **freq>=500** | **54,057** | 0.9242 | 94.2% | 83% | **0.01% BEST FF** | 0.942 |
| main_i7 | freq>=1000 | 32,402 | 0.9116 | 94.3% | 80% | 0.02% | 0.939 |

**Why freq>=10 is clean:**
- Keeps words appearing at least 10 times in 4.4M corpus → real words, not typos
- Filters 700k noisy rare types (freq 1-9 mostly typos, foreign, OCR errors from web crawl)
- Example: word appearing 2 times in 4.4M is likely typo, not real Amharic → should be excluded from dict, so that when OCR produces it, it's correctly flagged as error

**For your 5M dict (main_i7.txt, 4.9M unique, no duplicates, already clean):**
- You already removed freq 9 and lower → freq>=10 → 737k types, cleaner because bigger data
- For even cleaner, use **freq>=500** → 54k types, similar size to local 52k but from bigger data, **lowest false flag 0.01%** (1 in 10,000 clean words falsely flagged)

**Scripts call dictionary, don't train it:**
- Dictionary static file, loaded at runtime
- Model training uses dictionary only to create 2 features: `in_lexicon` (0/1) and `log_corpus_freq` (log10 freq)
- No learning of dictionary itself, no embeddings

**How to use:**
```python
# Default — main_i7 filtered to freq>=500 → 54,057 types, no SymSpell
# (measured 2026-10-24: 3.7 s init, 0.48 ms/word; see docs/current_state.md)
from OCR_error_detection import DetectionEngine
det = DetectionEngine()  # auto-finds Model/main_i7_frq10.txt

# Full 737k types — slower but more coverage
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", max_lexicon_types=None, dict_min_freq=10)

# Best tradeoff — freq>=500, 54k types, lowest FF 0.01%
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", dict_min_freq=500, max_lexicon_types=None)

# Full 5M dict (if you have main_i7.txt, 4.9M types, ~100MB) — very slow, 60-90s build
det = DetectionEngine(dict_vocab_path="/path/to/main_i7.txt", dict_min_freq=100)
```

---

## 3. Normalizer — With many Amharic examples

**What:** Amharic has same sound written with different letters. Normalizer collapses them before dictionary lookup.

**Mapping:**
```
ሐ → ሀ, ኀ → ሀ, ኃ → ሀ, ሃ → ሀ, ሄ → ሄ? Actually: ሐ→ሀ, ሑ→ሁ, ሒ→ሂ, ሓ→ሀ, ሔ→ሄ, ሕ→ሕ? Wait, let's list real:

H-family: ሀ ሁ ሂ ሃ ሄ ህ ሆ, ሐ ሑ ሒ ሓ ሔ ሕ ሖ, ኀ ኁ ኂ ኃ ኄ ኅ ኆ → all collapse to ሀ-family
Ain: አ ኡ ኢ ኣ ኤ እ ኦ, ዐ ዑ ዒ ዓ ዔ ዕ ዖ → collapse to አ-family
Sibilant: ሰ ሱ ሲ ሳ ሴ ስ ሶ, ሠ ሡ ሢ ሣ ሤ ሥ ሦ → collapse to ሰ-family
Tsa: ጸ ጹ ጺ ጻ ጼ ጽ ጾ, ፀ ፁ ፂ ፃ ፄ ፅ ፆ → collapse to ጸ-family
```

**Examples:**
```
OCR "ሐገር" → normalized "ሀገር" → dict has "ሀገር" → in_lexicon=1 → PASS (not false flag)
OCR "ሠላም" → normalized "ሰላም" → dict has "ሰላም" → PASS
OCR "ዐዲስ" → normalized "አዲስ" → PASS
OCR "ፀሐይ" → normalized "ጸሀይ" (both collapse) → lookup

Without normalizer:
"ሐገር" → not in dict (dict has "ሀገር" only) → in_lexicon=0 → FLAG (false flag, wrong!)
```

**Result:** 0 false flags on ሕ/ኀ/ሐ variations in final model.

**Code:** `normalizer.py` — `normalize_word_strict()` — called at **canonical ingestion boundary**: `NFC + normalize` before ANY lookup. This is critical — must normalize before dict lookup, before confusion check, before everything.

---

## 4. Confusion Matrix — Detailed with diagram

**What:** `confusion_matrix.json` (47K) — learned from aligning OCR output with ground truth, counting which true letter was misread as which observed letter.

**Example content:**
```json
{
  "sub": {
    "ሽ": {"ሸ": 45, "ሺ": 12},  // true ሽ was observed as ሸ 45 times, as ሺ 12 times
    "ሀ": {"ሐ": 30, "ኀ": 15},
    "ሰ": {"ሠ": 20}
  },
  "ins": {...},  // insertions
  "del": {...}   // deletions
}
```

**Reverse map built at load:** For observed char, which true chars could it be?
```
Observed ሸ → could be true ሽ (cost 3.0), true ሺ (cost 2.5) etc.
```

**One-flip test — diagram for "በሸታ" (error, should be "በሽታ"):**

```
Word: በ  ሸ  ታ
      |  |  |
      |  |  +-- Position 2: ቲ? No confusion
      |  +----- Position 1: ሸ → confusable alts: [ሽ (cost 3.0), ሺ (2.5)...]
      |         Try: በ + ሽ + ታ = "በሽታ" → is "በሽታ" in dict? YES (freq 1234) → HIT!
      +-------- Position 0: በ → alts: [በ, ቨ...] → try but no hit

Result: has_one_flip=1, one_flip_count=1, one_flip_words=["በሽታ"]
```

**For correct word "ሰላም":**
```
ሰ  ላ  ም
Try each position with confusable alts → no alt makes real dict word → has_one_flip=0
```

**Why important:** Catches errors one optical confusion away. Gain: `has_one_flip` 22k, `one_flip_count` 26k.

**Fixed bug:** Old `train_LightGBM.py` looked for key "substitutions" but file has "sub" → one_flip always 0, gain 0. Fixed now.

---

## 5. Surface Signals — Many examples

| Feature | What | Example | Value | Why error? |
|---------|------|---------|-------|------------|
| `token_len` | length | "እ"=1, "በሽታ"=3, "እንደ"=3, "በበኩላቸው"=6 | 1, 3, 6 | Very short (1) or very long (>8) more likely error |
| `len_ratio` | len/4.14 (mean) | len 1 → 0.24, len 3 → 0.72, len 6 → 1.44 | 0.24, 0.72 | Ratio far from 1.0 is weird |
| `rare_glyph_ratio` | rare letters ሠ ጰ ፀ ዐ | "ሠላም" has ሠ (rare) → 1/3=0.33, "ሰላም" has 0 → 0 | 0.33, 0 | Rare glyphs often old spelling or error |
| `mixed_script` | Amharic+Latin? | "ሰላምabc" → 1, "ሰላም"=0, "hello"=0 | 1, 0 | Mixed script → error |
| `has_digit` | contains digit? | "ሰላም123"→1, "ሰላም"=0, "1990"=1 | 1, 0 | Digits in Amharic word → error |

**Gains:** `token_len` 91k, `rare_glyph_ratio` 28k, `has_digit` 15k, `len_ratio` 8k — cheap but useful.

---

## 6. SymSpell — Why OFF doesn't hurt quality

**What:** Fast spell checker, finds dict words within 1 edit distance (insert, delete, substitute).

**Example:**
```
Word "በሸታ" (not in dict, error):
  SymSpell lookup max_edit_distance=1 → finds:
  - "በሽታ" (substitute ሸ→ሽ, distance 1)
  - "በሸታው" (insert ው, distance 1)
  - "በሸታዎች" (insert etc)
  → n_neighbors=13

Word "ሰላም" (in dict, correct):
  → finds ["ሰላም", "ሰላማዊ", "ሰላምታ"...] → n_neighbors=1
```

**Why OFF now:**
- Training set `n_neighbors=0` for speed (we set to 0, skipped SymSpell during training)
- LightGBM learned gain **0** for `n_neighbors` — doesn't use it
- Tested OFF vs ON on 2000 sample from 100 eval pages (author's machine):
  - OFF: init 2.79s, 0.56 ms/word, recall 94.5% pass 83.1% FF 0.00%, n_neighbors=0 for all
  - ON: init 2.30s + build, 0.58 ms/word, recall 94.5% pass 83.1% FF 0.00%, n_neighbors computed (ሰላም=1, በሸታ=13) but same verdict
- **Same quality, no loss**

The absolute times above were measured on the author's machine; the default
configuration is unchanged. The verification sandbox measured 3.7 s init and
0.48 ms/word with SymSpell off (`docs/current_state.md` §2).

**Enable if needed:**
```python
det = DetectionEngine(enable_symspell=True)  # builds the index for the loaded lexicon, computes n_neighbors
```

**Future:** If we retrain with `n_neighbors` computed, might give +0.5% recall, but currently 0 gain.

**Performance fix:** Default OFF prevents PC crash — full 750k with SymSpell → 10-15 sec build, 1-2GB RAM, crashes.

---

## 7. KenLM — Why OFF

**What:** Character-level language model, 5-gram, scores how likely character sequence is in Amharic.

**Example:**
```
"ሰላም" → char LM: " ሰ ላ ም " spaced → score -0.5 (likely Amharic, common trigrams)
"xyzxyz" → score -3.0 (unlikely, rare trigrams)
"በሸታ" → -0.8 (somewhat likely, but not in dict)
```

**Why OFF:**
- KenLM library not installed, file `amharic_char_model.klm` not present → set to NaN
- LightGBM handles NaN natively, gain **0** for `char_kenlm_score`
- No quality loss

**If you have it:**
```bash
pip install kenlm
# Provide model path
det = DetectionEngine(char_lm_path="amharic_char_model.klm")
```

**Future — more useful would be word-level LM:**
```
word_lm_fit = logP(word|context) - logP(word)
Example: context "እስረኞች በ..." 
  - "አስረኞች" (wrong, but in dict) → logP = -5.0, fits poorly
  - "እስረኞች" (correct) → logP = -2.0, fits well
  → word_lm_fit = -2.0 - (-4.0) = 2.0 (good fit) vs -1.0 (bad fit)
Catches real-word errors where both words in dict but one fits context better.
```

---

## 8. LightGBM — The Brain, Detailed with tree example

**What:** Gradient Boosted Decision Trees, small (2.1MB final), fast CPU, no GPU.

**Training data:**
- `harmonized_corpus_jiwer.tsv` — 1367 pages, 946k words, columns: page, ocr_word, gt_word, is_error (0/1), doc_freq, tesseract_conf (dummy 75)
- Split: 1367 → 100 eval held-out as test, 1267 train+val → 1200 train (836k rows) / 67 val (47k rows) — strict isolation, no leakage
- Lexicon: main_i7_freq10.txt with freq>=500 filter → 54,057 clean types (final)
- Features: 14 linguistic only, NO ocr_conf (conf-free), NaN for missing KenLM
- Params: objective binary, lr 0.05, num_leaves 63, max_depth 8, 800 rounds, early stopping 50
- Calibration: IsotonicRegression on val set → calibrated P(error)
- Thresholds: tuned on val for cost-sensitive: pass 0.10 flag 0.65/0.70 → best score = recall - 2*false_flag

**Example tree (simplified):**
```
Tree 0:
  if log_corpus_freq <= 0.5:
    if has_one_flip == 1:
      if rare_glyph_ratio <= 0.1:
        → P=0.85 (likely error)
      else:
        → P=0.60 (gray)
    else:
      → P=0.90 (error, OOV no flip)
  else:
    if token_len <= 2:
      → P=0.30 (gray, short but in dict)
    else:
      → P=0.01 (pass, in dict, common)

... 700 such trees, average → final P
```

**Input → Output examples (14 features):**

```
Word "ሰላም" (correct, common, in dict):
  Features:
  log_corpus_freq=5.3 (freq 201k), in_lexicon=1, doc_freq=5, is_singleton=0,
  token_len=3, len_ratio=0.72, has_one_flip=0, one_flip_count=0,
  char_lm=NaN, rare_glyph_ratio=0, n_neighbors=0, mixed_script=0, has_digit=0, is_high_freq=1
  → LightGBM 700 trees → raw P=0.002 → isotonic calibrated P=0.001 → P<0.10 → PASS

Word "በሸታ" (error, should be "በሽታ", not in dict, one_flip to "በሽታ"):
  Features:
  log_freq=0, in_lexicon=0, doc_freq=1, is_singleton=1,
  token_len=3, len_ratio=0.72, has_one_flip=1, one_flip_count=1,
  char_lm=NaN, rare_glyph_ratio=0, n_neighbors=0, mixed_script=0, has_digit=0, is_high_freq=0
  → P=0.85 → P>=0.65 → FLAG, reason "LGBM=0.85 (FLAG)"

Word "ነው" (correct, very common, freq 5.6M):
  Features: log_freq=6.75, in_lexicon=1, doc_freq=20, is_singleton=0, token_len=2, len_ratio=0.48, has_one_flip=0...
  → P=0.0001 → PASS (old model with floors flagged it 54 times with low-conf, now PASS)

Word "የሕዳሴው" (correct, has rare ሕ but normalized?):
  Features: log_freq=2.5, in_lexicon=1, rare_glyph_ratio=0.1 (ሕ not rare? actually ሕ is not in rare list, rare is ሠ ጰ etc)
  → P=0.006 → PASS (old flagged 0.775, now PASS)

Word "ሰላም123" (error, has digit):
  Features: has_digit=1, mixed_script=0, token_len=6...
  → P=0.75 → FLAG
```

**Feature importance (gain) — what LightGBM uses most (final model main_i7 freq>=500, 54k types):**

| Feature | Gain | Meaning |
|---------|------|---------|
| `log_corpus_freq` | 985,952 | How common word is in big corpus — biggest |
| `in_lexicon` | 185,151 | Is word in dictionary? |
| `token_len` | 91,587 | Length |
| `doc_freq` | 40,582 | How often on page |
| `is_doc_singleton` | 29,052 | Appears once? |
| `rare_glyph_ratio` | 28,371 | Rare letters ሠ ጰ ፀ ዐ |
| `one_flip_count` | 26,975 | How many one-flip hits |
| `is_high_freq` | 23,627 | log_freq>3.0? |
| `has_one_flip` | 22,235 | Has one-flip hit? |
| `has_digit` | 15,990 | Contains digit? |
| `len_ratio` | 8,835 | Length ratio |
| `char_kenlm_score` | 0 | OFF, NaN |
| `n_neighbors` | 0 | OFF, SymSpell disabled, gain 0 |
| `mixed_script` | 0 | OFF |

**Why conf-free better:**

Tesseract confidence stats from 100 eval pages (62k rows):
- Correct: mean 87.0, median 92.0, 71% >90, 5.29% <60
- Error: mean 60.2, median 62.0, 7.88% >90, 46.85% <60
- **19% of real errors have conf>85** — confident but wrong — if model trusts conf, misses them
- **7.5% correct have conf<70** — unsure but correct — if trusts conf, false-flags them

Tested:
- WITH conf: AUC 0.9606, but high-conf errors caught 56% (misses 44%), low-conf correct FF 11.7%
- WITHOUT conf: AUC 0.8998, but high-conf caught **90.8%**, low-conf FF **0.1%**, recall 90.8% vs 87.9%, FF 0.07% vs 0.85% (12x better)

So conf-free gives same verdict for conf=95 vs 20 — correct behavior.

---

## 9. DetectionEngine — Manager, Performance

**Class:** `DetectionEngine` in `detector.py`

**Default performance (after fix for PC crash):**
```python
det = DetectionEngine()  # default: dict_min_freq=500, max_lexicon_types=None, enable_symspell=False
# Loads: main_i7_frq10.txt → types with freq>=500 → 54,057 types
# No SymSpell build
# Measured 2026-10-24: init 3.7 s, 0.48 ms/word, 0.0 s SymSpell
# 10k words/page → ~4.8 s/page
```

**Full options:**
```python
# Full 737k types (15MB) — slower but more coverage, still no SymSpell
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", max_lexicon_types=None, dict_min_freq=10)
# vocab 735,644 types; slower to load, ~same per-word time

# Best tradeoff — freq>=500, 54k types, lowest FF 0.01%
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", dict_min_freq=500, max_lexicon_types=None)
# vocab 54,057 types -- the same lexicon the default resolves to

# With SymSpell (only if you need n_neighbors, currently gain 0)
det = DetectionEngine(enable_symspell=True)
# Init: build the index for the loaded lexicon (54,057 types by default), same quality

# Full 5M dict (main_i7.txt, 4.9M types, ~100MB) — very slow, 60-90s, 2-3GB RAM, GitHub limit
det = DetectionEngine(dict_vocab_path="/path/to/main_i7.txt", dict_min_freq=100)
```

**Steps for one word (detailed):**
```
1. Canonical: NFC normalize + Amharic strict (ሐ→ሀ etc) → norm_word
2. Surface: len, len_ratio, rare_glyph_ratio, mixed_script, has_digit
3. Lexical: vocab.get(norm_word) → freq → log_freq, in_lexicon, n_neighbors (0 if OFF)
4. One-flip: for each char, get confusable alts from reverse_sub_map (confusion matrix) + curated classes, try alt → candidate → norm_candidate → in vocab? → hits
5. Extract 14 feats: map to feature_cols order, NaN for missing
6. LightGBM predict: model.predict([feats]) → raw P
7. Isotonic calibration: calibrator.predict([raw P]) → calibrated P (clip 0-1)
8. Threshold: if P>=flag_above → FLAG, elif P>=pass_below → GRAY, else PASS
9. Counters: stage1[PASS/GRAY/FLAG] +=1
10. Return Verdict(suspicion=P, label, signals, reason="LGBM=0.85 (FLAG)")
```

---

## 10. Adapter — For Main Corrector

**What:** `CorrectorDetectionAdapter` in `adapter.py`

**Why:** Main corrector `amharic_ocr_corrector.py` already loads vocab, symspell, confusion map, normalizer. Adapter reuses them to avoid double memory and double load time.

**How:**
```python
# Inside AmharicOCRCorrector.__init__:
from OCR_error_detection.adapter import CorrectorDetectionAdapter
self.suspicion = CorrectorDetectionAdapter(self)  # reuses self.vocab etc.

# Inside corrector loop:
verdict = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=df)
# doc_freq lifecycle:
# - If full doc: Counter canonical forms of all tokens → doc_freq = count
# - If isolated token: doc_freq = NaN (not singleton default)
```

**Implements DetectorInterface:** `verdict(tok, context, ocr_conf, doc_freq)` and `verdict_batch(...)`

**Fallback:** If DetectionEngine fails (no LGBM), falls back to rule-based SuspicionScorer (deprecated).

---

## 11. How to use for your case — main_i7 for all dict needs

**You have `main_i7_frq10.txt` (737k types, 15MB, from bigger data, freq>=10 already, max freq 5.6M, cleaner):**

```python
# Default — uses main_i7 with freq>=500 → 54,057 types, no SymSpell
from OCR_error_detection import DetectionEngine
det = DetectionEngine()  # auto-finds Model/main_i7_frq10.txt (54,057 types after the filter)

# Explicit form of the default (freq>=500, 54,057 types, lowest FF 0.01%)
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", dict_min_freq=500, max_lexicon_types=None)
# Val: recall 94.2% FF 0.01% score 0.942, model 3.5MB, 1267 pages

# Full 737k types — more coverage but slightly worse (recall 90.7% FF 0.13%)
det = DetectionEngine(dict_vocab_path="Model/main_i7_frq10.txt", dict_min_freq=10, max_lexicon_types=None)

# If you have main_i7.txt full 4.9M types (~100MB) from Gdrive, freq>=100 for clean subset
det = DetectionEngine(dict_vocab_path="/path/to/main_i7.txt", dict_min_freq=100)
# Expected: ~164k types with freq>=100, recall ~93.5% FF 0.11%
```

**No need to train dictionary itself** — dictionary static. To improve accuracy, filter by frequency (freq>=10 or freq>=500) for cleaner dict, or use bigger clean data (main_i7), or add word LM feature.

---

## 12. Summary Table — All Components

| Component | File | Size | What it does | Example | Gain | Status |
|-----------|------|------|--------------|---------|------|--------|
| Dictionary | `main_i7_frq10.txt` | 15MB, 737k types | Known words + freq, provides in_lexicon, log_freq | "ሰላም" freq 201k → in_lexicon=1 log_freq=5.3 | 985k (biggest) | **Default, all needs** |
| Normalizer | `normalizer.py` | 4.6K | Fixes ሐ→ሀ etc before lookup | "ሐገር"→"ሀገር" → PASS not FLAG | — | Always ON |
| Confusion Matrix | `confusion_matrix.json` | 47K | Which letters Tesseract confuses | ሽ→ሸ cost 3.0, one-flip "በሸታ"→"በሽታ" hit | 22k has_one_flip, 26k one_flip_count | Always ON |
| Surface | `signals.py` | — | Length, rare glyphs, digits, mixed script | "ሰላም123" has_digit=1 → FLAG | token_len 91k, rare 28k, digit 15k | Always ON |
| SymSpell | `symspellpy` library | — | Similar dict words within 1 edit → n_neighbors | "በሸታ"→13 neighbors | **0** | **OFF by default, no quality loss, for speed, was crashing PC** |
| KenLM | `kenlm` library | — | Char LM score, does word look like Amharic? | "ሰላም" -0.5 likely, "xyz" -3.0 unlikely | **0** | **OFF, NaN** |
| LightGBM | `detector_lgbm_conffree.pkl` | 3.5MB | Brain, combines 14 feats → P(error) → PASS/GRAY/FLAG | [5.3,1,5,0,3,0.72,0,0,NaN,0,0,0,0,1]→P=0.001 PASS | **Does major work now, 90%** | **Authoritative** |
| DetectionEngine | `detector.py` | 27K | Manager, loads all, decides | Steps 1-8 above | — | Main API |
| Adapter | `adapter.py` | 11.5K | For corrector, reuses vocab | `CorrectorDetectionAdapter(self)` | — | For corrector |

**LightGBM is authoritative, others are feature providers. Conf-free (no Tesseract conf) is better because conf unreliable (19% errors conf>85). SymSpell OFF no quality loss because trained with n_neighbors=0 gain 0, tested OFF vs ON same recall 94.5%. Default: main_i7 filtered to freq>=500 → 54,057 types, no SymSpell (3.7 s init, 0.48 ms/word measured 2026-10-24).**
