# README claims vs. what the scripts actually produce

Audit date: 2026-10-15. Branch: `arena/7224c765-ocr-corrector`, commit `c9325d2`.

Environment: clean sandbox clone, Python 3.11.2, `pip install numpy joblib lightgbm
scikit-learn symspellpy==6.10.0 pandas opencv-python-headless Pillow fonttools pytest`.
No Tesseract binary, no KenLM model, no `/media/des` or `/mnt/mintdata` trees.

**Bottom line:** the pipeline code is largely sound and the *training* scripts reproduce
their own saved artifacts exactly. It is the **headline numbers in `README.md`,
`4_ocr_correction/error_detection/README.md`, `docs/components_guide.md`, `docs/architecture.md` and
`docs/project_structure.md` that do not reproduce** — and both runnable "How to use"
examples in the root README fail. Nothing below is a guess: every number was re-derived
by running the repo's own scripts.

> **Update — root cause found and fixed (§G).** Chasing the false-flag discrepancy turned
> up a genuine train/serve skew in the `has_digit` feature, not just a reporting error.
> `features.py` computed it from the raw token while every other column used the cleaned
> surface; the training corpus contains no ASCII digits at all, so the model learned that
> branch exclusively from rare Ethiopic-numeral words. Real OCR tokens carrying ASCII
> digits (`ከ20`, `በ5ሺህ`, `“3ኛውን`) fell into it and came back **FLAG at P=0.83–1.00 on
> clean dictionary words**. That one column produced **360 of the 419 false flags** on the
> held-out pages. Fixed by measuring `has_digit` on the clean surface — provably
> bit-identical on all 946,057 training rows, so no retraining was needed. Held-out FF
> **0.72% → 0.10%**, recall unchanged. The README's 0.01% claim is still ~10× optimistic,
> but the detector is now materially better than either number suggested.


---

## A. The FINAL detector's headline metrics are not reproducible

### A1. The README table contradicts the repo's own saved sweep report

The documented way to produce that table is

```bash
python 4_ocr_correction/error_detection/Model/train_LightGBM.py --sweep-freqs 10,100,500,1000
```

(which is exactly what `Model/train_sweep_report.json` records under `args`). Re-running it
reproduces `train_sweep_report.json` **to the digit**, and neither matches the README:

| Dictionary | Filter | Types | README AUC | **actual AUC** | README recall | **actual recall** | README FF | **actual FF** | README score | **actual score** |
|---|---|---|---|---|---|---|---|---|---|---|
| local_lexicon | freq>=1 | 719,804 ✓ | 0.9593 | 0.9571 | 92.1% | 92.45% | 0.10% | 0.12% | 0.919 | 0.923 |
| local_lexicon | freq>=3 | **162,893** (README: 150k) | 0.9485 | 0.9495 | 94.2% | 93.74% | 0.04% | 0.07% | 0.941 | 0.937 |
| local_lexicon | freq>=10 | 52,494 ✓ | 0.9346 | 0.9336 | 94.8% | 94.50% | 0.07% | 0.05% | 0.946 | **0.945 ← best** |
| main_i7 | freq>=10 | 735,644 ✓ | 0.9512 | 0.9529 | 90.7% | 90.90% | 0.13% | 0.16% | 0.904 | 0.907 |
| main_i7 | freq>=100 | 163,891 ✓ | 0.9423 | 0.9418 | 93.5% | 91.40% | 0.11% | 0.04% | 0.913 | 0.914 |
| **main_i7 "FINAL"** | **freq>=500** | **54,057 ✓** | 0.9242 | 0.9246 | **94.2%** | **92.64%** | **0.01%** | **0.06%** | **0.942** | **0.926** |
| main_i7 | freq>=1000 | 32,402 ✓ | 0.9116 | 0.9115 | 94.3% | 93.24% | 0.02% | 0.05% | 0.939 | 0.932 |

Most rows drift by a few tenths of a point (plausibly a different run/seed/machine).
The **"FINAL" row does not**: recall is overstated by 1.6 pts, false-flag is understated
**6×**, and the score by 0.016. `Model/train_sweep_report.json` — the artifact written by
the run that produced the shipped model — has said `recall 92.64% / FF 0.06% / score 0.926`
all along.

### A2. The model-selection conclusion is unsupported

The README declares main_i7 freq>=500 the winner ("BEST", score 0.942). Reproducibly it is
**4th of 7** by the repo's own `score = recall × (1 − FF)`:

```
local_lexicon freq>=10   0.945   <- best
main_i7       freq>=1000 0.932
main_i7       freq>=3    0.937   (local)
main_i7       freq>=500  0.926   <- shipped "FINAL"
```

The README's own table is internally inconsistent here too: it lists `local_lexicon
freq>=10` at score **0.946** and labels it "prev best", while calling the lower **0.942**
the FINAL winner.

### A3. On the held-out `Evaluation/` set, FF is 0.72%, not 0.01%

```bash
python 4_ocr_correction/error_detection/scripts/evaluate_child.py
```

```
Pages: 100, GT words: 62,427, OCR words: 62,479, Aligned: 62,385
WER: 6.84%  (substitutions 4,267), CER: 1.90%

ERROR words (4,267):
  FLAG 140 (3.3%), GRAY 3,823 (89.6%), PASS 304 (7.1%)
  Recall (flag+gray): 92.9%   (flag only: 3.3%)

CLEAN words (58,118):
  PASS 48,895 (84.1%), GRAY 8,804, FLAG 419
  False flag rate: 0.72%
```

0.01% of 58,118 clean words is ~6 words. The real number was **419 — 72× the claim**
(before the §G fix; **59** after it, still ~10× the claim).

Note these 100 pages are the `DEV_PAGES` genuinely held out of training, while the
sweep's 0.06% is measured on 67 val pages drawn from the *training* distribution. The
README quotes the in-distribution figure as if it were the operating figure.

**After the §G `has_digit` fix**, the same command reports:

```
ERROR words (4,267):
  FLAG 113 (2.6%), GRAY 3,849 (90.2%), PASS 305 (7.1%)
  Recall (flag+gray): 92.9%   (flag only: 2.6%)
CLEAN words (58,118):
  PASS 48,896 (84.1%), GRAY 9,163, FLAG 59
  False flag rate: 0.10%
```

Recall is unchanged (92.9%); false flags fall 419 → 59.

### A4. Three mutually inconsistent records of the same 100 pages

| Source | recall (flag+gray) | recall (flag only) | false flag |
|---|---|---|---|
| `README.md` / `4_ocr_correction/error_detection/README.md` | 94.2% | — | **0.01%** |
| committed `Evaluation/analysis.tsv` (62,521 rows) | 95.64% | 60.37% | **1.194%** |
| current code re-run (above) | 92.9% | **3.3%** | **0.72%** |

The shipped model + current code almost never emits `flag` (3.3% of real errors); nearly
everything lands in `gray`. Because the README describes correction as firing on FLAG, this
is a behavioural difference, not just a reporting one.

### A5. "100% high-conf caught, 0% low-conf FF" is contradicted by the committed artifact

From `Evaluation/analysis.tsv` (the repo's own per-word record):

* errors with `tesseract_conf > 85`: 808 → caught (flag+gray) **92.3%**, FLAG only **52.0%**
* clean words with `tesseract_conf < 70`: 4,327 → falsely FLAGged **4.51%** (not 0%)

---

## B. Both "How to use" examples in the root README fail

### B1. Child example — documented `FLAG 0.85`, actual `gray 0.154`

```python
det = DetectionEngine()
result = det.verdict("በሸታ", doc_freq=2)
```

| | documented | actual |
|---|---|---|
| label | `FLAG` | `gray` |
| suspicion | `0.85` | `0.1537` |

`docs/components_guide.md:73` goes further and prints the exact feature vector
`[0, 0, 1, 1, 3, 0.72, 1, 1, NaN, 0, 0, 0, 0, 0] → P=0.85 → FLAG`. Feeding that exact
vector into the shipped booster returns **P=0.3142 → GRAY** (raw 0.3023). The documented
example cannot be produced by the shipped model at any `doc_freq`:

```
doc_freq=1     -> GRAY  P=0.3142
doc_freq=2     -> GRAY  P=0.1537
doc_freq=None  -> GRAY  P=0.3142
```

The companion example `ክተማ` (documented elsewhere as the canonical OCR error for `ከተማ`)
also returns GRAY, not FLAG.

Labels are also lowercase in code (`"pass"`, `"gray"`, `"flag"`), so the README's
`print(result.label)  # FLAG 0.85` and the pseudo-code
`if verdict.label == "FLAG":` in both `README.md` and `4_ocr_correction/README.md`
never match.

### B2. Parent example — raises, cannot work from a clean clone

```python
corrector = _mod.AmharicOCRCorrector()
print(corrector.correct_text("ሰላም በሸታ"))
```

```
ValueError: AmharicOCRCorrector needs a KenLM language model: the Viterbi stage scores
every candidate with it. Pass kenlm_path=..., set AMHARIC_WORD_KENLM ...
```

There is **no KenLM model anywhere in the repo** (`find` for `*.klm|*.binary|*.arpa|*.lm`
returns nothing) and no in-repo copy is planned — `repo_paths.word_kenlm()` resolves to
`/mnt/mintdata/...`. So the documented parent example fails for every user, always.
`4_ocr_correction/README.md`'s `python -m 4_ocr_correction.test_corrector` fails the same
way (`FileNotFoundError: kenlm_path=... does not exist`).

Also note: README says the child has "KenLM OFF (gain 0)" while the parent hard-requires a
word 5-gram KenLM. Both are true, but the README never says the parent is unusable without
an external ~hundreds-of-MB model.

---

## C. Dependency and path defects that break documented commands

### C1. `requirements.txt` omits `pydantic` → morphology silently disabled

`vendor/segmenter/morph_types.py` raises `ImportError: morph_types requires pydantic`
without it. `pydantic` is **not** in `requirements.txt`, though
`vendor/segmenter/VENDORING.md:63` documents the requirement. After a clean
`pip install -r requirements.txt`:

```
[WARNING] amharic_ocr_corrector: Could not import AmharicMorphologicalSegmenter:
          No module named 'segmenter'
```

Per `VENDORING.md`, that means `self.segmenter = None`, `analyze()` returns
`known=False` for every OOV token, and "the entire `lambda_morph` scoring term silently
cancels out of the ranking". Vendoring was done specifically to stop this; the missing
dependency reinstates it. The README still advertises the corrector as
"SymSpell + confusion + morphology + KenLM".

### C2. `os.makedirs` on hardcoded author paths instead of `repo_paths.makedirs`

`repo_paths.makedirs()` exists precisely to turn a missing root into "set
TESSERACT_WORKSPACE …". These documented commands bypass it and die with a raw
`PermissionError: [Errno 13] Permission denied: '/media/des'`:

* `python 01_evaluate_full_corrector.py` — documented in `5_correction_evaluation/README.md`
  (lines 120/151, `CORRECTED_OUT_DIR`)
* `3_training_evaluation/02_benchmark_new_model.py:93`
* `1_data_preparation/07_generate_eval_pages.py:291`

`bash smoke_eval.sh` and the `2_training/*.sh` scripts at least read
`${TESSERACT_WORKSPACE:-…}`, but still default to a path that does not exist.

### C3. A typo'd dictionary path degrades silently to 8 words

`4_ocr_correction/error_detection/README.md` documents:

```python
DetectionEngine(dict_vocab_path="main_i7_freq10.txt", max_lexicon_types=None)
```

The real filename is `main_i7_frq10.txt`. Because a *caller-supplied* path that does not
exist falls into the "minimal fallback" branch, this loads an **8-word** lexicon with no
error and no warning:

```
lexicon_info: {'path': 'main_i7_freq10.txt', ..., 'types': 0, 'caller_supplied': True}
vocab size  : 8
verdict በሸታ: gray 0.2833
```

A nonexistent `dict_vocab_path` should raise, not silently invalidate every
`in_lexicon` / `log_corpus_freq` feature (which together carry **80%** of the model's gain).

---

## D. Stage 5: the corrector makes the text worse, and no README says so

`5_correction_evaluation/benchmarks/benchmark_full_symspell.json` (150 pages, committed):

| Metric | strict | canonical |
|---|---|---|
| raw CER → corrected CER | 1.178% → **1.930%** | 1.059% → **1.140%** |
| raw WER → corrected WER | 4.674% → **6.293%** | 4.465% → **4.652%** |
| character_error_reduction | **−63.8%** | **−7.68%** |
| clean tokens damaged | 2,462 / 60,596 (4.06%) | 1,618 / 60,766 (2.66%) |

Negative reduction = net harm. The root README poses stage 5 as the question
"does correction improve CER/WER?" but never reports the committed answer, and the
"One-line" summary presents the whole pipeline as a success. `docs/ocr_corrector_review.md`
§"Evaluation issues to fix before claiming improvement" and `docs/two_phase_benchmark.md`
("This is a benchmark procedure, not a claimed result") are honest about this; the top-level
READMEs are not.

---

## E. Smaller documentation inaccuracies

| Location | Claim | Actual |
|---|---|---|
| `README.md`, `4_ocr_correction/error_detection/README.md` | "Tests: 40 OK for child" | 56 pass (child); parent 159 pass / 24 skipped |
| `4_ocr_correction/README.md` | `amharic_ocr_corrector.py` "(80K lines)" | 1,817 lines (87 KB) |
| `docs/components_guide.md` | dict lines `በሸታ 0`, `በሽታ 1234` | real file: `በሸታ 263`, `በሽታ 64452` |
| `README.md` | local_lexicon "top 150k (freq>=3)" | 162,893 canonical types at freq>=3 |
| `README.md` | "detector.py — default 50k types **freq>=10**" | default is **freq>=500** → 54,057 (freq>=10 gives 735,644) |
| `README.md` | "Performance fix: Default 50k types **freq>=10**" | same contradiction |
| `README.md` | "Model/main_i7_frq10.txt 737k types **default**" | loaded with freq>=500 → 54,057 by default |
| `docs/architecture.md`, `docs/project_structure.md` | `dict_min_freq=500,  # 54k types, best FF 0.01%` | FF 0.06% (val), 0.72% (held-out eval) |

---

## F. What *does* verify — the code and data are in good shape

Claims that reproduce exactly, and should be kept:

**Tesseract / eval-set statistics** (from `evaluate_child.py` and `analysis.tsv`)
* WER 6.84% ≈ "WER 7%" ✓ · substitutions **4,267** = "4267 errors" ✓ (exact)
* CER 1.90% → 98.10% char acc ≈ "98.07%" ✓ · word acc 93.16% ≈ "93%" ✓
* mean conf: clean **87.0** / error **60.8** = "correct mean 87, error mean 60" ✓
* "19% errors conf>85" → **18.9%** ✓ · "7.5% correct conf<70" → **7.4%** ✓

**Data and artifact facts**
* `main_i7_frq10.txt` **737,106** lines ✓ · `local_lexicon.txt` **750,870** ✓
* `amh_final_training_text.txt` **4,420,102 tokens / 359,776 lines / 55.6 MB** = "4.4M tokens, 56MB, 359k lines" ✓
* `harmonized_corpus_jiwer.tsv` **946,057 rows / 1,367 pages / 48 MB** ✓
* `analysis.tsv` **62,521 rows** = "62k rows" ✓ · `Evaluation/` 100 GT + 100 OCR + 100 images ✓
* `confusion_matrix.json` **47,847 B** = "47K" ✓ · `detector_lgbm_conffree.pkl` **3,655,005 B** = "3.5MB" ✓
* **14 conf-free features**, gates **0.10 / 0.65**, **54,057 types at freq>=500** ✓
* shipped pkl is byte-identical (md5 `a918c717…`) to `archive/models/detector_lgbm_conffree_main_i7_freq500.pkl` ✓

**Feature-gain claims** (booster importance)
* "SymSpell OFF, gain 0" → `n_neighbors` gain **0.0**, 0 splits ✓
* "KenLM OFF, gain 0" → `char_kenlm_score` gain **0.0**, 0 splits ✓
* "log_freq (gain 985k biggest)" → `log_corpus_freq` gain **985,952.1** = **67.61%** ✓
  (`in_lexicon` 12.70%, `token_len` 6.28%; `mixed_script` also 0.0, undocumented)

**Performance**
* child init **3.29 s** (README "2.9s") ≈ ✓ · **0.32 ms/word** (README "0.56 ms/word") — better ✓

**Reproducibility**
* `train_LightGBM.py --sweep-freqs …` is deterministic and matched the committed
  `train_sweep_report.json` exactly on a re-run ✓
* `pytest 4_ocr_correction/error_detection/tests/` → **56 passed** ✓
* `python -m unittest discover -s tests -p 'test_*.py'` → **Ran 159 tests … OK (skipped=24)** ✓

---

## G. Root cause of the false-flag gap: a train/serve skew in `has_digit`

The 0.01% (README) vs 0.72% (measured) gap is not only a reporting error — most of it was
a real defect, now fixed.

### The mechanism

`features.py` documented `has_digit` as the one column that reads the **raw** token,
"because cleaning removes digits". But `clean_surface` uses `PUNCT_RE = r"[^\w\s]|\d"`,
and Ethiopic numerals `፩፪፫፬፭፮፯፰፱` (U+1369–U+1371) are Unicode category **No**, not
**Nd** — so `\w` keeps them and `\d` does not match them. Cleaning therefore strips
**ASCII** digits but **keeps Ethiopic** ones.

Consequences in the training corpus (`Model/harmonized_corpus_jiwer.tsv`, 946,057 rows):

* rows containing an ASCII digit or any punctuation: **0** — `clean_surface(w) == w` for
  **100.0000%** of rows
* rows with `has_digit = 1`: **812 (0.086%)**, and every one is an Ethiopic-numeral word
  (`መሆ፭ም`, `፬ዎቹ`, `፪ምሮ`, `ቋን፮`, …) where the numeral is part of the surface that
  `token_len`, `in_lexicon`, `log_corpus_freq` and the one-flip test all describe

So the trees learned `has_digit = 1` from one tiny, peculiar population — and split on it
enough to accumulate **24,972 gain** (15,990 in the shipped pickle).

At serving time the population is completely different. `evaluate_child.py` sends the raw
OCR token, and the parent corrector's `EthiopicPunctuationManager.decompose_token` strips
only leading/trailing punctuation and diacritics — **digits are not in `ALL_PUNCT`**, so
production passes cores like `ከ20`, `በ5ሺህ`, `3ኛውን`, `200`. For those, cleaning removes the
digit from every other column while `has_digit` stays 1:

| token | features that differ from its clean form | P(clean) | P(raw) |
|---|---|---|---|
| `ከ20` vs `ከ` | `has_digit` 0 → 1 | 0.0121 **PASS** | 0.8283 **FLAG** |
| `በ5ሺህ` vs `በሺህ` | `has_digit` 0 → 1 | 0.0154 **PASS** | 0.8051 **FLAG** |
| `“3ኛውን` vs `ኛውን` | `has_digit` 0 → 1 | 0.3599 GRAY | 1.0000 **FLAG** |

420 of the 62,479 held-out tokens (0.67%) contain an ASCII digit, and they account for
**360 of the 419 false flags**.

### The fix

Measure `has_digit` on the clean surface, like every other column
(`4_ocr_correction/error_detection/features.py`, `cheap_signals`). This is **provably a no-op on
training data** — `clean_surface(w) == w` for all 946,057 corpus rows, so both definitions
yield `has_digit = 1` for exactly the same 812 rows. No retraining, and no shipped model
changes behaviour on any input it was trained on.

Also updated: `tests/test_features.py`, where `RAW_ONLY_DIFF = {"has_digit"}` had
*codified* the skew — the parity test explicitly excused that column instead of catching
the bug. It is now `set()`, so `test_extract_frame_matches_detector` asserts parity on all
14 columns, and `test_has_digit_reads_the_raw_token` is replaced by
`test_has_digit_follows_the_clean_surface`, which asserts the Ethiopic/ASCII distinction
and the exact P(raw) == P(clean) equality.

### Verification: two independent protocols now agree

Before the fix, the trainer's held-out dev metric and `evaluate_child.py`'s alignment
metric disagreed by 3–9× on the *same* model. After it, they agree to three decimals —
which is the real evidence that parity is restored:

| config | FF, trainer dev rows | FF, `evaluate_child` protocol **before** | **after** |
|---|---|---|---|
| local_lexicon f>=10 | 0.0809% | 0.730% | **0.081%** ✓ |
| local_lexicon f>=3 | 0.4818% | 0.762% | **0.482%** ✓ |
| main_i7 f>=1000 | 0.0705% | 0.692% | **0.071%** ✓ |
| main_i7 f>=500 (retrained) | 0.2684% | 0.919% | **0.268%** ✓ |

Tests after the change: child **58 pass** (was 56, +2 new), parent **159 pass / 24 skipped**
(unchanged — no regression).

---

## H. Held-out comparison of the candidate configurations

Trained each candidate with the documented production recipe (`train_LightGBM.py`:
1,267 pages, 5-fold grouped CV, isotonic calibration, gates 0.10/0.65, seed 42) and served
every one — including the shipped pickle — through the identical
`scripts/compare_lexicon_configs.py` harness on the 100 held-out `Evaluation/` pages.
Numbers below are **after** the §G fix.

| config | lexicon | types | recall | flag-only | FF | score | ms/word |
|---|---|---|---|---|---|---|---|
| **shipped `detector_lgbm_conffree.pkl`** | main_i7 f>=500 | 54,057 | **92.85%** | 2.65% | 0.102% | **0.9276** | 1.48 |
| retrained local_lexicon f>=10 | local f>=10 | 52,494 | 92.24% | 4.50% | **0.081%** | 0.9217 | 1.46 |
| retrained main_i7 f>=1000 | main_i7 f>=1000 | 32,402 | 92.03% | 4.10% | **0.071%** | 0.9197 | 1.48 |
| retrained main_i7 f>=500 | main_i7 f>=500 | 54,057 | 91.00% | 4.87% | 0.268% | 0.9076 | 1.44 |
| retrained local_lexicon f>=3 | local f>=3 | 162,893 | 91.07% | 8.46% | 0.482% | 0.9063 | 1.49 |

### Verdict on the model-choice question: do not switch lexicon

1. **The shipped model is the best of the five on held-out pages** (score 0.9276). The
   in-distribution sweep that ranked it 4th of 7 (§A2) does not survive contact with the
   held-out set, so the case for switching to `local_lexicon f>=10` or `main_i7 f>=1000`
   is **not** made. Keep it.
2. **Lexicon choice barely matters.** All five configs span 1.85 recall points; three of
   them sit within 0.03 pp of each other on FF. The README's central premise — that
   main_i7 f>=500 is decisively best, with "7× better FF than local_lexicon" — is not
   supported: `local_lexicon f>=10` has *lower* FF (0.081% vs 0.102%) and
   `main_i7 f>=1000` lower still (0.071%). The differences the README magnifies are
   noise between two in-distribution estimates.
3. **`main_i7 f>=1000` (32,402 types) is the efficiency pick** if init time or RAM matters:
   same speed, lowest FF, 0.8 pp less recall.
4. **The shipped pickle is not reproducible from the documented recipe.** Retraining the
   identical configuration with the current trainer gives a *worse* model (0.9076 vs
   0.9276), because the shipped artifact is a pre-schema-2 pickle served with legacy
   imputation while a fresh one is schema-3 with NaN. `Model/train_LightGBM.py` today
   cannot regenerate `Model/detector_lgbm_conffree.pkl`. Either ship a retrained model or
   record the exact trainer revision that produced it.

### Two further detector issues this surfaced

* **The FLAG gate is far too high.** Flag-only recall is **2.65–8.46%** across every
  config: the three-way PASS/GRAY/FLAG design is effectively two-way in production, with
  ~90% of real errors sitting in GRAY and only ~3% reaching FLAG. The committed
  `Evaluation/analysis.tsv` recorded 60.4% flag-only for the same pages, so this is also a
  large behavioural drift (§A4). The trainer's own dev threshold sweep shows the flag gate
  barely moves anything (raising it 0.65 → 0.80 changes FF by ≤0.02 pp), i.e. the
  calibrated probabilities simply do not reach 0.65.
* **The README's `score = recall × (1 − FF)` is gameable.** `recall` counts GRAY as
  "caught" while `FF` counts only FLAG, so lowering the *pass* gate raises the score at
  zero measured cost. On the held-out dev pages for main_i7 f>=500: pass 0.10 → recall
  90.44%, score 0.902; pass **0.05** → recall **94.68%**, score **0.944**, with FF
  **unchanged at 0.268%**. Notably the README's recall column (94.2–94.8%) matches the
  **pass=0.05** readings, not the documented 0.10 — which is the most likely mechanical
  explanation for §A1. Any gate comparison must fix both gates, and a metric that treats
  GRAY as a catch should not be used to rank models.

---

## I. Gate tuning: a defensible PASS / FLAG pair

New tool: `4_ocr_correction/error_detection/scripts/tune_gates.py`. It collects the calibrated
P(error) for every aligned token on the 100 held-out pages once, then sweeps gates
exactly in memory.

Three methodology points, because the trainer's own `threshold_sweep` cannot answer this:

* **The four rates are separable.** `err_pass` and `clean_pass` depend only on the pass
  gate; `err_flag` and `clean_flag` only on the flag gate. So the full grid is exact from
  two 1-D passes — no sampling.
* **The README's `score` is unusable for this.** `recall` counts GRAY as caught while `FF`
  counts only FLAG, so `pass_below = 0.00` "captures" 100% of errors by passing nothing at
  all. My first attempt at an objective reproduced that same degeneracy. The fix is to
  treat the two gates as **two separate decisions with their own budgets**, never one score.
* **Selection is cross-validated over pages.** Gates are chosen on 4/5 of the pages and
  measured on the fifth, so the reported numbers come from gates that never saw those pages.

### The constraint that dominates everything: the probabilities are steppy

Isotonic calibration emits only **59 distinct values** over 62,385 tokens, and the top of
the range is thin:

| | p50 | p75 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|---|
| all tokens | 0.0028 | 0.0181 | 0.3142 | 0.3387 | 0.614 | 1.0 |
| real errors | 0.3142 | 0.3539 | 0.4159 | 0.641 | 1.0 | 1.0 |
| clean words | 0.0028 | 0.0081 | 0.2239 | 0.3142 | 0.4159 | 0.7058 |

Separation is good (AUC 0.92) but **only 0.276% of all tokens ever reach 0.65**. Many grid
points are therefore identical, and there is no gate that makes FLAG carry real volume.

### Decision 1 — the PASS gate (shipped model)

`attention` = clean words not passed. In the parent corrector both GRAY and FLAG go on to
candidate generation, so attention is real work and real damage risk. `attn precision` is
the share of the tokens the corrector would work on that really are errors — the ratio that
drives the net-harm result in §D.

| pass gate | err PASS | missed n | capture | attention | attn n | attn precision |
|---|---|---|---|---|---|---|
| 0.05 | 4.85% | 207 | **95.15%** | 17.74% | 10,308 | 28.3% |
| **0.10 (current)** | 7.15% | 305 | 92.85% | 15.87% | 9,222 | **30.1%** |
| 0.15 | 8.69% | 371 | 91.31% | 15.00% | 8,716 | 30.9% |
| 0.20 | 17.27% | 737 | 82.73% | 11.93% | 6,932 | 33.7% |
| 0.25 | 33.84% | 1,444 | 66.16% | 8.09% | 4,700 | 37.5% |

Marginal exchange rates (the band between two adjacent gates):

| move | errors gained/lost | clean words spared/added | precision of the marginal band |
|---|---|---|---|
| 0.10 → 0.05 | **+98 errors** | **−1,087 clean** (more work) | **8.3%** |
| 0.10 → 0.15 | −66 errors | +506 clean spared | 11.5% |
| 0.15 → 0.20 | −366 errors | +1,784 clean spared | 17.0% |

**Keep the pass gate at 0.10.** The band below it (0.05–0.10) is only **8.3% errors** — the
least useful real estate in the whole curve. Lowering the gate to chase the README's
documented ~94–95% recall spends 1,087 extra clean words to catch 98 more errors, and §D
shows correction is net-harmful on clean words. Raising to 0.15 is defensible only if you
weight a damaged clean word ~8× less than a missed error. Page CV agrees: **pass<0.11 was
chosen in 5/5 folds** at a 7.5% miss budget, i.e. the same plateau.

Note this also settles §A1 mechanically: the README's 94.2% recall is reachable only at
**pass 0.05** (95.15%), never at the documented 0.10 (92.85%).

### Decision 2 — the FLAG gate (shipped model)

`precision` = of the words FLAG sends to correction, the share that really were errors.
Below 50% the detector damages at least as many clean words as it fixes.

| flag gate | err FLAG | clean FLAG | n err | n clean | precision |
|---|---|---|---|---|---|
| 0.30 | 63.32% | 7.631% | 2,702 | 4,435 | 37.9% |
| 0.40 | 20.65% | 1.428% | 881 | 830 | 51.5% |
| 0.45–0.60 | 8.06% | 0.594% | 344 | 345 | 49.9% |
| **0.65 (current)** | 2.65% | 0.102% | 113 | 59 | **65.7%** |
| 0.75 | 2.58% | 0.064% | 110 | 37 | 74.8% |
| **0.85** | **2.55%** | **0.053%** | **109** | **31** | **77.9%** |
| 0.96 | 1.76% | 0.026% | 75 | 15 | 83.3% |

**Raise the flag gate 0.65 → 0.85.** It is close to a free improvement: **109 vs 113** true
errors flagged (−4) against **31 vs 59** clean words sent to a correction that damages them
(−28, a 47% reduction). Precision 65.7% → 77.9%, FF 0.102% → 0.053%. Going further to 0.96
buys precision 83.3% but gives up 38 error flags, which is a worse trade than 0.85's 4.
Page CV: flag ≥0.84 chosen in **4/5 folds** (the fifth picked 0.96); out-of-fold FLAG
precision **77.4%**, clean FLAG **0.052%**.

Note the 0.45–0.60 plateau: precision sits at **49.9%** — a coin flip. Any proposal to lower
the flag gate to make FLAG "actually fire" lands there, flagging 344 errors and 345 clean
words. That is not a usable operating region given §D.

### Cross-model robustness

The pass-gate curve has the same shape and the same plateau for every candidate
(capture ≈95% at pass ≈0.02–0.06, ≈93% at ≈0.09–0.11, ≈90–91% at ≈0.15). The flag-gate
optimum is **model-specific**, so it must be re-derived per model rather than hard-coded:

| model | FLAG precision at 0.65 | best gate at precision ≥70% | note |
|---|---|---|---|
| shipped conffree | 65.7% | **0.85** (77.9%) | 0.65 is on the edge of the plateau |
| local_lexicon f>=10 | 80.3% | **0.65** (80.3%) | already optimal; 0.85 loses 93 error flags |
| main_i7 f>=1000 | 81.0% | **0.60–0.65** (81.0%) | already optimal; 0.70 halves coverage |

So: **pass 0.10 for any of these models; flag 0.85 for the shipped one only.**

### The honest limit

Gate tuning cannot rescue the three-way design. At any gate with usable precision (≥65%),
FLAG covers only **2.3–4.5% of real errors** across all three models; reaching 8% requires
dropping to 0.45–0.55 where precision is ~50%. The probability mass simply is not there —
59 distinct isotonic values, 0.276% of tokens above 0.65. The realistic options are:

1. **Accept a small, high-precision FLAG band** (what the numbers support) and let GRAY plus
   the corrector's do-no-harm margin carry the work — which is what the parent already does,
   since it generates candidates for GRAY too.
2. **Collapse to two verdicts** (PASS / SUSPECT) and stop implying FLAG drives correction.
3. **Retrain with better top-end calibration** if FLAG is meant to carry real volume —
   e.g. calibrate on the held-out pages rather than out-of-fold, or drop isotonic for a
   smoother calibrator. This is the only route to a genuinely wider FLAG band.

Whichever is chosen, the README must stop describing 0.65 as producing meaningful FLAG
volume, and `Evaluation/analysis.tsv` (60.4% flag-only) must be retired: no model in this
repo reproduces it.

### Applying it

The payload's `pass_thresh` / `flag_thresh` used to win unconditionally, which made the
gates impossible to retune without rewriting the pickle. `DetectionEngine` now takes
`gates_override_model=True` to make caller-supplied gates authoritative:

```python
from error_detection import DetectionEngine
det = DetectionEngine(pass_below=0.10, flag_above=0.85, gates_override_model=True)
```

It is opt-in on purpose: `AmharicOCRCorrector` forwards its own concrete defaults
(`suspicion_pass_below=0.30`, `suspicion_flag_above=0.70`) through the adapter, so
honouring caller values unconditionally would silently move the parent's Stage-1 operating
point. Default behaviour is unchanged and the parent suite still passes (159 / 24 skipped).

---

## J. Recommended fixes, in priority order

Detector-scoped items first; parent-pipeline items are marked **[parent]** and were
deliberately left alone.

### Done in this pass

1. ✅ **Fix the `has_digit` train/serve skew** (§G) — `features.py` now measures it on the
   clean surface. Held-out FF 0.72% → 0.10%, recall unchanged, bit-identical on all
   training rows, no retraining. Tests updated so the parity test can no longer excuse the
   column that was broken.
2. ✅ **Raise on a nonexistent caller-supplied `dict_vocab_path`** (§C3) instead of
   silently degrading to an 8-word lexicon. The error names the real filename, since the
   child README documents a typo'd one (`main_i7_freq10.txt` vs `main_i7_frq10.txt`).
3. ✅ **`evaluate_child.py` can now evaluate any model** — `--model`, `--lexicon`,
   `--min-freq`. It previously always auto-resolved the shipped pickle, so no alternative
   configuration could be measured with the documented gate script.
4. ✅ **`scripts/compare_lexicon_configs.py`** (new) — serves every candidate and the
   shipped model through one identical held-out harness. This is what produced §H.
5. ✅ **`train_LightGBM.py` no longer mislabels provenance** — the saved `note` said
   "conf-free main_i7 …" even when trained from `local_lexicon.txt`.

### Next, in order

6. ✅ **Gate tuning done (§I)** — `scripts/tune_gates.py`, exact separable sweep plus
   page-level CV. Recommendation: **keep pass 0.10, raise flag 0.65 → 0.85** for the
   shipped model (109 vs 113 error flags, 31 vs 59 clean words damaged, precision
   65.7% → 77.9%). `gates_override_model=True` now makes that applicable without
   rewriting the pickle. Still open: the **three-way design itself** — FLAG covers only
   2.3–4.5% of errors at usable precision, so either accept a small high-precision band,
   collapse to two verdicts, or retrain the top-end calibration (§I, "The honest limit").
   That is a product decision, not a tuning one.
7. **Re-run the frequency sweep and retire the README table.** Post-fix, the sweep numbers
   in `Model/train_sweep_report.json` and both READMEs are stale. Regenerate with
   `--sweep-freqs`, then replace the 94.2% / 0.01% / 0.942 row (and the derived "7× /
   85× / 100× better", "1 in 10,000" claims in `README.md`,
   `4_ocr_correction/error_detection/README.md`, `docs/architecture.md`, `docs/components_guide.md`,
   `docs/project_structure.md`) with the held-out §H figures. **Keep main_i7 f>=500** —
   §H shows it wins on held-out data; only the justification needs rewriting.
8. **Fix or replace the `score` metric** (§H). Ranking models by `recall(flag+gray) ×
   (1 − FF(flag))` is gameable through the pass gate, and the README's recall column
   appears to have been measured at pass=0.05 while documenting 0.10. Report flag-only
   recall alongside, and always state both gates with any score.
9. **Make the shipped model reproducible.** `Model/train_LightGBM.py` cannot regenerate
   `Model/detector_lgbm_conffree.pkl` (retraining the same config gives a worse model,
   §H.4). Either ship a retrained schema-3 model with its report, or pin the trainer
   revision that produced the current one.
10. **Fix the child "How to use" example** (§B1) — capture real output. `verdict("በሸታ",
    doc_freq=2)` returns `gray 0.154`, not `FLAG 0.85`; labels are lowercase; and
    `docs/components_guide.md`'s printed feature vector yields P=0.3142, not 0.85.
11. **Refresh or retire `Evaluation/analysis.tsv`** so there is one authoritative record
    per evaluation set (§A4 — it currently disagrees with both the README and a fresh
    run).
12. **Remove the duplicate lexicon.** `4_ocr_correction/error_detection/main_i7_frq10.txt` is
    byte-identical (md5 `df11e6cb…`) to `4_ocr_correction/error_detection/Model/main_i7_frq10.txt` —
    15 MB tracked twice, contradicting the README's "no duplicates, single source". Only
    `detector.py`'s `CANDIDATE_LEXICON_PATHS` fallback references it, after the `Model/`
    entry, so deleting it is safe.
13. Sweep the minor inaccuracies in §E that are child-scoped (test count 40 → 58, the
    `በሸታ 0` / `በሽታ 1234` example dictionary lines, "top 150k" → 162,893, "default 50k
    types freq>=10" → 54,057 at freq>=500).

### Out of scope for this pass — parent pipeline

* **[parent]** Both README "How to use" corrector examples raise `ValueError`: no KenLM
  model exists in the repo (§B2).
* **[parent]** `requirements.txt` omits `pydantic`, so the vendored segmenter stays
  unimportable and `lambda_morph` silently cancels out (§C1).
* **[parent]** `os.makedirs` on hardcoded `/media/des` paths instead of
  `repo_paths.makedirs()` in `01_evaluate_full_corrector.py`, `02_benchmark_new_model.py`,
  `07_generate_eval_pages.py` (§C2).
* **[parent]** The committed stage-5 benchmark shows net harm (§D).

## Reproduction commands

```bash
python3 -m venv .venv
.venv/bin/pip install numpy joblib lightgbm scikit-learn symspellpy==6.10.0 pandas pytest

# A1/A2 — the README dictionary table (deterministic; matches Model/train_sweep_report.json)
.venv/bin/python 4_ocr_correction/error_detection/Model/train_LightGBM.py \
    --sweep-freqs 10,100,500,1000 --report /tmp/sweep.json
.venv/bin/python 4_ocr_correction/error_detection/Model/train_LightGBM.py \
    --lexicon 4_ocr_correction/error_detection/Model/local_lexicon.txt \
    --sweep-freqs 1,3,10 --report /tmp/sweep_local.json

# A3 — held-out 100-page evaluation with the documented gate script
.venv/bin/python 4_ocr_correction/error_detection/scripts/evaluate_child.py

# G — the has_digit skew, in one line
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from error_detection import DetectionEngine as D
e=D(lgbm_model_path='/tmp/det_cmp/loc_f10.pkl',
    dict_vocab_path='4_ocr_correction/error_detection/Model/local_lexicon.txt', dict_min_freq=10)
print([ (w, round(e.verdict(w,doc_freq=1).suspicion,4), e.verdict(w,doc_freq=1).label) for w in ('ከ','ከ20') ])"

# H — train the candidates, then compare them all on the held-out pages
for cfg in "main_i7_frq10.txt 500 m7_f500" "main_i7_frq10.txt 1000 m7_f1000" \
           "local_lexicon.txt 10 loc_f10" "local_lexicon.txt 3 loc_f3"; do
  set -- $cfg
  .venv/bin/python 4_ocr_correction/error_detection/Model/train_LightGBM.py \
      --lexicon "4_ocr_correction/error_detection/Model/$1" --min-freq "$2" \
      --out "/tmp/det_cmp/$3.pkl" --report "/tmp/det_cmp/$3.report.json"
done
.venv/bin/python 4_ocr_correction/error_detection/scripts/compare_lexicon_configs.py \
    --json-report /tmp/det_cmp/heldout.json

# I — gate tuning on the held-out pages (cache the probabilities, then sweep for free)
.venv/bin/python 4_ocr_correction/error_detection/scripts/tune_gates.py \
    --dump-probs /tmp/probs_shipped.npz --json-report /tmp/gates_shipped.json
.venv/bin/python 4_ocr_correction/error_detection/scripts/tune_gates.py \
    --load-probs /tmp/probs_shipped.npz \
    --miss-budget 7.5 --ff-budget 0.06 --min-precision 75 --folds 5
# per-model, served with the lexicon it was trained on
.venv/bin/python 4_ocr_correction/error_detection/scripts/tune_gates.py \
    --model /tmp/det_cmp/m7_f1000.pkl \
    --lexicon 4_ocr_correction/error_detection/Model/main_i7_frq10.txt --min-freq 1000

# B1 — the child "How to use" example
.venv/bin/python -c "from error_detection import DetectionEngine; d=DetectionEngine(); r=d.verdict('በሸታ', doc_freq=2); print(r.label, r.suspicion)"

# F/G — test suites (child 58, parent 159 + 24 skipped)
.venv/bin/python -m pytest 4_ocr_correction/error_detection/tests/ -q
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

