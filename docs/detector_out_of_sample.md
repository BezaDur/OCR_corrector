# Detector, out of sample — what the shipped model actually does

**Measured:** 2026-10-25, branch `arena/c4c81d74-ocr-corrector`.
**Artifact:** `OCR_error_detection/Model/detector_lgbm_conffree.pkl` (unmodified).
**Machine:** the verification sandbox — Python 3.11.2, lightgbm 4.7.0,
scikit-learn 1.9.1. No Tesseract, no KenLM (neither is needed).

This document answers one question: **is the headline "recall 94.2 %, false flag
0.01 %" a performance claim or a selection claim?** It is a selection claim. When
the shipped model is scored on data it was not fit on, it lands about two points
lower on recall and about ten times higher on false flags — and every number in
the README's table comes from the same 67-page slice that was used to *choose*
the model.

Everything below is reproducible from files in this repository; the two commands
are in §8.

---

## 1. The answer

|  | README claim (selection, 67 pages) | Document-grouped CV, out-of-fold, uncalibrated (51 documents, 883,578 rows) | Shipped model, 100 dev pages (62,479 rows) |
|---|---:|---:|---:|
| recall (flag + gray) | **94.2 %** | **91.8 %** (calibrated 91.3 %) | **92.4 %** |
| flag-only recall | — | 5.6 % | 2.7 % |
| false flag on clean words | **0.01 %** | **0.116 %** (calibrated 0.030 %) | **0.102 %** |
| AUC | 0.9242 | 0.9187 (calibrated 0.9193) | 0.9232 |
| AP | — | 0.4025 (calibrated 0.3912) | 0.4117 |
| gray on clean words | — | 16.5 % | 15.8 % |
| score = recall × (1 − FF) | 0.942 | 0.917 (calibrated 0.912) | 0.923 |
| calibration (ECE) | — | — | 0.0050 |

Three independent estimates of the same quantity, and they agree with each other
to within a point (the gates are applied to calibrated probabilities, so the
calibrated column is the comparable one):

1. **Document-grouped 5-fold CV** on the 1,267 production pages. Folds are whole
   documents (`--group-by document`, 51 groups), so no fold contains a page that
   continues a page in another fold. Out-of-fold, calibrated: recall 91.27 %,
   FF 0.0305 %.
2. **A freshly retrained model on the 100 dev pages**: recall 90.44 %,
   FF 0.2684 % (train-report JSON: `docs/out_of_sample_doccv.json`).
3. **The shipped pickle on the same 100 dev pages**: recall 92.39 %,
   FF 0.102 %, AUC 0.9232.

The first column used to be a **text-grouped** CV — connected components of shared
5-grams. That grouping does not work on this corpus: the pages are overlapping
windows of 55 documents, so every 5-gram occurs on 10–40 pages and the
`max_page_occurrences=5` rule throws away exactly the evidence that would link
them. Measured: 1,095 groups at cutoff 5 (1,074 singletons), 798 at 10, 48 at 20,
19 at 50 — there is no cutoff that recovers the 55 documents. The old number
(91.65 %/0.099 %) was therefore close to page-level CV. Grouping by document
instead, the result survives: 91.27 %/0.031 %.

The honest expectation for this model on new synthetic-scan pages of the same
kind is therefore **≈ 91–92 % recall (flag + gray) at ≈ 0.1 % false flags**, i.e.
about 10× the advertised false-flag rate. The ranking quality is not the
problem — AUC is 0.922–0.923 in all three settings — the advertised *operating
point* is.

Selection-vs-test is the whole story: 94.2 %/0.01 % was read off the 67
validation pages that also chose the dictionary filter, the round count and the
gates.

---

## 2. How the evaluation is set up

`Model/train_LightGBM.py` holds pages `page_0001`…`page_0100` out of the fit
(`DEV_PAGES = 100`) and trains on the remaining 1,267. Those 100 pages are the
evaluation set here. They are genuinely unseen by the model — and, as §3 shows,
also unseen *as text*.

Two details that matter for reading the numbers:

* **What counts as an error.** `is_error` in
  `Model/harmonized_corpus_jiwer.tsv` is binary over 946,057 rows: 884,976 clean
  (`MATCH`) and 61,081 errors, of which 59,992 are substitutions and **1,089 are
  insertions** (extra tokens the OCR produced; they have no `gt_word`). In the
  dev pages: 4,361 errors, 94 of them insertions (2.2 %). A word-level detector
  is not the right tool for insertions, and they are inside the recall
  denominator — the substitution-only recall is marginally higher than quoted.
* **The gaps are real.** Pages share no 5-grams across the dev/production
  boundary in a way that would let the model memorise them (§3), so the dev
  number is not flattered by leakage.

---

## 3. The dev pages are genuinely new text

This corrects an error in the earlier documentation (and in an earlier version of
`docs/current_state.md`). `docs/lgbm_step2_gate.md` §C6 reported *"eval 5-grams
found in training GT: 61,613/61,613 (100.0 %)"* and concluded that "the 100
evaluation pages are the training pages". **That check compared the dev pages
against a training set that contained the dev pages**, so it could only ever
return 100 %. Run against the pages the model actually trains on, the numbers are:

| Comparison | Result |
|---|---|
| dev GT lines (4+ words) verbatim in **production-page** GT | **259 / 4,148 = 6.2 %** |
| dev GT lines (4+ words) verbatim in the OCR training text (`amh_final_training_text.txt`) | **19 / 4,148 = 0.5 %** |
| dev word 5-grams also occurring on a production page | 516 / 4,127 = 12.5 % |
| dev pages with **no** shared 5-gram at all | **76 / 100** |
| dev pages whose 5-grams are ≥ 90 % present on production pages | 4 / 100 (`page_0096`–`page_0100`) |

The overlap is concentrated in a tail: 20 pages share between 0.1 % and 60 % of
their 5-grams with the training pages, and 4 pages are near-duplicates
(90.6 %–96.5 %). That tail is not boilerplate — see §3b: the split at page 100
cuts through the middle of a document, so the pages nearest the cut are literally
the same text as the first training pages.

Splitting the evaluation by that overlap gives the reassuring result:

| Tier | pages | rows | errors | AUC | recall | false flag | gray on clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| **text-disjoint** (0 %) | 76 | 47,291 | 3,191 | 0.9231 | **92.86 %** | 0.134 % | 16.35 % |
| partial (0–90 %) | 20 | 13,265 | 926 | 0.9252 | 91.25 % | 0.000 % | 13.57 % |
| near-duplicate (≥ 90 %) | 4 | 1,923 | 244 | 0.9189 | 90.57 % | 0.000 % | 16.56 % |
| all dev pages | 100 | 62,479 | 4,361 | 0.9232 | 92.39 % | 0.102 % | 15.77 % |

**Seeing the text does not help.** The model performs the same on pages whose
vocabulary it has never encountered as on pages it has partly seen. That is the
strongest thing that can be said about this detector, and it was not established
before: the earlier reviews could only call the numbers "in-sample" because the
model in circulation at the time had been fit on all 1,367 pages.

**A caveat that matters more than the table.** "New text" is true page by page
and misleading as a whole: the 100 dev pages are overlapping windows of **5
documents**, and they judge only about **4,100 distinct source words**. §3b
quantifies this.

---

## 3b. How much evidence is behind 62,479 rows

The corpus is not 1,367 independent pages. It is **55 documents cut into
overlapping windows**: the opening words of page *i+1* are normally found 10–60
words into page *i*, and only 54 of the 1,366 page transitions start new text.

| measure | value |
|---|---|
| pages / documents | 1,367 pages, **55 documents** |
| rows / distinct 5-grams | 944,968 rows, **105,596 distinct 5-grams** (89 % of rows repeat text already present) |
| dev pages / documents they cover | 100 pages, **5 documents** |
| dev rows / distinct source words | 62,385 rows, **~4,137 distinct words** (6.6 % of rows) |
| dev rows that are new 5-word OCR contexts | 18,541 of 62,479 (29.7 %) |
| recall on just those new contexts | 92.62 % ± 0.83 pp (n = 3,849 errors) |

Two honest readings of that:

* **The row-level number is stable.** Restricting to first-seen OCR contexts
  (29.7 % of the data) gives 92.62 % ± 0.83 pp, against 92.39 % ± 0.79 pp on all
  rows. The 62k rows are not a statistical mirage: the same words recur, but
  they recur on *different crops*, and Tesseract does not always make the same
  mistake twice.
* **The coverage is narrow.** Everything above is measured on ~4,100 word
  positions drawn from 5 news documents (Ethiopia/Eritrea dam coverage, Ukraine
  war reporting), one layout, one font set, one Tesseract build. The detector has
  not been shown to work on a *different kind of page* — only on new text of the
  same kind.

That is why §7 puts "one corpus, one OCR engine" first, and why the honest way to
move this forward is a new test set (`docs/new_test_set_plan.md`), not a tighter
confidence interval on the current one.

---

## 4. The probability is honest

The gates (`pass < 0.10 ≤ gray < 0.65 ≤ flag`) only mean something if
`P(error)` is calibrated on new data. It is: isotonic calibration was fit on
out-of-fold scores over the 1,267 pages and transfers to the dev pages.

| decile of P(error) | rows | mean P | observed error rate |
|---|---:|---:|---:|
| 1 | 6,248 | 0.0000 | 0.0005 |
| 2 | 6,248 | 0.0009 | 0.0037 |
| 3 | 6,248 | 0.0013 | 0.0018 |
| 4 | 6,248 | 0.0017 | 0.0026 |
| 5 | 6,248 | 0.0026 | 0.0038 |
| 6 | 6,248 | 0.0034 | 0.0048 |
| 7 | 6,248 | 0.0057 | 0.0038 |
| 8 | 6,248 | 0.0359 | 0.0503 |
| 9 | 6,248 | 0.2163 | 0.2297 |
| 10 | 6,247 | 0.3837 | 0.3972 |

**ECE = 0.0050**, and every decile is within a factor of ~1.5 of its predicted
rate. The one systematic bias is in the safest deciles, where the model is
*under*-confident (observed 3–4× the predicted value, at absolute rates below
0.4 %). It matters for threshold choices at the very low end, not for the gates
in use.

---

## 5. Where the remaining errors are

The single recall number hides the structure of the problem completely:

| lexicon state | rows | errors | error rate | recall | flag-only | false flag | gray on clean | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| out of lexicon | 13,341 | 3,904 | 29.3 % | **98.13 %** | 1.84 % | 0.170 % | **91.97 %** | 0.659 |
| **in lexicon (real-word errors)** | 49,138 | 457 | 0.9 % | **43.33 %** | 10.07 % | 0.088 % | 0.99 % | 0.856 |

* **Out-of-lexicon errors are effectively solved** — 98 % caught. But 92 % of
  *clean* out-of-lexicon words are also `gray`, because a name or a rare
  inflected form cannot be told apart from a corruption by lexical evidence
  alone. That is where the corrector's compute goes, and where over-correction
  risk lives.
* **Real-word errors are the open problem** — 457 of the 4,361 dev errors
  (10.5 %) are wrong words that spell a valid Amharic word (`ተከስተው → ተከሰተው`,
  `አውሮፕላን → አውሮኘላን`). Only 43 % are caught. They are also 78 % of the 332
  words the detector passes while wrong.
* Curiously, when the model *does* reach `flag`, it is 5× more likely to be on an
  in-lexicon error (10.07 %) than on an OOV one (1.84 %). Flag is doing
  real-word-error work; gray is doing OOV triage.

So the ceiling is not "more signal from the dictionary" — it is context. A
detector built on `in_lexicon` + frequency + surface shape cannot see that
`ስላም` is wrong when `ሰላም` is meant, because both are words. That needs the
KenLM/context features that the model currently does not use, which is a
modelling decision, not a bug.

---

## 5b. Which features do the work (fresh measurement)

The retrain in §1 (2) reports gain per feature, and it confirms two claims the
README makes without evidence:

| feature | gain | splits |
|---|---:|---:|
| `log_corpus_freq` | 1,019,671 | 4,857 |
| `in_lexicon` | 291,872 | 311 |
| `is_high_freq` | 151,045 | 169 |
| `token_len` | 103,972 | 2,859 |
| `doc_freq` | 38,991 | 2,851 |
| `is_doc_singleton` | 35,752 | 432 |
| `one_flip_count` | 32,683 | 1,290 |
| `rare_glyph_ratio` | 31,137 | 1,091 |
| `has_one_flip` | 23,849 | 904 |
| `has_digit` | 19,982 | 140 |
| `len_ratio` | 11,125 | 799 |
| **`char_kenlm_score`** | **0** | 0 |
| **`n_neighbors`** | **0** | 0 |
| **`mixed_script`** | **0** | 0 |

* **Lexical features carry the model**: `log_corpus_freq` alone is 50× the gain
  of the next feature, and it explains both why out-of-lexicon errors are easy
  (gain near zero means the word is wrong) and why real-word errors are hard
  (both spellings are frequent).
* **Three of the 14 columns are dead weight** — `char_kenlm_score` (no model was
  available), `n_neighbors` (SymSpell off) and `mixed_script`. They can be
  dropped without a loss, which simplifies the feature contract and the training
  data requirement. Conversely, the dead `char_kenlm_score` column is the
  obvious place to add the context signal that §5 says is missing.

---

## 6. What this changes in the repository

* `docs/lgbm_step2_gate.md` §C6 and its "Verdict" item 1 said the evaluation
  pages are the training pages. **That conclusion is withdrawn** — see §3 for
  the corrected measurement and the reason the original check could not fail.
* `docs/lgbm_detector_review.md` called the results in-sample. That was true of
  the model it reviewed (fit on all 1,367 pages); the shipped model is not.
* `docs/current_state.md` §5.2 and both READMEs now quote the numbers in §1
  instead of the selection numbers.
* `OCR_error_detection/Evaluation/analysis.tsv` remains **stale**: it predates
  the shipped model. The current engine agrees with only 85.6 % of its stored
  verdicts. The numbers in this document are computed from the corpus, not from
  that file.

---

## 7. Limits — what is still not measured

1. **One corpus, one OCR engine.** All 946,057 rows come from the same synthetic
   scan pipeline (11 fonts, three degradation levels) and the same Tesseract
   build. Nothing here says how the detector behaves on real archive scans,
   different paper, handwriting-adjacent fonts, or a different OCR engine.
2. **The dictionary is external.** `main_i7_frq10.txt` is a general Amharic
   lexicon from a much larger corpus, so its entries were never "trained on" the
   dev pages — that is realistic for deployment, but it means this is *not* a
   test on genuinely new vocabulary. A proper test needs a lexicon built without
   the test documents, or the non-dictionary synthetic set
   (`docs/new_test_set_plan.md`), which was built but has no recorded results.
3. **Insertions are in the denominator.** 2.2 % of dev errors are extra tokens
   (no `gt_word`); a word-level detector cannot fix them and arguably should not
   be scored on them.
4. **The thresholds are inherited.** 0.10 / 0.65 came from the same selection
   data. §5 of the JSON report contains a sweep on the dev pages; choosing gates
   from it would overfit the dev set, so don't. A third split is needed.
5. **Retraining is not bit-reproducible.** The retraining in §1 (2) reproduced
   the recipe but not the artifact: same features and folds, recall 90.4 % /
   FF 0.27 % versus the shipped 92.4 % / 0.10 %. The spread between those two is
   the run-to-run noise to expect when the model is retrained.
6. **Five documents of coverage.** §3b: the dev pages are overlapping windows of
   5 news documents (~4,100 distinct words). Nothing here shows the detector
   works on a different page type — a form, a table, a book page, a scan from a
   different collection. The per-row statistics are stable; the *scope* is one
   document family.

---

## 8. Reproducing

```bash
pip install -r requirements.txt pytest

# 1. Shipped model on the 100 development pages (~35 s), writes the JSON report
python OCR_error_detection/scripts/evaluate_out_of_sample.py \
    --report docs/out_of_sample_shipped.json

# 2. Document-grouped cross-validation over the 1,267 training pages (~2 min)
python OCR_error_detection/Model/train_LightGBM.py --no-char-lm --group-by document \
    --out /tmp/model.pkl --report docs/out_of_sample_doccv.json

# 3. Check every claim in this document against the data (~3 min, exits 1 on a miss)
python OCR_error_detection/scripts/verify_claims.py

# 4. Score a new evaluation folder (see docs/verify_claims.md)
python OCR_error_detection/scripts/make_eval_corpus.py --eval-dir fresh_eval --out /tmp/fresh.tsv
python OCR_error_detection/scripts/evaluate_out_of_sample.py --corpus /tmp/fresh.tsv --eval-all
```

`evaluate_out_of_sample.py` prints the tables in §1, §3, §4 and §5 and writes the
machine-readable version. It never writes to the repository except the report
path you pass.

---

## 9. Files

| File | Contents |
|---|---|
| `OCR_error_detection/scripts/evaluate_out_of_sample.py` | the evaluation script (§8) |
| `docs/out_of_sample_shipped.json` | shipped model, all tables, per-page overlap, threshold sweep |
| `docs/out_of_sample_doccv.json` | document-grouped CV report (51 documents, OOF + dev metrics, feature importance) |
| `OCR_error_detection/scripts/verify_claims.py` | re-derives every number here and tries to break it |
| `OCR_error_detection/scripts/make_eval_corpus.py` | builds a scorable corpus from a new `Evaluation/`-style folder |
| `docs/verify_claims.md` | how to check this without trusting it, including on new data |
| `OCR_error_detection/Model/train_LightGBM.py` | the trainer; `--group-by document` is the CV protocol used here |
