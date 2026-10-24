# LightGBM detector: integration review

Date: 2026-10-13. Branch: `arena/3ca31a9a-ocr-corrector`.

> **Status (2026-10-24): most of the "Bottom line" below no longer describes the code.**
>
> * "Not used by the corrector" — fixed. `CorrectorDetectionAdapter` now builds
>   `DetectionEngine` and delegates every verdict to it; `SuspicionScorer` is only
>   the fallback when the engine cannot be constructed.
> * "Verdict logic is patched by hard floors" — fixed. The floors were removed;
>   the calibrated payload thresholds (0.10 / 0.65) decide.
> * "Cannot be imported from the repo root" — fixed. `from OCR_error_detection
>   import DetectionEngine` works from anywhere; the imports are absolute.
> * "No feature-name guard" — fixed. `DetectionEngine.__init__` raises
>   `FeatureError` for a payload that asks for a column this code cannot compute.
> * "`doc_freq` default" — fixed. Missing context is NaN, never a magic number.
> * "Adapter bypasses the model" — fixed, as above.
>
> Still true, and still the important caveat: **the text of the 100 evaluation
> pages is inside the training corpus** (100 % 5-gram overlap; the pages
> themselves are held out via `DEV_PAGES = 100`), so §2–§3 measure seen
> vocabulary, not generalisation; and the stored `Evaluation/analysis.tsv`
> predates the shipped model (the current engine agrees with it on 85.6 % of a
> resampled 1,500 words). For what is true now see
> `docs/current_state.md` §2 and §5.2. The recommendations in §5 remain a good
> checklist.

> **Update.** Confidence is excluded from training (decided after this review). The
> recommendation to add confidence as a feature (sections 4.1 and 5, item 4) is withdrawn.
> The plain-language account of step 1, including the dictionary and confidence decisions,
> is in `docs/lgbm_step1_changes.md`.

Scope: the LightGBM model in `OCR_error_detection/` (`Model/detector_lgbm.pkl`,
`Model/train_LightGBM.py`, `detector.py`) and how it is used. The request said
"LightBLM"; I read this as LightGBM, the only gradient-boosting model in the repo.

Method: static reading of the code, loading the pickle, probing the inference path
on a small lexicon, and analysis of the saved `Evaluation/analysis.tsv` (100 pages,
62,385 aligned substitution and correct rows). I did not retrain. The production
training inputs (`/mnt/mintdata/...` master dictionary and KenLM model) are not in
this sandbox. Scripts used for the numbers are in `docs/lgbm_review/`.

## Bottom line

1. **The model is not used by the corrector.** It runs only inside
   `OCR_error_detection.DetectionEngine`. The corrector (`amharic_ocr_corrector.py`)
   runs `suspicion.SuspicionScorer` with hand-tuned weights, and nothing outside
   `OCR_error_detection/` references the model. Today the integration does not affect
   correction output.
2. **The reported results are in-sample.** The 100 evaluation pages are part of the
   training corpus, token for token. The final model is fit on all of it.
3. **The model is trained without its most useful signal.** OCR confidence is a
   placeholder (75.0) on every training row, and the model has no confidence feature.
   On a page-grouped check, confidence adds about 0.13 AUC.
4. **There are train/serve defects.** The one-flip feature is always 0 in training. At
   inference, a normalization mismatch makes correct words look out-of-lexicon, and a
   `doc_freq` default makes single-word verdicts depend on the call.
5. **The verdict logic is patched by hard floors.** The floors override the model in
   thousands of cases, and one floor is only 56% precise.

What does work: the model ranks errors well (AUC 0.95 on the eval set, in-sample; 0.89
for OCR confidence alone), it loads and runs at about 0.09 ms per word, and the
per-page `doc_freq` used in `analyze_methods.py` matches the training definition.

## 1. Integration status

| Path | Uses LightGBM? | Notes |
|---|---|---|
| `OCR_error_detection.DetectionEngine.verdict` | yes (primary) | floors applied after the model |
| `OCR_error_detection.adapter.CorrectorDetectionAdapter` | no | rule-based `combine_signals`, same name, different logic |
| `suspicion.SuspicionScorer` (corrector Stage 1) | no | `DEFAULT_WEIGHTS` hand-tuned |
| `detection/` (`ErrorDetector`) | no | separate logistic model (JSON) |

The `adapter.py` docstring says it "lets the main corrector use this detection module",
but it bypasses the model. The `DetectionEngine` path also cannot be imported from the
repo root (see 4.10), so it is only usable from scripts run inside the package folder.

**Decision needed:** which detector should be authoritative for the corrector? Options:
(a) make `SuspicionScorer` read the LightGBM payload; (b) keep the rules and treat
LightGBM as an offline analysis tool; (c) drop one of the two.

## 2. Evaluation caveats (in-sample)

- `harmonized_corpus_jiwer.tsv` has 1,367 pages, including `page_0001`–`page_0100`.
  Their OCR token sequences are identical to `Evaluation/OCRed/*.ocr.txt` (100/100 pages).
- `train_LightGBM.py` fits the production model on all rows
  (`lgb.train(params, final_trn, num_boost_round=400)`).
- Therefore `analysis.tsv` scores the model on data it was trained on. The gap between
  in-sample and out-of-sample numbers is not known, because the training run's
  out-of-fold metrics were not saved and the training inputs are missing here.

Recommendation: exclude the evaluation pages from production training, and report the
out-of-fold metrics from `GroupKFold` (they are printed, not saved).

## 3. Results as recorded (in-sample, for orientation)

Aligned rows: 62,385 (4,267 substitution errors, 58,118 correct).

| Metric | Value |
|---|---|
| AUC, LightGBM score | 0.954 |
| AUC, `-tesseract_conf` alone | 0.890 |
| AUC, `-in_lexicon` alone | 0.713 |
| Errors caught (flag + gray) | 96.5% |
| Flag precision (post-floor) | 55.8% (2,721 TP / 2,154 FP) |
| Correct words sent to gray | 22.1% (12,842) |

Threshold sweep on the raw score (floors excluded):

| pass gate | errors not passed | clean flagged | flag precision |
|---|---|---|---|
| 0.15 (current) | 93.1% | 12.4% | 0.36 |
| 0.30 | 84.4% | 7.4% | 0.46 |
| 0.60 (current flag gate) | 56.1% | 2.5% | 0.63 |

Note: the current pass gate of 0.15 sends about one in five correct words to gray.

## 4. Findings

### 4.1 Training never saw real OCR confidence (high)

`tesseract_conf` is `75.0` in all 946,057 rows of `harmonized_corpus_jiwer.tsv`
(every page). The eval run uses real per-word confidences from `*.conf.tsv`, so the two
data sources disagree. The model has no confidence feature, but the inference floors
use confidence, so the floors are the only place the model sees it.

Page-grouped 5-fold CV on the eval rows, using features available at inference (no
detector score, no KenLM, no master dictionary):

| Feature set | LightGBM AUC | LightGBM AP |
|---|---|---|
| lexicon + surface + doc_freq + one-flip (no confidence) | 0.798 | 0.465 |
| lexicon + confidence + surface + doc_freq (no one-flip) | 0.927 | 0.645 |

Confidence is the largest lever available. This CV uses smaller data and fewer
features than production, so it shows relative value, not the production AUC.

### 4.2 The one-flip feature is dead in training (high)

`train_LightGBM.py::load_confusion_map` reads `data.get("substitutions")`, but
`confusion_matrix.json` stores substitutions under `"sub"`. The map is empty, so
`one_flip_count` and `has_one_flip` are 0 for every row. The pickle agrees: both
features have zero gain.

The inference side builds the one-flip set correctly from `"sub"`, so the model
ignores the signal in training and sees it at inference. Its usefulness is also
doubtful: on the eval data, an out-of-lexicon word with a one-flip hit is an error 24%
of the time, versus 57% without one. The one-flip direction is inverted for OOV words.
The same concern applies to the corrector's hard FLAG rule and +0.85 weight in
`suspicion.py`, which read this signal the same way.

### 4.3 Normalization mismatch at inference flags correct words (high)

`DetectionEngine` builds `self.vocab` with normalized keys (for example `ዓ`→`አ`, `ሐ`→`ሀ`).
`_extract_lgbm_features` looks up the raw token:

```python
freq = self.vocab.get(clean_w, self.vocab.get(tok, 0))
```

So words with variant glyphs are seen as out-of-lexicon by the model, although the
cheap signals (which normalize) say they are in the lexicon. Probe results:

| word | cheap `in_lexicon` | LightGBM `in_lexicon` feature |
|---|---|---|
| `ሀገር` (no variant) | True | 1 |
| `ዓለም` | True | 0 |
| `ሐገር` | True | 0 |

Impact on the eval set (`norm_bug.py`): clean, in-lexicon words with an H-family
variant and no rare glyph are not `pass` 78.2% of the time (n = 1,055). The comparable
rate for plain in-lexicon clean words is 18.4%. These account for 369 of the 2,154
false flags. Examples: `የሕዳሴው` (correct, in lexicon) is flagged at 0.775; `የሕግ` and
`የውሃ` go to gray.

Training used raw `ocr_word` for the same lookups, so training and inference disagree.

### 4.4 `doc_freq` defaults to 1, which means "singleton" (high for API use)

`verdict(tok, ..., doc_freq=1)` makes every standalone word a document singleton.
Singletons are much more likely to be errors in training, so the verdict depends on
the call:

| call | p(error) | label |
|---|---|---|
| `verdict("ቤት")` (default `doc_freq=1`) | 0.178 | gray |
| `verdict("ቤት", doc_freq=5)` | 0.038 | pass |

Within `analyze_methods.py` the per-page count is passed, so the evaluation is
consistent. Any other caller (including the README example) gets the default.

### 4.5 No-KenLM operation is off-distribution (high for deployment)

`char_kenlm_score` is the second-highest-gain feature. `DEFAULT_CHAR_LM_PATH` points to
`/mnt/mintdata/...`. Without the file, every word gets the constant `-2.5`, which is
not a value the model saw as a typical score. The unit test
`test_passes_all_clean_words` fails because of this together with the `doc_freq`
default in 4.4 (`ቤት` → gray, p=0.178; with `doc_freq=5` it passes). Either
ship a model variant trained without KenLM, or make the missing feature explicit (NaN
with a trained missing-value branch) and document the degraded mode.

Possible train/serve skew (not verified, because KenLM is not in the sandbox): training
scores the raw OCR word (`kenlm_cache[w]`), while `_score_char_lm` normalizes and
strips punctuation first.

### 4.6 Calibration and thresholds (medium)

- `scale_pos_weight=2.0` inflates the positive class. The comment "Calibrated
  probability P(error)" is therefore not accurate. The score is a ranking, not a
  probability.
- `PASS_THRESH = 0.15` and `FLAG_THRESH = 0.60` are hard-coded, not chosen from
  out-of-fold results. The sweep above shows the trade-off; the 0.15 gate is loose.
- The constructor's `pass_below`/`flag_above` defaults (0.25/0.70) are silently
  replaced by the payload's values, even when the caller passes its own.

### 4.7 The floors override the model (medium)

The post-model floors change the label in large numbers (eval set):

| Floor | Effect | Precision of the flag |
|---|---|---|
| severe low-confidence (conf < 50, OOV) → flag | 750 correct words flagged; 604 of them had model p < 0.15 | 0.56 (952 of 1,702) |
| OOV → gray | 5,363 correct words moved from pass to gray | n/a |
| low-confidence floor (< 60) → gray | 1,812 correct words moved to gray | n/a |

The floors are compensating for weaknesses that the model should learn (the model
lacks confidence, so the floor works around it). After retraining with confidence,
re-check whether the floors still help.

### 4.8 Training script issues (medium)

- Early stopping uses the validation fold, so out-of-fold metrics are optimistic.
- The final model uses a fixed 400 rounds; `best_iteration` is 0 in the pickle.
- Out-of-fold metrics are printed but not saved, so they cannot be checked later.
- Insertion tokens (1,089 rows) are labelled as errors in the corpus; that is a
  reasonable choice, but it should be stated in the docs.

### 4.9 Evaluation pipeline (medium)

- `analyze_methods.py` maps confidences by position (`conf_map[idx]`). In 16 of 100
  pages the `.conf.tsv` token count differs from the `.ocr.txt` count. About 4,300
  tokens (6.8%) at or after the first mismatch may have the wrong confidence, which
  affects the floor results in `analysis.tsv`.
- The `in_lexicon` column in `analysis.tsv` uses the normalized lookup, so it does not
  show what the model received (see 4.3). Log the features the model actually used.
- No other label issues in the eval set: none of the 4,267 substitution errors are
  equivalent under the normalizer.

### 4.10 Package and tests (medium)

- `from OCR_error_detection import DetectionEngine` fails from the repo root
  (`ModuleNotFoundError: confusion`): `detector.py`, `signals.py` and others use bare
  imports that only resolve when the package folder is on `sys.path`. The README and
  the test command (`python -m unittest OCR_error_detection.tests.test_detector`)
  fail as written.
- With the package folder on the path, 40 tests run: 5 errors and 1 failure. Four
  errors call methods that do not exist on `DetectionEngine` (`verdict_batch`,
  `signals`, `reset_counters`, `detection_stats`). The `verdict_batch` example is also
  in the README. The failure is the `ቤት` case in 4.5.

## 5. Recommendations, in order

1. **Decide the wiring** (section 1). Until this is settled, no LightGBM change affects
   the corrector.
2. **Fix the lookup** in `_extract_lgbm_features`: use the normalized key
   (`self._norm(clean_w)`). Add a test that `ዓለም` gets `in_lexicon=1`.
3. **Fix the confusion-map key** in `train_LightGBM.py` (`"sub"`, not `"substitutions"`),
   then check that `has_one_flip` gains splits. If it does not help, remove the feature.
4. **Rebuild the corpus with real confidences** from `*.conf.tsv`, fix the position
   mapping, and add `tesseract_conf` as a model feature.
5. **Hold out the 100 evaluation pages** from production training. Report out-of-fold
   AUC and AP, and pick the pass and flag thresholds from those results against a
   precision target.
6. **Make `doc_freq` optional** (NaN when unknown) and train with that missing-value
   branch, instead of defaulting to 1.
7. **Make the no-KenLM mode explicit**: a separate model, or a documented NaN branch.
   Check that training and inference normalize words the same way before scoring.
8. **Replace `scale_pos_weight`** with a calibration step (for example isotonic on
   out-of-fold scores), or remove the claim that the output is calibrated.
9. **Re-check the floors** after retraining; remove those the model now captures.
10. **Add a feature-name guard** in `DetectionEngine`: fail if the payload's features
    are not all available, rather than defaulting them to 0.
11. **Fix the imports and tests**: absolute imports in the package, implement or remove
    the missing methods, and update the README.
12. **Revisit the one-flip signal** in both `suspicion.py` and the detector before
    relying on it for hard FLAG decisions.

## Reproducing the numbers

Scripts in `docs/lgbm_review/`:

- `probe.py`: inference-path probe (features, normalization mismatch, `doc_freq`, speed).
- `analyze_eval.py`: AUC/AP, threshold sweep, floor breakdown, normalization groups.
- `norm_bug.py`: the H-family group analysis in 4.3.
- `signals_check.py`: one-flip direction and the `tesseract_conf` distribution.
- `honest_cv.py`: page-grouped CV for the feature-set comparison in 4.1.
- `floors.py`: floor precision in 4.7.
- `label_check.py`: normalizer-equivalent label check in section 4.9.

They need `lightgbm`, `scikit-learn`, `pandas`, `joblib`, and `symspellpy`. They read
`OCR_error_detection/Evaluation/analysis.tsv`, `harmonized_corpus_jiwer.tsv`, and
`confusion_matrix.json`, and do not require the master dictionary or KenLM.

## Step 1 status (data pipeline and hygiene)

Done in code (no retraining, see "not verified" below):

| Item | Change | Where |
|---|---|---|
| Confusion key | training reads `"sub"`; the reverse map is built by `ConfusionMatrix` | `Model/train_LightGBM.py`, `features.py` |
| Canonical lookups | vocabulary and frequency lookups use the normalized key, in training and inference | `features.py`, `train_LightGBM.py::normalize_vocab` |
| Punctuation | punctuation and digits are stripped before any lookup; before this, `ቤት፡` was out of lexicon | `features.py::cheap_signals` |
| Char LM input | scored on the raw clean surface (no normalization), in training and inference | `features.py::FeatureContext.char_lm_score` |
| Missing context | `NaN`, not magic numbers: no KenLM, words under 2 characters, no page context (`doc_freq=None`) | `features.py`, `detector.py` |
| One definition | training and inference both call `features.py`; a parity test checks equal vectors | `tests/test_features.py` |
| Schema version | pickles carry `feature_schema_version`; schema 1 gets explicit legacy imputation and a warning | `detector.py`, `features.py` |
| Imports | `from OCR_error_detection import DetectionEngine` works from the repo root | `signals.py`, `detector.py` |

Tests: `python -m unittest OCR_error_detection.tests.test_features` runs 15 tests and all pass.
The existing `test_detector.py` still has the 5 stale-API errors and the `test_passes_all_clean_words` failure. Those belong to step 4.

### Not verified

- **Training was not run.** The master dictionary and KenLM model are not in this checkout. The training code is checked only by the parity and vocabulary tests.
- **The shipped pickle is still schema 1.** Missing `doc_freq` is imputed as a singleton, so `verdict("ቤት")` still gives 0.178 (gray). This is the intended legacy behaviour and goes away on retraining.

### Measured effect of the punctuation fix (counterfactual, legacy model, in-sample)

The model's inputs for punctuated tokens were already correct, so only the floors change.
The floors fired on 4,128 correct words whose cleaned form is in the lexicon. The raw OCR
tokens confirm the bug for 3,070 of them, and the remaining 1,058 are ambiguous (a bare form of the word also appears on the page).

| | Recorded | With punctuation fix |
|---|---|---|
| Correct words: pass / gray / flag | 43,122 / 12,842 / 2,154 | 49,613 / 7,069 / 1,436 |
| Correct words not pass | 25.8% | 14.6% |
| False flags (correct words flagged) | 3.71% | 2.47% |
| Errors caught (flag + gray) | 96.53% | 93.67% |

Trade-off: 122 errors (2.9%) that were caught only by the floors become pass. Those are
mostly low-confidence words that are in the lexicon. The model has no confidence feature, so
the floors were compensating for it. This is the strongest argument for putting confidence in
the model (see below).

### Confidence in training: open questions

- **Useful?** Yes. On the eval rows, a page-grouped model with confidence reaches AUC 0.927;
  without it, 0.798 (section 4.1). The 122 errors above show that the current floors are
  working around the same information.
- **Double counting.** Once confidence is a model feature, a separate confidence floor on top
  of the model counts it twice. The floors should go, as planned in Stage 2.
- **Data blocker.** The checkout has `*.conf.tsv` only for the 100 eval pages, and no images
  for the training pages. Confidence for the other 1,267 pages must come from wherever the
  corpus was built, from the same OCR engine and settings as deployment.
- **Missing confidence at inference.** The corrector can run without a TSV, so the model must
  also handle `NaN` confidence. Train with a fraction of rows masked, or keep a second model
  without confidence.

### Open question: is the dictionary independent of the evaluation pages?

`main_i7` is built from a corpus. If that corpus includes the ground truth of the 100 eval pages
(or of the training pages), `in_lexicon` and `log_corpus_freq` leak the labels. Both in
evaluation and in training, correct words would look in-lexicon because their ground truth
was in the dictionary build, which is not true for new text at deployment. Check which corpus
`main_i7` came from, and whether it contains the `harmonized_corpus_jiwer.tsv` pages.
