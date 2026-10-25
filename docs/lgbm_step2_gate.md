# Step 2: data checks (go / no-go gate)

> **Correction (2026-10-25): check C6 was tautological and its conclusion is
> withdrawn.** C6 compared the evaluation pages against a "training" set that
> still contained pages 1–100, so it could only report 100 % overlap. Run
> against the pages the detector actually trains on (101–1367), only **6.2 %** of
> evaluation lines (259/4,148) appear verbatim, and only **0.5 %** appear in the
> OCR training text. Under the strict 5-gram grouping used for cross-validation,
> all 100 evaluation pages are text-disjoint from the training pages.
> `docs/detector_out_of_sample.md` §3 has the measurements and §1 the resulting
> out-of-sample performance. Items 2 and 3 of the verdict below still hold and
> were acted on.

Date: 2026-10-13. Branch: `arena/3ca31a9a-ocr-corrector`.
Script: `tools/analysis/step2_data_checks.py`. Output: `tools/analysis/step2_results.txt`.

## Correction to my earlier report

My first version of this gate said that 50.7% of evaluation 5-grams were in the training
text. That number was wrong. It compared the evaluation text, which still had punctuation,
with the training text, which has none. With the same cleaning applied to both, the
overlap is **100%**. The check script now cleans both sides, and the numbers below come
from that version.

## Verdict

**Not ready for retraining as the data stands.** Two things have to be fixed first, and
one decision is needed:

1. ~~**The 100 evaluation pages are the training pages.**~~ **Withdrawn — see the
   correction above.** They are held out of the detector fit and their text is
   text-disjoint from the 1,267 training pages; 6.2 % of their lines occur
   verbatim anywhere in the training GT. The evaluation numbers *were* in-sample
   at the time this gate was written, because the model then in circulation had
   been fit on all 1,367 pages.
2. **The training rows contain no punctuation or digits, but deployment text does.**
   12.3% of the evaluation OCR tokens have punctuation or digits. The model never
   learns from them. This is the same class of problem as the punctuation bug fixed in
   step 1.
3. **Cross-validation has to be grouped by text, not by page.** The training pages
   overlap each other heavily.

   > **Update (2026-10-25):** grouped by *shared 5-grams* this does not work on
   > this corpus — the pages are overlapping windows, so every 5-gram occurs on
   > 10–40 pages and the `max_page_occurrences=5` rule drops exactly the linking
   > evidence (1,095 groups, 1,074 singletons). Use `--group-by document`
   > instead, which follows page-to-page continuation and finds 51 groups in the
   > 1,267 training pages. See `docs/detector_out_of_sample.md` §3b.

## What the harmonized file is

`harmonized_corpus_jiwer.tsv` is the harmonized form of your 1,367-page OCR run with the
`amh_fast` model. Harmonization removes punctuation and digits from both the OCR words
and the ground truth, and it drops tokens that are only punctuation or digits. Check C5
confirms this: after the same cleaning, the training rows match the evaluation OCR
output exactly (62,479 of 62,479 tokens). The 12% difference seen before is entirely
punctuation and digits.

## What was checked

| # | Check | Result | Status |
|---|---|---|---|
| C1 | Error label matches the words (an error exactly when OCR word ≠ ground truth) | 0 disagreements. 6.46% of tokens are errors (61,081 of 946,057) | Pass |
| C2 | Confidence column | Placeholder 75.0 on every row | Pass (confidence is excluded from training, as decided) |
| C3 | `doc_freq` means the same thing in training and in use | Both count occurrences on the page (642 of 642 tokens checked) | Pass |
| C4 | Page inventory | 1,367 pages. Missing from 1..1370: 1232, 1272, 1279 | Pass (minor) |
| C5 | Training rows vs evaluation OCR | 100% identical after cleaning | Pass (explained by harmonization) |
| C6 | Evaluation text in training text | 100% of evaluation 5-grams (61,613) and 100% of evaluation lines of 4+ words (4,148) are in the training ground truth. On the same-numbered training page, 99.1% of each page's 5-grams appear (minimum 90.6%) | **Fail** |
| C7 | Overlap among training pages, and page groups | 82.5% of distinct 5-grams occur on more than one page. Strict grouping gives 1,193 groups (largest 27 pages). Loose grouping gives 19 groups, one of them 1,147 pages | **Fail for page grouping** |
| C8 | Punctuation and digits | 0 training rows have them. 7,818 of 63,573 evaluation OCR tokens (12.3%) do | **Fail** |

## What C6 means

The 100 evaluation pages are the same pages as training pages 1–100 (plus overlapping
neighbours). The best match for an evaluation page is usually its own number. So the
evaluation set is not an independent test. The punctuation finding, the counterfactual,
and the dictionary coverage on evaluation pages were all measured on text the model and
dictionary were built from.

## What C7 means

With the strict grouping (5-grams that appear on at most 5 pages count as links; a pair
of pages is linked when they share at least 20), the training set splits into 1,193
groups. That is a reasonable basis for grouped cross-validation. With the loose grouping,
one group holds 1,147 pages, so cross-validation cannot be used. I recommend the strict
grouping. It is a judgment call, listed as decision 2 below.

## What C8 means

The detector's rules were designed around punctuated tokens, such as `ቤት፡`. Those tokens
are 12.3% of the evaluation OCR output, yet the training rows drop all of them. So the
model is being trained on a different distribution from the one it is used on. Step 1
fixed how the detector looks words up, but the model itself was never trained on
punctuated input.

The fix is to rebuild the training rows from the raw OCR output and the raw ground truth,
keeping punctuation and digits. The labels should use the same cleaned comparison as the
evaluation analysis, so that the labels do not change. This needs the raw per-page OCR
files and raw ground truth for the 1,367 pages. Please confirm they still exist.

## Dictionary note

The dictionary covers 94.2% of the correct words on the evaluation pages. Those pages are
part of the corpus the dictionary was built from. So about 6% of correct words from the
corpus are missing from the dictionary. The cause is not yet known. It could be the
normalization, a frequency cut, or a difference between the dictionary and the corpus.
It should be checked on your machine, using the coverage script on the training text.

On the training data itself, `in_lexicon` is therefore an in-sample feature. Correct
words are in the dictionary about 94% of the time, and errors about 52% of the time.
Once the model meets new text, the correct-word coverage is likely to be lower, which
means more false flags. The synthetic test set is the measurement for that.

## Checks that still need your machine

- **Dictionary coverage on the training text.** Run the coverage script on the training
  ground truth, and check why about 6% of correct words are missing.
- **The character model** (`amharic_char_model.klm`). Training without it works
  (`--no-char-lm`), but the model then loses that feature.

## Decisions for step 3

1. **Evaluation pages.** Treat the current 100 pages as development only. The synthetic
   set is the test.
2. **Grouping.** Use the strict grouping for cross-validation (1,193 groups), replacing
   the page grouping in `train_LightGBM.py`.
3. **Training rows.** Rebuild from raw OCR with punctuation and digits kept (C8). This is
   my recommendation. It needs the raw files for the 1,367 pages to exist.

Step 3 can start after decisions 2 and 3 are confirmed, and after the raw files are found.
