# Plan: a fresh synthetic test set that is not in the dictionary

Date: 2026-10-13. Branch: `arena/3ca31a9a-ocr-corrector`.

## Why

Every document in the current corpus is also in the dictionary, so the evaluation
cannot show how the detector behaves on genuinely new *vocabulary*.

> **Update (2026-10-25):** the second half of this paragraph was wrong. The 100
> evaluation pages are *not* part of the detector's training text — see the
> correction at the top of `docs/lgbm_step2_gate.md` and
> `docs/detector_out_of_sample.md` §3. Held-out-text performance is now measured
> (recall 91–92 % at ≈0.1 % false flags). What remains unmeasured is behaviour on
> documents whose *words* are not in the dictionary, which is still the reason to
> build the set below.

The fix is a new test set made from text that is not in the corpus at all. The test set
is synthetic: we collect the text, render it as page images, run OCR on the images, and
use the original text as the ground truth. Nobody transcribes anything by hand.

## How the synthetic set is made

1. **Collect source text** from documents that are not in the corpus. Keep a record of
   each source and when it was collected (the manifest).
2. **Render each text as a page image.** The rendering should match the real scans as
   closely as possible: same font family and size, same page layout, and the same
   scan-like degradations (blur, noise, skew, uneven contrast). The more the rendering
   differs from real scans, the less the test shows about real documents.
3. **OCR the image** with the deployment settings: Tesseract, `amh_fast`, page
   segmentation mode 3, with the per-word confidence file saved.
4. **Ground truth is the original text,** exactly as collected, with its punctuation.
5. **Record the rendering settings** for each page (font, size, degradation level), so
   that the test can be broken down by condition later.

## Requirements

1. **The text must not come from the corpus.** This is the one requirement that cannot
   be relaxed. Rendering corpus text into images would reproduce the overlap problem,
   even though the images are new. Before a page is used, check it against the corpus
   (the coverage check below shows the result).
2. **Render at the same resolution and format as the deployment.**
3. **OCR with the deployment settings, with confidence switched on.** Earlier runs for the
   large corpus were made without confidence, so check this for each batch.
4. **A manifest.** One row per page: page name, text source, date collected, font, size,
   degradation level, and notes.

## A limitation to keep in mind

Tesseract's errors on rendered text can differ from its errors on real scans. The rendered
pages are a controlled test of the detector, not a full substitute for real scans. Where
possible, keep a small set of real scans that are not in the corpus, so that the synthetic
results can be compared with real ones.

## Folder layout

Use the same layout as `OCR_error_detection/Evaluation/`, so the existing scripts work:

```
fresh_eval/
  images/        page_NNNN.png        (rendered pages)
  GroundTruth/   page_NNNN.gt.txt     (original source text)
  OCRed/         page_NNNN.ocr.txt    page_NNNN.conf.tsv
  manifest.csv   page, source, date_collected, font, size, degradation, notes
```

## How big

About 100 pages, similar to the current set, so results can be compared. This is a
suggestion. The set should not be smaller than about 50 pages, and more pages give
tighter numbers.

## Checks before anything is measured

1. **Dictionary coverage.** Run:

   ```
   python docs/lgbm_review/check_dictionary_coverage.py \
       --dict <path to main_i7.txt> --set fresh_eval
   ```

   The script reports the share of correct words that are in the dictionary. Compare it
   with the current set, where 94.2% of correct words are in the dictionary (and in
   training, the dictionary covers almost all of the words).
   - If the synthetic set is close to 94%, the dictionary feature is probably reliable.
   - If it is clearly lower, the dictionary feature is over-trusted, and the model has
     to be judged with that in mind.
   The gap between the two numbers is the main result of this test.

2. **No peeking.** Do not look at the detector's output on the synthetic set until the
   model and thresholds are frozen. Measure once, and report that result.

## Effect on the roadmap

- **Step 2 (data checks)** is recorded in `docs/lgbm_step2_gate.md`. The training file is
  the harmonized output of your 1,367-page `amh_fast` run. Still open there: rebuilding the
  training rows with punctuation kept, and the dictionary coverage on your machine.
- **Confidence rules stay, for detection only.** Confidence is not used in training. The
  rules are reviewed against the development set in step 3, and the synthetic set is
  used only for the final measurement.
- **Step 5 (one-time scoring)** uses the synthetic set with the thresholds frozen before
  scoring.

## What I need from you

1. **Source texts.** Which documents or publications can be used for the source text, and
   are they newer than the corpus? Please say where they come from, so I can record them
   in the manifest.
2. **Rendering.** Which fonts are used in the real documents, and roughly what scan
   quality they have? That lets the rendering match real pages.
3. **OCR settings.** Confirm that the engine, language model, and page segmentation mode
   used for the current 100 pages are the ones used in deployment. The synthetic set
   must use the same ones.
4. **Raw files for the 1,367 pages.** Do the raw per-page OCR outputs and raw ground truth
   (with punctuation) still exist? They are needed to rebuild the training rows (step 2,
   check C8).

## Status (2026-10-13): first synthetic set built

Source: `telegram_c4.txt` (new source, pulled from the branch). Built in `synthetic_eval/`,
which is git-ignored, so the images and text stay out of Git.

- **Selection** (`docs/lgbm_review/build_synthetic_source.py`): 35 consecutive posts per
  page, 100 pages sampled at random (seed 7). Posts sharing any word 5-gram with the
  training ground truth, the other corpus files, or the evaluation ground truth were
  removed (8,061 of 187,748 candidate posts). Final ground truth has 0 shared 5-grams with
  any of them.
- **Rendering** (`docs/lgbm_review/make_synthetic_set.py`): A4 at 300 dpi, 36 px text,
  11 legible fonts rotated page by page (Abyssinica SIL, Brana, FreeSerif, Noto Sans
  Ethiopic regular and bold, Nyala, Jiret, Jiret slant, Hiwua, WashRa BS, WashRa slant).
  Degradation rotates clean / light / medium. Five pages were too full; their last posts
  are not rendered and are not in the ground truth (manifest notes say so).
- **OCR** (`docs/lgbm_review/ocr_synthetic_api.py`): Tesseract 5.5.1 through the Python API,
  model `amh_fast` from `models/`, page segmentation mode 3. The command-line binary is not
  installed in this sandbox. On the saved preview images in `font_diag_temp/`, the API output
  matches the saved OCR exactly for Abyssinica and gives 0.952 word similarity for Noto and
  Nyala. The gap has not been explained. Check one page against `run_ocr.py` on your machine.
- **Difficulty:** the synthetic pages are easier than real scans. Wrong OCR tokens are 3.7%
  overall (2.8% clean, 3.9% light, 4.6% medium), against 7.0% on the real evaluation pages.
  The degradation settings are the lever. The decision is what your real scans look like
  (see "What I need from you" above).

Still to do:
1. Run `docs/lgbm_review/check_dictionary_coverage.py --dict <main_i7.txt> --set synthetic_eval`
   on your machine. It needs the dictionary file.
2. Decide whether the degradation settings should be harsher, to match the real error rate.
3. Run one page through `run_ocr.py` and compare with `synthetic_eval/OCRed/`.
