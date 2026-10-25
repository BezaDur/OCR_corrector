# How to test the claims before merging

Date: 2026-10-25. Branch: `arena/c4c81d74-ocr-corrector`.

This page exists because the detector's headline claim changed in this branch and
a reader should not have to trust the author of the change. Everything below runs
against data and code already in the repository, or against a folder you supply.

---

## 0. The three commands

```bash
# 1. Re-derive every number and try to break it. ~3 min. Exit 1 on a miss.
python OCR_error_detection/scripts/verify_claims.py

# 2. The independent cross-check: 5-fold CV grouped by document, not by page. ~2 min
python OCR_error_detection/Model/train_LightGBM.py --no-char-lm --group-by document \
    --out /tmp/model.pkl --report /tmp/report.json

# 3. Score the shipped pickle yourself (~35 s)
python OCR_error_detection/scripts/evaluate_out_of_sample.py --report /tmp/oos.json
```

Run 1 first: it performs 2 and 3 internally and compares them with the committed
numbers in `docs/out_of_sample_shipped.json`. Command 2 is worth running
separately when you want to see the fold-by-fold behaviour, and command 3 when you
want the raw report.

`--fast` skips the two slow checks (the label rebuild and the grouping
sensitivity), taking the run to about a minute.

---

## 1. What the checks are, and what each one can catch

| id | check | proves | cannot catch |
|---|---|---|---|
| A1 | corpus shape is 946,057 rows / 1,367 pages / 884,976 clean + 59,992 substitutions + 1,089 insertions | the data is the data the documents describe | — |
| A2 | the dev slice is 62,479 rows / 4,361 errors / 100 pages | the slice is the one that was measured | — |
| A3 | `is_error` and `match_type` rebuilt from `Evaluation/*.gt.txt` + `Evaluation/*.ocr.txt` match the corpus on **62,479 / 62,479 rows** | the labels were not hand-edited; the recipe is the one in `make_eval_corpus.py` | whether the ground truth itself is right |
| B1–B5 | recall 92.39 %, flag-only 2.71 %, false flag 0.102 %, AUC 0.9232, AP 0.4117 | the headline numbers reproduce | that they generalise |
| B6 | ECE 0.0050 | the probabilities are calibrated | calibration on other page types |
| B7–B8 | the three text tiers and the two lexicon states reproduce | the tables in the write-up are computed, not typed | — |
| C1 | shuffling the error labels gives AUC 0.50 ± 0.03 (five shuffles) | no label leakage in the metric path | — |
| C2 | page-cluster bootstrap of recall: 90.9–93.7 % | the number does not hinge on one page | that pages are independent (they are not, see D) |
| C3 | flagging every out-of-lexicon word scores recall 89.5 % at **16.2 %** false flags; the model gets 92.4 % at 0.102 % | the model is not a renaming of the obvious rule | — |
| C4 | two scoring passes are bit-identical | deterministic scoring | — |
| D1 | 944,968 rows contain only ~106,000 first-seen 5-grams | how much of the corpus is repeated text | — |
| D2 | the 100 dev pages judge ~4,137 distinct source words | the effective vocabulary behind 62k rows | — |
| D2b | recall on first-seen OCR contexts only: 92.62 % ± 0.83 pp (18,541 rows, 3,849 errors) | the row-level number is not a mirage | — |
| D3 | the corpus is 55 documents in 1,367 overlapping windows | the right unit of independence is the document | — |
| D4 | `page_0101` continues `page_0100` | the train/dev cut splits a document | — |
| D5 | shared-5-gram grouping gives 1,095 / 48 / 19 groups at cutoffs 5 / 20 / 50 | why the old "text-grouped" CV was really page-level | — |
| D6 | `--group-by document` finds 51 groups | the new evidence in §1 of the write-up is produced by the documented method | — |

D1–D5 are marked `INFO` because they are measurements, not pass/fail claims.
Everything else is a hard check and sets the exit code.

**What none of this catches:** all of it uses the same corpus. It can show that
the arithmetic is right and that the labels are how the documents say they are.
It cannot show that the detector works on a page type this corpus does not
contain. Only §3 below does that.

---

## 2. What the run says today (2026-10-25)

```
A: 3/3 passed        B: 11/11 passed      C: 4/4 passed      D: 7/7 passed
102 s, exits 0
```

The two findings that are not pass/fail:

* **89 % of the corpus is repeated text** (944,968 rows, 105,596 distinct
  5-grams), because pages are overlapping windows of 55 documents.
* **The 100 dev pages are 5 documents and ~4,137 distinct words.** That is the
  real sample size behind "92.4 %", and it is why `docs/detector_out_of_sample.md`
  §7 now lists page-type coverage as a limit rather than a footnote.

A check that *did* fail earlier in this round, and what it changed: `strict_text_groups`
(the old text grouping) returns 1,095 groups with 1,074 singletons, so the
"text-grouped CV" was page-level CV in disguise. The write-up now uses
`--group-by document` (51 groups) and the conclusion survives — 91.27 % recalled
at 0.030 % false flags out-of-fold, calibrated.

---

## 3. Testing on new data

The existing corpus cannot answer the question that matters ("does it work on
pages like mine?"). A new set can, and it does not need much:

### 3.1 What the new folder must look like

```
fresh_eval/
  GroundTruth/page_0001.gt.txt     # the correct text, exactly as written
  OCRed/page_0001.ocr.txt          # what the OCR engine produced for the same page
  OCRed/page_0001.conf.tsv         # optional: word<TAB>confidence
```

This is the layout of `OCR_error_detection/Evaluation/`, so every script works
unchanged. Page names can be anything; `make_eval_corpus.py` renames them
`page_0001…` in file-name order.

### 3.2 Build the corpus and check that it is really new

```bash
python OCR_error_detection/scripts/make_eval_corpus.py \
    --eval-dir fresh_eval --out /tmp/fresh.tsv --report /tmp/fresh.json
```

It prints, before anything is scored:

* rows, errors, and how many ground-truth words had no OCR counterpart
  (deletions have no OCR word to judge and are not scored — the shipped corpus
  has the same blind spot, 42 words on the dev pages);
* **text novelty**: how many pages share no 5-gram with the 1,267 training pages;
* **vocabulary novelty**: what share of OCR tokens never occur in the training
  text, and what share of errors sit on them.

If the novelty numbers are low, the set is not a test — it is a re-run.

### 3.3 Score it

```bash
python OCR_error_detection/scripts/evaluate_out_of_sample.py \
    --corpus /tmp/fresh.tsv --eval-all --report /tmp/fresh.json
```

`--eval-all` scores every row (there is no train/dev split in a new corpus) and
skips the overlap tiers, which need a training reference.

### 3.4 How many errors you need

Half-width of the 95 % interval on a recall near 92 %:

| errors in the set | ± half-width |
|---:|---:|
| 300 | 3.1 pp |
| 1,000 | 1.7 pp |
| 4,000 | 0.8 pp |
| 10,000 | 0.5 pp |

The dev pages have 4,361 error rows, but only ~4,100 distinct source words and 5
documents, so their *effective* precision is closer to the 1,000-error row. A new
set of 20–30 pages with ~1,000 errors is enough to confirm or refute "91–92 % at
0.1 %" with a different page type; it is not enough to resolve 90 % from 92 %.

### 3.5 Rules that keep the test honest

1. **Do not tune anything on the new set.** The gates (0.10 / 0.65), the lexicon
   and the feature set are fixed. Choose the operating point before you look.
2. **Do not reuse training text.** `make_eval_corpus.py` reports the overlap for
   exactly this reason.
3. **Keep the manifest** — page, source, date, font, degradation — as
   `docs/new_test_set_plan.md` requires, so a surprising number can be traced to
   a condition.
4. **Same OCR settings** as deployment (Tesseract `amh_fast`, psm 3). Confidence
   is not needed: the shipped model is conf-free and the corpus confidence column
   is a constant 75.0.
5. **Report the whole set.** A per-subset result chosen after the fact is a
   selection number again — the thing this branch exists to remove.

---

## 4. What you are actually approving

* **Solid:** on this corpus, the shipped model recalls 92.39 % of errors at
  0.102 % false flags; the probabilities are calibrated (ECE 0.0050); it clearly
  beats a dictionary rule; the labels are reproducible from raw GT/OCR; and the
  numbers hold out-of-fold when folds are whole documents.
* **Corrected here:** the "text-grouped" cross-validation was effectively
  page-level (D5), and the "94.2 % / 0.01 %" headline was selection on 67 pages.
* **Still unproven:** behaviour on a different page type, on real scans, on
  vocabulary outside the lexicon, and anything about the corrector's output —
  the corrector's own measurement in `docs/two_phase_benchmark.md` remains
  negative (CER 1.178 → 1.930 strict).

If you want the smallest honest merge: accept the correction, and treat
"≈91–92 % recall at ≈0.1 % false flags **on news-style pages from this
pipeline**" as the claim. Do not let it become "the detector is 92 % accurate".
