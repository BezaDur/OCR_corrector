# LightGBM detector: what changed in step 1, in plain language

Date: 2026-10-13. Branch: `arena/3ca31a9a-ocr-corrector`.

This note explains the step 1 changes to the LightGBM error detector without
the technical detail. The detailed findings are in `docs/lgbm_detector_review.md`.
Where this note and that review disagree, this note is newer and wins.

## 1. What the detector does

For each word that the OCR engine produced, the detector estimates how likely it
is to be a mistake. It places each word into one of three groups:

- **pass**: leave the word alone,
- **gray**: only change it if the evidence is strong,
- **flag**: the word is probably wrong, so correct it.

The detector is a LightGBM model (a decision-tree model) that looks at features
of each word: is it in the dictionary, how common is it, how long is it, does a
small optical change turn it into a real word, and so on. The same features must
be computed when the model is trained and when it is used. If they differ, the
model is wrong in ways that are hard to see.

## 2. The problems found, and what was changed

### 2.1 The confusion table was read under the wrong name

**Problem.** The table of common OCR character confusions is stored in a file under
the name `sub`. The training script looked for a name that does not exist, got
nothing back, and therefore trained the model without the "one-character change
gives a real word" signal. The model ignored that signal completely.

**Change.** Training now reads the table under the right name, through the same
code the detector uses.

**Effect.** The signal is now available to the model. Whether it helps is for the
retrained model to show.

### 2.2 Words with variant spellings were looked up incorrectly

**Problem.** The dictionary is stored in a normalized spelling. For example, the
letter `ዓ` is stored as `አ`, and `ሐ` is stored as `ሀ`. The detector looked words up
in their raw spelling, so a correct word such as `ዓለም` (world) looked
out-of-dictionary to the model. The model then tended to flag or hold back correct
words that contain these letters.

**Change.** Every dictionary lookup now uses the normalized spelling. Training does
the same when it builds the dictionary it uses.

**Effect.** Correct words with these letters are no longer treated as unknown.
On the saved evaluation run, about 78% of such correct words were not passed,
compared with 18% for ordinary dictionary words. This is the main cause of that gap.

### 2.3 Punctuation broke dictionary lookups

**Problem.** Words in the OCR output often end with punctuation, such as `ቤት፡`
(the Ethiopic colon) or `ሰላም።` (full stop). The detector kept the punctuation
when it looked the word up, so `ቤት፡` was treated as unknown while `ቤት` was known.
That word then went into the "unknown word" rule and was held back.

**Change.** Punctuation and digits are removed before every lookup.

**Effect.** This was the larger of the two bugs in practice. On the saved evaluation
run, the "unknown word" rule fired on 4,128 correct words that are in the dictionary.
For 3,070 of them, the raw OCR text confirms that punctuation caused it. The other
1,058 are ambiguous, because the same word also appears without punctuation
elsewhere on the page.

If the fix had been in place for that run, the share of correct words sent to
"gray" would have dropped from about 26% to about 15%. The share of correct words
flagged would have dropped from 3.7% to 2.5%. The catch rate of real errors would
have dropped from 96.5% to 93.7%. The next section explains that trade-off.

### 2.4 The character-level language model read a different string in training and in use

**Problem.** The character model scores how word-like a string is. Training scored
the raw word, but the detector scored a normalized version of it. The same word
could therefore get two different scores.

**Change.** Both now score the same string: the word with punctuation removed,
and without normalization. Normalization would hide exactly the letter variants
that OCR confuses, so it is not applied here.

### 2.5 Missing information was filled with made-up numbers

**Problem.** Three situations have missing information:

- no character model is loaded,
- a word is too short to score (under two letters),
- a word is classified alone, without its page, so the page frequency is unknown.

The detector filled these with fixed numbers. For example, a missing character-model
score was set to −2.5, and a missing page frequency was set to 1, which means
"appears once on the page". Those numbers were not values the model had learned
from, so the score for these words was distorted.

**Change.** Missing values are now recorded as "unknown" (NaN). The model is retrained
to handle unknown values directly.

**Effect.** The current model file was trained before this change. The code therefore
still uses the old fixed numbers for it, and prints a warning. The distortion
disappears when the model is retrained.

### 2.6 One definition of the features

**Problem.** Training and the detector each had their own code for computing features.
Any difference between the two was a silent error.

**Change.** A new file, `4_ocr_correction/error_detection/features.py`, holds the only definition
of the features. Training and the detector both call it. A new set of tests checks
that the two produce identical numbers for the same word.

### 2.7 Each model now records the feature version it was trained with

**Problem.** A saved model did not say which version of the features it expects.
Using an old model with new code would be silently wrong.

**Change.** The saved model now records a feature version number. The detector
checks it on loading. The current model file has no version number, so it is
treated as the old version, and the detector warns about it.

### 2.8 The package could not be imported from the repository root

**Problem.** `from error_detection import DetectionEngine` failed unless the
package folder itself was on the Python path.

**Change.** Internal imports were made to use full package names.

**Effect.** The import now works from the repository root. This was planned for a
later step, but the new shared feature file needs it.

## 3. Tests

- `4_ocr_correction/error_detection/tests/test_features.py`: 15 tests, all passing. They check
  the confusion table, the normalized lookups, punctuation handling, unknown values,
  the old-version handling, and that training and the detector produce the same numbers.
- `4_ocr_correction/error_detection/tests/test_detector.py`: the same 40 tests as before. Five of
  them call methods that do not exist yet (`verdict_batch`, `signals`,
  `reset_counters`, `detection_stats`), and one checks an old behaviour that step 2
  will change. These are scheduled for step 4 and were not changed in step 1.

## 4. What was not verified

- **The model has not been retrained.** The dictionary and the character model are
  not in this copy of the repository, so training cannot run here. The shipped
  model is still the old version.
- **The retraining has not been checked for accuracy.** The new numbers have to come
  from a retrained model.
- **The numbers in section 2.3 are estimates.** They come from recomputing the
  saved evaluation run. The model's own scores were not changed by the fix, so the
  estimate uses the same scores.

## 5. Decisions recorded from the review

### 5.1 Confidence stays out of the model's training

Tesseract's confidence score is not a reliable sign of an error. The saved
evaluation run shows this:

- 466 of the 4,267 errors (11%) have a confidence of 90 or higher. 327 of those
  are real dictionary words, so the dictionary check alone cannot catch them.
- 76% of the correct words also have a confidence of 90 or higher.

The model will therefore be trained without confidence. This overrides my earlier
recommendation in `docs/lgbm_detector_review.md` (sections 4.1 and 5). That
recommendation is withdrawn.

Two facts about the data support this decision:

- The large 1,367-page training set has no real confidence values. Every row has the
  same placeholder value, 75.0, because the OCR run that produced it did not record
  confidence. Confidence was switched on only for the later 100-page run. So there is
  nothing to put into training for most of the corpus.
- The 100 evaluation pages are the only pages with real confidence values. They are
  also the test set, so they cannot be used for training. Even for these pages, the
  training file holds the placeholder 75.0, so the real values were lost when the
  training file was built.

### 5.2 Decided: the confidence rules stay at inference

The detector uses confidence in rules on top of the model. Those rules catch the 122
real errors described in section 2.3. You decided to keep them for detection, because
confidence helps in the clearest cases. Confidence is still not used in training.

Step 2 keeps these rules and reviews their thresholds against the development set.
The question is whether they should apply only to the clearest cases. See
`docs/new_test_set_plan.md`.

### 5.3 The dictionary and the training data come from the same source

This is the most important consequence of the facts you described. The dictionary
was built from the same corpus as the training text. Its entries are normalized
and cleaned, and the training text is closer to raw. Almost every training word is
in the dictionary.

The evaluation numbers show the same pattern:

- 94.2% of correct words are in the dictionary,
- 51.6% of errors are in the dictionary.

The dictionary does not contain labels. It is a plain list of words with their
frequencies, and it includes inflected forms as well as stems.

**Why this matters.** The model learns that "not in the dictionary" usually means
"error". In training, almost every correct word is in the dictionary, so that rule
looks very reliable. On new text, many correct words will not be in the dictionary:
names, new terms, and inflected forms that were not collected. Those words will
then be flagged or held back more often than the evaluation suggests. The
evaluation numbers are therefore probably too optimistic for new documents.

I cannot measure how large this effect is from here, because the dictionary itself
is not in this copy. The options below were the measurement routes. Route 2 is now
the plan: the dictionary cannot be rebuilt for now, so new pages will be collected
(see `docs/new_test_set_plan.md`).

The measurement routes were:

1. **Documents that were not used to build the dictionary.** If the corpus has
   document boundaries, rebuild the dictionary without the test documents, then
   measure dictionary coverage on those documents. This is the most direct test.
2. **Fresh text.** Collect new documents, and measure the share of their correct
   words that are in the dictionary.
3. **Compare frequencies.** Check whether frequencies in the dictionary for words
   that occur only once in the source look like the source counts. This would show
   whether the dictionary holds the same text as the evaluation pages.

Option 1 is the most convincing. It needs to know whether the corpus can be split by
document.

## 6. Decisions and next steps

Decided:

1. **Confidence rules at inference stay** (section 5.2). Confidence is not used in training.
2. **The dictionary is not rebuilt for now.** Instead, a fresh test set of new pages
   that are not in the dictionary's source will be collected. The plan is in
   `docs/new_test_set_plan.md`. The fresh set is used once, for the final measurement.

Next:

3. **Step 2 (data checks)** starts with the dictionary-coverage comparison: the share
   of correct words in the dictionary on the new pages, against 94.2% on the current
   pages. The check script is `tools/analysis/check_dictionary_coverage.py`. It has to
   run where the dictionary file lives. Step 2 also reviews the confidence thresholds
   on the development set.
4. **Waiting on you:** sources for the new pages, who transcribes them, and confirmation
   that the OCR settings match deployment (`docs/new_test_set_plan.md`, "What I need
   from you").

## 7. Files changed in step 1

| File | Change |
|---|---|
| `4_ocr_correction/error_detection/features.py` | new: the only definition of the features |
| `4_ocr_correction/error_detection/detector.py` | uses the shared features; missing values are unknown, not fixed numbers; checks the model's feature version |
| `4_ocr_correction/error_detection/signals.py` | full-package import |
| `4_ocr_correction/error_detection/Model/train_LightGBM.py` | uses the shared features; reads the confusion table under the right name; normalizes the dictionary; records the feature version; can train without the character model |
| `4_ocr_correction/error_detection/tests/test_features.py` | new: 15 tests |
| `docs/lgbm_detector_review.md` | the review (the earlier confidence recommendation is superseded by this note) |
| `docs/lgbm_step1_changes.md` | this note |
| `tools/analysis/` | scripts used for the measurements in this note |

The retraining command, once the dictionary and character model are available:

```
python 4_ocr_correction/error_detection/Model/train_LightGBM.py
python 4_ocr_correction/error_detection/Model/train_LightGBM.py --no-char-lm   # without the character model
```
