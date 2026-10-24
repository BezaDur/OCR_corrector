# OCR Error Detection — the child project

> Verified numbers for everything on this page: `docs/current_state.md` §2 and §5.

**One job:** look at each OCR word and decide whether to keep it or fix it.

* `pass` — the word is fine (P(error) < 0.10)
* `gray` — not sure (0.10 ≤ P(error) < 0.65); the parent still tries candidates
* `flag` — wrong (P(error) ≥ 0.65)

This package is self-contained: code, model, lexicon, confusion matrix and
evaluation pages are all inside `OCR_error_detection/`. It is the only part of
the repository that runs end to end with nothing but `pip install -r requirements.txt`.
The parent corrector calls it through `adapter.py`.

---

## How to use

```python
from OCR_error_detection import DetectionEngine

det = DetectionEngine()                 # ~3.7 s; auto-finds Model/detector_lgbm_conffree.pkl
v = det.verdict("አና", doc_freq=2)
print(v.label, round(v.suspicion, 3))   # flag 1.0
```

Measured on this checkout (2026-10-24):

| token | label | P(error) | why |
|---|---|---:|---|
| `እና` | pass | 0.000 | in lexicon, corpus freq 4,065,980 |
| `በሽታ` | pass | 0.002 | in lexicon, freq 64,452 |
| `በሸታ` | **gray** | **0.154** | in `main_i7_frq10.txt` at freq 263, dropped by the freq ≥ 500 loader filter; one flip reaches `በሽታ` |
| `ABC` | gray | 0.283 | mixed script, no digit |
| `የአድሜ` | flag | 0.863 | OCR of `የእድሜ` |
| `ምዕፅራፍ` | flag | 0.867 | OCR of `ምዕራፍ` |
| `አና` | flag | 1.000 | OCR of `እና` |

(`doc_freq=2`; pass a page-level `Counter` value in real use, or `None` when the
word is isolated — inference then leaves the two context columns as NaN.)

> Older versions of this file said `verdict("በሸታ") → FLAG 0.85`. The shipped
> model returns **gray 0.154**, and `በሸታ` is not a clean "unknown word" example:
> only the default frequency filter makes it OOV. See
> `docs/current_state.md` §2.

---

## The model

`Model/detector_lgbm_conffree.pkl` — 3.5 MB, LightGBM with **529 trees and 14
features**, isotonic-calibrated. The pickle carries its own metadata:

| key | value |
|---|---|
| `pass_thresh` / `flag_thresh` | **0.10 / 0.65** (these win over constructor defaults) |
| `calibrator` | `IsotonicRegression(out_of_bounds='clip')` |
| `features` | the 14 columns defined once in `features.py` |
| `note` | `conf-free main_i7 freq>=500: 54057 types, 1267 pages, BEST FF 0.01%` |

The pickle has **no `feature_schema_version`**, so the loader treats it as a
pre-NaN-schema model: it logs one warning
(`model has no feature schema version (pre-schema-2); missing values are imputed
with the trained constants`) and applies `features.LEGACY_IMPUTE` to missing
`doc_freq` / `is_doc_singleton` / `char_kenlm_score`. That warning is normal.

**Signals that are deliberately off**

* **SymSpell** — `n_neighbors` is 0 in the training data, the model gives the
  column zero gain, and OFF vs ON measured the same recall. Building the index
  costs seconds and gigabytes on a large lexicon.
* **Character KenLM** — optional; without a model file the column is NaN.
* **Tesseract confidence** — the model is conf-free. Confidence is not a feature
  and is not needed at inference.

---

## Lexicon — the one knob that changes behaviour

`Model/main_i7_frq10.txt`: **737,106 types**, freq 10 … 5,672,129 (`ነው`), 15 MB,
from a much larger corpus than `local_lexicon.txt`.

`DetectionEngine()` keeps types with **freq ≥ 500 → 54,057 types**
(`DEFAULT_DICT_MIN_FREQ` in `detector.py`, because the pickle carries no
`lexicon_min_freq`). That is the vocabulary the model was trained against, so it
is the default. Other cuts, same file:

| `dict_min_freq` | types | effect |
|---:|---:|---|
| 10 | 735,644 | widest coverage; rare/typo-ish types enter the lexicon |
| 100 (sweep) | 163,891 | 93.5 % recall, 0.11 % false flag |
| **500 (default)** | **54,057** | **the shipped model's training lexicon** |
| 1000 (sweep) | 32,402 | 94.3 % recall, 0.02 % false flag |

`local_lexicon.txt` (750,870 types, built from the repo's 4.4 M-token corpus) is
the fallback if `main_i7_frq10.txt` is missing. It is smaller-domain and was
measured worse (0.07 % false flag at freq ≥ 10).

---

## Sweep behind the shipped model

Measured on the 1,267-page corpus, **1,200 train / 67 validation**:

| Dictionary | Filter | Types | Val AUC | Recall | False flag | Score |
|---|---|---|---:|---:|---:|---:|
| local_lexicon | freq ≥ 1 | 719k | 0.9593 | 92.1 % | 0.10 % | 0.919 |
| local_lexicon | top 150k (freq ≥ 3) | 150k | 0.9485 | 94.2 % | 0.04 % | 0.941 |
| local_lexicon | freq ≥ 10 | 52,494 | 0.9346 | **94.8 %** | 0.07 % | **0.946** |
| main_i7 | freq ≥ 10 (all) | 735,644 | 0.9512 | 90.7 % | 0.13 % | 0.904 |
| main_i7 | freq ≥ 100 | 163,891 | 0.9423 | 93.5 % | 0.11 % | 0.913 |
| **main_i7 (shipped)** | **freq ≥ 500** | **54,057** | 0.9242 | 94.2 % | **0.01 %** | 0.942 |
| main_i7 | freq ≥ 1000 | 32,402 | 0.9116 | 94.3 % | 0.02 % | 0.939 |

The shipped row was chosen for the lowest false-flag rate (≈1 clean word in
10,000) at equal recall. This is **validation inside the training corpus** (a
67-page slice of the 1,267 production pages), not a held-out test — the text of
the 100 dev pages is in the corpus too. Do not quote 94.2 %/0.01 % as
out-of-sample performance.

---

## Files

| Path | What it is |
|---|---|
| `Model/detector_lgbm_conffree.pkl` | the shipped model (3.5 MB) |
| `Model/main_i7_frq10.txt` | dictionary, 737,106 types, 15 MB |
| `Model/local_lexicon.txt` | fallback lexicon, 750,870 types, 15 MB |
| `Model/confusion_matrix.json` | optical confusion matrix, 318 substitution rows, 48 KB — the single source |
| `Model/harmonized_corpus_jiwer.tsv` | training corpus, 1,367 pages, 946,058 rows, 50 MB |
| `Model/train_LightGBM.py`, `train_final.py` | training entry points (need the in-repo corpus; KenLM optional) |
| `Evaluation/` | 100 ground-truth pages, 100 OCR pages, `analysis.tsv` (62,521 rows), page images |
| `detector.py` | `DetectionEngine`: load, features, verdicts |
| `features.py` | the single feature definition shared with training |
| `adapter.py` | `CorrectorDetectionAdapter` used by the parent corrector |
| `signals.py`, `scoring.py`, `confusion.py`, `normalizer.py`, `utils.py` | evidence collection, rule fallback, character tables |
| `run_ocr.py` | thin OCR helper |
| `scripts/audit_decision.py` | per-decision audit (also reachable as root `ocr_decision_audit.py`) |
| `tests/` | 56 tests |

`main_i7_frq10.txt` also exists one directory up; it is byte-identical and only
a fallback in the loader's candidate list.

---

## Evaluation status — what the stored numbers mean

`Evaluation/analysis.tsv` (62,521 rows, 62,385 of them scored: 4,267 errors,
58,118 correct) was
produced by an **earlier** model. The 100 pages are held out of the fit
(`DEV_PAGES = 100` in `Model/train_LightGBM.py`), but their text is entirely
present in the training corpus (100 % 5-gram overlap — `docs/lgbm_step2_gate.md`),
so this measures vocabulary the model has seen, not generalisation. From the
stored verdicts: flag recall 60.4 %, flag+gray recall 95.6 %, false-flag rate
1.19 % on clean tokens.

Re-running the current engine on a 1,500-row random resample agrees with those
stored verdicts only **85.6 %** of the time, so the file is stale. For current
behaviour, run the engine yourself; for a number that measures unseen
vocabulary, use the synthetic set built per `docs/new_test_set_plan.md`.

---

## Tests

```bash
python -m pytest OCR_error_detection/tests -q       # 56 passed, ~2 s
python -m unittest discover -s OCR_error_detection/tests -p 'test_*.py'
```

They cover feature/train-serve parity, the normalizer, the confusion matrix,
verdict gates, the adapter handover and the legacy-pickle path. No KenLM, no
Tesseract, no network.
