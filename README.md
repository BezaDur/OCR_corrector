# OCR Corrector — Amharic post-OCR pipeline

**Prepare data → train Tesseract → measure OCR → detect word errors → correct them → measure the correction.**

Two things live in this repository:

* **A parent pipeline** (`1_`…`5_`) that builds an Amharic OCR model, evaluates it
  and post-corrects its output.
* **A child package, `OCR_error_detection/`**, that decides word by word whether
  an OCR token is fine (`pass`), uncertain (`gray`) or wrong (`flag`). The parent
  imports it — never the other way round.

Everything in this README was checked against the code; the exact numbers,
with the commands that produce them, are in **[`docs/current_state.md`](docs/current_state.md)**.
Where older documents disagree with that page, the page is right and the
document is stale.

---

## Repository map

```
1_data_preparation/        synthetic pages, Tesseract boxes/.lstmf, character inventory
2_training/                Tesseract LSTM epoch scripts, packaged traineddata
3_training_evaluation/     OCR inference, CER/WER (ocr_metrics.py), error composition
4_ocr_correction/          the corrector (amharic_ocr_corrector.py), confusion-matrix builder
5_correction_evaluation/   end-to-end corrector benchmark, decision audit
OCR_error_detection/       CHILD: LightGBM detector + Model/ + Evaluation/ + tests/
vendor/segmenter/          vendored Amharic morphological suite (frozen copy)
archive/                   superseded code, old models, old reports
docs/                      design notes, reviews, and current_state.md
fonts/                     Ethiopic fonts used by the page generator
repo_paths.py              every external data path, resolved from env vars
```

Root-level Python files (`amharic_ocr_corrector.py`, `normalizer.py`,
`ocr_metrics.py`, `suspicion.py`, `tsv_gt.py`, …) are **thin import shims** kept
for backward compatibility. The real modules live in the stage folders above.
Typical use:

```python
from amharic_ocr_corrector import AmharicOCRCorrector   # shim -> 4_ocr_correction/
from OCR_error_detection import DetectionEngine          # the child package
```

---

## The detector (child)

```python
from OCR_error_detection import DetectionEngine

det = DetectionEngine()                       # ~3.7 s, no external files
v = det.verdict("አና", doc_freq=2)
print(v.label, round(v.suspicion, 3))         # flag 1.0
```

* **Model:** `Model/detector_lgbm_conffree.pkl` — LightGBM, 529 trees, 14
  features, isotonic-calibrated, **gates in the payload: pass < 0.10 ≤ gray <
  0.65 ≤ flag**.
* **Lexicon:** `Model/main_i7_frq10.txt` (737,106 types) filtered at load to
  **freq ≥ 500 → 54,057 types**, which is the lexicon the model was trained
  against. `DetectionEngine(dict_min_freq=10)` widens it to 735,644 types.
* **Features:** one definition in `features.py` shared by training and inference
  (`log_corpus_freq, in_lexicon, doc_freq, is_doc_singleton, token_len, len_ratio,
  has_one_flip, one_flip_count, char_kenlm_score, rare_glyph_ratio, n_neighbors,
  mixed_script, has_digit, is_high_freq`).
* **Deliberately off:** SymSpell (`n_neighbors = 0`), character KenLM (column is
  NaN when no model is supplied), OCR confidence (the model is conf-free).
* **Speed:** 0.48 ms/word measured here; no dictionary beyond the one file, so it
  is the part of the repo that runs offline end to end.
* **Held-out performance:** recall (flag + gray) **92.4 %**, false flag
  **0.10 %**, AUC 0.923, calibration ECE 0.005 on 100 pages that are held out of
  the fit *and* text-disjoint from the training pages. The README's older
  "94.2 % / 0.01 %" was a selection number. Details:
  `docs/detector_out_of_sample.md`.

Real verdicts on this checkout: `እና` pass 0.000 · `በሽታ` pass 0.002 ·
**`በሸታ` gray 0.154** (attested at freq 263, so below the 500 cut) ·
`የአድሜ` flag 0.863 · `ምዕፅራፍ` flag 0.867 · `አና` flag 1.000.

> The old examples in this README (`verdict("በሸታ") → FLAG 0.85`, "50k types",
> "40 tests") described an earlier model and are gone. See
> `docs/current_state.md` §2 for why.

---

## The corrector (parent)

`4_ocr_correction/amharic_ocr_corrector.py` (1,817 lines) is a CPU post-processor:

1. **Stage 1 — detection.** Every token goes through
   `CorrectorDetectionAdapter(self)`, which delegates to the child
   `DetectionEngine`. The engine loads its own 54k lexicon, then adopts the
   corrector's vocabulary on the first verdict (filtered back to the model's
   freq ≥ 500), so the corrector never keeps two full lexicons. `pass` keeps the
   token; `gray` and `flag` continue to candidate search. The rule-based
   `SuspicionScorer` is only a fallback if the engine cannot be built.
2. **Stage 2 — candidate generation.** Identity, ghost suffix, SymSpell (edit
   distance 1), single and pairwise optical reversals from
   `confusion_matrix.json`, plus a morphological peel guard.
3. **Stage 3 — KenLM Viterbi.** Beam 4 over
   `λ_lm·logP(w|h) + λ_opt·(−optical cost) + λ_morph·is_known_word + anchor`,
   with `</s>` scoring and do-no-harm anchor margins for verified/OOV originals.
4. **Stage 4 — merge/split** for OOV spans.

**A word KenLM model is required and is not in this repo.** Construction works
without it; the first correction raises a clear `ValueError`:

```bash
export AMHARIC_WORD_KENLM=/path/to/amharic_5gram.binary   # or AMHARIC_DATA_ROOT
python -c "
from amharic_ocr_corrector import AmharicOCRCorrector
print(AmharicOCRCorrector(verbose=False).correct_text('ሰላም በሸታ'))"
```

Everything else the corrector needs (confusion matrix, dictionary, morphemes,
the vendored segmenter) has an in-repo copy. `python repo_paths.py` prints what
resolves where, and every root can be overridden by an environment variable
(`AMHARIC_DATA_ROOT`, `TESSERACT_WORKSPACE`, `TESSERACT_TRAINING`,
`AMHARIC_WORD_KENLM`, `AMHARIC_CHAR_KENLM`, `AMHARIC_MORPH_PATHS`).

---

## Data and models

| What | Where | Size |
|---|---|---|
| Training text (4.42 M tokens, 638 k types) | `1_data_preparation/data/amh_final_training_text.txt` | 58 MB |
| Dictionary, freq ≥ 10 (737,106 types) | `OCR_error_detection/Model/main_i7_frq10.txt` | 15 MB |
| Fallback lexicon (750,870 types) | `OCR_error_detection/Model/local_lexicon.txt` | 15 MB |
| Optical confusion matrix (318 substitution rows) | `OCR_error_detection/Model/confusion_matrix.json` | 48 KB |
| Detector training corpus (1,367 pages) | `OCR_error_detection/Model/harmonized_corpus_jiwer.tsv` | 50 MB |
| Eval set (100 GT + 100 OCR pages + `analysis.tsv`) | `OCR_error_detection/Evaluation/` | 570 MB |
| Packaged Tesseract models | `2_training/models/`, `models/` | 8 MB + 5 MB |

`2_training/config/confusion_matrix.json` exists but is read by no code path —
the `Model/` copy is the single source.

---

## Does correction help? (current honest answer)

**Unknown for HEAD; negative for the last version that was benchmarked.**

* `5_correction_evaluation/benchmarks/benchmark_full_symspell.json` (150 pages)
  was produced by an earlier corrector revision and shows strict CER
  **1.18 % → 1.93 %**, clean-token damage 4.06 % strict / 2.66 % canonical, i.e.
  the corrector made OCR output worse. It cannot be attributed to the current
  code: that run used options (`candidate_backend`, `trie_search`,
  `trusted_*_frequency`) that no longer exist.
* `OCR_error_detection/Evaluation/analysis.tsv` is **stale** — the current engine
  agrees with only 85.6 % of its stored verdicts on a 1,500-row resample. Use
  `OCR_error_detection/scripts/evaluate_out_of_sample.py` instead; it recomputes
  every number from the corpus.

Re-running Phase B on the current code is the open task; see
`docs/two_phase_benchmark.md` for the procedure and `docs/current_state.md` §5
for the provenance of every number above.

---

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest OCR_error_detection/tests -q   # 56 passed  (~2 s)
python -m pytest tests -q                       # 142 passed, 52 skipped (~4 min)
```

The 52 skips are the retired `detection/` package (tests assert it is gone),
three quarantined suites that described a deleted corrector API (replaced by
`tests/test_head_contract.py`, 75 contract tests), and two environment-dependent
contract checks. Before the morphology fix in this pass, five more contract tests
skipped because the vendored analyzer could not import (`pydantic`).
`python -m unittest discover -s tests` also works.

---

## Docs

| Document | What it is |
|---|---|
| **`docs/current_state.md`** | **verified facts: defaults, verdicts, timings, data counts, test counts, known gaps** |
| **`docs/detector_out_of_sample.md`** | **held-out performance of the shipped detector: what the headline numbers really are** |
| `docs/repo_layout.md` | where everything lives today, including the root shims |
| `docs/architecture.md` | parent vs child, data flow, how the adapter wires them |
| `docs/components_guide.md` | plain-language tour of dictionary, normalizer, confusion matrix, SymSpell, KenLM, LightGBM |
| `docs/architecture_review.md` | architectural review + the §7/§8 work log (historical after §8) |
| `docs/lgbm_detector_review.md` | detector review (2026-10-13) and what has since been fixed |
| `docs/segmenter_integration.md` | exactly which segmenter methods are used and why |
| `docs/two_phase_benchmark.md` | how to run the corrector benchmark, and what damage metrics mean |
| `docs/decision_audit.md` | how per-token repair/damage decisions are computed |
| `docs/principles_review.md` | framework scorecard for the corrector |
| `docs/lgbm_step1_changes.md`, `lgbm_step2_gate.md`, `new_test_set_plan.md` | detector work log and the plan for a non-dictionary test set |

---

## Known limitations

1. No benchmark of the **current** corrector exists; the last one was negative
   (see above).
2. The detector is now measured out of sample (92.4 % recall / 0.10 % false
   flags on text-disjoint pages), but only on one synthetic-scan corpus with one
   OCR engine, and never on documents whose *words* are outside the dictionary.
   A non-dictionary synthetic set was built following `docs/new_test_set_plan.md`
   (in the external `synthetic_eval/` workspace, git-ignored), but no detector
   results on it are recorded here.
3. The corrector needs an external KenLM binary to run at all.
4. The repository is large (~960 MB tracked, 570 MB of it evaluation PNGs) and
   still tracks some build junk (`__pycache__`, `augraphy_cache/`,
   `font_diag_temp/`).
5. `docs/current_state.md` §7 keeps the running list of defects found while
   verifying the docs.
