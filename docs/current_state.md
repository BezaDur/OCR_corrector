# Current state — verified facts

**Verified:** 2026-10-24 on branch `arena/c4c81d74-ocr-corrector`, base commit `c9325d2`.
**Environment:** Python 3.11.2, numpy 2.4.6, lightgbm 4.7.0, scikit-learn 1.9.1,
pandas 3.1.0, symspellpy 6.10.0, joblib. `kenlm`, `tesseract-ocr` and the
`segmenter` package's `pydantic` dependency are **not** installed there.

**Why this file exists.** The READMEs quote sizes, defaults, timings, verdicts
and test counts, and those drift silently: the front page still said "default
50k types", "2.9 s init" and `verdict("በሸታ") → FLAG 0.85` long after all three
stopped being true. This page is the one place where such numbers are measured,
dated and reproducible. **If another document disagrees with this one, the
number here was measured — fix the other document.**

Each section ends with the command that produces its numbers (§8 has them all
in one block).

---

## 1. What runs without external data

| Component | Runs from a clean checkout? | Needs |
|---|---|---|
| `OCR_error_detection` detector (model, features, verdicts) | **yes** | pip deps only |
| `pytest OCR_error_detection/tests` | **yes** | `pytest` |
| `pytest tests` (repo-wide contract suite) | **yes** | `pytest` |
| `AmharicOCRCorrector()` construction | **yes** | pip deps only |
| `corrector.correct_text(...)` | **no** | word KenLM binary + `kenlm` |
| morphology (`segment_word`) | **yes**, once `pydantic` is installed (it is now in `requirements.txt`) | vendored path (§7.1, fixed) |
| `2_training`, `3_training_evaluation/02_run_ocr_eval.py` | **no** | Tesseract + workspace roots |
| `1_data_preparation/01_generate_eval_pages.py` | **no** | OpenCV/Pillow + fonts + writable workspace |
| `5_correction_evaluation/04_evaluate_full_corrector.py` | **no** | KenLM + pages + OCR output |

`repo_paths.py` is the only place that knows where the external trees live
(`AMHARIC_DATA_ROOT`, `TESSERACT_WORKSPACE`, `TESSERACT_TRAINING`,
`AMHARIC_WORD_KENLM`, …). `python repo_paths.py` prints what resolves where.

---

## 2. Detector — the numbers that matter

```python
from OCR_error_detection import DetectionEngine
det = DetectionEngine()
```

| Fact | Value | Where it comes from |
|---|---|---|
| Model file | `OCR_error_detection/Model/detector_lgbm_conffree.pkl` (3.5 MB) | `_resolve_first_existing(CANDIDATE_MODEL_PATHS)` |
| Payload | `model, features, pass_thresh, flag_thresh, calibrator, note` | `joblib.load` |
| Booster | 529 trees, 14 features | `booster.num_trees()` |
| Features | `log_corpus_freq, in_lexicon, doc_freq, is_doc_singleton, token_len, len_ratio, has_one_flip, one_flip_count, char_kenlm_score, rare_glyph_ratio, n_neighbors, mixed_script, has_digit, is_high_freq` | `features.FEATURE_NAMES` |
| Gates | **pass < 0.10 ≤ gray < 0.65 ≤ flag** | payload (`pass_thresh` / `flag_thresh`); they override the constructor defaults |
| Calibration | `IsotonicRegression(out_of_bounds='clip')` | payload |
| Lexicon | `Model/main_i7_frq10.txt`, 737,106 types → filtered at load to freq ≥ 500 → **54,057 types** | `det.lexicon_info` |
| SymSpell | off → `n_neighbors = 0.0` | `features.FeatureContext.neighbor_count` |
| Char KenLM | absent → `char_kenlm_score` is NaN | path resolution |
| Init (measured) | **3.7 s** | §8 |
| Throughput (measured) | **0.48 ms/word** (2,066 words/s), 5,000 eval tokens, `doc_freq=2` | §8 |

**The shipped pickle has no `feature_schema_version`.** The loader therefore
treats it as a pre-NaN-schema model: it logs
`model has no feature schema version (pre-schema-2); missing values are imputed
with the trained constants` once per load and applies `features.LEGACY_IMPUTE`
to missing `doc_freq`, `is_doc_singleton` and `char_kenlm_score`. The warning is
expected, not an error.

### Verdicts, measured on this checkout (`doc_freq=2`)

| token | label | P(error) | comment |
|---|---|---:|---|
| `እና` | pass | 0.0000 | in lexicon, corpus freq 4,065,980 |
| `ትምህርት` | pass | 0.0000 | freq 332,618 |
| `ለመሸፈን` | pass | 0.0017 | out of the top-frequency cut but a real word |
| `በሽታ` | pass | 0.0017 | freq 64,452 |
| `በሸታ` | **gray** | **0.1537** | attested at freq 263 → below the freq≥500 cut, so OOV; one flip reaches `በሽታ` |
| `ABC` | gray | 0.2833 | mixed script, no digit |
| `የአድሜ` | flag | 0.8632 | OCR of `የእድሜ` |
| `ምዕፅራፍ` | flag | 0.8669 | OCR of `ምዕራፍ` |
| `አና` | flag | 1.0000 | OCR of `እና` |
| `ብሏል፭` | flag | 1.0000 | trailing Ethiopic digit |

Three consequences worth carrying into every other document:

1. **The old README example was wrong twice.** `verdict("በሸታ")` returns
   **gray 0.15**, not FLAG 0.85 — and `በሸታ` is a poor "not in the dictionary"
   example because it *is* in `main_i7_frq10.txt` at freq 263. It is OOV only
   because the default loader drops freq < 500.
2. **`gray` is not "keep the word".** The parent corrector adds `gray_margin`
   and still runs candidate generation for gray tokens; only `pass` short-circuits.
3. **The freq≥500 filter is a behaviour choice, not just a memory cap.** With
   `dict_min_freq=10` the same file yields 735,644 types and `በሸታ` moves to
   `in_lexicon=1`, gray 0.2689. `in_lexicon` and `log_corpus_freq` are the two
   features with the largest gain, so the cut changes verdicts.

---

## 3. Corrector — the defaults that are actually in the code

`4_ocr_correction/amharic_ocr_corrector.py` — **1,817 lines, 88 KB** (not
"80K lines"; that number was bytes, mislabelled).

| Knob | Default | Note |
|---|---|---|
| `beam_width` | 4 | Viterbi beam |
| `max_optical_cost` | 3.6 | pairwise ceiling = 1.6 × = 5.76 |
| `max_reverse_sub_cost` | 4.0 | component-level gate, independent of the pairwise sum |
| `reserved_pairwise_slots` / `max_candidates` | 20 / 24 | pairwise candidates cannot be starved |
| `lambda_optical` / `lambda_lm` / `lambda_morph` | 1.2 / 1.8 / 1.5 | UNTUNED |
| `anchor_margin` / `oov_anchor_margin` / `gray_margin` | 3.5 / 3.0 / 3.0 | do-no-harm margins |
| `stage1_detection` | `True` | every token goes through Stage 1 |
| `suspicion_pass_below` / `suspicion_flag_above` | 0.30 / 0.70 | superseded by the model payload (0.10 / 0.65) when the adapter builds |
| `enable_merge_split` / `merge_split_penalty` | `True` / 2.0 | Stage 4, UNTUNED |
| `score_eos` | `True` | `</s>` is scored |
| `indel_cost_cap` | 2.4 | |
| `canonicalize_surface` | `False` | surface spelling restored on PASS when enabled |
| `use_morphology` | `None` | use the analyzer when importable |

**KenLM.** `AmharicOCRCorrector()` **constructs** without a language model; the
first `correct_text()` raises `ValueError` naming `AMHARIC_WORD_KENLM`,
`AMHARIC_DATA_ROOT`, `kenlm_path=` and `lm=`. Verified directly. Construction in
this sandbox takes ~18 s because it loads the 737k lexicon and builds the Stage
1 adapter; in production terms that is the 222 s initialization the last
benchmark recorded on the author's machine, dominated by the LM and index.

**Stage 1 wiring.** The corrector creates `CorrectorDetectionAdapter(self)`,
which builds the child `DetectionEngine`, hands it the corrector's own
vocabulary (`adopt_lexicon`, model's freq filter still applied) and delegates
every verdict to it. Rule-based `SuspicionScorer` is only the fallback when the
engine cannot be constructed. If the handover fails, the adapter records
`last_error` and keeps correcting — it never raises into the beam.

---

## 4. Data inventory (exact counts, this checkout)

| Path | Bytes | Rows / lines | Notes |
|---|---:|---:|---|
| `1_data_preparation/data/amh_final_training_text.txt` | 58,248,932 (55.6 MiB) | 359,776 lines, 4,420,102 tokens, 638,125 unique | the "4.4M tokens" corpus; 22,714,994 characters |
| `1_data_preparation/data/mined_rare_sentences.txt` | 58,248,932 | 359,776 | same size and line count as the training text but a **different 51M-byte content**; scripts read `repo_paths.workspace("mined_rare_sentences.txt")`, i.e. the external workspace copy, not this one |
| `1_data_preparation/data/amh.training_text` | 24,896,171 | — | Tesseract training text built from the corpus (with `amh.unicharset`, `amh.lstm-unicharset`, 20 KB each) |
| `OCR_error_detection/Model/main_i7_frq10.txt` | 15,052,074 | 737,106 | the dictionary; min freq 10, max 5,672,129 (`ነው`) |
| `OCR_error_detection/main_i7_frq10.txt` | 15,052,074 | 737,106 | **byte-identical duplicate**, only a fallback in `CANDIDATE_LEXICON_PATHS` |
| `OCR_error_detection/Model/local_lexicon.txt` | 14,809,281 | 750,870 | fallback lexicon, not the default |
| `OCR_error_detection/Model/harmonized_corpus_jiwer.tsv` | 50,595,285 | 946,058 | 1,367-page training corpus |
| `OCR_error_detection/Model/detector_lgbm_conffree.pkl` | 3,655,005 | — | the shipped model |
| `OCR_error_detection/Model/confusion_matrix.json` | 47,847 | 318 `sub` rows | keys `sub, del, ins, default_sub, default_del, default_ins` |
| `2_training/config/confusion_matrix.json` | 48 KB | — | **different content, read by no code path** (§7.3) |
| `OCR_error_detection/Evaluation/` | 570 MB | 100 GT pages, 100 OCR pages, `analysis.tsv` 62,521 rows | the 100-page eval set |
| `5_correction_evaluation/data/correction_decisions.tsv` | 3.1 MB | 21,927 rows | 976 TRUE_POSITIVE, 1,348 FALSE_POSITIVE, 1,370 SUBOPTIMAL_FLIP, 18,233 FALSE_NEGATIVE |
| `5_correction_evaluation/data/failed_cases.tsv` | 712 KB | 7,979 rows | |
| `5_correction_evaluation/benchmarks/benchmark_full_symspell.json` | 160 KB | 150 pages | see §5.1 |

`git ls-files` totals **1,143 tracked files / ~959.6 MB**, of which 570 MB is
the evaluation PNG set.

---

## 5. Evaluation status — read the provenance before quoting a number

### 5.1 `benchmark_full_symspell.json` — the corrector's last committed measurement

150 pages, 63,518 reference words. **Produced by an earlier corrector revision,
not by HEAD** (it passes options — `candidate_backend`, `trie_search`,
`trusted_*_frequency` — that no longer exist). It is a measurement of a corrector
that is not in this tree.

| | raw | corrected |
|---|---:|---:|
| strict char accuracy | 98.82 % | 98.07 % |
| strict CER | 1.178 % | 1.930 % |
| strict WER | 4.674 % | 6.293 % |
| canonical CER | 1.059 % | 1.140 % |
| clean-token damage | — | 4.06 % strict / 2.66 % canonical |
| character-error reduction | — | **−63.8 %** strict, **−7.7 %** canonical |

Initialization 222 s, 55.3 tokens/worker-second. **No benchmark of HEAD exists.**
Until one is run, the honest statement is "the last committed version of this
corrector made OCR output worse; whether HEAD still does is unknown".

### 5.2 `OCR_error_detection/Evaluation/analysis.tsv` — held out by page, not by text, and stale

62,521 rows (4,267 error tokens, 58,118 correct). The stored verdicts imply:
flag recall 60.4 %, flag+gray recall 95.6 %, false-flag rate 1.19 % on clean
tokens. Two caveats:

* **Held out by page, not by text.** `Model/train_LightGBM.py` sets
  `DEV_PAGES = 100` and fits on `page_0101`…`page_1367`, so these 100 pages are
  not training rows. But their *text* is entirely present in the training corpus
  (`docs/lgbm_step2_gate.md`: 100 % 5-gram overlap once punctuation is handled),
  so results here are not evidence of generalisation to new vocabulary.

* **Stale.** Re-running the current engine on a 1,500-row random resample agrees
  with the stored labels only **85.6 %** of the time (39.3 % of stored `flag`
  rows now come out `gray`). Quote these numbers as "an older model on the dev
  pages", not as current performance.

In short: page-level held out, vocabulary-level seen, and produced by a previous
model. The sweep in §5.4 is different again — it is validation inside the
production fit.

### 5.3 A held-out synthetic set exists but has no results here

`docs/new_test_set_plan.md` (§Status, 2026-10-13) records that a 100-page
synthetic set was built in the external `synthetic_eval/` workspace from
`telegram_c4.txt`, with zero shared 5-grams with the training ground truth. It is
git-ignored, and **no detector or corrector result on it is recorded in this
repository.** It is the natural first target for the out-of-sample work.

### 5.4 The dictionary/threshold sweep (94.2 % recall, 0.01 % false flag)

Measured on the 1,267 production pages (`page_0101`…`page_1367`), split
1,200 train / 67 validation. It is the selection evidence for the shipped model,
and the only measurement behind the "FINAL: recall 94.2 %, false flag 0.01 %"
headline. It is **validation inside the training corpus**, not a held-out test:
the 67 validation pages are part of the same 1,267, and the text of the 100 dev
pages overlaps it (see §5.2).

---

## 6. Tests

| Suite | Command | Result (this checkout) |
|---|---|---|
| Child detector | `pytest OCR_error_detection/tests -q` | **56 passed** in 1.6 s |
| Repo-wide | `pytest tests -q` | **142 passed, 52 skipped, 2,930 subtests** in 236 s |

Skip reasons: the retired `detection/` package (tests assert it is absent), the
three quarantined suites that described a deleted corrector API
(`test_ocr_corrector.py`, `test_evaluator.py`, `test_phase_a.py` — replaced by
`tests/test_head_contract.py`, now 75 contract tests), and two
environment-dependent contract checks.

**The morphology fix in §7.1 changed these numbers.** Before it,
`tests/test_head_contract.py` ran 70 passed / 6 skipped because five
morphology-dependent tests skipped (the vendored analyzer could not import);
after it, 75 passed / 1 skipped. That is why the repo-wide total moved from
137/57 to 142/52. `unittest discover` works too, but the counts above are
pytest's.

---

## 7. Known gaps and defects found while verifying

1. ~~**The vendored segmenter is unreachable after the reorg.**~~
   `_vendored_segmenter_path()` built `<dir of the corrector>/vendor/segmenter`,
   and the corrector now lives in `4_ocr_correction/`, so it looked for
   `4_ocr_correction/vendor/segmenter` — which does not exist, leaving
   morphology silently disabled and `lambda_morph` cancelled out. **Fixed** in
   this pass: the path now resolves through `repo_paths.REPO_ROOT`. Verified
   after the fix: `HAS_MORPH=True`, `MORPH_PATHS[0]=<repo>/vendor/segmenter`,
   `segmenter = AmharicMorphologicalSegmenter`, and the analyzer is actually
   called on OOV tokens (e.g. `ተከትሎማቸውም` → `known=False`, `pos=UN`,
   `stem=ተከትሎማቸ`; `በሸታና` → `known=True`, `pos=VB:PERFECTIVE:TEMPORAL_PAST`).
   Side effect on the test suite: five contract tests that used to skip because
   the analyzer could not import now run (§6).
2. **`vendor/segmenter/morph_types.py` needs `pydantic`** (documented in
   `vendor/segmenter/VENDORING.md`) but it was missing from `requirements.txt`,
   which is why gap 1 stayed invisible for so long: with no `pydantic` the
   import fails even on a correct path. Added to `requirements.txt`.
3. **Two confusion matrices.** `2_training/config/confusion_matrix.json` differs
   from `OCR_error_detection/Model/confusion_matrix.json` and is read by no code
   path; the `Model/` copy is the single source. Treat `2_training/config/` as a
   stale artefact.
4. **15 MB duplicated dictionary** (`OCR_error_detection/main_i7_frq10.txt` ==
   `Model/main_i7_frq10.txt`). The root copy is only a fallback.
5. **Build junk is tracked**: `__pycache__/*.pyc` (3), `augraphy_cache/*.png` (9),
   `font_diag_temp/` (48 images). `.gitignore` lists them, but tracked files are
   not affected by ignore rules.
6. **The README's flagship example was wrong** (§2) and **`40 OK`** in the test
   section was three generations of model old. Both fixed in this pass.
7. **`5_correction_evaluation/README.md` never mentioned that the corrector
   measured worse than raw OCR.** §5.1 added there.
8. **`docs/lgbm_review/build_synthetic_source.py` looked for its corpora at the
   repository root** (they moved into the stage folders) and for
   `harmonized_corpus_jiwer.tsv` outside `Model/`. Its `os.path.exists` guards
   meant the decontamination step would silently run against nothing — the
   "0 shared 5-grams" property would have been reported as intact while it was
   not. Paths fixed and missing files now abort. The other `docs/lgbm_review/`
   helper scripts had hard-coded `/home/user/OCR_corrector/...` paths; they now
   resolve them from `__file__`.
9. **`2_training/data/` contains only a README** — the Tesseract inputs
   (`amh.training_text`, `amh.unicharset`, `amh.lstm-unicharset`) are in
   `1_data_preparation/data/`. The old README claimed otherwise; it is now a
   pointer note.

---

## 8. How these numbers were produced

```bash
# detector facts
python - <<'PY'
import joblib
p = joblib.load('OCR_error_detection/Model/detector_lgbm_conffree.pkl')
print(list(p), p['pass_thresh'], p['flag_thresh'], p['features'])
print('trees', p['model'].num_trees())
PY

# defaults + lexicon + timing
python - <<'PY'
import time
from OCR_error_detection import DetectionEngine
t0 = time.time(); d = DetectionEngine(); print('init', round(time.time()-t0, 2), 's')
print(d.lexicon_info)
for w in ["እና", "በሽታ", "በሸታ", "አና", "የአድሜ", "ምዕፅራፍ"]:
    v = d.verdict(w, doc_freq=2)
    print(w, v.label, round(v.suspicion, 4))
PY

# tests
pytest OCR_error_detection/tests -q
pytest tests -q

# corpus counts
python - <<'PY'
t = open('1_data_preparation/data/amh_final_training_text.txt', encoding='utf-8').read()
print(len(t.splitlines()), len(t.split()), len(set(t.split())))
PY
```

Measured on the sandbox described at the top; timings are machine-specific and
should be re-measured rather than copied into new claims.

---

## 9. Change log of this file

* **2026-10-24** — created. Detector defaults, verdicts, timings, data counts,
  benchmark provenance, test counts, gaps 1–9. Same day, after this file was
  written: gaps 1–3 and 8 were fixed (vendored-segmenter path, `pydantic` in
  `requirements.txt`, `docs/lgbm_review/` paths), which raised the root test
  total from 137/57 to 142/52 (§6).
