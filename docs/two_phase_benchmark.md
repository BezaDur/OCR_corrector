# Two-phase SymSpell validation

> **Status (2026-10-24): procedure is current, some flags are not.** The
> evaluator exists and runs, but: `--backend trie` no longer exists (only
> `symspell` is accepted; the trie backend was removed), `--trie-search` and
> `--trusted-{long,short}-word-frequency` are rejected with an explanation, and
> `requirements-symspell.txt` does not exist in this tree — `symspellpy==6.10.0`
> is in `requirements.txt`. The evaluator's CLI defaults (`--max-optical-cost
> 4.8`, `--beam-width 5`, `--lambda-optical 0.9`, `--lambda-lm 2.0`) also differ
> from the corrector's constructor defaults (`3.6 / 4 / 1.2 / 1.8`), so pass
> every value explicitly and record it in the JSON report. Last committed result
> and its provenance: `docs/current_state.md` §5.1.

This is a benchmark procedure, not a claimed result. The supplied 6,928→903 ms measurement is an approximately 87% reduction for that run, but the old script timed candidates before sentences, warming its caches. Adding those token timings to sentence timing is invalid. SymSpell hash probes are average O(1); whole lookups also generate signatures, scan collision buckets, verify suggestions and rerank them. Exact candidate lists and 15–30 ms sentence latency cannot be promised. Full-vocabulary RAM and morphology costs may still dominate.

## Phase A — targeted regressions and cold/warm measurements

Install the optional dependency in the environment with KenLM and your segmenter:

```bash
python -m pip install -r requirements-symspell.txt
```

Run from the repository root. `test_page1_fixes.py` retains the original machine's KenLM/dictionary defaults, uses the repository confusion matrix, and now defaults to SymSpell distance 1. Override paths with `--kenlm`, `--dictionary`, and `--confusion` when needed.

```bash
python 4_ocr_correction/tests/test_page1_fixes.py --backend trie --repeats 5 \
  --json-report benchmark_results/phase_a_trie.json

python 4_ocr_correction/tests/test_page1_fixes.py --backend symspell --symspell-distance 1 --repeats 5 \
  --json-report benchmark_results/phase_a_symspell.json
```

Use separate processes so you do not keep two multi-million-word indexes in RAM or mix process high-water RSS. Both commands use identical beam/scoring settings: beam 4, optical ceiling 4.0, optical weight 1.0, LM weight 1.8, morphology weight 1.5. Frequency trust defaults to > 1,000 for 4+ character words and > 4,000 for 2–3 character words (no one-character shortcut). Override both benchmarks consistently using `--trusted-long-word-frequency` and `--trusted-short-word-frequency`. Deep morphology caching defaults to 65,536 entries.

The script:
- Measures initialization, vocabulary size, and peak process RSS (where supported).
- Records analyzer calls/time, failures, frequency-trust bypasses, and cache hits/misses separately for each cold/warm sentence. These snapshot deltas include morphology invoked during candidate generation and decoder scoring. The timer wraps analyzer execution plus tag handling, not the entire validation pipeline.
- Clears application caches before each cold sentence repetition, then immediately measures a warm repetition. OS/model pages and analyzer-internal caches are **not** reset; this is not machine-cold startup latency.
- Checks both occurrences of `አና -> እና` at their positions, plus `አንደሚያደርጉ -> እንደሚያደርጉ`, `ሶስትዮሽቨ -> ሶስትዮሽ`, and `በሀዳሴው -> በህዳሴው` in every cold/warm output.
- Profiles cold/warm token generation in an independent pass after sentence measurements, and checks target candidate recall.
- Records full measured outputs, timings, candidates, and non-target lexical changes for review. Target success is **not** proof the entire sentence is accurate. Non-target changes are flagged, not automatically judged against unannotated OCR.
- Returns nonzero if a target is missing, output is unstable, or candidate recall fails. Requires a working segmenter unless `--allow-no-morphology` is explicitly requested for diagnostics; that override is not full-stack validation.

Do not proceed to Phase B until the target checks pass and any other changed words have been reviewed. The 5M-vocabulary initialization itself must complete successfully on your machine; local fixture tests cannot establish that.

## Phase B — 150 pages, unchanged OCR baseline

Use the *same* raw OCR directory for both backends and distinct corrected-output directories. This command explicitly matches Phase A weights instead of silently using the evaluator's older defaults:

```bash
python 5_correction_evaluation/01_evaluate_full_corrector.py \
  --kenlm /path/to/amharic_5gram.binary \
  --dictionary /path/to/main_i7.txt \
  --confusion confusion_matrix.json \
  --gt-dir /path/to/synthetic_eval/ground_truth \
  --raw-dir /path/to/synthetic_eval/ocr_new_model \
  --output-dir /path/to/synthetic_eval/corrected_symspell \
  --backend symspell --symspell-distance 1 \
  --beam-width 4 --max-optical-cost 4.0 \
  --lambda-optical 1.0 --lambda-lm 1.8 --lambda-morph 1.5 \
  --workers 1 --expected-pages 150 \
  --json-report benchmark_results/phase_b_symspell.json
```

Repeat with `--backend trie`, a separate output directory, and a separate JSON report. The evaluator calculates raw accuracy from these pages; **98.86% is not hard-coded or assumed**.

Both strict and canonical metrics are produced in **one correction run**:
- Raw/corrected character accuracy, CER, WER, character-error reduction.
- Alignment-derived clean-token damage range, its clean-token denominator, and excluded ambiguous raw positions.
- Additive per-page counts, corpus totals, correction-only worker seconds, input tokens/worker-second, settings and file paths.
- Missing inputs or a page count other than 150 fail rather than silently reducing coverage.

`--canonical-metrics` selects the mode for an optional gate; it no longer suppresses strict reporting. To enforce a zero-damage gate on canonical text, add:

```bash
--canonical-metrics --max-clean-damage-rate 0
```

Choose a deployment threshold on held-out development evidence, not on this single page. A gate fails if the upper alignment-derived rate exceeds the threshold or its denominator is zero. JSON is saved before gate failure for inspection. A low damage rate alone is not sufficient: demand net CER/WER improvement too.

Correction timing excludes alignment, metrics, and I/O. End-to-end evaluation can still be slow: word alignment uses exact quadratic-time DP, and character CER uses Python Levenshtein (linear memory but quadratic time). The single-worker wall timer excludes initial model load; multiworker wall time includes worker startup, with per-worker initialization reported as unavailable. These are not comparable startup measurements. Begin with one worker.

## What clean-token damage means here

`ocr_metrics.py` replaces positional guessing with two reference-anchored word alignments. Optimality is minimum unit-cost word edits, then maximum exact matches.

1. A gold token is included in the clean denominator only if it matches exactly in **every** co-optimal raw alignment. Raw positions whose cleanliness depends on a tie are excluded and counted.
2. Over all co-optimal corrected alignments, compute the minimum and maximum losses of these clean gold tokens.
3. Damage rates are `100 * losses / clean tokens`; aggregate counts before division. An empty denominator is N/A (`null` in JSON), not zero.

Repeated-word ties can yield a range. For gold/raw `A A` and corrected `A`, exactly one clean word is lost regardless of which copy aligns, so the range is 50–50%, not 0–100%. Inserted tokens have no gold position; they contribute to WER and insertion counts, not this damage numerator.

These values are **alignment-derived estimates**, not causal token provenance or statistical confidence intervals. Arbitrary reordering/duplicate words can mask which observed token was changed. Canonical tokenization can also change the denominator. Inspect affected pages and preserve strict results. Truly exact edit provenance would require source-span tracking through normalization/decoding, which is not implemented here. The conservative anchor does not guarantee zero damage.

## Local validation and outstanding work

60 automated tests pass, including the real SymSpell fixture tests and exhaustive tiny-sequence oracles for alignment/tie counts and for optimal mapping options, gap slots, and adjacent-edit flags. Phase A harness and evaluator tests use fake correction/LM components where necessary; none of those timings measure the full model.

The production KenLM binary, external segmenter, full lexicon, and 150 evaluation pages remain unavailable in the sandbox. Run the commands on the machine holding them and share the JSON reports for interpretation. Script 05 is now alignment-derived (see `docs/decision_audit.md`), but it is a per-token decision audit and still must not be used for this damage metric.
