# 5_correction_evaluation — measure what the corrector changes

**Parent project, step 5: does correction help?**

## Read this first

The last committed benchmark, `benchmarks/benchmark_full_symspell.json`, shows the
corrector **making OCR worse**:

| 150 pages, 63,518 reference words | raw | corrected |
|---|---:|---:|
| strict character accuracy | 98.82 % | 98.07 % |
| strict CER | 1.178 % | 1.930 % |
| strict WER | 4.674 % | 6.293 % |
| canonical CER | 1.059 % | 1.140 % |
| clean-token damage | — | 4.06 % strict / 2.66 % canonical |
| character-error reduction | — | **−63.8 %** strict / **−7.7 %** canonical |

That run was produced by an **earlier corrector revision** — it passes options
(`candidate_backend`, `trie_search`, `trusted_*_frequency`) that no longer exist —
so it is not a measurement of HEAD. There is currently **no benchmark of the
current code**; running one is the open task. Provenance and caveats:
`docs/current_state.md` §5.1.

## What it does

| Script | What it does |
|---|---|
| `01_evaluate_full_corrector.py` | end-to-end benchmark: runs the corrector over `<page>.ocr.txt`, produces corrected files, strict + canonical CER/WER, clean-token damage, optional damage gate, JSON report |
| `02_collect_error_cases.py` | alignment-derived per-token triage into `TRUE_POSITIVE` / `FALSE_POSITIVE` / `SUBOPTIMAL_FLIP` / `FALSE_NEGATIVE` (uses `ocr_metrics.align_tokens`, not `SequenceMatcher`) |
| `03_audit_correction_decisions.py` | summarize a decisions TSV into repair/damage patterns |
| `04_stage1_precision.py` | Stage 1 precision/recall against a verdict dump |
| `05_fit_stage1_scores.py` | fit Stage 1 score weights |
| `stage1_labels.py`, `stage1_fit.py` | label/sample helpers for the above |
| `smoke_eval.sh` | quick run over the first N pages |

## How to run

```bash
python 5_correction_evaluation/01_evaluate_full_corrector.py \
  --kenlm /path/to/amharic_5gram.binary \
  --confusion 4_ocr_correction/error_detection/Model/confusion_matrix.json \
  --dictionary 4_ocr_correction/error_detection/Model/main_i7_frq10.txt \
  --gt-dir  /path/to/synthetic_eval/ground_truth \
  --raw-dir /path/to/synthetic_eval/ocr_new_model \
  --output-dir /path/to/synthetic_eval/corrected \
  --expected-pages 150 \
  --json-report benchmark_results/phase_b.json

python 5_correction_evaluation/05_collect_error_cases.py \
  --gt-dir ... --raw-dir ... --corrected-dir ... \
  --normalization canonical \
  --output-tsv benchmark_results/decisions_canonical.tsv \
  --output-summary benchmark_results/patterns_canonical.txt
```

The evaluator needs a real KenLM binary and a matching OCR/GT pair; it fails
rather than silently reducing coverage. `--max-clean-damage-rate 0` turns the
canonical damage metric into a pass/fail gate. Full flag list:
`--help`; the procedural notes are in `docs/two_phase_benchmark.md`.

Note that the evaluator's CLI defaults (`--max-optical-cost 4.8`,
`--beam-width 5`, `--lambda-optical 0.9`, `--lambda-lm 2.0`) differ from the
corrector's constructor defaults (3.6 / 4 / 1.2 / 1.8). Pass values explicitly
and record them with the report, or the run is not comparable to another one.

## Files

| Path | Contents |
|---|---|
| `data/correction_decisions.tsv` | 21,927 audited token events: 976 `TRUE_POSITIVE`, 1,348 `FALSE_POSITIVE`, 1,370 `SUBOPTIMAL_FLIP`, 18,233 `FALSE_NEGATIVE` |
| `data/failed_cases.tsv` | 7,979 rows, e.g. `OVER_CORRECTION` (`የሕዳሴው → የህዳሴው`), `UNREPAIRED_TYPO` |
| `benchmarks/benchmark_full_symspell.json` | the 150-page benchmark summarized above (older revision) |

The decision TSVs come from `02_collect_error_cases.py` and are alignment-derived
estimates, not causal token provenance — see `docs/decision_audit.md`. The
`FALSE_POSITIVE` rows are dominated by morphological over-peels
(`ተከትሎም → ተከትሎ`, `የሚታይም → የሚታይ`), which is what
`is_unsupported_morpheme_peel()` exists to stop.
