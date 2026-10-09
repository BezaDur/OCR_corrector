# Three-way decision audit

The supplied Step 13 script creates two `SequenceMatcher` instances but never
uses either. It compares `gold[i]`, `raw[i]`, and `corrected[i]` only up to the
shortest sequence. A single inserted/deleted token shifts subsequent comparisons;
equal overall lengths do not establish correspondence either. Unmatched tails
are silently discarded. Its four category totals are therefore not validated
repair/damage counts. The old repository auditor also used proportional raw
indices and omitted successful repairs.

`05_collect_error_cases.py` now uses `ocr_decision_audit.py` and the exact
word-edit DP in `ocr_metrics.py`. It does not load the corrector or rerun OCR.
There are no additional dependencies beyond Python's standard library.

## Run against the SAME raw files used to generate corrected output

```bash
python 05_collect_error_cases.py \
  --gt-dir /media/des/Tess/Tesseract2/synthetic_eval/ground_truth \
  --raw-dir /media/des/Tess/Tesseract2/synthetic_eval/ocr_new_model \
  --corrected-dir /media/des/Tess/Tesseract2/synthetic_eval/ocr_corrected \
  --expected-pages 150 --normalization strict \
  --output-tsv benchmark_results/decisions_strict.tsv \
  --output-summary benchmark_results/patterns_strict.txt

python 05_collect_error_cases.py \
  --gt-dir /media/des/Tess/Tesseract2/synthetic_eval/ground_truth \
  --raw-dir /media/des/Tess/Tesseract2/synthetic_eval/ocr_new_model \
  --corrected-dir /media/des/Tess/Tesseract2/synthetic_eval/ocr_corrected \
  --expected-pages 150 --normalization canonical \
  --output-tsv benchmark_results/decisions_canonical.tsv \
  --output-summary benchmark_results/patterns_canonical.txt
```

Missing raw/corrected pairs fail before report creation. `--allow-missing` permits
skips but lists every skipped page; `--expected-pages` still checks the number
of complete triples. Outputs must be distinct and must not overwrite inputs.
Make sure corrected files were generated from these exact raw files, not from
a different OCR run. Alignment cannot detect this provenance mistake reliably.

Strict mode compares whitespace-delimited surface tokens without punctuation
canonicalization. Canonical mode applies the same `normalize_for_metric`
function as the evaluator to all three texts before splitting. The current
canonicalizer does **not** equate `ዓ/ም` and `ዓ.ም`, or remove word-internal
hyphens. Those edits can remain errors in canonical mode.

## Attribution policy and limitations

For gold/raw, gold/corrected, and raw/corrected separately, DP minimizes
`(word edit count, -number of exact token matches)`. It retains mapping options
across every path tied under both objectives. This is not the set of all
minimum-edit-only paths: the exact-match tie-break is an explicit policy.

A gold-position row enters the five ordinary decision categories only when:

- its correspondence is invariant, one-to-one, and cycle-consistent in all
  three pairwise alignments;
- no gap occurs at that token or an immediately adjacent token/boundary in any
  optimal alignment;
- it is outside a multi-token edit block (including adjacent nonmatches on any
  optimal path).

Ordinary statuses are `TRUE_NEGATIVE`, `TRUE_POSITIVE`, `FALSE_POSITIVE`,
`FALSE_NEGATIVE`, and `SUBOPTIMAL_FLIP`. The TSV includes true negatives so
coverage and denominators are visible; the four top-pattern tables exclude them.
Other rows are retained as `AMBIGUOUS_ALIGNMENT`, `WORD_BOUNDARY`,
`COMPLEX_EDIT_BLOCK`, or `UNRESOLVED_CORRESPONDENCE`. Concatenation-preserving
whitespace shifts are flagged as boundary cases, even when token counts match.
This intentionally abstains on some genuine adjacent typos rather than
mislabeling displaced phrases as single-token OCR errors.

There is one TSV row per gold position, plus separate `RAW_UNMATCHED` and
`CORRECTED_UNMATCHED` rows from the deterministic traceback for unmatched
hypothesis tokens, including tails. Ambiguous unmatched rows carry an
`_AMBIGUOUS` suffix. Raw and corrected unmatched rows can describe the same
underlying event; summing them does not give an edit count. When a gold token
has no counterpart, its raw/corrected cell is empty (a gap).

Index columns are zero-based. On gold rows, `RAW_OPTIONS` and
`CORRECTED_OPTIONS` list possible hypothesis indices; on unmatched rows the
populated options column lists possible gold indices. `<gap>` denotes no
counterpart. Unresolved rows show one representative traceback in the token
columns: those columns alone must NOT be treated as certain correspondence.
Top-pattern tables include only resolved ordinary decisions.

**These remain alignment-derived diagnostics, not proof of causal edit
provenance.** Even invariant edit-distance correspondence can be linguistically
wrong, particularly for unrelated spans or reordered text. The auditor does not
reconstruct a true simultaneous three-sequence alignment, identify the actual
corrector edit operations, or decide whether gold contains an annotation error.
Use sentence review/edit traces for those questions. Use the existing evaluator's
strict/canonical CER, WER, and alignment-derived clean-token damage range for
corpus acceptance; this deliberately conservative audit does not replace them.

The real 150-page data is not available in this workspace. The previously
pasted 21,927-event report cannot be recalculated locally. No accuracy improvement
is claimed by fixing the auditor, and corrector scoring is unchanged in this step.

## Validation

Portable tests cover isolated decisions, insertion/deletion shifts, duplicates,
reordering, split/merge cases, equal-length whitespace shifts, empty sequences,
unmatched tails, normalization, missing files, page counts, and overwrite
protection. An independent exhaustive tiny-sequence oracle verifies all optimal
mapping options, gap slots, adjacent-edit flags, and deterministic traceback.

```bash
python -m unittest discover -s tests -v
```

The implementation uses quadratic page-sized word DP, three pairwise passes;
it is not intended for concatenating an entire corpus into one sequence.
