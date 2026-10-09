# Corrector improvements and optional SymSpell backend

## What changed

- Optional `candidate_backend="symspell"`, leaving `"trie"` as the default. SymSpell indexes the same lexicon and segmenter vocabulary; no second populated trie is built in this mode.
- Every SymSpell suggestion is rescored with **directional weighted Levenshtein** (`true -> observed`). SymSpell frequency/distance does not decide the final spelling. `Verbosity.ALL` allows real-word alternatives rather than stopping at an exact dictionary match.
- Cheap reverse-confusion substitutions are checked against the vocabulary before either backend. These also apply to frequent words that otherwise skip the trie walk.
- `max_optical_cost` and per-call `max_cost` now define a real **acceptance ceiling**, included in candidate cache keys. Heuristic prefix/trailing-character repairs obey it too.
- Original tokens always reserve a candidate slot; verified abbreviations remain protected. Single-fidel words are admitted.
- Per-instance morphology validation caching (32,768 entries), retry after transient exceptions, and Enum-aware UNKNOWN handling. Singleton candidates avoid unnecessary morphology rewards while still advancing LM state.
- Trie DP precomputes insertion costs and reuses character-cost rows. Decoder paths use backpointers instead of repeatedly copying entire paths.
- End-of-sentence scoring is on by default (`score_eos=False` disables it).
- Retained surface words keep combining marks. Numeric ASCII colons such as `12:30` survive normalization; repeated Ethiopic commas no longer become full stops.
- Missing explicit dictionary paths and invalid optical costs fail instead of silently degrading the model.

## Retrieval budgets versus scoring

The acceptance ceiling is **not** a guarantee of exhaustive retrieval. By default `trie_search="adaptive"` retains the old fallback radii (2.4 for known/morphology-valid words, 2.6 for short unknown words, 3.0 for medium unknown words, 2.2 for long unknown words), additionally capped by `max_optical_cost`. This avoids accidentally widening every long-token walk over a multi-million-word trie.

Reverse-confusion neighbors use the full configured ceiling, so documented one-character errors such as `በሸታ -> በሽታ` (3.3322) and `ክተማ -> ከተማ` (2.7451) are now reachable at the default constructor ceiling of 4.0, even with adaptive trie search.

`trie_search="full"` uses the full ceiling for the fallback walk; use it for recall experiments, not unmeasured production speed. Both trie modes retain the existing `len(observed)+1` candidate-length bound and frequent-word shortcut. SymSpell instead has its own edit-distance bound. All modes retain only `max_candidates`, including the original.

The existing insertion/deletion cap of 2.4 is retained for compatibility and exposed as `indel_cost_cap`. Set it to `None` to experiment with the stored matrix values, but retune and reevaluate: this changes scoring substantially. Log bases and weights have deliberately not been silently changed.

Call `clear_caches()` after changing search/scoring attributes that affect candidate generation or replacing the segmenter. Backend/index changes require constructing a new corrector. Direct mutation of the cost matrix is unsupported; rebuild the corrector so reverse-confusion tables remain consistent.

## Enable SymSpell

In your existing Python environment, which must already have working KenLM and the external segmenter installed/importable:

```bash
python -m pip install -r requirements-symspell.txt
```

```python
from amharic_ocr_corrector import AmharicOCRCorrector

corrector = AmharicOCRCorrector(
    kenlm_path="/path/to/amharic_5gram.binary",
    confusion_matrix_path="confusion_matrix.json",
    dict_vocab_path="/path/to/main_i7.txt",
    candidate_backend="symspell",
    symspell_max_edit_distance=1,
    max_optical_cost=4.0,
)
print(corrector.correct_text_block("ክተማ በሸታ"))
```

Start with **edit distance 1** and one worker. Distance 2 generates a larger delete index and can substantially increase RAM and startup cost for a multi-million-word vocabulary. A SymSpell index is not guaranteed to be smaller than the old trie. It is rebuilt on initialization; index persistence is not implemented here.

SymSpell can retrieve generic one-edit neighbors that are absent from the confusion matrix, but those still need to pass optical scoring. Its unweighted edit distance (including transpositions) is not equivalent to the weighted OCR channel. It cannot by itself generate unseen morphological forms or solve split/merge errors; those remain future work. More candidates can improve recall **or** increase over-correction—measure both.

## Compare backends on your evaluation pages

The evaluator now accepts paths and backend settings from the command line, uses identical options in sequential and spawned worker processes, reports correction-only worker time separately from I/O/metric computation, and uses linear-memory edit distance. Missing raw pages fail the run.

```bash
python 04_evaluate_full_corrector.py \
  --kenlm /path/to/amharic_5gram.binary \
  --dictionary /path/to/main_i7.txt \
  --confusion confusion_matrix.json \
  --gt-dir /path/to/synthetic_eval/ground_truth \
  --raw-dir /path/to/synthetic_eval/ocr_new_model \
  --output-dir /path/to/synthetic_eval/corrected_symspell \
  --backend symspell --symspell-distance 1 \
  --max-optical-cost 4.0 --workers 1
```

Repeat with `--backend trie` and a different output directory, keeping other settings identical. The evaluator's historical default optical ceiling is 4.8, so specifying 4.0 above is intentional; always compare equal ceilings. Use `--no-eos` for an EOS ablation. `--trie-search full` enables the wider fallback experiment.

The evaluator now reports **both strict and canonical metrics in one run**. `--canonical-metrics` selects the mode for the optional damage-rate gate. Canonical scoring applies the existing normalizer equally to gold, raw, and corrected text; it does not rewrite saved output. CER still excludes whitespace. Worker-time throughput is not parallel wall-clock throughput.

See [two_phase_benchmark.md](two_phase_benchmark.md) for Phase A cold/warm target checks, Phase B JSON reports, exact-150-page validation, matched scoring options, and the clean-token damage definition.

**Not fixed in this patch:** script 05's proportional raw alignment and old input directory. Do not use its old category counts as evidence of new correction precision. The evaluator now has alignment-derived damage reporting independent of script 05, with explicit tie ambiguity; causal edit-provenance auditing and CER including spaces remain future work. Production measurements still require the full assets. The main decoder still processes physical lines and normalizes whitespace; this patch does not promise layout preservation or true sentence segmentation.

## Tests and validation limits

```bash
python -m unittest discover -s tests -v
```

Tests use tiny lexicons and the checked-in confusion matrix. LM scoring tests use a deterministic fake LM; optional SymSpell tests use the real package. Tests cover threshold/cache semantics, directed costs, trie/oracle agreement under the declared bounds, morphology caching/recovery, single-fidel entries, protected abbreviations, surface marks, EOS selection, backpointer order, and evaluator metrics/worker configuration. `test_page1_fixes.py` is now an import-safe Phase A CLI benchmark; `test_corrector.py` remains a model-dependent smoke script. Use the explicit `tests` discovery command above.

In the sandbox, 39 tests passed with `symspellpy==6.10.0` installed. KenLM compilation was attempted but blocked by missing Python development headers (`Python.h`). The production KenLM binary, full lexicon, segmenter and evaluation pages are also absent. Therefore there is **no claimed production speedup or accuracy gain** yet. The prior `ocr_corrector_review.md` describes the pre-change baseline, not the current implementation.
