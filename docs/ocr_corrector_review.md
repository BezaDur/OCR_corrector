# Amharic post-OCR corrector: speed and accuracy review

## Scope and limits

Reviewed the active `amharic_ocr_corrector.py`, its older `(copy)` version, punctuation normalizer, scripts 01–12, smoke tests, confusion matrix, and checked-in error reports. The training-data/font utilities provide upstream context, but this review focuses on post-processing rather than retraining Tesseract.

The checkout does **not** include the KenLM binary, `main_i7.txt`, the external morphological segmenter, or the paired evaluation-page directories. Neither `kenlm` nor `segmenter` is installed here. Consequently, no production throughput or new CER/WER result is claimed. The segmenter's linguistic accuracy and return-schema semantics cannot be verified from this repository.

Small candidate-generation probes used the real confusion JSON and a tiny trie, with no LM scoring. A counting stub tested repeated morphology calls only. These establish local code behavior, not end-to-end model accuracy.

## Current pipeline

1. `07_generate_eval_pages.py` renders text in multiple fonts and applies degradation.
2. `01_run_ocr_eval.py` runs the older configured OCR model into `ocr_pred`; `02_benchmark_new_model.py` runs the configured newer model into `ocr_new_model`.
3. `01_build_confusion_matrix.py` aligns ground truth to `ocr_new_model`, estimates directional edit costs, and writes the JSON.
4. The active corrector loads KenLM, the external segmenter, and a nested-dictionary trie seeded from segmenter frequencies plus `main_i7.txt` and abbreviations.
5. Each line is normalized and split on whitespace; outer punctuation is peeled and selected combining marks are removed.
6. Frequent words bypass search. Other words get weighted trie candidates, plus an authenticated prefix repair and a trailing-character removal heuristic. The observed word is normally retained at zero cost.
7. A left-to-right beam combines KenLM, optical cost, boolean morphology reward, and an observed-word anchor bonus.
8. `01_evaluate_full_corrector.py` calculates raw/corrected CER and WER. `02_collect_error_cases.py` classifies residual correction decisions from exact word alignments (see `docs/decision_audit.md`). `06_audit_raw_ocr_sentences.py` separately canonicalizes punctuation before a raw-OCR audit.

**Good foundations:** directional confusion costs; always retaining the observed token; separating punctuation from lexical LM input; caching candidate sets; and a conservative bias toward keeping recognized words. Keep these ideas.

**Important limitation:** morphology currently validates candidates; it is not a general morphological candidate generator. Inflected OOV repairs are mostly unreachable unless the full form exists in the trie or happens to match the one prefix rule.

## Confirmed problems

### 1. Candidate thresholds silently ignore configuration

Location: `generate_candidates`, `_generate_candidates_uncached`.

`generate_candidates(ocr_token, max_cost=...)` never uses `max_cost`. The constructor's `max_optical_cost` is stored but never consulted by search. Thus the evaluator's `max_optical_cost=4.8` has no effect.

Actual thresholds are:

| Token class | Actual radius |
| --- | ---: |
| In dictionary or morphology-valid | 2.4 |
| Unknown, 1–3 Unicode characters | 2.6 |
| Unknown, 4–6 characters | 3.0 |
| Unknown, 7+ characters | 2.2 |

Amharic fidel characters are not Latin letters: `በሸታ` and `ክተማ` each have length **3**. They hit the short-token radius.

Using the checked-in matrix and a tiny lexicon containing the gold words:

| Gold ← observed | Substitution cost | Returned gold candidate? |
| --- | ---: | --- |
| `በሽታ` ← `በሸታ` | 3.3322 | No |
| `ውስጥ` ← `ውሰጥ` | 5.7643 | No |
| `ከተማ` ← `ክተማ` | 2.7451 | No |
| `እንደ` ← `አንደ` | 2.1666 | Yes |
| `በህዳሴው` ← `በሀዳሴው` | 2.6911 | Yes |

Even requesting `max_cost=10` leaves `በሽታ` absent for `በሸታ`.

Only **27 of 425 stored substitution entries** fit radius 2.4; 50 fit 3.0. This is entry coverage, not a frequency-weighted recall estimate. Nevertheless, several prominent errors are demonstrably excluded before KenLM can help.

**Action:** define threshold semantics explicitly and include the resolved search policy in cache keys. Separate candidate-retrieval budgets from final optical scores. Do not simply raise the radius globally over a multi-million-word Python trie.

### 2. Morphological work is repeated despite the candidate cache

Location: `validate_morphology` and `correct_sentence`.

Morphology is checked during candidate generation, again for the observed-word anchor, and again for each candidate during every occurrence in decoding. The candidate LRU does not cache those later checks. A counting segmenter receives 100 `segment_word` calls for 100 validations of the same word.

**Action:** cache morphological analyses per corrector instance, including ordinary unknown results; derive validation/reward from that cached analysis. Avoid permanently caching transient analyzer failures. If no segmenter exists or `lambda_morph == 0`, avoid reward-related analysis. For a singleton candidate, its morphology/anchor bonuses cannot affect which path wins; those particular reward calls can be skipped, but the lexical token must still advance KenLM state.

Treat POS presence as evidence, not proof of correct spelling. Verify whether the actual analyzer returns enums, guessed POS, known stems, or confidence. `str(pos)` is not necessarily comparable to literal `UNKNOWN`.

### 3. One-character Amharic words are discarded

Location: `_add_to_trie`.

All words of length less than two are rejected, including `በ` and `ከ`, although both appear in the ambiguity list. They cannot become dictionary-generated alternatives through the current trie.

**Action:** admit verified one-fidel lexical words. Protect standalone digits and punctuation through token typing rather than a blanket minimum length.

### 4. Frequency bypass makes real-word OCR errors unreachable

Location: `common_words` and the first return in candidate generation.

A frequent observed token produces only itself, regardless of context or optical neighbors. The small hard-coded ambiguity list cannot cover every valid-word substitution. A large lexicon may also include frequent source-OCR misspellings; frequency alone is not a correctness certificate.

**Action:** retain the fast path, but permit a cheap shortlist of directional confusion neighbors for real-word errors. Use a stricter replacement margin for valid/high-confidence originals instead of eliminating alternatives entirely. Keep verified abbreviations protected and test them explicitly.

### 5. Optical costs are partly overwritten and incompletely calibrated

Locations: `_get_del_cost`, `_get_ins_cost`, `01_build_confusion_matrix.py`.

Both insertion and deletion costs are capped at 2.4. The stored insertion costs range from 8.5155 to 11.5129, so **every stored insertion gets exactly 2.4**. Relative insertion evidence is lost. A substitution costing more than 4.8 can be represented as one deletion plus one insertion costing 4.8, where the search budget permits it.

The builder uses natural logarithms; KenLM returns base-10 log scores. Tuned weights can absorb a constant conversion, but the current score is not a directly calibrated probabilistic combination. Match costs are zero rather than estimated channel likelihoods, and unseen substitutions default to 4.5 even when measured rare substitutions have larger costs. A rare observed event can therefore be penalized more than an unseen one.

There is a zero-cost deletion for Latin `C`. The current `n+1` length bound prevents an unbounded search, but cheap/zero deletions weaken pruning. Do not derive a safe maximum edit count or candidate length from a positive minimum cost unless the matrix guarantees one.

**Action:** save counts and denominators alongside costs; smooth sparse events; estimate match/error evidence consistently. Keep full channel scores separate from cheaper retrieval heuristics. Convert log bases before retuning weights. Do not remove the cap without recalibrating retrieval and acceptance: that alone would suppress many existing repairs.

### 6. Decoder omits EOS and uses physical lines as sentences

Location: `correct_sentence`, `correct_text_block`.

The beam never scores `</s>`, so the last word lacks an important right-boundary signal. Line-by-line processing resets context at line wraps, while multiple sentences on one physical line share a history. Punctuation is skipped rather than delimiting LM sentences. Interior punctuation attached without spaces remains inside lexical tokens.

**Action:** establish sentence/token conventions consistent with the LM's training preprocessing. Add EOS to hypothesis scores and reset at actual sentence boundaries. Preserve line-break layout independently of linguistic context. A narrow beam may prune a correct candidate before later right context arrives; measure candidate recall separately from beam survival.

The 1.6 anchor is an accumulated score feature, not a guaranteed local confidence threshold. It favors preservation, but cannot guarantee that each replacement has enough evidence.

### 7. Normalization changes content and layout independently of spelling

Locations: `EthiopicPunctuationManager`, `amharic_canonical_normalizer.py`.

Examples: `12:30` becomes `12 30` in the corrector; `,,`/`፣፣` become a full stop there but a comma in the canonical normalizer. Combining marks such as U+135F are removed from the returned surface token even when no correction is selected. Splitting/rejoining normalizes whitespace, and block processing loses indentation, original line-ending style, and a final newline.

**Action:** separate surface spans from lookup-normalized words. Preserve the original surface when retaining a token. Offer an explicit punctuation-normalization mode; do not confuse dictionary lookup normalization with publication spelling. Handle periods at outer boundaries without breaking internal dots in abbreviations or decimal numbers. Add token typing for numbers, URLs, Latin text, abbreviations, and Ethiopic numerals.

### 8. Candidate truncation and hard-coded shortcuts can hide alternatives

The prefix rule returns early without applying the ordinary candidate limit or optical threshold. It preserves the original but can omit a better third candidate. `MAX_CAND_LEN = n + 1` blocks repairs requiring two missing fidel characters even when their costs would otherwise fit. Sorting all results before slicing is unnecessary when a shortlist alone is needed. Zero-cost alternatives can theoretically displace the original under a small `max_candidates` limit.

**Action:** reserve a slot for the observed token; validate positive beam/candidate limits; apply consistent policies to heuristic candidates. Keep prefix rules as candidate sources, not exclusive exits. Tag candidate sources for debugging.

## Evaluation issues to fix before claiming improvement

- **Dataset mismatch:** scripts 03/04 use `ocr_new_model`; several upstream diagnostics still use `ocr_pred`. Reports may compare different OCR model generations. *Partially fixed:* script 05 now defaults to `ocr_new_model`; scripts 02/06 remain on `ocr_pred`.
- **Raw-token misalignment:** *Fixed for script 05.* It previously estimated raw slices by proportional position in the gold page, so a split, merge, dropped line, or earlier insertion shifted everything; its 884 over-corrections and 2,363 segmentation cases (still committed as `error_summary_report.txt` / `failed_cases.tsv`) are reported classifications, not trustworthy exact event counts. Script 05 now uses the exact word-edit DP and abstains on shifted, boundary, and multi-token cases. See `docs/decision_audit.md`.
- **Normalization mismatch:** evaluator 04 compares uncanonicalized text, while audit 12 canonicalizes punctuation. Many reported failures are `፡፡` → `።`, not lexical mistakes. Report strict and canonical metrics separately, using the same declared policy on gold, raw, and corrected text. Include CER with spaces to expose boundary errors, as well as the current no-whitespace CER.
- **Alignment contamination:** the matrix builder zips replacement line blocks. A fused/dropped line can create many false character confusions. Use layout-aware matching or well-aligned text spans and inspect difficult alignments. `'-'` as a gap sentinel collides with real hyphens; use `None` or a unique sentinel.
- **SequenceMatcher is not edit-distance DP:** audit 12's docstring promises DP, but the implementation uses SequenceMatcher. Its default popular-token heuristic can distort repetitive pages. Use an explicit alignment method appropriate to the desired metric/classification.
- **Timing contamination:** evaluator 04's page duration includes correction, file I/O, and four full-matrix edit-distance calculations. This is not isolated correction latency. Replace scalar edit-distance calculations with a tested native implementation or two-row DP and time them separately.
- **Coverage:** missing pages are silently skipped while some summaries use total discovered pages. Report completed/missing/failed counts; fail on unexpected missing inputs.
- **Configuration:** sequential and multiprocessing evaluators use different correction parameters. Use one factory/configuration and record model/matrix/lexicon hashes.
- **Holdout:** confusion estimation and correction evaluation currently point to the same dataset. Reserve separate channel-estimation, tuning, and untouched test documents. Check sentence/document overlap with OCR and LM training; include real scans, names, numerals, and uncommon inflections. Image-level random splits do not establish text/source independence.

The checked-in confusion report says 98.91% character accuracy on its evaluated sample. This is not equivalent to 98.91% word accuracy or a guarantee of real-document performance. With a strong OCR model, damage to correct tokens is a first-class metric.

## Recommended candidate architecture

**Tier 0 — original and protected token types.** Always preserve the original candidate and surface form. Verified abbreviations should not need a general trie walk.

**Tier 1 — reverse-confusion neighbors.** Precompute `observed_char -> [(true_char, channel_cost)]` from the directional matrix. Enumerate selected single-fidel replacements, then use O(1) lexicon membership. This retrieves high-cost but documented errors without widening a global weighted radius. Candidate retrieval is driven by edit count/fan-out; final ranking still uses the original directional cost.

**Tier 1b — controlled insertion/deletion and order variants.** Try deleting an observed fidel and inserting likely missing fidels. Same-consonant vowel-order variants are useful hypotheses but must use a verified Unicode/fidel table: Ethiopic blocks are not uniformly safe seven-character arithmetic. Measure additional variants' precision. Do not equate phonetic spelling equivalence with faithful OCR transcription.

**Tier 2 — OOV morphological repair.** Use the actual analyzer to recover plausible affix/stem structure, repair the stem or uncertain affix, reconstruct the full form, and validate it. Start with restricted confusion-based OOV variants and cap analyzer calls. Prefer authenticated stems over merely non-UNKNOWN POS. Avoid hand-written exhaustive affix combinations until the analyzer's behavior is known.

**Tier 3 — bounded fallback.** For unresolved suspicious tokens, retain weighted trie search with explicit edit-count/expansion limits and observable truncation. Later evaluate a compact trie/DAWG or native implementation. A fixed-edit SymSpell index is an experiment, not a drop-in equivalent to arbitrary weighted Levenshtein, especially with zero-cost operations and multi-million vocabularies.

**Boundary layer — split/merge candidates.** This requires a token-span lattice, not another per-token spelling rule. An edge should consume one or two observed tokens and emit one or two corrected words, with a calibrated space-error penalty. Advance KenLM for every emitted word. Protect abbreviations, numerals, and legitimate inflections from indiscriminate splitting.

Keep candidate sets capped and diverse: original, lexical repairs, morphological repairs, boundary repairs. Rank/accept using optical evidence, LM gain, morphology evidence, and preservation/change features. If OCR TSV/hOCR confidences become available, calibrate them on held-out data and use them to prioritize search; raw Tesseract confidence is not a probability.

## Speed improvements, in order

1. Cache morphological analyses; precompute candidate-local features outside beam expansion.
2. Precompute per-token insertion costs and per-character substitution rows; avoid nested JSON lookups and method calls in each DP cell.
3. Add the cheap confusion-neighbor retrieval tier; avoid unrestricted searches for common/protected tokens.
4. Use backpointers instead of copying the entire path for every beam expansion. Today path copying grows with sentence length across approximately `beam_width * candidates` expansions per position.
5. Merge equivalent LM histories where supported by KenLM's state API, keeping the best score. Verify equivalence; never merge merely by last word for a 5-gram model. Use top-k selection instead of sorting every expansion if profiling supports it.
6. Profile trie memory and load time. Nested Python dictionaries can dominate storage independently of KenLM's binary mapping. Benchmark compact storage before choosing it; the segmenter may hold its own vocabulary too.
7. Only then add parallelism. Current Python trie DP is not made CPU-parallel by ordinary threads. Multiple worker processes duplicate large objects unless a measured sharing strategy is used; portability and KenLM/segmenter behavior matter. Use identical parameters in every worker.

## Proposed implementation sequence and acceptance gates

### Phase A — trustworthy baseline, minimal semantic changes

- One portable CLI/config for paths and correction settings.
- True raw/gold/corrected alignment and strict/canonical metrics.
- Separate initialization, correction, scoring, and I/O timers; collect peak RSS, words/sec, cold-cache and warm-cache p50/p95 token latency.
- Log candidate counts, source, search truncation, morphology calls/cache hit rate, and correction decisions.
- Add dependency-free unit tests using tiny tries/counting analyzers plus real-KenLM integration tests when assets are supplied. Existing `test_*.py` files instantiate external models at import time and are smoke scripts rather than portable automated tests.
- Implement analysis caching and local DP cost precomputation; test candidate/decoded-output parity on a frozen baseline.

### Phase B — candidate recall and calibrated ranking

- Fix threshold API semantics, single-fidel vocabulary admission, original retention, and consistent heuristic handling.
- Add bounded reverse-confusion retrieval and real-word alternatives.
- Add EOS and consistent log-base handling, then retune LM/optical/change/morphology weights. Do not change units under old weights and assume accuracy is preserved.
- Measure **oracle candidate recall** on aligned OCR errors: whether gold is in the set before decoding. Break it down by substitution, insertion/deletion, real-word errors, OOV inflections, length, and domain. Report beam survival separately.
- Measure correction precision, repair recall, clean-token damage rate, strict/canonical CER/WER, and runtime. Review names/abbreviations/numbers separately.

### Phase C — structural errors and scale

- Add morphology-backed reconstructed candidates and split/merge lattice edges.
- Compare compact/native search and parallel execution only after profiling.
- Run an ablation table: baseline; caching only; retrieval; EOS/scoring; morphology; boundaries. Tune on development data and accept on untouched documents. Use document-level uncertainty estimates rather than pooling correlated glyphs as independent samples.

**Bottom line:** first make evaluation reliable and eliminate repeated analysis. Then broaden candidate recall cheaply rather than enlarging the Python trie radius everywhere. KenLM cannot choose a gold word that never reaches the beam, and a post-corrector for ~98% OCR must explicitly earn each change to already-correct text.
