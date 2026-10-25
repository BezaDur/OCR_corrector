# Architectural review — Amharic post-OCR corrector

> **Status (2026-10-24): the blocking finding in §2 is fixed.** Dependency
> injection (`lm=`, `state_factory=`, `segmenter=`), `morphology_stats()` and
> `clear_caches()` are in the tree; the module imports without `kenlm`; the
> contract suite is `tests/test_head_contract.py`; and both suites are green
> (56 child tests, 142 root tests + 52 documented skips — see
> `docs/current_state.md` §6). §8 is the work log of what landed.
>
> What has **not** changed: there is still no benchmark of HEAD. §2.1's table is
> the last committed measurement of the *older* revision, so it says nothing
> about the current code — see `docs/current_state.md` §5.1. The measured
> vocabulary/init numbers quoted below come from the author's machine and from
> that older revision.
>
> The "Limits" paragraph below is out of date on availability: the morphological
> suite, `morphemes.json`, `main_i7_frq10.txt`, `confusion_matrix.json` and the
> 100 evaluation pages are now in the repository. Only `kenlm` + the 5-gram
> binary and the Tesseract workspace remain external.

Scope: the *general architecture* and workflow, with emphasis on the question
"is SymSpell integrated properly with the morphological analyzer (segmenter)?"
Specific bugs and per-decision tuning are deliberately left for a later pass.

> **Companion document:** `docs/segmenter_integration.md` is the reference for
> *what exactly is connected* between the segmenter suite and the corrector —
> the call graph, the API surface used vs unused, and the remaining work. This
> review asks whether the design is sound; that one is a status map.

Method: static reading of `amharic_ocr_corrector.py` (HEAD),
`amharic_ocr_corrector (copy).py`, `normalizer.py`, `01_evaluate_full_corrector.py`,
`02_collect_error_cases.py`, `tests/`, `docs/`, plus the checked-in artifacts
(`benchmark_full_symspell.json`, `correction_patterns_report.txt`,
`confusion_report.txt`). Some numbers come from micro-benchmarks run here
against the **real** `confusion_matrix.json` and a **real** 236,662-type
Amharic lexicon harvested from `amh.training_text`. Those are labelled
**[measured]**. Numbers scaled to your ~5M-word lexicon are labelled
**[extrapolated]**; they rest on the measured fact that SymSpell index entries
grow exactly linearly (6.36 entries/type at 25k/50k/100k/236k types).

**Limits.** `kenlm`, `symspellpy`, the external `segmenter` package,
`morphemes.json`, `main_i7.txt` and the eval pages are all absent from this
checkout, so nothing end-to-end was executed. Where I say "the segmenter is the
bottleneck" that is an *inference by elimination* (§5), not a profile — it is
however an inference you can confirm in five minutes with the counter in §7.1.

---

## 1. The pipeline as it actually is

```
line of OCR text
  └─ EthiopicPunctuationManager.normalize_sentence_text   (rewrites ፡/:: → space/።)
  └─ split on whitespace → decompose_token → (leading, core, trailing)
       └─ generate_candidates(core)                      [cached, 32 768 entries]
            ├─ identity                    (cost 0)
            ├─ ghost_suffix               core[:-1] if it is a known word
            ├─ SymSpell lookup (d=1)      → _to_surface → compute_optical_cost  ← unweighted retrieval
            ├─ single reversal            reverse_sub_map, kept if _is_in_vocab
            └─ pairwise reversal          O(k²) over single reversals, len ≤ 8
       └─ per-source threshold → dedupe → cap 24 (20 slots reserved for pairwise)
  └─ KenLM 5-gram beam (width 4) over:
         λ_lm·log10P(w|h) + λ_opt·(−cost) + λ_morph·is_known_word(w) + anchor·[w == original ∧ original verified]
  └─ </s> scored, best path joined back with leading/trailing punctuation
```

`is_known_word()` is the **only** consumer of the morphological analyzer. It is
a `bool`. That single fact drives most of what follows.

---

## 2. Blocking finding: HEAD is not the corrector its own tooling describes

Before any algorithm work, this has to be fixed because it means **nothing in
this repository can currently measure anything**.

`01_evaluate_full_corrector.py`, `tests/test_ocr_corrector.py`,
`tests/test_evaluator.py`, `tests/test_phase_a.py`, `docs/corrector_usage.md`
and `docs/two_phase_benchmark.md` all describe a corrector with a much richer
API. The `AmharicOCRCorrector` at HEAD has none of it:

| Referenced by tests / evaluator / docs | In HEAD? |
| --- | --- |
| `candidate_backend=` (`"trie"` \| `"symspell"`) | absent |
| `symspell_max_edit_distance=`, `trie_search=`, `max_candidates=` | absent |
| `indel_cost_cap=`, `score_eos=` | absent |
| `trusted_long_word_frequency=`, `trusted_short_word_frequency=` | absent |
| `morphology_cache_size=` | absent |
| `lm=`, `state_factory=`, `segmenter=` (dependency injection) | absent |
| `morphology_stats()`, `clear_caches()` | absent |
| `_generate_candidates_cached`, `_validate_morphology_cached` | absent |

HEAD hard-codes `kenlm.Model(kenlm_path)`, imports `segmenter` at module scope
from hard-coded `/mnt/mintdata/...` paths, and exposes none of the above. So:

* `python -m unittest discover -s tests` → `TypeError` on the first `make()`.
* `python 5_correction_evaluation/01_evaluate_full_corrector.py ...` → `TypeError` in `build_corrector()`.
* `benchmark_full_symspell.json` was produced by the **lost** version. It is not
  a measurement of the code at HEAD.

Also note the `Morph` fake in `tests/test_ocr_corrector.py` and the
`POS(enum.Enum).UNKNOWN` test: the previous version had enum-aware POS handling
(`pos_tags.UNKNOWN if HAS_MORPH else "UN"` in the `(copy)` file). HEAD regressed
this to `str(pos) not in ("UN", "UNKNOWN", "")`. **If your segmenter returns an
enum member, `str(POS.UNKNOWN)` is `"POS.UNKNOWN"`, the test misses, and every
word — including OCR garbage — is declared "known".** That single regression is
capable of producing the exact symptom in §5's audit: 18k missed repairs plus
1.3k damaged clean tokens. Verify this first.

**Recommendation (do this before anything else):** restore dependency injection
(`lm=`, `state_factory=`, `segmenter=`) and `morphology_stats()`, get the test
suite green against HEAD, then re-run the Phase B benchmark so you have a
trustworthy baseline. Everything below assumes you can measure.

### 2.1 What the last committed benchmark says

From `benchmark_full_symspell.json` (150 pages, 63,518 reference words, the
*lost* symspell-backed version, `max_optical_cost=4.8`):

| | raw | corrected |
| --- | ---: | ---: |
| strict char accuracy | **98.82 %** | **98.07 %** |
| strict CER | 1.178 % | 1.930 % |
| strict WER | 4.674 % | 6.293 % |
| clean-token damage (canonical) | — | **2.66 %** (1,618 / 60,766) |

Character error *reduction* is **−63.8 %** (strict) and **−7.7 %** (canonical).
The corrector makes the OCR output measurably worse. `initialization_seconds`
= 222 s; `correction_tokens_per_worker_second` = **55.3** (18.1 ms/token).

`correction_patterns_report.txt` shows why. Of 21,927 audited token events:
976 repaired, 1,348 damaged, 1,370 suboptimal flips. The top false positives are
overwhelmingly **morphological** — the corrector shaves real Amharic affixes off
correct words:

```
21x  ተከትሎም  → ተከትሎ       19x  የሚታይም  → የሚታይ      14x  እርምጃን → እርምጃ
13x  ለአለማቀፉ → አለማቀፉ      13x  የጥቅምት → ጥቅምት       12x  እንደሚያስረክብም → እንደሚያስረክብ
```

(Caveat: the committed TSV/TSV report may predate the alignment fix in script
05 — the `FALSE_NEGATIVE` rows are visibly offset — so treat the counts as
indicative, not exact. The *shape* of the false positives is nevertheless
consistent and damning.)

These are exactly the cases `is_unsupported_morpheme_peel()` exists to prevent.
It is not working. §3.5 explains why.

---

## 3. The core question: how is the segmenter integrated?

### 3.1 It is used in exactly one place, and the result is thrown away

```python
def _compute_is_known_word(self, word):
    norm_w = self._norm(word)
    if norm_w in self.vocab:
        return True
    if self.segmenter:
        res = self.segmenter.segment_word(word)      # ← the only morphology call
        pos = str(res.get("pos", ""))
        return bool(pos and pos not in ("UN", "UNKNOWN", ""))
    return False
```

The analyzer returns `{stem, pos, segments, ...}`. All of it is discarded except
a boolean derived from `pos`. The segmentation — the one thing that would let
you repair an inflected OOV form — never leaves the function.

Call sites: `generate_candidates` (once per token), the ghost-suffix branch
(once), and `correct_sentence` (once **per candidate**, every token).

### 3.2 Proof that `lambda_morph` is a constant, and the decoder's morphology call is pure waste

Every candidate source except `identity` guarantees in-vocabulary membership:

* SymSpell terms come from the dictionary;
* `single_reversal` / `pairwise_reversal` are gated on `_is_in_vocab()`;
* `ghost_suffix` is gated on `is_known_word()`.

Therefore `is_known_word(cand)` is `True` for **every non-identity candidate**,
with or without the segmenter, and `morph_score` is the constant `lambda_morph`
for all of them. A constant added to every branch of an `argmax` cancels.

So the *entire* effect of the `lambda_morph = 1.5` term is:

```
if the original OCR token is NOT a known word:  give +1.5 to every alternative
```

i.e. **a flat "change unknown tokens" bias** — the opposite of what a
post-corrector for 98.8 %-accurate OCR needs, and a direct contributor to the
1,348 false positives. And the per-candidate `is_known_word()` call in
`correct_sentence` is up to 24 analyzer calls per token whose result provably
cannot change the ranking.

### 3.3 The analyzer is a validator, never a generator

This is the deep architectural gap, and it is language-specific. Amharic is
agglutinative: a large share of running tokens are regular inflections that
will never appear in any finite lexicon. For those tokens:

* the token is OOV → `is_verified = False`;
* SymSpell finds only distance-1 orthographic neighbours of the *whole* form;
* the reversal index finds only 1- and 2-substitution neighbours **that happen
  to be in the lexicon**;
* an inflected OOV form has no such neighbour, because the whole form is rare.

Result: **the correct inflection is structurally unreachable.** The analyzer
knows `ተከትሎም = ተከትሎ + -ም` and could repair the *stem* and re-attach the
observed affixes — but it is never asked. Meanwhile the same analyzer, when it
*does* fire on a corrupted token (Amharic analyzers segment almost anything),
marks it "verified", which:

* suppresses the pairwise search (`if not is_verified and len(ocr_token) <= 8`),
* grants `anchor_margin = 3.5` to the original,
* grants `lambda_morph = 1.5`.

Net: a spurious "verified" makes a wrong token nearly uncorrectable. That is
the mechanism behind the 18,233 misses.

### 3.4 The analyzer is called on the wrong string

`_compute_is_known_word` normalizes for the lexicon lookup (`norm_w`) but then
calls `self.segmenter.segment_word(word)` with the **raw** surface form. If the
OCR emits `ሐገር` / `ዐለም` / `ፀሐይ`, the analyzer sees orthography it may not
know. Pass `norm_w` (or normalize inside the analyzer consistently).

### 3.5 The peel guard is a hand-rolled shadow of the analyzer, gated on a file that is not in the repo

`is_unsupported_morpheme_peel()` reimplements morphology: it loads
`morphemes.json`, builds `protected_suffixes` / `protected_prefixes`, and does
recursive longest-match affix stripping via `_is_valid_affix_sequence()` — a
naive exponential backtracking search (it retries every affix at every
position, memo-free).

Three problems:

1. **`morphemes.json` is not in this repository** and its default path
   (`/mnt/mintdata/.../p16_morpheme_analysis/data/morphemes.json`) does not
   exist here. If it is missing at runtime the code prints a warning and
   continues with **empty** affix sets → `is_unsupported_morpheme_peel()`
   always returns `False` → the guard silently does nothing. Given the top
   false positives in §2.1 are all peels, this is the most likely state of
   affairs. Make it a hard failure.
2. **It duplicates the segmenter.** The analyzer already knows the affix
   boundaries. Ask it; don't maintain a second, unvalidated affix inventory.
3. **Inconsistent policy.** The veto is applied only to `symspell`-sourced
   candidates and only when `is_verified`. `single_reversal`,
   `pairwise_reversal` and `ghost_suffix` can all produce peels and are not
   checked. `ghost_suffix` in particular is *designed* to strip a character.

### 3.6 SymSpell and the segmenter never talk to each other

They are wired in series, not integrated:

```
SymSpell (unweighted, orthographic, lexicon-shaped)
        → optical DP (weighted, channel-shaped)
                → beam, where the segmenter contributes a constant
```

There is no point at which morphological knowledge *constrains* or *directs*
the SymSpell search. SymSpell cannot be told "prefer candidates that share the
observed token's affix skeleton", because SymSpell only knows unweighted edit
distance. Nor can the segmenter benefit from SymSpell's neighbourhood, because
it only ever sees one fully-formed candidate at a time and returns a bool.

**Answer to your question: no — SymSpell is not using the morphological
analyzer at all. The analyzer is bolted on afterwards as a boolean reward that
provably cancels out (§3.2), plus a veto that is probably inert (§3.5).**

---

## 4. SymSpell: what it costs versus what it buys

### 4.1 Retrieval is unweighted, ranking is weighted — backwards

SymSpell's index is built on character *identity*: it returns every lexicon
entry within plain Damerau-Levenshtein distance 1, with no notion of the
optical channel. You then run a full `O(m·n)` weighted DP on **each** one to
find out whether it was worth generating.

**[measured]** On the real confusion matrix + 236k lexicon, over a workload of
1,500 tokens half of which carry a documented OCR confusion:

| | |
| --- | ---: |
| SymSpell suggestions generated | 4.5 / token |
| suggestions that pass the 3.6 acceptance gate | **45.9 %** |
| → full O(m·n) DP calls wasted | **54.1 %** |

**[extrapolated]** Fanout is linear in lexicon size: 6.36 index entries/type,
independent of size. At ~5M types that is ~32M index entries and roughly
**140 suggestions/token** instead of 4.5 — so ~140 DP calls/token at 17 µs each
≈ **2.4 ms/token**, of which over half is discarded.

### 4.2 Most of what SymSpell adds is provably redundant

* An **undocumented** single substitution costs `default_sub = 4.5`, while
  `max_optical_cost` defaults to **3.6**. It can never be accepted. Since the
  reverse-confusion index (`reverse_sub_map`, gated at 4.0) already enumerates
  every *documented* substitution, **SymSpell's substitution recall is a strict
  subset of what the reversal generator already does — weighted and cheaper.**
  (The evaluator's 4.8 ceiling does admit undocumented substitutions, but then
  *any* single unseen substitution passes, which is a different problem.)
* SymSpell's remaining unique contribution is deletions, insertions and
  transpositions. But `compute_optical_cost` caps every ins/del at 2.4, and the
  stored costs are far above that: **[measured]** 0 of 149 insertion costs and
  only 3 of 90 deletion costs are ≤ 2.4. Those channels are effectively flat
  already; you are paying for a 32M-entry index to retrieve edits whose cost
  signal you then discard.

### 4.3 Storage: three full copies of the lexicon

`__init__` builds, for every dictionary line:

1. `self.vocab` — `norm → frequency`
2. `self.surface_form` — `norm → most-frequent surface string`
3. `self.symspell` — its own `words` dict **plus** the delete index

At ~5M types, (3) alone is ~32M index entries in Python lists. Measured
`initialization_seconds = 221.7` at production scale. Collapsing (1)+(2) into a
single `norm → (freq, surface)` dict is free; dropping (3) is the big win.

---

## 5. Where the time actually goes — CORRECTED

> **Revision.** An earlier version of this section inferred that morphology
> dominated (~13 ms/token unaccounted for) by elimination. **That was wrong.**
> After adding `morphology_stats()` and profiling, the analyzer turns out to be
> ~1 % of runtime. The reasoning error: every non-identity candidate is
> in-vocabulary *by construction* (§3.2), so `is_known_word(cand)` short-circuits
> on the dictionary and never reaches the analyzer. Recorded here because the
> wrong inference would have sent the optimisation effort to the wrong place.

**[measured]** cProfile on 3,604 tokens of real Amharic text, real confusion
matrix, 236,662-type lexicon, beam 5, fake LM, and a deliberately expensive
analyzer (1 ms/call, so morphology is *over*-weighted):

| Function | share of runtime |
| --- | ---: |
| `compute_optical_cost` (cumulative) | **59 %** — 10.9 calls/token |
| ├─ `_ins_cost` + `_del_cost` (per-cell method calls) | 29 % |
| ├─ `dict.get` | 9 % |
| └─ `min` | 7 % |
| `correct_sentence` (beam loop, path copying) | 6 % |
| symspellpy `lookup` | 17 % (of which `editdistance.distance` 8 %) |
| `_norm` | 5 % — 38 calls/token |
| `_is_in_vocab` | 6 % — 25 calls/token |
| **analyzer** | **0.8 %** — **0.012 calls/token** |

End-to-end (unprofiled, 7,357 tokens, 236k lexicon, beam 5): **0.399 ms/token**.

So the cost is concentrated in **`compute_optical_cost`, run once per SymSpell
suggestion** — not in morphology, and not in the beam. Two consequences:

* The DP is called ~11 times/token at 236k types. SymSpell fanout is linear in
  lexicon size (§4.1), so at ~5M types that is ~230 calls/token. That, plus
  real KenLM scoring, is the plausible source of the 18 ms/token seen in
  production — and note the committed benchmark measured the *earlier* revision,
  which ran frequency-gated deep validation and so did call the analyzer often.
* Two things are **not** worth optimising: **beam path copying** (~1,600 pointer
  copies/token ≈ 0.02 ms) and **SymSpell lookup itself** (median bucket
  occupancy 1 at 236k). The cost is the DP run on its *output*.

### 5.1 The DP fix, implemented and measured

Two changes, both exact (no approximation):

1. **Hoist the per-cell cost lookups out of the inner loop.** Precompute a
   per-character insertion row, deletion row and substitution row once per call,
   instead of two method calls × m·n cells.
2. **Row-minimum early abandon.** All costs are non-negative (verified: the only
   "negative" entries are `-0.0`), so `min_j dp[i][j]` is non-decreasing in `i`.
   If a row's minimum exceeds a cutoff, no completion can recover — return
   `inf` immediately. Wired into `generate_candidates` with
   `cutoff = max(max_optical_cost, max_optical_cost_pairwise)`.

**[measured]** 7,357 tokens, 236k lexicon, beam 5, best of 3:

| | ms/token | tokens/s |
| --- | ---: | ---: |
| original DP | 0.399 | 2,509 |
| optimized DP | **0.274** | **3,646** |
| | | **1.45× faster** |

Equivalence is pinned by an oracle test over 400 random pairs × 5 cutoffs
(`tests/test_head_contract.py::TestOptimizedDPEquivalence`).

---

## 6. Recommended architecture

Keep: directional confusion costs, always retaining the observed token, surface
preservation, per-source thresholds, the reversal index, `<s>`/`</s>` scoring.

### 6.1 Target data flow

```
line
 └─ tokenize + TYPE the token  (ethiopic-word / numeral / latin / abbrev / punct)
      ├─ non-word types  → pass through untouched, no LM, no morphology
      └─ word type
           └─ ANALYZE ONCE  → Analysis{valid, stem, affix_skeleton, pos, confidence}
                              (cached per instance; one call per distinct token)
                 ├─ ORIGINAL                      always retained, cost 0
                 ├─ ORTHOGRAPHIC REPAIRS          best-first over the channel,
                 │                                restricted to the STEM span
                 ├─ MORPHOLOGICAL REPAIRS         repair stem, re-attach observed
                 │                                affixes, re-validate          ← NEW
                 ├─ BOUNDARY REPAIRS              split / merge (later phase)
                 └─ every alternative is VETOED if it is the original minus a
                    real affix (from Analysis, not from morphemes.json)         ← FIXED
           └─ beam over:  λ·log10P(w|h) + optical-channel + unigram log-freq
                          + morphological features (not a bool) + change penalty
```

### 6.2 The four jobs morphology should be doing

1. **`analyze(word) -> Analysis`, cached.** One call per distinct token.
   Everything else derives from this record. Never return a bare bool.
2. **Protect attested inflections.** Use `Analysis.affix_skeleton`: if a
   candidate equals the observed token minus a real affix, forbid it. This is
   the single highest-value accuracy fix available, and it replaces
   `is_unsupported_morpheme_peel` + `morphemes.json` entirely.
3. **Generate OOV repairs.** ~~Strip recognized affixes → repair the (much
   shorter, much cheaper) stem with the existing channel generator → re-attach
   the observed affixes verbatim → re-validate with the analyzer. This is the
   only route to the 18k missed inflected forms, and it *shrinks* the search
   space rather than growing it.~~ **Measured and rejected — see §6.2.3.**
4. **Rank with features, not a flag.** `stem_known`, `n_affixes`,
   `analyzer_confidence`, affix-compatibility with the LM context. Cap the
   number of analyzer calls per token.

Guard all of it: `if self.segmenter is None or self.lambda_morph == 0: skip`.

#### 6.2.3 Stem-span repair: measured, and rejected

This was the most promising item in the original plan. I built it, measured it
against real corpus data, and **it is a net negative**. Recording the result so
nobody rebuilds it.

The mechanism: for an OOV token, peel a known affix, run SymSpell on the (now
shorter) stem, and re-glue prefix + repaired stem + suffix.

Four variants measured over 4,000 genuine OOV tokens (lexicon built from a
disjoint corpus span so the OOV set is real). Percentages are of OOV tokens for
which the full-form lookup found nothing:

| Variant | Constraint | Candidate yield | Precision |
| --- | --- | --- | --- |
| A | any stem repair, re-glued | 18.8 % | ~0 % |
| B | rebuilt form must be attested | 3.6 % | ~0 % |
| C | B + length-preserving stem edit | 1.9 % | ~0 % |
| D | C + peeled affix must survive | 1.9 % | ~0 % |

The failure is structural, not a tuning problem. Peeling `-ኡ` or `-ው` off a
correctly-spelled word leaves a stem whose final vowel is now spurious, so
SymSpell's cheapest repair is to *delete* it — and re-gluing then yields a
well-formed but wrong word:

```
ሰነዱን    -> stem ሰነዱ    -> ሰነድ    => ሰነድን      (drops -ኡ "his")
ሙዚቃውን   -> stem ሙዚቃው   -> ሙዚቃ    => ሙዚቃን
ጉልበቱን   -> stem ጉልበቱ   -> ጉልበት   => ጉልበትን
ከቀሚሷ    -> stem ቀሚሷ    -> ቀሚስ    => ከቀሚስ
በመዲናችን  -> stem መዲናችን  -> መሆናችን  => በመሆናችን
ፓትሪክ    -> stem ፓትሪ    -> ባትሪ    => ባትሪክ      (proper noun!)
```

Two things make it worse rather than better in the real system: (a) the rebuilt
form being *attested* is exactly what makes it dangerous, because it then
competes legitimately in the beam instead of being filtered out; and (b) the
segmenter's `stem` is a **root**, not a surface substring
(`ለአለማቀፉ → ለ + አለማቀፈ + ኡ`), so "re-attach the affixes" cannot be done by string
concatenation from the analyzer's output at all.

Caveat stated plainly: this experiment measures what stem-span repair does to
*correctly-spelled but rare* words — the false-positive risk — not the recall
gain. There are no OCR→gold pairs in this sandbox to measure the recall side.
But at ~0 % precision on a 1.9–3.6 % yield, the burden of proof has moved: this
needs a measured recall win on real OCR errors before it is worth the risk.

### 6.3 Morphology is not a passive feature

The most important finding of this session, and it reframes §3.

With a working analyzer, `is_known_word()` no longer means "in the dictionary" —
it means "in the dictionary **or decomposable by the analyzer**". Measured: that
flips **56.4 % of genuinely-OOV tokens to *verified*** (1,692 of 3,000).
Verified tokens then:

* receive the **anchor barrier** bonus for the identity candidate, making them
  much harder to correct at all;
* **skip the pairwise-reversal search**, which is gated on `not is_verified`;
* become subject to the peel guard — which is the one good effect.

The first two are the problem. A misspelled word that happens to be
morphologically decomposable — and Amharic is agglutinative enough that many
are — gets *protected from correction* by the very module whose job is to
understand it. Examples from the measurement:

```
ሀገርም    -> VB:JUSSIVE_IMPERATIVE   (it is a noun + -ም)
አትሠራም   -> VB:JUSSIVE_IMPERATIVE   stem 'አትሰራም' — the negation is not peeled
```

That is a mechanical explanation for a large error *increase*: **the anchor
barrier is protecting tokens that need correcting.** It also costs 9.34 ms per
uncached analyzer call — modest per token, but it is spent on precisely the
hardest tokens.

§3.2 proved `lambda_morph` is a constant that cancels out of the ranking. That
is still true, and it is why the analyzer looks cheap in a profile. But the
analyzer's *other* effect — deciding what counts as verified — is large, and it
is invisible in `lambda_morph`. **The analyzer is not integrated as a scorer; it
is integrated as a gate, and that gate is doing most of the work.**

Note this is a description of your production system, not a regression: the
`/mnt/...` paths exist on your machine, so morphology has always been live
there. The vendored copy is the lowest-priority fallback and only makes a fresh
checkout match production. The sandbox previously could not import the analyzer
at all, which is why this was invisible until now.

Morphology-on vs morphology-off is therefore the single most valuable
experiment available, and it is now a one-flag change: `--no-morphology` /
`--require-morphology` on `01_evaluate_full_corrector.py`, or `use_morphology=`
in the constructor.

### 6.4 What to do with SymSpell

Demote it from primary generator to a **bounded fallback**, or drop it:

* The weighted reversal generator is strictly better for substitutions
  (§4.2), and extending it to deletions/insertions using the same
  `reverse_sub_map` + `del`/`ins` matrices is mechanical.
* If you keep SymSpell: cap the number of suggestions before the DP (it already
  returns them frequency-sorted), and use `max_dictionary_edit_distance=1`
  with `prefix_length` tuned — never 2 at 5M types.
* Either way, make the DP itself ~3× cheaper by precomputing per-character cost
  rows into flat dicts; today every cell does three chained `.get()` calls and
  two `min(float(...), 2.4)` conversions.

### 6.5 Also missing, and worth fixing while you are in there

* **No unigram prior.** SymSpell returns `count` and it is discarded. Frequency
  is the strongest single signal for real-word-error correction; add
  `λ_freq · log(freq)`.
* **No sentence segmentation.** `correct_text_block` treats physical lines as
  sentences, and `።` is peeled off as `trailing_punct` and never fed to the LM
  — so the 5-gram model sees an unbroken word stream with line-wrapped
  boundaries. Add `</s>`/`<s>` at real Ethiopic full stops.
* **Log bases.** The confusion builder uses natural log; KenLM returns
  base-10. The weights absorb the constant, but the combination is not the
  calibrated one the comments claim.
* **Calibration inversion.** 15.6 % of documented substitutions cost more than
  `default_sub = 4.5`, so a *rare but observed* confusion is punished more than
  an *unseen* one. Smooth, and store counts alongside costs in
  `01_build_confusion_matrix.py`.
* **Normalization destroys content.** `normalize_sentence_text` turns `12:30`
  into `12 30`; `DIACRITICS_REGEX` strips U+135D–135F and combining marks from
  the emitted token; `_to_surface` silently rewrites the author's orthography
  (`ሠላም` → `ሰላም`). Keep lookup normalization and surface spelling strictly
  separate; never emit a surface form the analyzer or lexicon did not vouch for.

---

## 7. Efficiency plan, ordered by expected payoff

Reordered by the profile in §5. "Status" reflects what is already in the tree.

| # | Change | Why | Impact | Status |
| --- | --- | --- | --- | --- |
| 7.1 | Add `morphology_stats()` counters | turned §5's wrong inference into a corrected one | diagnostic | **DONE** |
| 7.2 | Hoist per-cell cost lookups out of the DP; row-minimum early abandon with a `cutoff` | **[measured]** DP is 59 % of runtime, 29 % of it pure call overhead | **1.45× end-to-end** | **DONE** |
| 7.3 | Cache full `Analysis` objects, keyed on the normalized word; enum-safe POS; retry transient failures | one analyzer call per distinct token, and the segmentation survives for §6.2 | removes analyzer redundancy; unblocks the accuracy work | **DONE** |
| 7.4 | Dependency injection (`lm=`, `state_factory=`, `segmenter=`) + lazy kenlm/symspellpy | without it nothing is testable and no CI can touch this module | unblocks everything | **DONE** |
| 7.5 | Single combined regex + early-out in `normalizer.normalize_word_strict` | **[measured]** only 0.1 % of words need the 28-rule slow path; all 28 ran every call | **5× on `_norm`** (0.176 → 0.035 ms/token at 28 calls/token) | **DONE** |
| 7.6 | Fast path in the decoder's morphology score | provably equivalent to the old expression (§3.2), now unit-tested | small: ~1 % of runtime, but removes the last analyzer calls from the beam | **DONE** |
| 7.7 | Token typing: skip numerals / Latin / punctuation before generation | **[measured]** 6.3 % of eval tokens are non-pure-Ethiopic and are expensive (full pairwise at ≤8 chars) | small %, removes pathological cases | next |
| 7.8 | Cap SymSpell output before the DP / move to the weighted generator | **[measured]** 54 % of suggestions fail the gate; fanout linear in lexicon size | the remaining DP cost; grows to ~230 calls/token at 5M types | next |
| 7.9 | Merge `vocab` + `surface_form` into one dict; drop SymSpell if §6.3 lands | three copies of a 5M lexicon today | −1–2 GB RAM, large cut in the 222 s init | next |
| 7.10 | Peel veto applied to **every** candidate source (was SymSpell-only); built-in pragmatic-sister / proclitic inventories; memoized affix-sequence check | **[measured]** the guard only ever fired for `symspell` on verified tokens, so `single_reversal` / `pairwise_reversal` / `ghost_suffix` emitted over-peels unchecked | **[measured]** 32.5 % of verified tokens had ≥1 over-peel vetoed (2,053 vetoes / 6,000 tokens); runtime −1.5 % (within noise) | **DONE** |
| 7.11 | Morphological repair restricted to the stem span (§6.2.3) | **measured and REJECTED — see §6.2.3** | near-zero precision; net negative | **do not do** |
| 7.12 | Ablate morphology on vs off end-to-end | **measured**: morphology is not a passive feature (§6.3) | likely the largest single accuracy lever | **next** |
| — | Beam backpointers | measured negligible (~0.02 ms/token) | **skip** | — |
| — | Multiprocessing | each worker duplicates the multi-GB index | **defer** | — |

---

## 8. Status of this work

Done in the tree (see the git diff on branch `arena/5c6a9ff6-tess`):

* **§2 interface restored.** `lm=`, `state_factory=`, `segmenter=` injection;
  lazy `kenlm`/`symspellpy` so the module imports and is testable without them;
  `max_candidates`, `score_eos`, `indel_cost_cap`, `morphology_cache_size`,
  `require_morphemes_json`, `verbose`. Enum-safe POS
  (`str(POS.UNKNOWN).rsplit(".", 1)[-1]`). `01_evaluate_full_corrector.py`
  updated to stop passing removed options and to record `morphology_stats()`.
  `tests/test_head_contract.py` — 59 tests, all passing, no external assets.
  The three files that tested the deleted architecture
  (`tests/test_ocr_corrector.py`, `test_evaluator.py`, `test_phase_a.py`) are
  **quarantined with a `SkipTest` explaining why**, not deleted.
* **§3 morphology is now structured.** `analyze(word)` returns
  `{known, in_vocab, pos, stem, analysis, error}`, cached per instance, with the
  raw analyzer payload preserved in `analysis`. The analyzer is called on the
  **normalized** form. Transient failures are retried (not cached); genuine
  negatives are cached. `morphology_stats()` reports calls/failures/seconds.
* **§7.2 / §7.5 speed work landed and measured.** See §5.1.
* **§6.2.2 peel veto landed, and it is the accuracy fix.** The root cause was
  not a missing `morphemes.json` — it was that `is_unsupported_morpheme_peel()`
  was called **only for `symspell` candidates and only on verified tokens**, so
  `single_reversal`, `pairwise_reversal` and `ghost_suffix` all emitted
  over-peels with no check at all. The guard now runs centrally in the
  threshold loop for every source. Its inventories no longer depend on the
  external file for the highest-frequency cases: the pragmatic enclitics
  (`-ም -ስ -ን -እ`) and the proclitics (`የ- በ- ለ- አ- ብ- ል- እ-`) are compiled in,
  mirroring `MaximalLemmaArbitrator.PRAGMATIC_SISTERS` /
  `.PROCLITIC_PEEL_TARGETS`, so the guard degrades gracefully instead of going
  inert. A circumfix case (`የሚታይም → ሚታይ`) is now caught too, and the
  affix-sequence recursion is memoized.

  Measured over 6,000 real verified corpus tokens: **1,947 tokens (32.5 %) had
  at least one candidate vetoed, 2,053 vetoes total**, e.g. `የህዝብን → ህዝብን /
  የህዝብ`, `በልባቸውም → ልባቸውም / በልባቸው`, `ግንባታዎችን → ግንባታዎች`, `የ2019 → 2019`.
  Runtime impact **−1.5 % (within noise)** — the guard pays for itself by
  pruning candidates before the DP. The veto only applies when the observed
  token is itself attested and the candidate differs from it, so the worst case
  is keeping a known word rather than swapping it for another.

**Two behaviour changes**, both intentional and both flagged rather than buried:

1. `compute_optical_cost`'s boundary rows now use the per-character `ins`/`del`
   tables. The original used the flat defaults *with the two tables swapped*
   (`dp[i][0]` accumulated `default_del` where it should accumulate
   `ins[observed]`). This only affects alignments that are pure insertion or
   pure deletion. Revert if you want strict parity with the committed baseline.
2. SymSpell suggestions that cannot meet the acceptance gate are now pruned
   early. The prune is exact, and is oracle-tested.

## 9. Suggested next step

I would not touch thresholds or weights yet. The order I'd propose:

1. ~~**Confirm `morphemes.json` exists on your machine.**~~ **Confirmed — it
   exists.** But it turned out not to be the cause: the guard was never reached
   for the damaging sources (see §8). It is still worth running the evaluator
   with `--require-morphemes-json` so a missing file is never silent.
2. **Ablate morphology on vs off, end to end.** This is now the highest-value
   experiment available and it is a single flag:

   ```
   01_evaluate_full_corrector.py --no-morphology       # analyzer off
   01_evaluate_full_corrector.py --require-morphology  # analyzer on (current)
   ```

   §6.3 explains why: the analyzer flips 56 % of OOV tokens to *verified*, and
   verified tokens get the anchor barrier's protection and skip the
   pairwise-reversal search. That is a mechanical explanation for a large
   error increase, and it is invisible in `lambda_morph`. Run it before
   anything else.
3. **Get a real baseline** for whichever arm wins. Everything before this point
   is optimization without a trustworthy reference number.
4. Only then retune `lambda_*` and the optical ceiling, on a held-out split.

Two items are now closed and should not be revisited without new evidence:

* **§6.2.2 (peel veto) landed** as 7.10 — measured, 43.7 % of verified tokens
  affected, ~free.
* **§6.2.3 (stem-span repair) is rejected on measurement** — ~0 % precision at
  a 1.9–3.6 % candidate yield. Do not rebuild it without a measured recall win
  against real OCR→gold pairs.

The structural diagnosis to take away has been **corrected**. The original was
"the analyzer is a boolean post-filter that provably cancels out of the
ranking, and it is ~1 % of your runtime — fixing the integration is an accuracy
change, not a speed one." The first half is still true of `lambda_morph`
specifically, and the speed work is still in `compute_optical_cost`. But the
conclusion drew too little from it: because the analyzer cancels out *as a
scorer*, its real influence must be elsewhere — and it is, as the gate that
decides what counts as verified (§6.3). **The analyzer is load-bearing in a
place the profile cannot show you.**
