# Segmenter ↔ OCR Corrector: integration status

How the Amharic morphological segmenter suite is wired into
`amharic_ocr_corrector.py` today, what is deliberately *not* wired, and what
would move accuracy if it were.

This is a companion to `docs/architecture_review.md`. The review asks *whether*
the integration is sound; this document is the reference for *what exactly is
connected*, with the API surface and measurements.

---

## 1. Scope and one-line summary

**The segmenter is connected, but only as a boolean gate.** One method out of
the suite's ~92 public methods is called, and one field of its result is read.
Everything else — affix peeling, commutation testing, POS tagging, lemma
arbitration — is present in the tree and unused.

The connected part is not passive, though. It decides what counts as a *known*
word, and that decision changes which candidates are even generated. So the
integration is shallow but load-bearing, which is the worst of both worlds:
little of the suite's linguistic power is used, and what is used has an
outsize, partly unintended effect.

---

## 2. What the segmenter is

`vendor/segmenter/` — a frozen snapshot of the external morphological suite
(see `vendor/segmenter/VENDORING.md` for provenance, patches and precedence).
13 modules plus `data/`.

| Module | Role | Public API used by the corrector |
| --- | --- | --- |
| `segmenter.py` | Top-level orchestrator | **`segment_word()`** — the only call |
| `morphology.py` | Root/template engine, 51 methods | none |
| `affix_peeler.py` | Multi-split affix decomposition | none |
| `maximal_lemma_arbitrator.py` | Commutation testing, 16 methods | none |
| `pos_tagger.py` | Stem/POS classification, 10 methods | none |
| `morpheme_disambiguator.py` | Morpheme role disambiguation | none |
| `lemma_candidate_generator.py` | Lemma candidate generation | none |
| `morph_types.py` | `DecomposeContext`, `MorphSplit`, config | none |
| `phonology.py` | Boundary phonology | none |
| `pos_tags.py` | POS tag vocabulary | none (loaded transitively) |
| `macro_categories.py` | Macro category mapping | none |
| `seg_normalizer.py` | Suite's own normalizer (**renamed**) | none |
| `project_paths.py` | Data path resolution | none (loaded transitively) |

Measured usage:

```
class                               public  used
AmharicMorphologicalSegmenter            7     1
AmharicMorphologyEngine                 51     0
AffixPeeler                              2     0
MaximalLemmaArbitrator                  16     0
PosTagger                               10     0
AmharicMorphemeDisambiguator             3     0
LemmaCandidateGenerator                  3     0
```

**1 of 92.** That is the headline number for "how integrated is it".

### 2.1 What `segment_word()` returns

```python
>>> c.segmenter.segment_word("ለአለማቀፉ")
{'word': 'ለአለማቀፉ', 'prefix': 'ለ', 'stem': 'አለማቀፈ', 'suffix': 'ኡ',
 'prefixes': ['ለ'], 'suffixes': ['ኡ'], 'pos': 'VB:PERFECTIVE'}
```

Seven fields. Note the `stem` is a **root**, not a surface substring:
`ለአለማቀፉ → ለ + አለማቀፈ + ኡ` — the surface `ፉ` is reanalysed as `ፈ` + `ኡ`. This
matters for anything that wants to "peel affixes and glue back" (see §6.3).

---

## 3. What is integrated

### 3.1 Loading and precedence

`amharic_ocr_corrector.py` builds `MORPH_PATHS` and inserts each existing entry
at `sys.path[0]` **in list order**, so later entries end up with higher
priority:

```
MORPH_PATHS = [vendor/segmenter,            <- lowest priority (fallback)
               /mnt/.../p16_morpheme_analysis,
               /mnt/.../p16_morpheme_segmenter]   <- highest priority
```

The vendored copy is listed first deliberately, making it the fallback. A live
checkout via `AMHARIC_MORPH_PATHS` or the machine paths still wins.

**Consequence:** on the development machine, behaviour is *unchanged* by
vendoring — morphology was always live there. The snapshot only makes a fresh
checkout match production. Before vendoring, a clean clone logged one WARNING
(`Could not import AmharicMorphologicalSegmenter`), set `self.segmenter = None`,
and silently ran with morphology disabled — which is why all earlier profiling
in this repo measured a morphology-free system.

`data/morphemes.json` follows the same rule: the vendored copy is used only when
the *default* path is missing. An explicitly passed `morphemes_json_path=` is
honoured even when absent, so `--require-morphemes-json` keeps its teeth.

### 3.2 The single call site

Everything goes through `AmharicOCRCorrector.analyze(word)` (line 607; the
analyzer call itself is at ~line 634), which:

1. normalizes the word,
2. checks a bounded per-instance cache (`morphology_cache_size`, default 65536),
3. tests `norm_w in self.vocab`,
4. **only if not in vocab**, and only if `segmenter is not None` and
   `lambda_morph != 0`, calls `self.segmenter.segment_word(norm_w)`,
5. decides `known` via `_analysis_is_unknown()` (POS-sentinel test),
6. caches and returns.

Transient analyzer exceptions are **not** cached (so one broken call cannot
poison a word for the life of the process); genuine negatives are.

`analyze()` returns six fields:

```python
{'known': bool, 'in_vocab': bool, 'pos': str|None,
 'stem': str|None, 'analysis': dict|None, 'error': str|None}
```

**The corrector reads exactly one of them: `known`.** `pos` and `stem` are
populated and stored in the cache, but nothing consumes them. `analysis` (the
full raw dict, including `prefixes`/`suffixes`) is preserved but unread.

### 3.3 Where `known` propagates — five consequences

`is_known_word()` → `analyze()['known']`. That single boolean drives:

| # | Site | Effect when `known` is True |
| --- | --- | --- |
| 1 | `is_orig_verified` (decoder) | Tokens gets the **anchor barrier** bonus for the identity candidate → much harder to correct |
| 2 | `pairwise_reversal` generation | **Skipped entirely** (gated on `not is_verified`) |
| 3 | `ghost_suffix` generation | Skipped (gated on `not is_verified`) |
| 4 | Peel guard | **Enabled** — the only protective effect |
| 5 | `_morphology_score` | Returns `lambda_morph`… for every non-identity candidate, i.e. a constant |

Item 5 is why the analyzer looks free in a profile: `lambda_morph` provably
cancels out of the ranking (proved in `architecture_review.md` §3.2), because
every non-identity candidate is in-vocabulary by construction. **The analyzer's
real influence is items 1–3, none of which appear in any score.**

### 3.4 The peel guard — *not* driven by the analyzer

This is a common misreading, so it is worth stating plainly: the morpheme-peel
veto does **not** use the segmenter. It is a hand-rolled check against static
affix inventories loaded from `morphemes.json`, plus compiled-in constants:

```python
PRAGMATIC_SISTERS      = ("ም", "ስ", "ን", "እ")            # -ም -ስ -ን -እ
PROCLITIC_PEEL_TARGETS = ("የ", "በ", "ለ", "አ", "ብ", "ል", "እ")
```

…mirroring `MaximalLemmaArbitrator`'s own constants, so the guard degrades
gracefully rather than going inert when `morphemes.json` is absent.

It runs **centrally** in the per-source threshold loop, so it covers every
candidate source. This was the core bug fix: it previously ran only for
`symspell` candidates on verified tokens, so `single_reversal`,
`pairwise_reversal` and `ghost_suffix` emitted over-peels unchecked.

Measured over 6,000 real verified corpus tokens:

| Inventories | Tokens affected | Vetoes | Runtime |
| --- | --- | --- | --- |
| compiled-in only | 32.5 % | 2,053 | −1.5 % (noise) |
| + real `morphemes.json` (123 sfx / 123 pfx) | **43.7 %** | **2,844** | **+1.9 %** |

### 3.5 The morphology switch

`use_morphology=None|True|False` on the constructor, and
`--no-morphology` / `--require-morphology` on `04_evaluate_full_corrector.py`.

Default `None` preserves historical behaviour (on when importable). `True`
raises `RuntimeError` if the suite is not importable. An injected `segmenter=`
always wins over the switch.

### 3.6 Instrumentation

`morphology_stats()` reports:

```python
{'segmenter_calls': int, 'segmenter_failures': int,
 'segmenter_seconds': float, 'trusted_bypasses': int,
 'peel_guard_blocks': int, 'cache': {...}}
```

Recorded by `04_evaluate_full_corrector.py`.

---

## 4. What is **not** integrated

### 4.1 Unused modules

`AffixPeeler`, `MaximalLemmaArbitrator`, `PosTagger`,
`AmharicMorphemeDisambiguator`, `LemmaCandidateGenerator`: **zero calls**. They
are imported transitively (the segmenter loads them internally) but the
corrector never calls them.

### 4.2 Unused fields of the one call that *is* made

Of `segment_word()`'s seven fields, the corrector uses `pos` (via the unknown
sentinel) and discards `prefix`, `stem`, `suffix`, `prefixes`, `suffixes`.

### 4.3 Unused methods on the class that *is* used

`AmharicMorphologicalSegmenter` exposes seven public methods; only
`segment_word` is called:

| Method | What it would give the corrector |
| --- | --- |
| `analyze_word` | (richer analysis than `segment_word`) |
| `classify_pos` / `get_fine_grained_pos` | POS for affix-compatibility features |
| `explain_arbitration` | why a parse was chosen |
| `get_effective_stem_freq` | **stem frequency** — a real ranking feature |
| `get_verbal_lemma_freq` | **lemma frequency** — ditto |

### 4.4 Not vendored

`data/main_i7.txt` (103 MB) — the main dictionary. The segmenter resolves it
lazily; `AmharicMorphologicalSegmenter()` instantiates in ~0 s and
`segment_word()` works without it. If a code path later needs the full lexicon,
copy it in or point `AMHARIC_MORPH_PATHS` at a live checkout.

---

## 5. Measured behaviour of the current integration

All figures from real corpus data (`amh_final_training_text.txt`), lexicon and
evaluation spans kept disjoint.

| Measurement | Value |
| --- | --- |
| OOV tokens flipped to *verified* by the analyzer | **56.4 %** (1,692 / 3,000) |
| Analyzer cost, uncached | **9.34 ms/token** |
| Verified tokens with ≥1 candidate vetoed | **43.7 %** (2,844 vetoes / 6,000) |
| Peel-guard runtime | +1.9 % |
| Top POS assigned to flipped tokens | `VB:JUSSIVE_IMPERATIVE`, `VB:PERFECTIVE`, `VB:PERFECTIVE:RELATIVE` |

**The 56.4 % figure is the important one.** It means the majority of
out-of-vocabulary tokens are treated as known words — and therefore protected
by the anchor barrier and excluded from pairwise repair, per §3.3.

Some of those flips are wrong, and the errors are systematic:

```
ሀገርም    -> VB:JUSSIVE_IMPERATIVE   (it is a noun + -ም)
አትሠራም   -> VB:JUSSIVE_IMPERATIVE   stem 'አትሰራም' — the negation is not peeled
```

Amharic is agglutinative enough that a misspelled word is often still
decomposable, so **the analyzer protects tokens that need correcting.** That is
a mechanical explanation for a large error increase, and it is invisible in
`lambda_morph`.

---

## 6. What remains, and what to do about it

Ordered by expected value. Items 1–2 are cheap and high-value; 3–5 are the
deeper integration.

### 6.1 Ablate morphology on vs off — **do this first**

One flag, no code:

```bash
04_evaluate_full_corrector.py --no-morphology       # analyzer off
04_evaluate_full_corrector.py --require-morphology  # analyzer on (current default)
```

This is the single most valuable experiment available, and it must come before
any of the work below: §5 shows the analyzer changes the treatment of 56 % of
OOV tokens, so every other accuracy change is being measured on top of an
unvalidated gate.

### 6.2 Decouple "verified" from "protected"

Today one boolean does two unrelated jobs: *is this a real word* and *should we
refrain from correcting it*. Split them:

* `in_vocab` → already computed and already in `analyze()`'s result.
* `morphologically_decomposable` → also already there.

Use `in_vocab` for the anchor barrier (protect only genuine dictionary words),
and use decomposability only to *enable* morphology-aware repair — never to
suppress it. This is likely a large win and is contained to
`is_known_word`/`_compute_is_known_word` and the two call sites that read
`is_orig_verified`.

Watch out: the pairwise/ghost-suffix gates (`not is_verified`) would also need
to move to `in_vocab`, or the search space grows.

### 6.3 Replace the static peel guard with commutation testing

The guard is a static inventory check. The suite contains a principled
alternative that is **already vendored and unused**:

```python
MaximalLemmaArbitrator.verify_proclitic_peeling(...)   # PROBE A
MaximalLemmaArbitrator.verify_pragmatic_commutation(...)
MaximalLemmaArbitrator.verify_suffix_commutation(...)
MaximalLemmaArbitrator.verify_verbal_clitic_commutation(...)
MaximalLemmaArbitrator.verify_determiner_commutation(...)
```

These do *commutation testing*: swap the peeled affix for its sisters,
synthesize the surface forms (with vowel coalescence), check corpus attestation
at a frequency threshold, and accept the split only if enough sisters are
attested. That is a much stronger test than "is this string in my affix list",
and it is exactly what the observed false positives need.

Cost is the concern — commutation means several synthesized lookups per
candidate — so it should run only on candidates that survive the cheap static
guard, i.e. as a *second stage*, not a replacement.

### 6.4 Use `AffixPeeler` for the affix skeleton

`AffixPeeler.decompose(word, ctx)` returns **multiple ranked splits, each with
a named rule and a strength level** — far more than `segment_word()` gives:

```
ለአለማቀፉ
  segment_word : pfx=['ለ'] stem='አለማቀፈ' sfx=['ኡ'] pos=VB:PERFECTIVE
  peeler       : stem='ለአለማቀፈ' sfx=('ኡ',) rule='verb_plural_u_S0_perfective'        strength=3
                stem='ለአለማቀፋ' sfx=('ኡ',) rule='verb_plural_u_S4_weak'              strength=3
                stem='ለአለማቀፍ' sfx=('ኡ',) rule='verb_plural_u_S6_imperative'        strength=3
                stem='ለአለማቀፈ' sfx=('ኡ',) rule='possessive_3MS'                     strength=3

የተማሪዎች
  peeler       : stem='የተማሪ'   sfx=('ዎች',) rule='plural_allomorph_direct'
                stem='የተማሪዋ'  sfx=('ኦች',) rule='matrix_cell_S3_V6'
                pfx=('የ',) stem='ተማሪዎች' rule='prefix_literal:የ'                strength=1
```

Two properties make this useful:

* `strength` (STRONG/NORMAL/WEAK) is a **confidence signal** the ranking could
  consume directly.
* Multiple splits mean an edit can be tested against *several* analyses rather
  than being forced into one.

Note `decompose()` needs a `DecomposeContext` (stem checkers, flags) — it is not
a drop-in for `segment_word`.

### 6.5 Rank with features instead of a constant

`_morphology_score` currently returns a constant. With §6.4 wired in it could
return real features:

* `stem_known` — is the extracted stem attested
* `get_effective_stem_freq` / `get_verbal_lemma_freq` — available today, unused
* `n_affixes`, `strength`, `rule`
* affix-compatibility with the LM context

Cap analyzer calls per token; the cache already dedupes per distinct word.

### 6.6 Don't: stem-span repair by re-gluing

Measured and rejected (`architecture_review.md` §6.2.3). Peel → repair stem →
re-glue yields 1.9–3.6 % candidates at **~0 % precision**, because peeling
`-ኡ`/`-ው` makes the stem's final vowel spurious so SymSpell deletes it
(`ሰነዱን → ሰነድን`, `ሙዚቃውን → ሙዚቃን`, `ፓትሪክ → ባትሪክ`). The segmenter's `stem` being a
root rather than a substring (§2.1) means re-gluing cannot be done correctly by
concatenation at all. Leave it alone unless a measured recall win appears.

---

## 7. Memory: why `test_page1_fixes.py` gets OOM-killed

That script passes `main_i7.txt` (103 MB on disk) as `dict_vocab_path`. That is
a multi-million-type lexicon, and SymSpell is the problem.

### 7.1 Measured cost per structure

Built from real corpus text, 314,755 types:

| Stage | Cumulative RSS | Delta |
| --- | --- | --- |
| baseline | 32 MB | — |
| `vocab` + `surface_form` dicts | 76 MB | +44 MB |
| **+ SymSpell index** | **430 MB** | **+399 MB** |
| + segmenter | 430 MB | +0 MB (lazy) |

**SymSpell costs ~1.03 KB per dictionary type** — roughly 4.5× the two plain
dicts combined, and ~91 % of the total. The segmenter is free (it resolves
`MAIN_DICTIONARY_PATH` lazily).

Extrapolating linearly, an uncapped lexicon costs ~1.27 KB per type overall:

| Lexicon types | Uncapped | Capped at 500k | Capped at 300k |
| --- | --- | --- | --- |
| 1 M | ~1.3 GB | ~0.6 GB | ~0.4 GB |
| 3 M | ~3.8 GB | ~0.6 GB | ~0.4 GB |
| 5 M | ~6.4 GB | ~0.6 GB | ~0.4 GB |

Plus KenLM's 5-gram binary on top. That is the OOM.

### 7.2 Capping is also an accuracy improvement

Measured over 4.42 M running tokens of real corpus text (638,126 types):

| Top-N types | Token coverage | Est. SymSpell |
| --- | --- | --- |
| 50,000 | 78.16 % | 50 MB |
| 100,000 | 84.11 % | 101 MB |
| 200,000 | 89.57 % | 201 MB |
| **300,000** | **92.35 %** | **302 MB** |
| 500,000 | 96.88 % | 503 MB |

And the distribution is extremely long-tailed:

```
types seen exactly once : 415,414  (65.1 % of types)
types seen <= 2 times   : 499,428  (78.3 % of types, 13.2 % of tokens)
```

**65 % of types are hapax.** In a lexicon mined from OCR output, most of those
singletons *are* OCR errors. They cost ~65 % of the memory and let SymSpell
"correct" one misspelling into another. Capping is not a memory/accuracy
trade-off so much as a win on both.

### 7.3 What was added

`max_lexicon_types` and `dict_min_freq` on the constructor, and
`--max-lexicon-types` / `--dict-min-freq` on `04_evaluate_full_corrector.py`.

The loader was restructured into two passes. Previously SymSpell was built
*during* the file read, so a cap would have discarded an index we had already
paid to build. Now: read → cap → `gc.collect()` → build SymSpell from the
surviving set. Measured effect:

```
no cap                  types=314,755   +399 MB
max_lexicon_types=200k  types=200,000   +207 MB
max_lexicon_types=100k  types=100,000    +79 MB
dict_min_freq=3         types= 59,189    +15 MB
```

Also fixed: `_affix_seq_memo` (added earlier this session) was an unbounded
dict keyed on word substrings, so it grew for the life of a long-running
process. It now uses the existing bounded cache.

**Suggested starting point:** `--dict-min-freq 3`, or `--max-lexicon-types
500000`. Measure before and after — do not assume a specific value is right for
`main_i7.txt`, whose type count I could not measure here (the file is 103 MB
and was not vendored).

### 7.4 Still to do

Review item 7.9 — merge `vocab` + `surface_form` into one dict — would remove
another full copy. Dropping SymSpell entirely in favour of the weighted
reversal generator (review §6.4) would remove the dominant structure outright.

## 8. CLI compatibility with the pre-refactor evaluator

Four options present at HEAD pointed at a corrector revision that no longer
exists. At HEAD they were *accepted by argparse* and then passed to a
constructor that rejected them, so any run using them died with a `TypeError`
rather than a clean error.

| Option | Status |
| --- | --- |
| `--symspell-distance N` | **Restored and working.** Wired to a real `symspell_max_edit_distance` constructor arg (previously hard-coded to 1). Warns above 1 — the index grows steeply with edit distance and will OOM on a large lexicon. |
| `--trie-search` | **Removed.** Fails with an explanation. The trie backend is gone; SymSpell plus the weighted optical reversal generator replaced it. |
| `--trusted-long-word-frequency N` | **Removed.** Fails with an explanation. Superseded by the anchor barrier plus the peel guard. |
| `--trusted-short-word-frequency N` | **Removed.** Fails with an explanation. Same. |
| `--backend {trie,symspell}` | **Narrowed** to `symspell` only; kept so old command lines still parse, and not forwarded to the constructor. |

The three removed options are still registered with argparse (hidden) purely so
the failure message explains what happened instead of saying
`unrecognized arguments`.

## 9. Risks and caveats

* **The vendored snapshot will drift** from the live suite. It is a *frozen*
  copy; treat any refresh as a behaviour change worth measuring.
* **The one vendored patch** is the `normalizer.py → seg_normalizer.py` rename
  (plus its two importers). It exists because the suite's normalizer would
  otherwise shadow this repo's root `normalizer.py` and silently change
  normalization process-wide. Re-apply it after any refresh.
* **`morphology.py` is ~51 public methods of untested-here surface.** Only
  `segment_word` has been exercised in this repo; anything deeper needs its own
  tests.
* **No end-to-end run in this sandbox.** `kenlm` is absent, so
  `04_evaluate_full_corrector.py` cannot run here. Every number in this
  document is from candidate generation and analysis only — never from a full
  decode. Treat the ablation in §6.1 as unmeasured until you run it.
* **§5's 56.4 % is measured on correctly-spelled-but-rare words**, which is the
  right population for assessing false-positive risk from the *gate*, not for
  measuring recall on real OCR errors.

---

## 10. Quick reference

| Question | Answer |
| --- | --- |
| Where is the suite? | `vendor/segmenter/` |
| How is it found? | `MORPH_PATHS`, vendored = lowest priority |
| What is called? | `segment_word()` only |
| From where? | `analyze()` (~line 634), cached |
| What is read? | `known` (via `is_known_word`); `pos`/`stem` stored, unused |
| Is the peel guard analyzer-driven? | **No** — static `morphemes.json` + compiled-in constants |
| How do I turn it off? | `use_morphology=False` / `--no-morphology` |
| How do I see the cost? | `morphology_stats()` |
| How do I bound memory? | `max_lexicon_types=` / `dict_min_freq=` (§7) |
| Biggest single lever? | §6.1 ablation, then §6.2 decoupling |
