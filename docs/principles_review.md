# Principles review — does the code follow the post-OCR framework?

Reference: the "Framework" document (98% initial accuracy; over-correction is the
main risk; act like a surgical tool: fast, conservative, triggered only when needed).

This file records (a) where the code matched the framework, (b) where it had
drifted, (c) what was fixed, and (d) what still needs real data. Every
"[probe]" result comes from a small script run against the code; the scripts
used a fake LM because `kenlm`, the 5-gram model and the eval pages are not in
this checkout.

---

## 1. Scorecard

| Framework component | HEAD status | Now |
| --- | --- | --- |
| Overall: conservative, triggered only when needed | **Drifted.** Stage 1 was off by default, so every token went to candidate search and the LM. | Stage 1 on by default. |
| Stage 1: lexicon miss → suspicious | Implemented (`suspicion.py`). | Kept. |
| Stage 1: low OCR confidence → suspicious | Implemented, but no caller ever passed confidences. | `correct_sentence(..., confidences=)` added. Mismatched counts are dropped. |
| Stage 1: punctuation / numeral anomalies | **Missing.** `ETHIOPIC_PUNCT` is defined but unused; `decompose_token` strips punctuation silently. | Still missing. See §4. |
| Stage 1 gate protects clean text | **Bug.** PASS tokens returned before the LM ran, so the LM only ever scored `</s>` [probe]. | Fixed: PASS tokens advance the LM state. |
| Stage 2: SymSpell, d=1 (d=2 fallback) | d=1 implemented. d=2 deliberately absent (memory; docs §7). | Unchanged. Documented. |
| Stage 2: weighted confusion costs (0.5 / 1.0 / 1.2) | Weighted by the learned matrix, which has a different scale (0–11). Visually similar pairs had no curated cost. | Curated pairs enter the same table at 0.5 where the learned matrix is silent. Learned values win. |
| Stage 2 / Stage 1 consistency | **Inconsistent.** Stage 1's one-flip used curated pairs that Stage 2 could never propose (ኘ→ነ costs 4.5, above the 3.6 ceiling) [probe]. | One table. Both stages see the same pairs. |
| Stage 3: KenLM in context | Implemented (sentence-level beam, width 4). A superset of the per-token window. | Kept. |
| Stage 3: `α·logP − β·edit + γ·OCR_conf` | Weights differ (λ_lm 1.8, λ_opt 1.2, no γ term). An extra `λ_morph` lexicon bonus is not in the framework. | `λ_morph` is kept but is now bounded by the do-no-harm margin (below). Weights untuned. See §4 on γ. |
| Stage 4: split / merge (word segmentation) | **Missing.** | Implemented: merge of two OOV tokens; split of one OOV token into two lexicon words. Penalty is UNTUNED. |
| Stage 4: "do no harm" for OOV / names | **Bug.** The margin applied only to *verified* originals. The gray-zone margin never reached OOV tokens: outputs were identical for margins 0, 3 and 10 [probe]. Worse, `λ_morph=1.5` made an in-lexicon candidate beat an OOV original even when the LM preferred the original by 0.5 nats [probe]. | Fixed: every unverified original gets `oov_anchor_margin` (3.0, UNTUNED), plus the gray margin when Stage 1 says gray. |
| Lexicon matching: weighted similarity, not identity | `_norm` collapses homoglyph families (ሐ/ኀ→ሀ, ዐ/ዓ→አ, ሠ→ሰ, ፀ→ጸ) and applies 28 bilabial regex rules. These are hard equivalences, not 0.5-cost edits. | **Not changed.** Flagged in §4. |

---

## 2. What changed in code

`amharic_ocr_corrector.py`

* **Stage 1 on by default** (`stage1_detection=True`). It can still be turned off.
* **PASS tokens advance the LM state** (`_advance_kept`). Before, a PASS token
  was invisible to the LM. Also, LM scoring of every later token used the wrong
  left context.
* **Do-no-harm margin on the original** for unverified tokens
  (`oov_anchor_margin`, default 3.0). Gray adds `gray_margin` on top. A flag does
  not remove the margin: the framework applies the rule to all OOV tokens.
* **Curated confusions injected once** into `costs["sub"]` at
  `curated_confusion_cost` (default 0.5), only where the learned matrix has no
  entry. Stage 1 and Stage 2 read the same table.
* **Stage 4 split/merge** (`enable_merge_split`, `merge_split_penalty`). Beams
  are now kept per input position so that a merge can consume two tokens. Path
  entries carry a `span`. Change accounting counts a merge as one change even
  when the letters are unchanged.
* **OCR confidences** can be passed to `correct_sentence(sentence, confidences=…)`
  on the 0–100 scale. They feed Stage 1.

`tests/test_head_contract.py`

* The Stage 1 test helper now injects `state_factory`. Without it, 7 tests
  errored on machines without kenlm. (They were not testing anything wrong.)
* `test_stage1_off_by_default` became `test_stage1_on_by_default`, plus an
  explicit opt-out check.

`tests/test_principles.py` (new, 14 tests)

* LM context kept through PASS tokens.
* OOV original survives a small LM gap, with and without Stage 1 and with the
  morphology bonus on.
* Large LM gap still corrects.
* Curated pair present in the cost table; learned entries are never overwritten;
  every one-flip Stage 1 uses is reachable by Stage 2.
* Merge, merge disabled, merge counted, split, split refused when the LM does not
  prefer it.
* Confidence count mismatch is dropped, not fatal; matched confidences reach
  Stage 1.

On the unmodified HEAD, 11 of these 14 fail; the other 3 error only because they
pass keyword arguments HEAD does not have. So they guard the defects above.

**Test status:** `python -m unittest discover -s tests` → 112 tests, OK (9 skipped).
Setup note: `symspellpy` must be installed (`requirements-symspell.txt`). On a
system Python without it, 68 tests error at import. In this sandbox I used a
virtualenv (`/home/user/.venv_ocr`) because the system Python is PEP 668 managed.

---

## 3. Behaviour changes you should expect

These are deliberate, and they change outputs. Re-run the evaluator before
adopting them.

1. Stage 1 is on by default. Scripts that construct the corrector without the
   flag now get the gate.
2. Unverified (OOV) originals are kept unless beaten by `oov_anchor_margin`. This
   is more conservative than HEAD. It will reduce repairs. It should also reduce
   damage to names and loanwords.
3. Curated pairs (AIN/PHARYNGEAL, H-family, sibilants, nasals, labials, etc.)
   become cheap candidates at 0.5. This adds candidates the learned matrix did
   not propose. The docs call the curated set "unmeasured", so this is the
   change most likely to over-correct. Raise `curated_confusion_cost`
   (for example to 1.0 or more) if validation shows it over-corrects.
4. Merge/split can fire. Each needs both tokens (merge) or the whole token
   (split) to be out of lexicon, and it must beat two separate readings after
   penalty.

---

## 4. Still open (not fixed, needs data or a decision)

* **Stage 3 OCR-confidence term (γ).** As written, `γ·OCR_conf(W_i)` is the same
  for every candidate of token i, so it cannot change which candidate wins. Adding
  it naively does nothing. The useful version is confidence-scaled: low confidence
  should lower the original's margin. That needs real confidences first.
* **Punctuation anomalies (Stage 1 item 3).** Not implemented as a signal. Tokens
  with attached punctuation are handled by decomposition, which is probably the
  right behaviour. `ETHIOPIC_PUNCT` is dead code.
* **Confidence source.** Nothing in this repository captures Tesseract TSV or
  `image_to_data` word confidences. With no confidences, Stage 1 treats tokens as
  high-confidence (`ocr_conf is None` passes the lexicon gate). That is fail-open.
* **Homoglyph collapse in `_norm`.** Treating ሐ/ኀ/ሀ or ዐ/ዓ/አ as identical is a
  hard equivalence. The framework prescribes weighted 0.5-cost edits instead. The
  collapse can make a genuinely different spelling count as "verified" and get
  the anchor. It is the first thing to test on real data.
* **λ weights.** λ_lm=1.8, λ_opt=1.2 and λ_morph=1.5 are not the framework's
  α≈1, β≈1.5–2 ratio. They are untuned here.
* **Untuned new knobs:** `oov_anchor_margin=3.0`, `merge_split_penalty=2.0`,
  `curated_confusion_cost=0.5`. These need a grid search on paired (OCR, GT) data.
* **d=2 SymSpell fallback** remains intentionally absent for memory reasons.

---

## 5. What has NOT been verified

* No end-to-end accuracy run. `kenlm`, the 5-gram model, the segmenter and the
  eval pages are absent here, so these results come from fake-LM probes and unit
  tests only.
* The last committed benchmark (`benchmark_full_symspell.json`) was produced by a
  lost version of the corrector. It reports strict CER **worsening** from 1.18%
  to 1.93% after correction. That is the over-correction problem the framework
  warns about, and this change is aimed at it. It has not been re-measured.

**Next step:** run `01_evaluate_full_corrector.py` on the same 150-page set with
(a) HEAD defaults and (b) the new defaults, and compare strict CER, clean-token
damage, and tokens changed. Tune `oov_anchor_margin` and `curated_confusion_cost`
first.
