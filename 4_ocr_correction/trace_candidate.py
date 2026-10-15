#!/usr/bin/env python3
"""
Forensic Diagnostic Tracer: Inspects candidate generation and decoding gates
for any given (observed_ocr_token -> expected_word) pair in context.

Usage:
  python trace_candidate.py <observed_ocr_token> <expected_word> [optional_context_sentence]
"""

# Make the repo-root shared modules importable from this subfolder.
# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "repo_paths.py")):
    _root = _os.path.dirname(_root)
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import repo_paths  # noqa: E402  (imported after the path bootstrap)
# ----------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys
import kenlm

from amharic_ocr_corrector import AmharicOCRCorrector, EthiopicPunctuationManager

KENLM_BIN = repo_paths.word_kenlm()
CONFUSION_JSON = repo_paths.confusion_matrix()
DICT_FILE = repo_paths.main_i7_dict()


def get_sub_cost(corrector, true_ch, obs_ch):
    return corrector.costs.get("sub", {}).get(true_ch, {}).get(obs_ch)


def replay_candidate_sources(corrector, ocr_token: str):
    """
    Re-runs the same logic as _compute_candidates, but keeps track of WHICH
    stage produced each candidate and WHICH threshold it was judged against.
    """
    is_verified = corrector.is_known_word(ocr_token)
    norm_token = corrector._norm(ocr_token)

    pairwise_threshold = getattr(
        corrector, "max_optical_cost_pairwise", corrector.max_optical_cost
    )
    source_thresholds = {
        "identity": corrector.max_optical_cost,
        "ghost_suffix": corrector.max_optical_cost,
        "symspell": corrector.max_optical_cost,
        "single_reversal": corrector.max_optical_cost,
        "pairwise_reversal": pairwise_threshold,
    }

    raw = [(ocr_token, 0.0, "identity")]

    if len(ocr_token) >= 4 and not is_verified:
        stem = ocr_token[:-1]
        if corrector.is_known_word(stem):
            cost = min(float(corrector.costs.get("ins", {}).get(ocr_token[-1], corrector.default_ins)), 2.4)
            raw.append((stem, cost, "ghost_suffix"))

    from symspellpy import Verbosity
    for s in corrector.symspell.lookup(norm_token, Verbosity.ALL, max_edit_distance=1):
        cand = corrector._to_surface(s.term) if hasattr(corrector, "_to_surface") else s.term
        if is_verified and corrector.is_unsupported_morpheme_peel(ocr_token, cand):
            continue
        opt_cost = corrector.compute_optical_cost(ocr_token, cand)
        raw.append((cand, opt_cost, "symspell"))

    single_reversals = []
    for i, obs_ch in enumerate(ocr_token):
        for true_ch, cost in corrector.reverse_sub_map.get(obs_ch, []):
            variant = ocr_token[:i] + true_ch + ocr_token[i + 1:]
            # Mirror _compute_candidates: these generators gate on the strict
            # vocab test (_is_in_vocab), not on is_known_word (which also
            # consults the segmenter).
            if (corrector._is_in_vocab(variant) if hasattr(corrector, "_is_in_vocab")
                    else corrector.is_known_word(variant)):
                raw.append((variant, cost, "single_reversal"))
            single_reversals.append((i, true_ch, cost))

    if not is_verified and len(ocr_token) <= 8:
        for idx1, c1, cost1 in single_reversals:
            t1 = ocr_token[:idx1] + c1 + ocr_token[idx1 + 1:]
            for idx2, c2, cost2 in single_reversals:
                if idx2 > idx1:
                    t2 = t1[:idx2] + c2 + t1[idx2 + 1:]
                    if (corrector._is_in_vocab(t2) if hasattr(corrector, "_is_in_vocab")
                            else corrector.is_known_word(t2)):
                        raw.append((t2, cost1 + cost2, "pairwise_reversal"))

    return raw, source_thresholds


def trace_pair(corrector, ocr_token, expected, context_sentence=None):
    print("\n" + "=" * 75)
    print(f"      FORENSIC TRACE: {ocr_token!r}  -->  Expected: {expected!r}")
    print("=" * 75)

    # 1. Observed token status
    is_orig_known = corrector.is_known_word(ocr_token)
    norm_ocr = corrector._norm(ocr_token)
    in_vocab_ocr = norm_ocr in corrector.vocab

    print("\n[STAGE 1: OBSERVED TOKEN STATUS]")
    print(f"  • Raw Token              : {ocr_token!r} (length: {len(ocr_token)})")
    print(f"  • Normalized Form        : {norm_ocr!r}")
    print(f"  • In 5M Vocabulary?      : {in_vocab_ocr} (frequency: {corrector.vocab.get(norm_ocr, 0):,})")
    print(f"  • is_known_word()        : {is_orig_known}")
    if is_orig_known:
        print("    ↳ NOTE: Token is VERIFIED. Protected by anchor_margin; pairwise")
        print("      reversal search is SKIPPED ENTIRELY for verified tokens.")

    # 2. Expected target status
    norm_exp = corrector._norm(expected)
    in_vocab_exp = norm_exp in corrector.vocab
    is_target_known = corrector.is_known_word(expected)

    print("\n[STAGE 2: EXPECTED TARGET WORD STATUS]")
    print(f"  • Target Word            : {expected!r} (length: {len(expected)})")
    print(f"  • Normalized Form        : {norm_exp!r}")
    print(f"  • In 5M Vocabulary?      : {in_vocab_exp} (frequency: {corrector.vocab.get(norm_exp, 0):,})")
    print(f"  • is_known_word()        : {is_target_known}")
    if not is_target_known:
        print("    ⚠️ CRITICAL FAILURE: Expected target does not exist in vocabulary or morphology!")
        return

    # 3. Character-level differences (reads live thresholds, no hardcoded text)
    reverse_cutoff = getattr(corrector, "max_reverse_sub_cost", 2.8)
    print("\n[STAGE 3: CHARACTER-LEVEL DIFFERENCES]")
    if len(ocr_token) != len(expected):
        print(f"  • Length Mismatch: OCR={len(ocr_token)} vs Expected={len(expected)} "
              f"(insertion/deletion case — pairwise-substitution reasoning below does not apply)")
    else:
        diffs = [(i, o, e) for i, (o, e) in enumerate(zip(ocr_token, expected)) if o != e]
        print(f"  • Total Substitutions Required: {len(diffs)}")

        costs = []
        any_missing = False

        for i, obs_ch, true_ch in diffs:
            options = corrector.reverse_sub_map.get(obs_ch, [])
            match = [c for (t, c) in options if t == true_ch]
            raw_cost = get_sub_cost(corrector, true_ch, obs_ch)

            print(f"    Index {i}: observed '{obs_ch}' -> true '{true_ch}'")
            print(f"      - In reverse_sub_map now?   {bool(match)}")
            print(f"      - Raw confusion_matrix cost: {raw_cost}  "
                  f"(live reverse_sub_map COMPONENT cutoff = {reverse_cutoff})")

            if raw_cost is None:
                any_missing = True
            else:
                costs.append(float(raw_cost))

        if any_missing:
            print(f"  • Sum of Optical Sub Costs: N/A — at least one substitution has NO "
                  f"entry in confusion_matrix.json. Structurally unreachable until the "
                  f"matrix is patched.")
        else:
            total = sum(costs)
            single_thresh = corrector.max_optical_cost
            pair_thresh = getattr(corrector, "max_optical_cost_pairwise", single_thresh)
            print(f"  • Sum of Optical Sub Costs      : {total:.4f}")
            print(f"  • max_optical_cost (single-src) : {single_thresh}")
            print(f"  • max_optical_cost_pairwise     : {pair_thresh}")
            if len(diffs) <= 1:
                print(f"  • Passes single-source filter?  : {total <= single_thresh}")
            else:
                print(f"  • Passes pairwise filter?       : {total <= pair_thresh}  "
                      f"(this pair needs the PAIRWISE threshold, not the single one)")

    # 3b. Search space diagnostics — catches combinatorial explosion directly
    if hasattr(corrector, "debug_generation_stats"):
        stats = corrector.debug_generation_stats(ocr_token)
        print("\n[STAGE 3b: SEARCH SPACE DIAGNOSTICS]")
        print(f"  • Component cutoff (max_reverse_sub_cost)   : {stats['max_reverse_sub_cost']}")
        print(f"  • Pairwise sum cutoff (max_optical_cost_pairwise): {stats['max_optical_cost_pairwise']}")
        print(f"  • Reserved pairwise slots                   : {stats['reserved_pairwise_slots']}")
        print(f"  • Single-char reversals admitted             : {stats['n_single_reversals_admitted']}")
        print(f"  • Pairwise combinations attempted            : {stats['n_pairwise_combinations_attempted']}")
        print(f"  • Pairwise combos matching real words        : {stats['n_pairwise_hits_that_are_real_words']}  "
              f"(these compete for the reserved slots above)")

    # 4. Candidate generation audit, with source tagging
    print("\n[STAGE 4: CANDIDATE GENERATION AUDIT]")
    candidates = corrector.generate_candidates(ocr_token)
    cand_dict = dict(candidates)

    print(f"  • Total Candidates Returned (post-filter): {len(candidates)}")
    for rank, (w, c) in enumerate(candidates, 1):
        is_target = " <=== [EXPECTED TARGET]" if w == expected else ""
        print(f"    [{rank:2d}] '{w}' (optical cost: {c:.4f}){is_target}")

    expected_in_final = expected in cand_dict
    if expected_in_final:
        print(f"  ✅ SUCCESS: Expected word {expected!r} WAS generated (cost: {cand_dict[expected]:.4f}).")
    else:
        print(f"  ❌ FAILURE: Expected word {expected!r} was NOT in the final (post-filter) candidate pool.")
        print(f"\n  --- Replaying raw generation (PRE-filter/PRE-truncation) to find out why ---")
        raw, thresholds = replay_candidate_sources(corrector, ocr_token)
        matches = [(cand, cost, src) for cand, cost, src in raw if cand == expected]
        if not matches:
            print(f"  The expected word was never produced by ANY generation stage.")
            print(f"  Fix needed: confusion-matrix coverage, or the edit requires >2")
            print(f"  substitutions / mixed indel+sub, which the algorithm doesn't search.")
        else:
            for cand, cost, src in matches:
                thresh = thresholds.get(src, corrector.max_optical_cost)
                passes_threshold = cost <= thresh
                print(f"  Found via source={src!r}  cost={cost:.4f}  "
                      f"applicable_threshold={thresh}  passes={passes_threshold}")
                if not passes_threshold:
                    print(f"    ↳ ROOT CAUSE: rejected by the '{src}' threshold. Tuning issue.")
                else:
                    print(f"    ↳ Passed its threshold but was likely squeezed out during")
                    print(f"      top-{12} truncation by competing candidates. Check Stage 3b")
                    print(f"      counts above — consider raising reserved_pairwise_slots.")
        print()
        return

    # 5. Contextual decoder decision — full score breakdown at this token's step
    if context_sentence:
        print("\n[STAGE 5: CONTEXTUAL DECODER DECISION]")
        print(f"  • Input Context : {context_sentence}")

        normalized_text = EthiopicPunctuationManager.normalize_sentence_text(context_sentence)
        raw_tokens = normalized_text.split()
        token_tuples = [EthiopicPunctuationManager.decompose_token(t) for t in raw_tokens]

        target_step = None
        for idx, (lead, w, trail) in enumerate(token_tuples):
            if w == ocr_token:
                target_step = idx
                break

        if target_step is None:
            print(f"  ⚠️ Token {ocr_token!r} not found as a standalone token in the context "
                  f"sentence after punctuation decomposition. Tokens seen: "
                  f"{[t[1] for t in token_tuples]}")
        else:
            start_state = kenlm.State()
            corrector.lm.BeginSentenceWrite(start_state)
            beam = [(0.0, start_state, [])]

            for step, (lead, w, trail) in enumerate(token_tuples):
                if not w:
                    beam = [(sc, st, p + [(lead + trail, "", "")]) for sc, st, p in beam]
                    continue

                cands = corrector.generate_candidates(w)
                is_verified_here = corrector.is_known_word(w)
                new_beam = []

                show_detail = (step == target_step)
                if show_detail:
                    print(f"\n  --- Full score breakdown at step {step} (token {w!r}) ---")

                for cand_word, opt_cost in cands:
                    opt_score = -corrector.lambda_optical * opt_cost
                    morph_known = corrector.is_known_word(cand_word)
                    morph_score = corrector.lambda_morph if morph_known else 0.0
                    anchor_bonus = corrector.anchor_margin if (cand_word == w and is_verified_here) else 0.0

                    for acc_score, prev_state, path in beam:
                        next_state = kenlm.State()
                        lm_score = corrector.lm.BaseScore(prev_state, corrector._norm(cand_word), next_state)
                        total = acc_score + (corrector.lambda_lm * lm_score) + opt_score + morph_score + anchor_bonus

                        if show_detail:
                            flag = "  <=== EXPECTED" if cand_word == expected else ""
                            print(f"    cand={cand_word!r:15} opt_cost={opt_cost:5.2f} "
                                  f"opt_score={opt_score:7.3f} morph={morph_score:4.2f} "
                                  f"anchor={anchor_bonus:4.2f} lm={lm_score:7.3f} "
                                  f"-> total={total:8.3f}{flag}")

                        new_beam.append((total, next_state, path + [(lead, cand_word, trail)]))

                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[: corrector.beam_width]

                if show_detail:
                    print(f"  Surviving beam after step {step}:")
                    for sc, _, p in beam:
                        words = " ".join(x[1] for x in p if x[1])
                        print(f"    score={sc:8.3f}  path={words!r}")

            finalized = []
            for score, state, path in beam:
                end_state = kenlm.State()
                eos = corrector.lm.BaseScore(state, "</s>", end_state)
                finalized.append((score + corrector.lambda_lm * eos, path))
            finalized.sort(key=lambda x: x[0], reverse=True)

            best_path = finalized[0][1]
            reconstructed = []
            for lead, w, trail in best_path:
                reconstructed.append(lead if not w else f"{lead}{w}{trail}")
            corrected_out = " ".join(reconstructed)

            print(f"\n  • Decoder Output: {corrected_out}")
            if expected in corrected_out:
                print(f"  ✅ WINNER: Decoder selected the expected word!")
            else:
                print(f"  ❌ OVERRULED: Decoder chose another candidate despite the expected")
                print(f"     word surviving candidate generation. Check the score breakdown above.")

    print("=" * 75 + "\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python trace_candidate.py <ocr_token> <expected_word> [optional_sentence]")
        print("Example: python trace_candidate.py ፕርጀከት ፕሮጀክት 'ትልቁ ፕርጀከት ነው'")
        return

    ocr_tok = sys.argv[1]
    expected_word = sys.argv[2]
    ctx = sys.argv[3] if len(sys.argv) > 3 else None

    corrector = AmharicOCRCorrector(
        kenlm_path=KENLM_BIN,
        confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE,
    )

    trace_pair(corrector, ocr_tok, expected_word, ctx)


if __name__ == "__main__":
    main()