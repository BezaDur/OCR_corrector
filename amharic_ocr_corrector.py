#!/usr/bin/env python3
"""
Production CPU-based Amharic OCR Spelling Corrector & Post-Processor.

Architecture:
  1. EthiopicPunctuationManager:
     - Isolates leading and trailing punctuation from lexical tokens.
     - Preserves internal abbreviation markers ('/' and '.').
  2. AmharicNormalizer Integration:
     - Normalizes tokens for dictionary lookup while preserving surface spelling
       via a norm -> most-frequent-surface-form reverse map.
  3. SymSpell + Reverse Optical Mapping:
     - High-speed candidate generation (O(1)) driven by confusion_matrix.json.
  4. AmharicMorphologicalSegmenter:
     - Linguistic prior validating agglutinated Out-of-Vocabulary forms.
  5. KenLM 5-Gram Viterbi Beam Decoder:
     - Contextual sequence scoring with a protective anchor barrier and
       explicit end-of-sentence scoring.

Fixes applied across this revision history:
  - [BUG] Swapped insertion/deletion costs in compute_optical_cost DP recurrence.
  - [BUG] Added explicit </s> scoring when finalizing the beam.
  - [BUG] Tiered optical-cost thresholds: pairwise (2-substitution) candidates
    are judged against max_optical_cost_pairwise, not the stricter single-edit
    max_optical_cost, since they structurally sum two costs.
  - [BUG] Decoupled the reverse_sub_map COMPONENT-level cutoff
    (max_reverse_sub_cost) from the pairwise SUM-level cutoff
    (max_optical_cost_pairwise). Tying them together caused O(n^2)
    combinatorial explosion in the pairwise search, flooding results with
    spurious real-word hits that crowded out legitimate repairs.
  - [BUG] Candidate truncation to top-12 now reserves dedicated slots for
    pairwise-sourced candidates instead of a single global cost-rank slice,
    which previously starved out valid (but necessarily pricier) pairwise
    repairs once enough cheap single-edit noise existed.
  - Hardcoded filesystem paths now configurable via constructor args / env vars.
  - Dictionary now preserves surface spelling (norm -> best surface form map)
    instead of always emitting the normalized form as the "correction".
  - lru_cache on bound methods replaced with bounded, instance-local caches
    (no more cross-instance leaks).
  - Bare `except Exception: pass`-style swallowing replaced with logged warnings.
  - Affix inventories pre-sorted once instead of on every recursive call.
  - Added a CLI entry point for quick manual testing, with tunable thresholds.
  - Added debug_generation_stats() for diagnosing search-space explosion.
"""

import os
import sys
import re
import json
import logging
import collections
import argparse
import kenlm
from symspellpy import SymSpell, Verbosity

# ----------------------------- Logging setup ----------------------------------
logging.basicConfig(
    level=os.environ.get("AMHARIC_LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amharic_ocr_corrector")

# ----------------- Dynamic Import of NLP & Morphological Suite -----------------
# Configurable via AMHARIC_MORPH_PATHS (os.pathsep-separated). Falls back to the
# original hardcoded defaults so existing setups keep working unchanged.
_DEFAULT_MORPH_PATHS = [
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis",
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_segmenter",
]
_env_morph_paths = os.environ.get("AMHARIC_MORPH_PATHS", "")
MORPH_PATHS = [p for p in _env_morph_paths.split(os.pathsep) if p] or _DEFAULT_MORPH_PATHS

for p in MORPH_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from normalizer import AmharicNormalizer
    HAS_NORM = True
    print("[OK] Successfully imported AmharicNormalizer.")
except ImportError as e:
    print(f"[WARN] Could not import AmharicNormalizer: {e}")
    HAS_NORM = False

try:
    from segmenter import AmharicMorphologicalSegmenter
    HAS_MORPH = True
    print("[OK] Successfully imported AmharicMorphologicalSegmenter.")
except ImportError as e:
    print(f"[WARN] Could not import AmharicMorphologicalSegmenter: {e}")
    HAS_MORPH = False
# -----------------------------------------------------------------------------

_DEFAULT_MORPHEMES_JSON = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/morphemes.json"


class _BoundedCache:
    """Simple instance-local LRU cache (avoids lru_cache-on-bound-method leaks)."""

    def __init__(self, maxsize: int = 65536):
        self.maxsize = maxsize
        self._data = collections.OrderedDict()

    def get(self, key, default=None):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return default

    def __setitem__(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key):
        return key in self._data


class EthiopicPunctuationManager:
    DIACRITICS_REGEX = re.compile(r'[\u135D-\u135F\u0300-\u036F]')

    ALL_PUNCT = set(
        "።!?፧…፣፤፥፦፠፨"
        ":;,!?"
        "“”‘’«»‹›„\'\""
        "()[]{}"
        "|\\—–-"
    )

    @classmethod
    def normalize_sentence_text(cls, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'([:፡]{2,}|\.{2,}|፣{2,})', '።', text)
        text = re.sub(r'[:፡]\s*$', '።', text, flags=re.MULTILINE)
        text = re.sub(r'[:፡]\s*([”"»‹\n])', r'።\1', text)
        text = re.sub(r'[:፡]', ' ', text)
        text = re.sub(r'[ ]+', ' ', text)
        return text.strip()

    @classmethod
    def decompose_token(cls, raw_token: str):
        if not raw_token:
            return "", "", ""
        start = 0
        while start < len(raw_token) and raw_token[start] in cls.ALL_PUNCT:
            start += 1
        leading_punct = raw_token[:start]
        end = len(raw_token)
        while end > start and raw_token[end - 1] in cls.ALL_PUNCT:
            end -= 1
        trailing_punct = raw_token[end:]
        core_word = cls.DIACRITICS_REGEX.sub('', raw_token[start:end])
        return leading_punct, core_word, trailing_punct


class AmharicOCRCorrector:
    def __init__(
        self,
        kenlm_path: str,
        confusion_matrix_path: str,
        dict_vocab_path: str,
        morphemes_json_path: str = None,
        beam_width: int = 4,
        max_optical_cost: float = 3.6,
        max_optical_cost_pairwise: float = None,
        max_reverse_sub_cost: float = None,
        reserved_pairwise_slots: int = 6,
        lambda_optical: float = 1.2,
        lambda_lm: float = 1.8,
        lambda_morph: float = 1.5,
        anchor_margin: float = 3.5,
        known_word_cache_size: int = 131072,
        candidate_cache_size: int = 32768,
    ):
        self.beam_width = beam_width
        self.max_optical_cost = max_optical_cost

        # Pairwise (2-substitution) repairs structurally need ~2x the single-edit
        # budget. Defaulting to 1.6x rather than exactly 2x keeps a modest penalty
        # for stacking two uncertain edits vs. one confident one.
        self.max_optical_cost_pairwise = (
            max_optical_cost_pairwise if max_optical_cost_pairwise is not None
            else max_optical_cost * 1.6
        )

        # Component-level gate: deliberately INDEPENDENT of
        # max_optical_cost_pairwise. This controls how many individual
        # substitutions are admitted as candidate building blocks feeding the
        # pairwise search. Setting this too loose (e.g. equal to the pairwise
        # SUM threshold) causes O(n^2) combinatorial explosion in the pairwise
        # search, flooding results with spurious real-word hits that crowd out
        # legitimate multi-substitution repairs. Keep this comfortably below
        # default_sub.
        self.max_reverse_sub_cost = (
            max_reverse_sub_cost if max_reverse_sub_cost is not None else 4.0
        )

        self.reserved_pairwise_slots = reserved_pairwise_slots

        self.lambda_optical = lambda_optical
        self.lambda_lm = lambda_lm
        self.lambda_morph = lambda_morph
        self.anchor_margin = anchor_margin

        print(f"Loading KenLM Language Model: {kenlm_path}...", flush=True)
        self.lm = kenlm.Model(kenlm_path)

        print(f"Loading Optical Confusion Matrix: {confusion_matrix_path}...", flush=True)
        with open(confusion_matrix_path, "r", encoding="utf-8") as f:
            self.costs = json.load(f)

        self.default_sub = float(self.costs.get("default_sub", 4.5))
        self.default_del = float(self.costs.get("default_del", 2.4))
        self.default_ins = float(self.costs.get("default_ins", 2.4))

        # Bounded instance-local caches (replaces lru_cache-on-method leak hazard)
        self._known_word_cache = _BoundedCache(known_word_cache_size)
        self._candidate_cache = _BoundedCache(candidate_cache_size)

        # 1. Normalizer
        self.normalizer = AmharicNormalizer() if HAS_NORM else None

        # 2. Morphological Segmenter & Declarative Morpheme Inventory
        self.segmenter = AmharicMorphologicalSegmenter() if HAS_MORPH else None
        if self.segmenter:
            print("Morphological Segmenter ready.", flush=True)

        # 3. Load Exhaustive Declarative Morphemes from morphemes.json
        self.protected_suffixes = set()
        self.protected_prefixes = set()
        morphemes_json_path = morphemes_json_path or os.environ.get(
            "AMHARIC_MORPHEMES_JSON", _DEFAULT_MORPHEMES_JSON
        )

        if os.path.exists(morphemes_json_path):
            with open(morphemes_json_path, "r", encoding="utf-8") as mj:
                m_data = json.load(mj)
                for k, v in m_data.items():
                    if "suffix" in k and isinstance(v, (list, set, tuple)):
                        self.protected_suffixes.update(v)
                    elif "prefix" in k and isinstance(v, (list, set, tuple)):
                        self.protected_prefixes.update(v)
            print(
                f"Loaded {len(self.protected_suffixes)} declarative suffixes and "
                f"{len(self.protected_prefixes)} prefixes from morphemes.json.",
                flush=True,
            )
        else:
            print(f"[WARN] morphemes.json not found at {morphemes_json_path}", flush=True)

        # Pre-sort affix inventories once (longest-first) instead of re-sorting
        # on every recursive call inside _is_valid_affix_sequence.
        self._suffixes_sorted = sorted(self.protected_suffixes, key=len, reverse=True)
        self._prefixes_sorted = sorted(self.protected_prefixes, key=len, reverse=True)

        # 4. Vocabulary & SymSpell
        print(f"Loading Lexicon into SymSpell & Hash Set: {dict_vocab_path}...", flush=True)
        self.vocab = {}
        # norm_w -> best-known surface spelling (preserves surface form on output)
        self.surface_form = {}
        self.symspell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

        with open(dict_vocab_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0]
                freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
                norm_w = self._norm(w)

                prev_freq = self.vocab.get(norm_w, 0)
                if freq >= prev_freq:
                    # Keep the most frequent surface spelling for this normalized form
                    self.surface_form[norm_w] = w
                self.vocab[norm_w] = max(prev_freq, freq)
                self.symspell.create_dictionary_entry(norm_w, freq)

                if idx % 1000000 == 0:
                    print(f"  ... loaded {idx:,} dictionary words", flush=True)

        # Precompute reverse optical substitutions. Gated by max_reverse_sub_cost
        # (the TIGHT component-level cutoff), NOT by max_optical_cost_pairwise.
        self.reverse_sub_map = collections.defaultdict(list)
        for true_ch, obs_map in self.costs.get("sub", {}).items():
            if isinstance(obs_map, dict):
                for obs_ch, cost in obs_map.items():
                    c_flt = float(cost)
                    if c_flt <= self.max_reverse_sub_cost and true_ch != obs_ch:
                        self.reverse_sub_map[obs_ch].append((true_ch, c_flt))

        print(f"Corrector initialized successfully ({len(self.vocab):,} active words).\n", flush=True)

    def _norm(self, word: str) -> str:
        if not word or not self.normalizer:
            return word
        return self.normalizer.normalize_word_strict(word)

    def _to_surface(self, norm_or_word: str) -> str:
        """Map a (possibly normalized) form back to its preferred surface spelling."""
        return self.surface_form.get(norm_or_word, norm_or_word)

    # ---------------------------------------------------------------- #
    # is_known_word: bounded-cache version (no more lru_cache-on-self) #
    # ---------------------------------------------------------------- #
    def is_known_word(self, word: str) -> bool:
        if not word:
            return False
        cached = self._known_word_cache.get(word)
        if cached is not None:
            return cached
        result = self._compute_is_known_word(word)
        self._known_word_cache[word] = result
        return result

    def _compute_is_known_word(self, word: str) -> bool:
        norm_w = self._norm(word)
        if norm_w in self.vocab:
            return True
        if self.segmenter:
            try:
                res = self.segmenter.segment_word(word)
                pos = str(res.get("pos", ""))
                return bool(pos and pos not in ("UN", "UNKNOWN", ""))
            except Exception:
                logger.warning("segmenter.segment_word failed for %r", word, exc_info=True)
                return False
        return False

    def compute_optical_cost(self, ocr_str: str, cand_str: str) -> float:
        m, n = len(ocr_str), len(cand_str)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.default_del
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.default_ins

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c_obs = ocr_str[i - 1]
                c_cand = cand_str[j - 1]
                sub_cost = (
                    0.0
                    if c_obs == c_cand
                    else self.costs.get("sub", {}).get(c_cand, {}).get(c_obs, self.default_sub)
                )
                # ins[c] = cost that OCR spuriously *produced* character c
                #          -> applies when consuming an extra OCR char (dp[i-1][j])
                # del[c] = cost that true character c was *dropped* by OCR
                #          -> applies when consuming a candidate char with no OCR
                #             counterpart (dp[i][j-1])
                ins_cost = min(float(self.costs.get("ins", {}).get(c_obs, self.default_ins)), 2.4)
                del_cost = min(float(self.costs.get("del", {}).get(c_cand, self.default_del)), 2.4)

                dp[i][j] = min(
                    dp[i - 1][j] + ins_cost,
                    dp[i][j - 1] + del_cost,
                    dp[i - 1][j - 1] + sub_cost,
                )

        return dp[m][n]

    def _is_valid_affix_sequence(self, chunk: str, sorted_inventory) -> bool:
        """Recursively validates if a stripped chunk is a valid layered affix sequence.
        `sorted_inventory` must already be sorted longest-first.
        """
        if not chunk:
            return True
        for aff in sorted_inventory:
            if aff and chunk.startswith(aff):
                if self._is_valid_affix_sequence(chunk[len(aff):], sorted_inventory):
                    return True
        return False

    def is_unsupported_morpheme_peel(self, orig_word: str, cand_word: str) -> bool:
        """
        Universally proscribes candidates formed by stripping authentic prefixes
        or suffixes from an already authenticated Amharic word.
        """
        if len(cand_word) >= len(orig_word):
            return False

        # Guard against suffix stripping (e.g. እርምጃን -> እርምጃ, ተከትሎም -> ተከትሎ)
        if orig_word.startswith(cand_word):
            stripped_suffix = orig_word[len(cand_word):]
            if self._is_valid_affix_sequence(stripped_suffix, self._suffixes_sorted):
                return True

        # Guard against prefix stripping (e.g. ለአለማቀፉ -> አለማቀፉ, የጥቅምት -> ጥቅምት)
        if orig_word.endswith(cand_word):
            stripped_prefix = orig_word[:-len(cand_word)]
            if self._is_valid_affix_sequence(stripped_prefix, self._prefixes_sorted):
                return True

        return False

    # -------------------------------------------------------------------- #
    # generate_candidates: bounded-cache version + surface-form preserving #
    # -------------------------------------------------------------------- #
    def generate_candidates(self, ocr_token: str):
        if not ocr_token:
            return (("", 0.0),)
        cached = self._candidate_cache.get(ocr_token)
        if cached is not None:
            return cached
        result = self._compute_candidates(ocr_token)
        self._candidate_cache[ocr_token] = result
        return result

    def _compute_candidates(self, ocr_token: str):
        is_verified = self.is_known_word(ocr_token)
        norm_token = self._norm(ocr_token)

        # Each entry: (candidate_word, optical_cost, source_tag)
        raw_candidates = [(ocr_token, 0.0, "identity")]

        # 1. Terminal Ghost Loop check (e.g. ሶስትዮሽቨ -> ሶስትዮሽ)
        if len(ocr_token) >= 4 and not is_verified:
            stem_stripped = ocr_token[:-1]
            if self.is_known_word(stem_stripped):
                cost = min(float(self.costs.get("ins", {}).get(ocr_token[-1], self.default_ins)), 2.4)
                raw_candidates.append((stem_stripped, cost, "ghost_suffix"))

        # 2. Fast SymSpell 1-Edit lookup with Declarative Morpheme Guard.
        suggestions = self.symspell.lookup(norm_token, Verbosity.ALL, max_edit_distance=1)
        for s in suggestions:
            cand = self._to_surface(s.term)
            if is_verified and self.is_unsupported_morpheme_peel(ocr_token, cand):
                continue
            opt_cost = self.compute_optical_cost(ocr_token, cand)
            raw_candidates.append((cand, opt_cost, "symspell"))

        # 3. Direct Optical Reversal (single substitution)
        single_reversals = []
        for i, obs_ch in enumerate(ocr_token):
            for true_ch, cost in self.reverse_sub_map.get(obs_ch, []):
                variant = ocr_token[:i] + true_ch + ocr_token[i + 1:]
                if self.is_known_word(variant):
                    raw_candidates.append((variant, cost, "single_reversal"))
                single_reversals.append((i, true_ch, cost))

        # 4. Pairwise optical substitution for unverified words
        if not is_verified and len(ocr_token) <= 8:
            for idx1, c1, cost1 in single_reversals:
                t1 = ocr_token[:idx1] + c1 + ocr_token[idx1 + 1:]
                for idx2, c2, cost2 in single_reversals:
                    if idx2 > idx1:
                        t2 = t1[:idx2] + c2 + t1[idx2 + 1:]
                        if self.is_known_word(t2):
                            tot_cost = cost1 + cost2
                            raw_candidates.append((t2, tot_cost, "pairwise_reversal"))

        # Per-source thresholds: pairwise repairs get a looser budget since they
        # structurally represent two stacked edits.
        source_thresholds = {
            "identity": self.max_optical_cost,
            "ghost_suffix": self.max_optical_cost,
            "symspell": self.max_optical_cost,
            "single_reversal": self.max_optical_cost,
            "pairwise_reversal": self.max_optical_cost_pairwise,
        }

        best_cost = {}
        best_source = {}
        for cand, cost, source in raw_candidates:
            threshold = source_thresholds.get(source, self.max_optical_cost)
            if cost <= threshold:
                if cand not in best_cost or cost < best_cost[cand]:
                    best_cost[cand] = cost
                    best_source[cand] = source

        # Reserve dedicated slots for pairwise-sourced candidates instead of a
        # single global cost-rank truncation. A flat top-12-by-cost slice
        # systematically starves out legitimate (necessarily pricier) pairwise
        # repairs once enough cheap single-edit candidates exist to fill the cap.
        TOTAL_CAP = 12
        RESERVED_PAIRWISE_SLOTS = self.reserved_pairwise_slots

        all_sorted = sorted(best_cost.items(), key=lambda x: x[1])

        pairwise_sorted = [
            (c, cost) for c, cost in all_sorted if best_source[c] == "pairwise_reversal"
        ][:RESERVED_PAIRWISE_SLOTS]

        remaining_slots = TOTAL_CAP - len(pairwise_sorted)
        non_pairwise_sorted = [
            (c, cost) for c, cost in all_sorted if best_source[c] != "pairwise_reversal"
        ][:remaining_slots]

        final = sorted(non_pairwise_sorted + pairwise_sorted, key=lambda x: x[1])
        return tuple(final[:TOTAL_CAP])

    def debug_generation_stats(self, ocr_token: str) -> dict:
        """
        Diagnostic helper: reports the size of the reverse-substitution search
        space for a given token, to catch combinatorial explosion in the
        pairwise search before it silently crowds out legitimate candidates.
        """
        is_verified = self.is_known_word(ocr_token)
        single_reversals = []
        for i, obs_ch in enumerate(ocr_token):
            for true_ch, cost in self.reverse_sub_map.get(obs_ch, []):
                single_reversals.append((i, true_ch, cost))

        n_attempted = 0
        n_valid = 0
        if not is_verified and len(ocr_token) <= 8:
            for idx1, c1, cost1 in single_reversals:
                t1 = ocr_token[:idx1] + c1 + ocr_token[idx1 + 1:]
                for idx2, c2, cost2 in single_reversals:
                    if idx2 > idx1:
                        n_attempted += 1
                        t2 = t1[:idx2] + c2 + t1[idx2 + 1:]
                        if self.is_known_word(t2):
                            n_valid += 1

        return {
            "n_single_reversals_admitted": len(single_reversals),
            "n_pairwise_combinations_attempted": n_attempted,
            "n_pairwise_hits_that_are_real_words": n_valid,
            "max_reverse_sub_cost": self.max_reverse_sub_cost,
            "max_optical_cost_pairwise": self.max_optical_cost_pairwise,
            "reserved_pairwise_slots": self.reserved_pairwise_slots,
        }

    def correct_sentence(self, sentence: str) -> str:
        normalized_text = EthiopicPunctuationManager.normalize_sentence_text(sentence)
        if not normalized_text:
            return ""

        raw_tokens = normalized_text.split()
        if not raw_tokens:
            return ""

        token_tuples = [EthiopicPunctuationManager.decompose_token(tok) for tok in raw_tokens]

        start_state = kenlm.State()
        self.lm.BeginSentenceWrite(start_state)
        beam = [(0.0, start_state, [])]

        for leading, word, trailing in token_tuples:
            if not word:
                beam = [(score, state, path + [(leading + trailing, "", "")]) for score, state, path in beam]
                continue

            candidates = self.generate_candidates(word)
            new_beam = []
            is_orig_verified = self.is_known_word(word)

            for cand_word, opt_cost in candidates:
                opt_score = -self.lambda_optical * opt_cost

                # Morphological legitimacy prior
                morph_score = self.lambda_morph if self.is_known_word(cand_word) else 0.0

                # Protective Anchor Barrier on verified original words
                anchor_bonus = self.anchor_margin if (cand_word == word and is_orig_verified) else 0.0

                for acc_score, prev_state, path in beam:
                    next_state = kenlm.State()
                    lm_score = self.lm.BaseScore(prev_state, self._norm(cand_word), next_state)

                    total_score = (
                        acc_score
                        + (self.lambda_lm * lm_score)
                        + opt_score
                        + morph_score
                        + anchor_bonus
                    )
                    new_beam.append((total_score, next_state, path + [(leading, cand_word, trailing)]))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[: self.beam_width]

        # Fold in end-of-sentence scoring before picking the winner, so the
        # final word isn't chosen ignoring P(</s> | context).
        finalized = []
        for score, state, path in beam:
            end_state = kenlm.State()
            eos_score = self.lm.BaseScore(state, "</s>", end_state)
            finalized.append((score + self.lambda_lm * eos_score, path))
        finalized.sort(key=lambda x: x[0], reverse=True)
        best_path = finalized[0][1]

        reconstructed_tokens = []
        for lead, word, trail in best_path:
            if not word:
                reconstructed_tokens.append(lead)
            else:
                reconstructed_tokens.append(f"{lead}{word}{trail}")

        return " ".join(reconstructed_tokens)

    def correct_text_block(self, text: str) -> str:
        lines = text.splitlines()
        res = [self.correct_sentence(l) if l.strip() else "" for l in lines]
        return "\n".join(res)


# --------------------------------- CLI ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Amharic OCR Spelling Corrector")
    parser.add_argument("--kenlm", required=True, help="Path to KenLM model (.arpa/.bin/.klm)")
    parser.add_argument("--confusion-matrix", required=True, help="Path to confusion_matrix.json")
    parser.add_argument("--vocab", required=True, help="Path to frequency dictionary/vocab file")
    parser.add_argument(
        "--morphemes-json",
        default=None,
        help="Path to morphemes.json (defaults to AMHARIC_MORPHEMES_JSON env var or built-in default)",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-optical-cost", type=float, default=3.6)
    parser.add_argument("--max-optical-cost-pairwise", type=float, default=None)
    parser.add_argument("--max-reverse-sub-cost", type=float, default=None)
    parser.add_argument("--reserved-pairwise-slots", type=int, default=6)
    parser.add_argument("--text", default=None, help="Single line/sentence to correct")
    parser.add_argument("--file", default=None, help="Path to a text file to correct, line by line")
    args = parser.parse_args()

    corrector = AmharicOCRCorrector(
        kenlm_path=args.kenlm,
        confusion_matrix_path=args.confusion_matrix,
        dict_vocab_path=args.vocab,
        morphemes_json_path=args.morphemes_json,
        beam_width=args.beam_width,
        max_optical_cost=args.max_optical_cost,
        max_optical_cost_pairwise=args.max_optical_cost_pairwise,
        max_reverse_sub_cost=args.max_reverse_sub_cost,
        reserved_pairwise_slots=args.reserved_pairwise_slots,
    )

    if args.text is not None:
        print(corrector.correct_sentence(args.text))
    elif args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        print(corrector.correct_text_block(text))
    else:
        print("Reading lines from stdin (Ctrl+D to end)...", file=sys.stderr)
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line.strip():
                print("")
                continue
            print(corrector.correct_sentence(line))


if __name__ == "__main__":
    main()