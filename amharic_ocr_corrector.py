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
import time
# kenlm / symspellpy are imported lazily (see _load_kenlm / _load_symspell) so that the
# module can be imported, unit-tested and introspected on machines that do not
# have them installed. Production behaviour is unchanged: the constructor still
# requires a real LM unless one is injected.
kenlm = None
SymSpell = Verbosity = None


def _load_kenlm():
    """Import kenlm on first use (it is a heavy native extension)."""
    global kenlm
    if kenlm is None:
        import kenlm as _kenlm
        kenlm = _kenlm
    return kenlm


def _load_symspell():
    """Import symspellpy on first use. Returns (SymSpell, Verbosity)."""
    global SymSpell, Verbosity
    if SymSpell is None:
        from symspellpy import SymSpell as _SymSpell, Verbosity as _Verbosity
        SymSpell, Verbosity = _SymSpell, _Verbosity
    return SymSpell, Verbosity

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
except ImportError as e:
    logger.warning("Could not import AmharicNormalizer: %s", e)
    HAS_NORM = False

try:
    from segmenter import AmharicMorphologicalSegmenter
    HAS_MORPH = True
except ImportError as e:
    logger.warning("Could not import AmharicMorphologicalSegmenter: %s", e)
    HAS_MORPH = False
# -----------------------------------------------------------------------------

# Sentinel values that mean "the analyzer could not identify this word".
# Compared against the *string form* of the returned POS, because the analyzer
# may return either a plain string or an enum member (str(POS.UNKNOWN) ==
# "POS.UNKNOWN", which is why every candidate must be stringified before the
# membership test -- see _analysis_is_unknown).
UNKNOWN_POS_VALUES = ("", "UN", "UNKNOWN", "UNK", "NONE", "NULL")

_DEFAULT_MORPHEMES_JSON = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/morphemes.json"


class _BoundedCache:
    """Simple instance-local LRU cache (avoids lru_cache-on-bound-method leaks)."""

    def __init__(self, maxsize: int = 65536):
        self.maxsize = maxsize
        self._data = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return default

    def __setitem__(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key):
        return key in self._data


class _CacheInfoView:
    """
    Minimal stand-in for functools.lru_cache's cache_info(), backed by a
    _BoundedCache. The evaluator prints these; the bounded cache tracks hits
    and misses differently, so report what it actually knows.
    """

    def __init__(self, cache):
        self._cache = cache

    def cache_info(self):
        hits = getattr(self._cache, "hits", 0)
        misses = getattr(self._cache, "misses", 0)
        return collections.namedtuple("CacheInfo", "hits misses maxsize currsize")(
            hits, misses, self._cache.maxsize, len(self._cache._data)
        )


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
        reserved_pairwise_slots: int = 20,
        lambda_optical: float = 1.2,
        lambda_lm: float = 1.8,
        lambda_morph: float = 1.5,
        anchor_margin: float = 3.5,
        known_word_cache_size: int = 131072,
        candidate_cache_size: int = 32768,
        # --- dependency injection / testability -----------------------------
        # lm: an object exposing BeginSentenceWrite(state) and
        #     BaseScore(prev_state, word, next_state). Defaults to a real
        #     kenlm.Model built from kenlm_path.
        lm=None,
        state_factory=None,
        # segmenter: an object exposing segment_word(word) -> dict. Defaults to
        #     AmharicMorphologicalSegmenter(); pass None explicitly to disable
        #     morphology (tests, ablations, analyzer-free deployments).
        segmenter=...,
        # --- search / scoring knobs -----------------------------------------
        max_candidates: int = 24,
        score_eos: bool = True,
        indel_cost_cap: float = 2.4,
        morphology_cache_size: int = 65536,
        require_morphemes_json: bool = False,
        verbose: bool = True,
    ):
        self.beam_width = beam_width
        self.max_optical_cost = max_optical_cost
        self.max_candidates = max_candidates
        self.score_eos = score_eos
        self.indel_cost_cap = indel_cost_cap
        self.require_morphemes_json = require_morphemes_json
        self.verbose = verbose

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

        # --- Language model (injectable) ------------------------------------
        if lm is not None:
            self.lm = lm
        else:
            _kenlm = _load_kenlm()
            if self.verbose:
                print(f"Loading KenLM Language Model: {kenlm_path}...", flush=True)
            self.lm = _kenlm.Model(kenlm_path)

        if state_factory is not None:
            self._state_factory = state_factory
        else:
            self._state_factory = _load_kenlm().State

        if self.verbose:
            print(f"Loading Optical Confusion Matrix: {confusion_matrix_path}...", flush=True)
        with open(confusion_matrix_path, "r", encoding="utf-8") as f:
            self.costs = json.load(f)

        self.default_sub = float(self.costs.get("default_sub", 4.5))
        self.default_del = float(self.costs.get("default_del", 2.4))
        self.default_ins = float(self.costs.get("default_ins", 2.4))

        # Bounded instance-local caches (replaces lru_cache-on-method leak hazard)
        self._known_word_cache = _BoundedCache(known_word_cache_size)
        self._candidate_cache = _BoundedCache(candidate_cache_size)
        # Caches the FULL analyzer result (not just a bool), so downstream
        # morphology work can reuse the segmentation instead of re-analyzing.
        self._analysis_cache = _BoundedCache(morphology_cache_size)

        # --- morphology counters (see morphology_stats) ---------------------
        self._morph_stats = {
            "segmenter_calls": 0,
            "segmenter_failures": 0,
            "segmenter_seconds": 0.0,
            "trusted_bypasses": 0,
        }
        self._analysis_cache_hits = 0

        # 1. Normalizer
        self.normalizer = AmharicNormalizer() if HAS_NORM else None

        # 2. Morphological Segmenter & Declarative Morpheme Inventory
        #    `segmenter=...` means "use the module default"; `segmenter=None`
        #    explicitly disables morphology.
        if segmenter is not ...:
            self.segmenter = segmenter
        else:
            self.segmenter = AmharicMorphologicalSegmenter() if HAS_MORPH else None
        if self.segmenter and self.verbose:
            print("Morphological Segmenter ready.", flush=True)

        # 3. Load Exhaustive Declarative Morphemes from morphemes.json
        self.protected_suffixes = set()
        self.protected_prefixes = set()
        morphemes_json_path = morphemes_json_path or os.environ.get(
            "AMHARIC_MORPHEMES_JSON", _DEFAULT_MORPHEMES_JSON
        )

        self.morphemes_json_path = morphemes_json_path

        if os.path.exists(morphemes_json_path):
            with open(morphemes_json_path, "r", encoding="utf-8") as mj:
                m_data = json.load(mj)
                for k, v in m_data.items():
                    if "suffix" in k and isinstance(v, (list, set, tuple)):
                        self.protected_suffixes.update(v)
                    elif "prefix" in k and isinstance(v, (list, set, tuple)):
                        self.protected_prefixes.update(v)
            if self.verbose:
                print(
                    f"Loaded {len(self.protected_suffixes)} declarative suffixes and "
                    f"{len(self.protected_prefixes)} prefixes from morphemes.json.",
                    flush=True,
                )
        else:
            # NOTE: with empty affix inventories is_unsupported_morpheme_peel()
            # is a no-op, so every "peel" damage (ተከትሎም -> ተከትሎ, የሚታይም ->
            # የሚታይ, ...) goes through unchecked. That failure mode is silent
            # and looks exactly like a modelling bug, so make it fatal unless
            # the caller explicitly opts out.
            msg = f"morphemes.json not found at {morphemes_json_path}"
            if require_morphemes_json:
                raise FileNotFoundError(msg)
            logger.warning("%s -- the morpheme-peel guard is DISABLED.", msg)
            if self.verbose:
                print(f"[WARN] {msg} (peel guard disabled)", flush=True)

        # Pre-sort affix inventories once (longest-first) instead of re-sorting
        # on every recursive call inside _is_valid_affix_sequence.
        self._suffixes_sorted = sorted(self.protected_suffixes, key=len, reverse=True)
        self._prefixes_sorted = sorted(self.protected_prefixes, key=len, reverse=True)

        # 4. Vocabulary & SymSpell
        if self.verbose:
            print(f"Loading Lexicon into SymSpell & Hash Set: {dict_vocab_path}...", flush=True)
        self.vocab = {}
        # norm_w -> best-known surface spelling (preserves surface form on output)
        self.surface_form = {}
        _SymSpell, _ = _load_symspell()
        self.symspell = _SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

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

                if self.verbose and idx % 1000000 == 0:
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

        if self.verbose:
            print(f"Corrector initialized successfully ({len(self.vocab):,} active words).\n", flush=True)

    def _norm(self, word: str) -> str:
        if not word or not self.normalizer:
            return word
        return self.normalizer.normalize_word_strict(word)

    def _to_surface(self, norm_or_word: str) -> str:
        """Map a (possibly normalized) form back to its preferred surface spelling."""
        return self.surface_form.get(norm_or_word, norm_or_word)

    # ------------------------------------------------------------------ #
    # Morphology: cached, structured analysis                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _analysis_is_unknown(analysis, pos) -> bool:
        """
        True when the analyzer could not identify the word.

        The analyzer may return a plain string ("UNKNOWN") or an enum member.
        str(POS.UNKNOWN) == "POS.UNKNOWN", so a bare `pos not in ("UN",
        "UNKNOWN")` test silently passes for enums and marks *every* token as
        known -- which is catastrophic (it suppresses the pairwise search and
        grants the anchor bonus to OCR garbage). Compare the tail of the
        stringified form instead.
        """
        if analysis is None:
            return True
        if pos is None:
            return True
        text = str(pos).strip()
        if not text:
            return True
        tail = text.rsplit(".", 1)[-1].upper()
        return tail in UNKNOWN_POS_VALUES

    def analyze(self, word: str):
        """
        Return the analyzer's structured result for `word`, cached per instance.

        Always dict-shaped, even for absent/failing/negative analyses, so
        callers need no None checks:
            {"known": bool, "in_vocab": bool, "pos": str|None,
             "stem": str|None, "analysis": <raw analyzer dict>|None,
             "error": str|None}
        The raw analyzer payload is preserved in "analysis" -- that is the part
        the corrector currently throws away, and the part a morphological
        candidate generator needs (affix skeleton, stem, confidence).
        """
        if not word:
            return {"known": False, "in_vocab": False, "pos": None,
                    "stem": None, "analysis": None, "error": "empty"}

        norm_w = self._norm(word)
        cached = self._analysis_cache.get(norm_w)
        if cached is not None:
            self._analysis_cache_hits += 1
            return cached

        in_vocab = norm_w in self.vocab
        result = {"known": in_vocab, "in_vocab": in_vocab, "pos": None,
                  "stem": None, "analysis": None, "error": None}

        if not in_vocab and self.segmenter is not None and self.lambda_morph != 0:
            t0 = time.perf_counter()
            try:
                # Analyze the NORMALIZED form: the raw surface may use
                # orthographic variants (ሐ/ዐ/ፀ ...) the analyzer does not know.
                raw = self.segmenter.segment_word(norm_w)
                self._morph_stats["segmenter_calls"] += 1
                pos = raw.get("pos") if isinstance(raw, dict) else None
                result["analysis"] = raw if isinstance(raw, dict) else {"pos": pos}
                result["pos"] = None if pos is None else str(pos)
                result["stem"] = result["analysis"].get("stem")
                result["known"] = not self._analysis_is_unknown(result["analysis"], pos)
            except Exception as exc:  # transient analyzer failure
                self._morph_stats["segmenter_failures"] += 1
                logger.warning("segmenter.segment_word failed for %r", word, exc_info=True)
                self._morph_stats["segmenter_seconds"] += time.perf_counter() - t0
                # Deliberately NOT cached: a transient failure must not poison
                # this word for the life of the process. The cost is one
                # analyzer call per occurrence while the analyzer is broken,
                # which is the correct side to err on. Genuine *negative*
                # analyses (see below) ARE cached.
                return dict(result, known=False, error=f"{type(exc).__name__}: {exc}")
            self._morph_stats["segmenter_seconds"] += time.perf_counter() - t0
        elif not in_vocab:
            # No analyzer available (or morphology disabled): don't pretend.
            result["error"] = "no_segmenter" if self.segmenter is None else "morphology_disabled"

        self._analysis_cache[norm_w] = result
        return result

    def validate_morphology(self, word: str) -> bool:
        """Public morphology predicate (used by callers and diagnostics)."""
        return bool(self.analyze(word)["known"])

    def morphology_stats(self) -> dict:
        """
        Cumulative analyzer cost/hit counters.

        Use this to confirm or refute the claim that morphology dominates
        correction latency: snapshot before/after a correction run and compare
        `segmenter_calls` against the number of tokens processed.
        """
        data = self._analysis_cache._data
        return {
            "segmenter_calls": self._morph_stats["segmenter_calls"],
            "segmenter_failures": self._morph_stats["segmenter_failures"],
            "segmenter_seconds": round(self._morph_stats["segmenter_seconds"], 6),
            "trusted_bypasses": self._morph_stats["trusted_bypasses"],
            "cache": {
                "hits": self._analysis_cache_hits,
                "size": len(data),
                "maxsize": self._analysis_cache.maxsize,
            },
        }

    def clear_caches(self) -> None:
        """Drop memoized candidates / analyses / known-word flags."""
        self._known_word_cache = _BoundedCache(self._known_word_cache.maxsize)
        self._candidate_cache = _BoundedCache(self._candidate_cache.maxsize)
        self._analysis_cache = _BoundedCache(self._analysis_cache.maxsize)
        self._analysis_cache_hits = 0

    # ---------------------------------------------------------------- #
    # is_known_word: bounded-cache version (no more lru_cache-on-self) #
    # ---------------------------------------------------------------- #
    def is_known_word(self, word: str) -> bool:
        if not word:
            return False
        cached = self._known_word_cache.get(word)
        if cached is not None:
            return cached
        # Reuse the structured analysis so the analyzer is never called twice
        # for the same word (previously this function called it independently).
        result = self.analyze(word)["known"]
        self._known_word_cache[word] = result
        return result

    def _is_in_vocab(self, word: str) -> bool:
        """
        Strict, vocab-only membership check (no segmenter fallback). Used in the
        O(n) / O(n^2) candidate-generation hot loops (single/pairwise optical
        reversal), where we specifically want "is this exactly a known real
        word" -- not "could this structurally parse as a word" (the segmenter
        fallback in is_known_word), which is both slower and a weaker signal
        for accepting reversal-generated candidates.
        """
        return self._norm(word) in self.vocab

    def _compute_is_known_word(self, word: str) -> bool:
        """Kept for backwards compatibility; delegates to the cached analysis."""
        return self.analyze(word)["known"]

    def _ins_cost(self, observed_ch: str) -> float:
        """Cost of OCR spuriously *producing* `observed_ch` (no true counterpart)."""
        raw = float(self.costs.get("ins", {}).get(observed_ch, self.default_ins))
        return raw if self.indel_cost_cap is None else min(raw, self.indel_cost_cap)

    def _del_cost(self, true_ch: str) -> float:
        """Cost of OCR *dropping* true character `true_ch` (no observed counterpart)."""
        raw = float(self.costs.get("del", {}).get(true_ch, self.default_del))
        return raw if self.indel_cost_cap is None else min(raw, self.indel_cost_cap)

    def optical_distance(self, cand_str: str, ocr_str: str) -> float:
        """Alias with the (true, observed) argument order used elsewhere."""
        return self.compute_optical_cost(ocr_str, cand_str)

    def _cost_rows(self):
        """
        Resolve the three cost tables and their capped defaults once per call.

        The DP inner loop used to do, per cell, two Python method calls each
        performing a 3-level dict lookup plus float() and min(). Profiling
        showed those two helpers alone were ~29% of total correction time and
        dict.get another ~9%. Hoisting them out of the loop is the single
        biggest available win in the corrector.
        """
        cap = self.indel_cost_cap
        return (
            self.costs.get("sub", {}), self.default_sub,
            self.costs.get("ins", {}), self.default_ins,
            self.costs.get("del", {}), self.default_del,
            cap,
        )

    def compute_optical_cost(self, ocr_str: str, cand_str: str, cutoff: float = None) -> float:
        """
        Directional weighted Levenshtein: cost that OCR would emit `ocr_str`
        given that the true text was `cand_str`.

        If `cutoff` is given, the search abandons as soon as it can prove the
        optimal alignment exceeds it, returning float('inf'). This is exact,
        not a heuristic: every cell cost is non-negative, so min_j(dp[i][j]) is
        non-decreasing in i; once a row's minimum exceeds the cutoff, no
        completion of that row can recover.
        """
        m, n = len(ocr_str), len(cand_str)
        (sub_table, default_sub, ins_table, default_ins,
         del_table, default_del, cap) = self._cost_rows()

        # Degenerate alignments: all-insertion or all-deletion.
        if n == 0:
            total = 0.0
            for c in ocr_str:
                v = ins_table.get(c, default_ins)
                total += cap if (cap is not None and v > cap) else v
            return total if cutoff is None or total <= cutoff else float("inf")
        if m == 0:
            total = 0.0
            for c in cand_str:
                v = del_table.get(c, default_del)
                total += cap if (cap is not None and v > cap) else v
            return total if cutoff is None or total <= cutoff else float("inf")

        # Boundary rows use the SAME per-character tables as the recurrence
        # body (previously they used the flat defaults, and with the two
        # tables swapped): dp[i][0] means "every observed char so far was
        # spurious" -> ins[observed]; dp[0][j] means "every candidate char so
        # far was dropped by OCR" -> del[candidate].
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        ins0 = [0.0] * (m + 1)
        for i in range(1, m + 1):
            c = ocr_str[i - 1]
            v = ins_table.get(c, default_ins)
            if cap is not None and v > cap:
                v = cap
            ins0[i] = ins0[i - 1] + v
            dp[i][0] = ins0[i]
        del0 = [0.0] * (n + 1)
        for j in range(1, n + 1):
            c = cand_str[j - 1]
            v = del_table.get(c, default_del)
            if cap is not None and v > cap:
                v = cap
            del0[j] = del0[j - 1] + v
            dp[0][j] = del0[j]

        # Per-character rows for the body, resolved once instead of per cell.
        ins_row = []
        for c in ocr_str:
            v = ins_table.get(c, default_ins)
            ins_row.append(cap if (cap is not None and v > cap) else v)
        del_row = []
        sub_rows = []
        for c in cand_str:
            v = del_table.get(c, default_del)
            del_row.append(cap if (cap is not None and v > cap) else v)
            sub_rows.append(sub_table.get(c, {}))

        for i in range(1, m + 1):
            c_obs = ocr_str[i - 1]
            ins_cost = ins_row[i - 1]
            prev = dp[i - 1]
            cur = dp[i]
            for j in range(1, n + 1):
                if c_obs == cand_str[j - 1]:
                    sub_cost = 0.0
                else:
                    row = sub_rows[j - 1]
                    v = row.get(c_obs)
                    sub_cost = default_sub if v is None else v
                a = prev[j] + ins_cost
                b = cur[j - 1] + del_row[j - 1]
                d = prev[j - 1] + sub_cost
                cur[j] = a if (a <= b and a <= d) else (b if b <= d else d)
            if cutoff is not None and prev is not None:
                # Row minima are non-decreasing, so this is a sound prune.
                if min(cur) > cutoff:
                    return float("inf")

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

    # --- backwards-compatible aliases -----------------------------------
    # 04_evaluate_full_corrector.py and tests/ call these and inspect
    # .cache_info(); keep them working, backed by the bounded caches.
    @property
    def _generate_candidates_cached(self):
        return _CacheInfoView(self._candidate_cache)

    @property
    def _validate_morphology_cached(self):
        return _CacheInfoView(self._analysis_cache)

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
        #
        # SymSpell's index is unweighted: it returns every lexicon entry within
        # plain edit distance 1, and most of them are optically implausible.
        # Measured on the real matrix + a 236k lexicon, ~54% of suggestions
        # fail the acceptance gate after paying for a full O(m*n) DP. Pass the
        # gate as a `cutoff` so the DP abandons the moment it can prove the
        # alignment cannot qualify (exact, not heuristic -- see the docstring).
        suggestions = self.symspell.lookup(norm_token, Verbosity.ALL, max_edit_distance=1)
        sym_cutoff = max(self.max_optical_cost, self.max_optical_cost_pairwise)
        for s in suggestions:
            cand = self._to_surface(s.term)
            if is_verified and self.is_unsupported_morpheme_peel(ocr_token, cand):
                continue
            opt_cost = self.compute_optical_cost(ocr_token, cand, cutoff=sym_cutoff)
            if opt_cost == float("inf"):
                continue
            raw_candidates.append((cand, opt_cost, "symspell"))

        # 3. Direct Optical Reversal (single substitution)
        single_reversals = []
        for i, obs_ch in enumerate(ocr_token):
            for true_ch, cost in self.reverse_sub_map.get(obs_ch, []):
                variant = ocr_token[:i] + true_ch + ocr_token[i + 1:]
                if self._is_in_vocab(variant):
                    raw_candidates.append((variant, cost, "single_reversal"))
                single_reversals.append((i, true_ch, cost))

        # 4. Pairwise optical substitution for unverified words
        if not is_verified and len(ocr_token) <= 8:
            for idx1, c1, cost1 in single_reversals:
                t1 = ocr_token[:idx1] + c1 + ocr_token[idx1 + 1:]
                for idx2, c2, cost2 in single_reversals:
                    if idx2 > idx1:
                        t2 = t1[:idx2] + c2 + t1[idx2 + 1:]
                        if self._is_in_vocab(t2):
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
        TOTAL_CAP = self.max_candidates
        RESERVED_PAIRWISE_SLOTS = min(self.reserved_pairwise_slots, TOTAL_CAP)

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

    def _morphology_score(self, word: str, cand_word: str, is_orig_verified: bool) -> float:
        """
        The morphological legitimacy bonus for `cand_word` replacing `word`.

        Legacy form (kept here for reference, see tests/test_head_contract.py):
            self.lambda_morph if self.is_known_word(cand_word) else 0.0

        Every candidate source except `identity` guarantees in-vocabulary
        membership by construction (SymSpell terms come from the dictionary;
        single/pairwise reversal are gated on _is_in_vocab; ghost_suffix on
        is_known_word). So `is_known_word(cand_word)` returned True for every
        non-identity candidate anyway -- meaning lambda_morph was a *constant*
        across all alternatives, and each of those calls cost a full analyzer
        invocation whenever the candidate was OOV.

        The fast path below is therefore semantically identical to the legacy
        expression while issuing no analyzer calls for constructed candidates:
        it checks the cheap vocab test first and only falls through to the
        analyzer if that fails.
        """
        if self.lambda_morph == 0:
            return 0.0
        if cand_word == word:
            cand_known = is_orig_verified
        else:
            cand_known = self._is_in_vocab(cand_word) or self.is_known_word(cand_word)
            if cand_known:
                self._morph_stats["trusted_bypasses"] += 1
        return self.lambda_morph if cand_known else 0.0

    def correct_sentence(self, sentence: str) -> str:
        normalized_text = EthiopicPunctuationManager.normalize_sentence_text(sentence)
        if not normalized_text:
            return ""

        raw_tokens = normalized_text.split()
        if not raw_tokens:
            return ""

        token_tuples = [EthiopicPunctuationManager.decompose_token(tok) for tok in raw_tokens]

        start_state = self._state_factory()
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

                morph_score = self._morphology_score(word, cand_word, is_orig_verified)

                # Protective Anchor Barrier on verified original words
                anchor_bonus = self.anchor_margin if (cand_word == word and is_orig_verified) else 0.0

                norm_cand = self._norm(cand_word)

                for acc_score, prev_state, path in beam:
                    next_state = self._state_factory()
                    lm_score = self.lm.BaseScore(prev_state, norm_cand, next_state)

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
        if self.score_eos:
            finalized = []
            for score, state, path in beam:
                end_state = self._state_factory()
                eos_score = self.lm.BaseScore(state, "</s>", end_state)
                finalized.append((score + self.lambda_lm * eos_score, path))
            finalized.sort(key=lambda x: x[0], reverse=True)
        else:
            finalized = [(score, path) for score, _, path in beam]
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
    parser.add_argument("--reserved-pairwise-slots", type=int, default=20)
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