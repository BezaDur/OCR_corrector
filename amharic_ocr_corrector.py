#!/usr/bin/env python3
"""
Production CPU-based Amharic OCR Spelling Corrector & Post-Processor.

Architecture:
  1. EthiopicPunctuationManager:
     - Isolates leading and trailing punctuation from lexical tokens.
     - Preserves abbreviation symbols ('/' and '.') inside words.
     - Strips non-spacing combining diacritics for dictionary lookups.
  2. Candidate retrieval & Vocabulary:
     - Bounded weighted trie or optional SymSpell retrieval with optical reranking.
     - Cheap reverse-confusion neighbors also cover frequent real-word errors.
     - Seeded with main_i7.txt and segmenter's lexical frequencies.
  3. Morphological Prior (AmharicMorphologicalSegmenter):
     - Trusts high-frequency dictionary entries; deeply validates lower-frequency
       dictionary words and OOV agglutinated forms, with cached results.
     - Protects legitimate morphological forms from destructive over-correction.
  4. Conservative Viterbi Decoder (KenLM 5-gram):
     - Beam search decoder with an explicit Anchor Margin (ANCHOR_MARGIN).
     - Prevents high-frequency language model words from displacing verified OCR words.
"""

import os
import sys
import re
import json
try:
    import kenlm
except ImportError:
    kenlm = None
from functools import lru_cache
import math
from time import perf_counter

_AUTO_SEGMENTER = object()

# ----------------- Dynamic Import of Morphological Suite -----------------
for p in [
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_segmenter",
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis"
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from segmenter import AmharicMorphologicalSegmenter
    HAS_MORPH = True
    print("[OK] Successfully imported AmharicMorphologicalSegmenter.")
except ImportError as e:
    print(f"[WARN] Could not import AmharicMorphologicalSegmenter: {e}")
    HAS_MORPH = False
# ------------------------------------------------------------------------


class EthiopicPunctuationManager:
    """
    Architectural manager for Ethiopic/Latin punctuation and combining diacritics.
    Handles token-level peeling, diacritic stripping, and boundary normalization.
    """
    # Combining Diacritics (U+135D to U+135F, and general Unicode combining marks)
    DIACRITICS_REGEX = re.compile(r'[\u135D-\u135F\u0300-\u036F]')

    # Punctuation characters to peel from outer token boundaries.
    # Note: Forward slash '/' and period '.' are omitted here to protect
    # abbreviations such as 'ም/ቤት', 'ዶ/ር', and 'ዓ.ም'.
    ALL_PUNCT = set(
        "።!?፧…፣፤፥፦፠፨"      # Ethiopic punctuation marks
        ":;,!?"             # ASCII punctuation (excluding / and .)
        "“”‘’«»‹›„\'\""     # Typographic and ASCII quotes
        "()[]{}"            # Parentheses and brackets
        "|\\—–-"            # Pipe, backslash, hyphens, and dashes
    )

    @classmethod
    def normalize_sentence_text(cls, text: str) -> str:
        """Normalizes inter-word and boundary keyboard substitutions."""
        if not text:
            return ""

        # 1. Double colons, double hulet neteb, or double periods -> Arat Neteb (።)
        text = re.sub(r'([:፡]{2,}|\.{2,})', '።', text)

        # 2. Terminal/trailing single colons or hulet neteb before newlines/quotes -> ።
        text = re.sub(r'[:፡]\s*$', '።', text, flags=re.MULTILINE)
        text = re.sub(r'[:፡]\s*([”"»‹\n])', r'።\1', text)

        # 3. Inter-word isolated single colon/hulet neteb -> clean space
        text = re.sub(r'(?<![0-9])[:፡]|[:፡](?![0-9])', ' ', text)

        # 4. Collapse spaces
        text = re.sub(r'[ ]+', ' ', text)
        return text.strip()

    @classmethod
    def decompose_token(cls, raw_token: str, strip_diacritics: bool = True):
        """
        Splits a surface token into (leading_punct, clean_word, trailing_punct).
        e.g., '“ፕርጀከት' -> ('“', 'ፕርጀከት', '')
              'ነው፡፡”'    -> ('', 'ነው', '።”')
              'ም/ቤት'     -> ('', 'ም/ቤት', '')  [slash is preserved intact]
        """
        if not raw_token:
            return "", "", ""

        # Peel leading punctuation
        start = 0
        while start < len(raw_token) and raw_token[start] in cls.ALL_PUNCT:
            start += 1
        leading_punct = raw_token[:start]

        # Peel trailing punctuation
        end = len(raw_token)
        while end > start and raw_token[end - 1] in cls.ALL_PUNCT:
            end -= 1
        trailing_punct = raw_token[end:]

        # Core lexical word with combining diacritics stripped for dictionary lookup
        core_word = raw_token[start:end]
        if strip_diacritics:
            core_word = cls.DIACRITICS_REGEX.sub('', core_word)

        return leading_punct, core_word, trailing_punct


class AmharicOCRCorrector:
    def __init__(
        self,
        kenlm_path: str,
        confusion_matrix_path: str,
        dict_vocab_path: str = None,
        beam_width: int = 5,
        max_optical_cost: float = 4.0,
        lambda_optical: float = 1.0,
        lambda_lm: float = 1.8,
        lambda_morph: float = 1.5,
        max_candidates: int = 25,
        *,
        candidate_backend: str = "trie",
        trie_search: str = "adaptive",
        symspell_max_edit_distance: int = 1,
        symspell_prefix_length: int = 7,
        indel_cost_cap: float = 2.4,
        anchor_margin: float = 1.6,
        score_eos: bool = True,
        trusted_long_word_frequency: int = 1000,
        trusted_short_word_frequency: int = 4000,
        morphology_cache_size: int = 65536,
        lm=None,
        state_factory=None,
        segmenter=_AUTO_SEGMENTER
    ):
        if beam_width < 1 or max_candidates < 1:
            raise ValueError("beam_width and max_candidates must be positive")
        if not math.isfinite(max_optical_cost) or max_optical_cost < 0:
            raise ValueError("max_optical_cost must be finite and nonnegative")
        if candidate_backend not in ("trie", "symspell"):
            raise ValueError("candidate_backend must be 'trie' or 'symspell'")
        if indel_cost_cap is not None and (not math.isfinite(indel_cost_cap) or indel_cost_cap < 0):
            raise ValueError("indel_cost_cap must be nonnegative or None")
        if trie_search not in ("adaptive", "full"):
            raise ValueError("trie_search must be 'adaptive' or 'full'")
        for name, value in (("trusted_long_word_frequency", trusted_long_word_frequency),
                            ("trusted_short_word_frequency", trusted_short_word_frequency)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(morphology_cache_size, bool) or not isinstance(morphology_cache_size, int) or morphology_cache_size < 1:
            raise ValueError("morphology_cache_size must be a positive integer")
        self.trusted_long_word_frequency = trusted_long_word_frequency
        self.trusted_short_word_frequency = trusted_short_word_frequency
        self.candidate_backend = candidate_backend
        self.trie_search = trie_search
        self.indel_cost_cap = indel_cost_cap
        self.anchor_margin = anchor_margin
        self.score_eos = score_eos
        if lm is None:
            if kenlm is None:
                raise ImportError("Install kenlm to load a language model")
            print(f"Loading KenLM Language Model: {kenlm_path}...", flush=True)
            lm = kenlm.Model(kenlm_path)
        self.lm = lm
        self._state_factory = state_factory or (kenlm.State if kenlm else None)
        if self._state_factory is None:
            raise ValueError("An injected LM requires state_factory when kenlm is unavailable")
        self.symspell = None
        if candidate_backend == "symspell":
            try:
                from symspellpy import SymSpell, Verbosity
            except ImportError as exc:
                raise ImportError("Install requirements-symspell.txt for the SymSpell backend") from exc
            if symspell_max_edit_distance < 1 or symspell_prefix_length <= symspell_max_edit_distance:
                raise ValueError("SymSpell requires distance >= 1 and prefix_length > distance")
            self.symspell = SymSpell(
                max_dictionary_edit_distance=symspell_max_edit_distance,
                prefix_length=symspell_prefix_length,
            )
            self._symspell_verbosity = Verbosity.ALL
            self.symspell_max_edit_distance = symspell_max_edit_distance

        print(f"Loading Optical Confusion Matrix: {confusion_matrix_path}...", flush=True)
        with open(confusion_matrix_path, "r", encoding="utf-8") as f:
            self.costs = json.load(f)

        self.beam_width = beam_width
        self.max_optical_cost = max_optical_cost
        self.lambda_optical = lambda_optical
        self.lambda_lm = lambda_lm
        self.lambda_morph = lambda_morph
        self.max_candidates = max_candidates

        self.default_sub = float(self.costs.get("default_sub", 4.5))
        self.default_del = float(self.costs.get("default_del", 5.0))
        self.default_ins = float(self.costs.get("default_ins", 5.0))
        all_costs = [self.default_sub, self.default_del, self.default_ins]
        all_costs.extend(value for row in self.costs.get("sub", {}).values() for value in row.values())
        all_costs.extend(self.costs.get("ins", {}).values())
        all_costs.extend(self.costs.get("del", {}).values())
        if any(not math.isfinite(float(cost)) or float(cost) < 0 for cost in all_costs):
            raise ValueError("Optical costs must be finite and nonnegative")

        # Initialize Morphological Segmenter
        self.segmenter = None if segmenter is _AUTO_SEGMENTER else segmenter
        if segmenter is _AUTO_SEGMENTER and HAS_MORPH:
            print("Initializing Amharic Morphological Segmenter...", flush=True)
            self.segmenter = AmharicMorphologicalSegmenter()
            print("Segmenter ready.", flush=True)

        self.reverse_confusions = {}
        for true_ch, observations in self.costs.get("sub", {}).items():
            for observed_ch, cost in observations.items():
                if len(true_ch) == len(observed_ch) == 1:
                    self.reverse_confusions.setdefault(observed_ch, []).append((true_ch, float(cost)))

        # Build Core Prefix Trie and Vocabulary
        print(f"Building {candidate_backend} index & vocabulary...", flush=True)
        self.vocab = {}
        self.trie = {}

        # 1. Seed Trie with Segmenter's built-in vocabulary frequencies
        if self.segmenter and hasattr(self.segmenter, "vocab_freq"):
            for w, f in self.segmenter.vocab_freq.items():
                self._add_to_trie(w, f)

        # 2. Seed Trie with main_i7.txt lexicon
        if dict_vocab_path:
            # A misspelled path must not silently leave an abbreviations-only index.
            if not os.path.isfile(dict_vocab_path):
                raise FileNotFoundError(dict_vocab_path)
            with open(dict_vocab_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    w = parts[0]
                    freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
                    self._add_to_trie(w, freq)
                    if i % 1000000 == 0:
                        print(f"  ... loaded {i:,} dictionary lines ({len(self.vocab):,} unique words)", flush=True)

        # 3. Seed verified standard abbreviations so they are never altered
        COMMON_ABBRS = [
            "ም/ቤት", "ት/ቤት", "ፍ/ቤት", "ጽ/ቤት", "ክ/ከተማ", "ዶ/ር", "ጠ/ሚኒስትር",
            "ወ/ሮ", "ወ/ሪት", "መ/ር", "ዓ.ም", "እ.ኤ.አ", "ቤ/ክርስቲያን", "ፕ/ር"
        ]
        self.protected_words = frozenset(COMMON_ABBRS)
        for abbr in COMMON_ABBRS:
            self._add_to_trie(abbr, freq=1000)

        print(f"Index successfully built with {len(self.vocab):,} active words.", flush=True)

        # High-frequency ambiguity list (words that must evaluate real-word optical neighbors)
        self.ambiguous_frequent_words = {"አና", "እና", "በ", "ከ", "ይህ", "ይሀ"}

        # Use the same frequency policy for dictionary trust and the trie fast path.
        # Ambiguous words still get the trie walk even when trusted for scoring.
        # All frequent words still get reverse-confusion/SymSpell alternatives.
        print("Extracting high-frequency word set...", flush=True)
        self.common_words = {
            w for w in self.vocab
            if self.is_trusted_dictionary_word(w) and w not in self.ambiguous_frequent_words
        }
        print(f"Frequent-word set active with {len(self.common_words):,} words.", flush=True)
        # Bind caches after index construction, avoiding millions of empty cache clears.
        self._validate_morphology_cached = lru_cache(maxsize=morphology_cache_size)(self._validate_morphology_uncached)
        self._morphology_counters = dict(trusted_bypasses=0, segmenter_calls=0,
                                        segmenter_failures=0, segmenter_seconds=0.0)
        self._generate_candidates_cached = lru_cache(maxsize=32768)(self._generate_candidates_tuple)

    def _add_to_trie(self, word: str, freq: int = 1):
        if not word:
            return
        old_freq = self.vocab.get(word, 0)
        self.vocab[word] = max(old_freq, freq, 1)
        if self.symspell is not None:
            self.symspell.create_dictionary_entry(word, self.vocab[word] - old_freq)
        else:
            node = self.trie
            for ch in word:
                node = node.setdefault(ch, {})
            node["$"] = True
        if hasattr(self, "common_words"):
            if self.is_trusted_dictionary_word(word) and word not in self.ambiguous_frequent_words:
                self.common_words.add(word)
            else:
                self.common_words.discard(word)
        if hasattr(self, "_generate_candidates_cached"):
            self._generate_candidates_cached.cache_clear()

    def is_trusted_dictionary_word(self, word: str) -> bool:
        """Frequency-based lexical prior, NOT proof of a morphological analysis.

        Lookup-token lengths are Unicode characters (normally fidel). Length >= 4
        requires frequency > trusted_long_word_frequency; lengths 2–3 require
        frequency > trusted_short_word_frequency. One-character words never pass.
        Abbreviation protection is a separate explicit policy.
        """
        size = len(word)
        if size < 2:
            return False
        threshold = (self.trusted_long_word_frequency if size >= 4
                     else self.trusted_short_word_frequency)
        return self.vocab.get(word, 0) > threshold

    def _get_sub_cost(self, c_true: str, c_obs: str) -> float:
        if c_true == c_obs:
            return 0.0
        return float(self.costs.get("sub", {}).get(c_true, {}).get(c_obs, self.default_sub))

    def _get_del_cost(self, c_true: str) -> float:
        cost = float(self.costs.get("del", {}).get(c_true, self.default_del))
        return cost if self.indel_cost_cap is None else min(cost, self.indel_cost_cap)

    def _get_ins_cost(self, c_obs: str) -> float:
        cost = float(self.costs.get("ins", {}).get(c_obs, self.default_ins))
        return cost if self.indel_cost_cap is None else min(cost, self.indel_cost_cap)

    def _validate_morphology_uncached(self, word: str) -> bool:
        if not self.segmenter or not word:
            return False
        self._morphology_counters['segmenter_calls'] += 1
        start = perf_counter()
        try:
            res = self.segmenter.segment_word(word)
            pos = res.get("pos")
            # Support string tags and Enum values without authenticating UNKNOWN.
            tag = getattr(pos, "name", pos)
            tag = str(tag or "").upper().rsplit(".", 1)[-1]
            return bool(tag and tag not in ("UN", "UNKNOWN", "NONE"))
        except Exception:
            self._morphology_counters['segmenter_failures'] += 1
            raise
        finally:
            self._morphology_counters['segmenter_seconds'] += perf_counter() - start

    def validate_morphology(self, word: str) -> bool:
        """Accept trusted dictionary words, otherwise use cached deep validation.

        This legacy method now returns lexical OR morphological legitimacy for
        scoring. A trusted word gets the same reward even without an analyzer.
        The shortcut does not remove competing correction candidates.
        """
        # Check before the LRU: trusted dictionary words consume no analysis entries,
        # and a newly trusted word cannot be hidden by an older cached rejection.
        if self.is_trusted_dictionary_word(word):
            self._morphology_counters['trusted_bypasses'] += 1
            return True
        try:
            return self._validate_morphology_cached(word)
        except Exception:
            # Exceptions are not cached: a transient analyzer failure can recover.
            return False

    def morphology_stats(self):
        """Cumulative counters since construction, plus current deep-validation LRU stats.

        clear_caches resets LRU statistics, not these counters. Snapshot differences
        isolate a measured call. segmenter_seconds includes result/tag handling.
        """
        return {**self._morphology_counters,
                'cache': self._validate_morphology_cached.cache_info()._asdict()}

    def clear_caches(self):
        """Call after changing scoring/search settings or replacing the analyzer."""
        self._validate_morphology_cached.cache_clear()
        self._generate_candidates_cached.cache_clear()

    def _generate_candidates_tuple(self, ocr_token: str, max_cost: float) -> tuple:
        return tuple(self._generate_candidates_uncached(ocr_token, max_cost))

    def generate_candidates(self, ocr_token: str, max_cost: float = None) -> list:
        """Return candidates with optical cost <= max_cost (defaults to max_optical_cost).

        This is an acceptance ceiling, not exhaustive retrieval: adaptive trie search,
        SymSpell edit distance, frequent-word shortcuts, and max_candidates limit recall.
        """
        budget = self.max_optical_cost if max_cost is None else float(max_cost)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("max_cost must be finite and nonnegative")
        return list(self._generate_candidates_cached(ocr_token, budget))

    def optical_distance(self, true_word: str, observed_word: str) -> float:
        """Weighted Levenshtein in channel direction P(observed | true), not SymSpell distance."""
        ins_costs = [self._get_ins_cost(ch) for ch in observed_word]
        row = [0.0]
        for cost in ins_costs:
            row.append(row[-1] + cost)
        for true_ch in true_word:
            deletion = self._get_del_cost(true_ch)
            subs = self.costs.get("sub", {}).get(true_ch, {})
            cur = [row[0] + deletion]
            for j, obs_ch in enumerate(observed_word, 1):
                sub = 0.0 if true_ch == obs_ch else float(subs.get(obs_ch, self.default_sub))
                cur.append(min(cur[-1] + ins_costs[j-1], row[j] + deletion, row[j-1] + sub))
            row = cur
        return row[-1]

    def _trie_radius(self, word: str, max_cost: float) -> float:
        if self.trie_search == "full":
            return max_cost
        if word in self.vocab or self.validate_morphology(word):
            radius = 2.4
        elif len(word) <= 3:
            radius = 2.6
        elif len(word) <= 6:
            radius = 3.0
        else:
            radius = 2.2
        return min(max_cost, radius)

    def _generate_candidates_uncached(self, ocr_token: str, max_cost: float):
        if not ocr_token or ocr_token in self.protected_words:
            return [(ocr_token, 0.0)]
        results = {ocr_token: 0.0}

        def add(word):
            cost = self.optical_distance(word, ocr_token)
            if cost <= max_cost:
                results[word] = min(results.get(word, float("inf")), cost)

        # Cheap known optical substitutions, including real-word errors on the fast path.
        for i, observed_ch in enumerate(ocr_token):
            for true_ch, cost in self.reverse_confusions.get(observed_ch, ()):
                if cost <= max_cost:
                    candidate = ocr_token[:i] + true_ch + ocr_token[i+1:]
                    if candidate in self.vocab:
                        add(candidate)

        if self.symspell is not None:
            # ALL is essential: TOP/CLOSEST can suppress alternatives for known words.
            for suggestion in self.symspell.lookup(
                ocr_token, self._symspell_verbosity,
                max_edit_distance=self.symspell_max_edit_distance,
            ):
                add(suggestion.term)
        elif ocr_token not in self.common_words:
            # Keep fallback walks bounded; cheap confusion neighbors use the full ceiling.
            radius = self._trie_radius(ocr_token, max_cost)
            n = len(ocr_token)
            ins_costs = [self._get_ins_cost(ch) for ch in ocr_token]
            initial = [0.0]
            for cost in ins_costs:
                initial.append(initial[-1] + cost)

            def search(node, prefix, prev):
                # Retain the old explicit length guard: zero-cost deletions exist.
                if len(prefix) >= n + 1:
                    return
                for ch, child in node.items():
                    if ch == "$":
                        continue
                    deletion = self._get_del_cost(ch)
                    subs = self.costs.get("sub", {}).get(ch, {})
                    cur = [prev[0] + deletion]
                    for j, obs_ch in enumerate(ocr_token, 1):
                        sub = 0.0 if ch == obs_ch else float(subs.get(obs_ch, self.default_sub))
                        cur.append(min(cur[-1] + ins_costs[j-1], prev[j] + deletion, prev[j-1] + sub))
                    if min(cur) <= radius:
                        word = prefix + ch
                        if "$" in child and cur[-1] <= radius:
                            results[word] = cur[-1]
                        search(child, word, cur)

            search(self.trie, "", initial)

        # Heuristics are additional sources, never exclusive early exits.
        if len(ocr_token) >= 6 and ocr_token.startswith("አንደ"):
            repaired = "እንደ" + ocr_token[3:]
            if repaired in self.vocab or self.validate_morphology(repaired):
                add(repaired)
        if len(ocr_token) >= 4 and ocr_token not in self.vocab and not self.validate_morphology(ocr_token):
            stem = ocr_token[:-1]
            if stem in self.vocab or self.validate_morphology(stem):
                add(stem)

        # Always reserve one slot for the original, even with zero-cost alternatives.
        alternatives = sorted(
            ((word, cost) for word, cost in results.items() if word != ocr_token),
            key=lambda item: (item[1], -self.vocab.get(item[0], 0), len(item[0]), item[0]),
        )
        return [(ocr_token, 0.0)] + alternatives[:self.max_candidates - 1]

    def correct_sentence(self, sentence: str) -> str:
        """
        Decodes the optimal sequence using Viterbi Beam Search with a
        Conservative Replacement Margin (ANCHOR_MARGIN) to prevent over-correction.
        """
        normalized_text = EthiopicPunctuationManager.normalize_sentence_text(sentence)
        if not normalized_text:
            return ""

        raw_tokens = normalized_text.split()
        if not raw_tokens:
            return ""

        surface_tuples = [
            EthiopicPunctuationManager.decompose_token(tok, strip_diacritics=False)
            for tok in raw_tokens
        ]
        token_tuples = [
            (lead, EthiopicPunctuationManager.DIACRITICS_REGEX.sub('', word), trail)
            for lead, word, trail in surface_tuples
        ]

        # Beam state: (score, LM state, backpointer); only surviving paths are reconstructed.
        start_state = self._state_factory()
        self.lm.BeginSentenceWrite(start_state)
        beam = [(0.0, start_state, None)]

        # Anchor margin: score buffer that a candidate must overcome to displace an observed word
        ANCHOR_MARGIN = self.anchor_margin

        for token_index, (leading, orig_word, trailing) in enumerate(token_tuples):
            if not orig_word:
                # Standalone punctuation token (e.g. '“' or '።')
                surface = leading + surface_tuples[token_index][1] + trailing
                beam = [(score, state, (path, (surface, "", "")))
                        for score, state, path in beam]
                continue

            candidates = self.generate_candidates(orig_word)
            new_beam = []

            has_alternatives = len(candidates) > 1
            is_orig_known = has_alternatives and (
                orig_word in self.vocab or self.validate_morphology(orig_word)
            )

            for cand_word, opt_cost in candidates:
                opt_score = -self.lambda_optical * opt_cost

                # Lexical/morphological legitimacy reward (frequency-gated bypass).
                morph_score = self.lambda_morph if (
                    has_alternatives and self.lambda_morph and self.validate_morphology(cand_word)
                ) else 0.0

                # Anchor bonus: protect verified words from being flipped by LM frequency bias
                anchor_bonus = ANCHOR_MARGIN if (cand_word == orig_word and is_orig_known) else 0.0

                surface_word = surface_tuples[token_index][1] if cand_word == orig_word else cand_word
                token = (leading, surface_word, trailing)
                for acc_score, prev_state, path in beam:
                    next_state = self._state_factory()
                    lm_score = self.lm.BaseScore(prev_state, cand_word, next_state)

                    total_score = (
                        acc_score
                        + (self.lambda_lm * lm_score)
                        + opt_score
                        + morph_score
                        + anchor_bonus
                    )
                    new_beam.append((total_score, next_state, (path, token)))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_width]

        if self.score_eos:
            def final_score(hypothesis):
                score, state, _ = hypothesis
                return score + self.lambda_lm * self.lm.BaseScore(state, "</s>", self._state_factory())
            best_score, best_state, best_path = max(beam, key=final_score)
        else:
            best_score, best_state, best_path = beam[0]

        tokens = []
        while best_path is not None:
            best_path, token = best_path
            tokens.append(token)
        reconstructed_tokens = []
        for lead, word, trail in reversed(tokens):
            if not word:
                reconstructed_tokens.append(lead)
            else:
                reconstructed_tokens.append(f"{lead}{word}{trail}")

        return " ".join(reconstructed_tokens)

    def correct_text_block(self, full_text: str) -> str:
        """Processes multi-line paragraphs or pages line by line."""
        lines = full_text.splitlines()
        corrected_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                corrected_lines.append("")
                continue
            corrected_lines.append(self.correct_sentence(line_str))
        return "\n".join(corrected_lines)