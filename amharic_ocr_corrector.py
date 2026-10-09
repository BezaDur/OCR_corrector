#!/usr/bin/env python3
"""
Production CPU-based Amharic OCR Spelling Corrector & Post-Processor.

Architecture:
  1. EthiopicPunctuationManager:
     - Isolates leading and trailing punctuation from lexical tokens.
     - Preserves abbreviation symbols ('/' and '.') inside words.
     - Strips non-spacing combining diacritics for dictionary lookups.
  2. Core Trie & Vocabulary:
     - In-memory nested prefix tree for bounded Levenshtein optical distance search.
     - Seeded with main_i7.txt and segmenter's lexical frequencies.
  3. Morphological Prior (AmharicMorphologicalSegmenter):
     - Validates Out-of-Vocabulary agglutinated forms (prefixes + stems + suffixes).
     - Protects legitimate morphological forms from destructive over-correction.
  4. Conservative Viterbi Decoder (KenLM 5-gram):
     - Beam search decoder with an explicit Anchor Margin (ANCHOR_MARGIN).
     - Prevents high-frequency language model words from displacing verified OCR words.
"""

import os
import sys
import re
import json
import kenlm
from functools import lru_cache

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
        text = re.sub(r'([:፡]{2,}|\.{2,}|፣{2,})', '።', text)

        # 2. Terminal/trailing single colons or hulet neteb before newlines/quotes -> ።
        text = re.sub(r'[:፡]\s*$', '።', text, flags=re.MULTILINE)
        text = re.sub(r'[:፡]\s*([”"»‹\n])', r'።\1', text)

        # 3. Inter-word isolated single colon/hulet neteb -> clean space
        text = re.sub(r'[:፡]', ' ', text)

        # 4. Collapse spaces
        text = re.sub(r'[ ]+', ' ', text)
        return text.strip()

    @classmethod
    def decompose_token(cls, raw_token: str):
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
        max_candidates: int = 25
    ):
        print(f"Loading KenLM Language Model: {kenlm_path}...", flush=True)
        self.lm = kenlm.Model(kenlm_path)

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

        # Initialize Morphological Segmenter
        self.segmenter = None
        if HAS_MORPH:
            print("Initializing Amharic Morphological Segmenter...", flush=True)
            self.segmenter = AmharicMorphologicalSegmenter()
            print("Segmenter ready.", flush=True)

        # Build Core Prefix Trie and Vocabulary
        print("Building Core Trie & Vocabulary...", flush=True)
        self.vocab = {}
        self.trie = {}

        # 1. Seed Trie with Segmenter's built-in vocabulary frequencies
        if self.segmenter and hasattr(self.segmenter, "vocab_freq"):
            for w, f in self.segmenter.vocab_freq.items():
                self._add_to_trie(w, f)

        # 2. Seed Trie with main_i7.txt lexicon
        if dict_vocab_path and os.path.exists(dict_vocab_path):
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
        for abbr in COMMON_ABBRS:
            self._add_to_trie(abbr, freq=1000)

        print(f"Trie successfully built with {len(self.vocab):,} active words.", flush=True)

        # High-frequency ambiguity list (words that must evaluate real-word optical neighbors)
        self.ambiguous_frequent_words = {"አና", "እና", "በ", "ከ", "ይህ", "ይሀ"}

        # Build in-memory O(1) fast-bypass table:
        # 1. Any word of len >= 4 with freq >= 100
        # 2. Any word of len >= 2 with freq >= 1000 (excluding known ambiguous pairs like አና)
        print("Extracting high-frequency clean word set for O(1) bypass...", flush=True)
        self.common_words = {
            w for w, freq in self.vocab.items()
            if (freq >= 1000 or (freq >= 100 and len(w) >= 4))
            and w not in self.ambiguous_frequent_words
        }
        print(f"Fast-bypass set active with {len(self.common_words):,} verified words.", flush=True)

    def _add_to_trie(self, word: str, freq: int = 1):
        if not word or len(word) < 2:
            return
        self.vocab[word] = max(self.vocab.get(word, 0), freq)
        node = self.trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True

    def _get_sub_cost(self, c_true: str, c_obs: str) -> float:
        if c_true == c_obs:
            return 0.0
        return self.costs.get("sub", {}).get(c_true, {}).get(c_obs, self.default_sub)

    def _get_del_cost(self, c_true: str) -> float:
        raw_cost = self.costs.get("del", {}).get(c_true, self.default_del)
        return min(float(raw_cost), 2.4)

    def _get_ins_cost(self, c_obs: str) -> float:
        raw_cost = self.costs.get("ins", {}).get(c_obs, self.default_ins)
        return min(float(raw_cost), 2.4)

    def validate_morphology(self, word: str) -> bool:
        """
        Validates whether an Out-of-Vocabulary token decomposes cleanly
        into legitimate stems + affixes via the morphological segmenter.
        """
        if not self.segmenter or not word:
            return False
        try:
            res = self.segmenter.segment_word(word)
            pos = str(res.get("pos", ""))
            return bool(pos and pos not in ("UN", "UNKNOWN", ""))
        except Exception:
            return False

    @lru_cache(maxsize=32768)
    def _generate_candidates_cached(self, ocr_token: str) -> tuple:
        """Internal cached worker returning an immutable tuple of (word, cost)."""
        return tuple(self._generate_candidates_uncached(ocr_token))

    def generate_candidates(self, ocr_token: str, max_cost: float = None) -> list:
        # Wrap candidate generation with LRU process cache for microsecond repeat lookups
        return list(self._generate_candidates_cached(ocr_token))

    def _generate_candidates_uncached(self, ocr_token: str):
        """
        Length-Adaptive Candidate Generator:
          1. O(1) Fast-Path: Words in top-common set (len >= 4, freq >= 100) bypass Trie immediately.
          2. Length-Adaptive Search Radius:
             - Short tokens (len <= 3): search radius 2.6 (prevents real-word blindness: አና -> እና).
             - Medium tokens (4 <= len <= 6): search radius 3.0.
             - Long tokens (len >= 7): search radius 2.4 (prunes exponential 5M sub-branch explosion).
          3. Fast Prefix Anchor for agglutinated compounds: checks known subordinator slips (አንደ- -> እንደ-).
        """
        if not ocr_token:
            return [("", 0.0)]

        # --- TIER 1: O(1) FAST BYPASS FOR HIGH-FREQUENCY LONG WORDS ---
        if hasattr(self, 'common_words') and ocr_token in self.common_words:
            return [(ocr_token, 0.0)]

        token_freq = self.vocab.get(ocr_token, 0)
        is_valid_morph = self.validate_morphology(ocr_token)

        results = {}
        # Observed token is ALWAYS candidate #1 with zero optical cost
        results[ocr_token] = 0.0

        n = len(ocr_token)

        # --- FAST PREFIX ANCHOR FOR AGGLUTINATED LONG TOKENS ---
        # e.g., 'አንደሚያደርጉ' starts with 'አንደ' (slip of subordinator 'እንደ')
        if n >= 6 and ocr_token.startswith("አንደ"):
            repaired_cand = "እንደ" + ocr_token[3:]
            if repaired_cand in self.vocab or self.validate_morphology(repaired_cand):
                results[repaired_cand] = self._get_sub_cost("እ", "አ")
                # Fast exit: The exact agglutinated repair is found and authenticated.
                # Avoid walking 5 million Trie nodes for 1000 ms.
                return [(repaired_cand, self._get_sub_cost("እ", "አ")), (ocr_token, 0.0)]

        # --- TERMINAL GHOST-LOOP HANDLER ---
        # e.g., 'ሶስትዮሽቨ' -> 'ሶስትዮሽ'
        # --- TERMINAL GHOST-LOOP HANDLER ---
        # Only strip trailing character if the original token is NOT a clean dictionary word
        if n >= 4 and token_freq == 0 and not is_valid_morph:
            stem_stripped = ocr_token[:-1]
            if (stem_stripped in self.vocab) or self.validate_morphology(stem_stripped):
                results[stem_stripped] = self._get_ins_cost(ocr_token[-1])

        # --- ADAPTIVE DEPTH PRUNING BASED ON TOKEN LENGTH ---
        if is_valid_morph or token_freq > 0:
            effective_cost = 2.4
        elif n <= 3:
            effective_cost = 2.6  # Shallow depth: fast & explores real-word neighbors (አና -> እና)
        elif 4 <= n <= 6:
            effective_cost = 3.0
        else:
            effective_cost = 2.2  # Long tokens: bounds exponential branch explosion

        # Length bound: Candidate words cannot exceed input token length by more than 1 character
        MAX_CAND_LEN = n + 1

        def search_trie(node, prefix, prev_row):
            # Prune search branches that exceed reasonable candidate word lengths
            if len(prefix) >= MAX_CAND_LEN:
                return

            for ch, next_node in node.items():
                if ch == "$":
                    continue

                cur_row = [prev_row[0] + self._get_del_cost(ch)]
                for j in range(1, n + 1):
                    obs_ch = ocr_token[j - 1]
                    ins = cur_row[j - 1] + self._get_ins_cost(obs_ch)
                    dlt = prev_row[j] + self._get_del_cost(ch)
                    sub = prev_row[j - 1] + self._get_sub_cost(ch, obs_ch)
                    cur_row.append(min(ins, dlt, sub))

                if min(cur_row) <= effective_cost:
                    if "$" in next_node and cur_row[-1] <= effective_cost:
                        matched_word = prefix + ch
                        results[matched_word] = min(results.get(matched_word, float("inf")), cur_row[-1])
                    search_trie(next_node, prefix + ch, cur_row)

        init_row = [0.0]
        for c in ocr_token:
            init_row.append(init_row[-1] + self._get_ins_cost(c))

        search_trie(self.trie, "", init_row)

        ranked = sorted(
            results.items(),
            key=lambda x: (x[1], -self.vocab.get(x[0], 0), len(x[0]))
        )
        return ranked[:self.max_candidates]

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

        token_tuples = [EthiopicPunctuationManager.decompose_token(tok) for tok in raw_tokens]

        # Beam state: (accumulated_score, kenlm_state, [(leading, cand_word, trailing)])
        start_state = kenlm.State()
        self.lm.BeginSentenceWrite(start_state)
        beam = [(0.0, start_state, [])]

        # Anchor margin: score buffer that a candidate must overcome to displace an observed word
        ANCHOR_MARGIN = 1.6

        for leading, orig_word, trailing in token_tuples:
            if not orig_word:
                # Standalone punctuation token (e.g. '“' or '።')
                beam = [(score, state, path + [(leading + trailing, "", "")]) for score, state, path in beam]
                continue

            candidates = self.generate_candidates(orig_word)
            new_beam = []

            is_orig_known = (orig_word in self.vocab) or self.validate_morphology(orig_word)

            for cand_word, opt_cost in candidates:
                opt_score = -self.lambda_optical * opt_cost

                # Morphological legitimacy reward
                morph_score = self.lambda_morph if self.validate_morphology(cand_word) else 0.0

                # Anchor bonus: protect verified words from being flipped by LM frequency bias
                anchor_bonus = ANCHOR_MARGIN if (cand_word == orig_word and is_orig_known) else 0.0

                for acc_score, prev_state, path in beam:
                    next_state = kenlm.State()
                    lm_score = self.lm.BaseScore(prev_state, cand_word, next_state)

                    total_score = (
                        acc_score
                        + (self.lambda_lm * lm_score)
                        + opt_score
                        + morph_score
                        + anchor_bonus
                    )
                    new_beam.append((total_score, next_state, path + [(leading, cand_word, trailing)]))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_width]

        best_score, best_state, best_path = beam[0]

        reconstructed_tokens = []
        for lead, word, trail in best_path:
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