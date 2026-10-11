#!/usr/bin/env python3
"""
Production CPU-based Amharic OCR Spelling Corrector & Post-Processor.
Architecture:
  1. EthiopicPunctuationManager (Decomposes tokens into leading/core/trailing,
     normalizes keyboard artifacts, and strips combining diacritics).
  2. Optical Confusion Matrix (Loaded from confusion_matrix.json).
  3. Trie-based Weighted Dynamic Candidate Search.
  4. Amharic Morphological Segmenter as Linguistic Prior.
  5. KenLM 5-Gram Viterbi Beam Search Contextual Decoder.
"""

import os
import sys
import json
import re
import math
import kenlm

# ----------------- Dynamic Import of Morphological Suite -----------------
MORPH_PROJECT_DIR = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis"
if MORPH_PROJECT_DIR not in sys.path:
    sys.path.insert(0, MORPH_PROJECT_DIR)

try:
    from segmenter import AmharicMorphologicalSegmenter
    import pos_tags
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
    # Combining Diacritics (U+135D, U+135E, U+135F, etc.)
    DIACRITICS_REGEX = re.compile(r'[\u135D-\u135F\u0300-\u036F]')

    # Punctuation characters to isolate from lexical tokens
    ALL_PUNCT = set(
        "።!?፧…፣፤፥፦፠፨"      # Ethiopic punctuation
        ":;,.!?"            # Latin equivalents
        "“”‘’«»„\'\""       # Quotations
        "()[]{}"            # Brackets
        "|/\\—–-"           # Connectors & separators
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

        # Core lexical word with combining diacritics stripped for dictionary matching
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
        max_optical_cost: float = 4.8,
        lambda_optical: float = 0.9,
        lambda_lm: float = 2.0,
        lambda_morph: float = 1.5
    ):
        print(f"Loading KenLM Language Model: {kenlm_path}...")
        self.lm = kenlm.Model(kenlm_path)

        print(f"Loading Optical Confusion Matrix: {confusion_matrix_path}...")
        with open(confusion_matrix_path, "r", encoding="utf-8") as f:
            self.costs = json.load(f)

        self.beam_width = beam_width
        self.max_optical_cost = max_optical_cost
        self.lambda_optical = lambda_optical
        self.lambda_lm = lambda_lm
        self.lambda_morph = lambda_morph

        # Initialize Morphological Segmenter
        self.segmenter = None
        if HAS_MORPH:
            print("Initializing Amharic Morphological Segmenter...")
            self.segmenter = AmharicMorphologicalSegmenter()

        # Build Vocabulary & Trie with Frequency Metadata
        print("Building Core Trie & Vocabulary...")
        self.vocab = {}  # Map: word -> frequency
        self.trie = {}

        # 1. Seed Trie with Segmenter's built-in vocabulary frequencies
        if self.segmenter and hasattr(self.segmenter, "vocab_freq"):
            for w, f in self.segmenter.vocab_freq.items():
                self._add_to_trie(w, f)

        # 2. Seed Trie with external dictionary file (parsing frequencies)
        if dict_vocab_path and os.path.exists(dict_vocab_path):
            with open(dict_vocab_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    w = parts[0]
                    freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
                    self._add_to_trie(w, freq)

        print(f"Trie successfully built with {len(self.vocab):,} active words.")

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
        return self.costs["sub"].get(c_true, {}).get(c_obs, self.costs.get("default_sub", 4.5))

    def _get_del_cost(self, c_true: str) -> float:
        return self.costs["del"].get(c_true, self.costs.get("default_del", 5.0))

    def _get_ins_cost(self, c_obs: str) -> float:
        return self.costs["ins"].get(c_obs, self.costs.get("default_ins", 5.0))

    def generate_candidates(self, ocr_token: str, max_cost: float = None):
        """
        Explores Trie via weighted branch-and-bound dynamic programming.
        Uses two-tier frequency pruning: words with freq >= 50 and length >= 3
        hit the zero-cost fast path immediately, bypassing the 5M Trie search.
        """
        if max_cost is None:
            max_cost = self.max_optical_cost

        token_freq = self.vocab.get(ocr_token, 0)

        # FAST PATH: Common dictionary words (freq >= 50, len >= 3) are clean
        if token_freq >= 50 and len(ocr_token) >= 3:
            return [(ocr_token, 0.0)]

        results = {}
        if token_freq > 0:
            results[ocr_token] = 0.0

        def search_trie(node, prefix, prev_row):
            for ch, next_node in node.items():
                if ch == "$":
                    continue

                cur_row = [prev_row[0] + self._get_del_cost(ch)]
                for j in range(1, len(ocr_token) + 1):
                    obs_ch = ocr_token[j - 1]
                    ins = cur_row[j - 1] + self._get_ins_cost(obs_ch)
                    dlt = prev_row[j] + self._get_del_cost(ch)
                    sub = prev_row[j - 1] + self._get_sub_cost(ch, obs_ch)
                    cur_row.append(min(ins, dlt, sub))

                if min(cur_row) <= max_cost:
                    if "$" in next_node and cur_row[-1] <= max_cost:
                        matched_word = prefix + ch
                        results[matched_word] = min(results.get(matched_word, float("inf")), cur_row[-1])
                    search_trie(next_node, prefix + ch, cur_row)

        init_row = [0.0]
        for c in ocr_token:
            init_row.append(init_row[-1] + self._get_ins_cost(c))

        search_trie(self.trie, "", init_row)

        # Morphological Prior Check for the OCR token itself (OOV rescue)
        if self.segmenter and ocr_token not in results:
            analysis = self.segmenter.segment_word(ocr_token)
            if analysis.get("stem") and analysis.get("pos") not in ("UN", pos_tags.UNKNOWN if HAS_MORPH else "UN"):
                results[ocr_token] = 1.0

        # Fallback if no candidate passed the threshold: return original token
        if not results:
            results[ocr_token] = 4.0

        return list(results.items())

    def validate_morphology(self, word: str) -> bool:
        """
        Returns True if the word is grammatically valid according to the segmenter.
        """
        if not self.segmenter:
            return False
        try:
            res = self.segmenter.segment_word(word)
            pos = res.get("pos", "")
            return bool(pos and pos not in ("UN", pos_tags.UNKNOWN if HAS_MORPH else "UN"))
        except Exception:
            return False

    def correct_sentence(self, sentence: str) -> str:
        """
        Decodes the most probable sentence path using Viterbi Beam Search,
        strictly isolating trailing/leading punctuation from lexical tokens.
        """
        normalized_text = EthiopicPunctuationManager.normalize_sentence_text(sentence)
        if not normalized_text:
            return ""

        raw_tokens = normalized_text.split()
        if not raw_tokens:
            return ""

        # Decompose tokens into (leading_punct, clean_word, trailing_punct)
        token_tuples = [EthiopicPunctuationManager.decompose_token(tok) for tok in raw_tokens]

        # Beam state: (accumulated_score, kenlm_state, [(leading, cand_word, trailing)])
        start_state = kenlm.State()
        self.lm.BeginSentenceWrite(start_state)
        beam = [(0.0, start_state, [])]

        for leading, word, trailing in token_tuples:
            if not word:
                # Standalone punctuation token (e.g. '“' or '።')
                beam = [(score, state, path + [(leading + trailing, "", "")]) for score, state, path in beam]
                continue

            candidates = self.generate_candidates(word)
            new_beam = []

            for cand_word, opt_cost in candidates:
                opt_score = -self.lambda_optical * opt_cost
                morph_score = self.lambda_morph if self.validate_morphology(cand_word) else 0.0

                for acc_score, prev_state, path in beam:
                    next_state = kenlm.State()
                    lm_score = self.lm.BaseScore(prev_state, cand_word, next_state)

                    total_score = (
                        acc_score
                        + (self.lambda_lm * lm_score)
                        + opt_score
                        + morph_score
                    )
                    new_beam.append((total_score, next_state, path + [(leading, cand_word, trailing)]))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:self.beam_width]

        # Best hypothesis path
        best_score, best_state, best_path = beam[0]

        # Re-assemble tokens with their original punctuation
        reconstructed_tokens = []
        for lead, word, trail in best_path:
            if not word:
                reconstructed_tokens.append(lead)
            else:
                reconstructed_tokens.append(f"{lead}{word}{trail}")

        return " ".join(reconstructed_tokens)

    def correct_text_block(self, full_text: str) -> str:
        """
        Processes multi-line paragraphs or pages line by line.
        """
        lines = full_text.splitlines()
        corrected_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                corrected_lines.append("")
                continue
            corrected_lines.append(self.correct_sentence(line_str))
        return "\n".join(corrected_lines)