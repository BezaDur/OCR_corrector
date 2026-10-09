#!/usr/bin/env python3
"""
Production CPU-based Amharic OCR Spelling Corrector & Post-Processor.
Architecture:
  1. EthiopicPunctuationManager:
     - Isolates leading and trailing punctuation from lexical tokens.
     - Preserves internal abbreviation markers ('/' and '.').
  2. AmharicNormalizer Integration:
     - Normalizes tokens for dictionary lookup while strictly preserving surface spelling.
  3. SymSpell + Reverse Optical Mapping:
     - High-speed candidate generation (O(1)) driven by confusion_matrix.json.
  4. AmharicMorphologicalSegmenter:
     - Linguistic prior validating agglutinated Out-of-Vocabulary forms.
  5. KenLM 5-Gram Viterbi Beam Decoder:
     - Contextual sequence scoring with a protective anchor barrier.
"""

import os
import sys
import re
import json
import collections
import kenlm
from functools import lru_cache
from symspellpy import SymSpell, Verbosity

# ----------------- Dynamic Import of NLP & Morphological Suite -----------------
MORPH_PATHS = [
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis",
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_segmenter"
]
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
        beam_width: int = 4,
        max_optical_cost: float = 3.6,
        lambda_optical: float = 1.2,
        lambda_lm: float = 1.8,
        lambda_morph: float = 1.5,
        anchor_margin: float = 3.5
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
        self.anchor_margin = anchor_margin

        self.default_sub = float(self.costs.get("default_sub", 4.5))
        self.default_del = float(self.costs.get("default_del", 2.4))
        self.default_ins = float(self.costs.get("default_ins", 2.4))

        # 1. Normalizer
        self.normalizer = AmharicNormalizer() if HAS_NORM else None

        # 2. Morphological Segmenter
        self.segmenter = AmharicMorphologicalSegmenter() if HAS_MORPH else None
        if self.segmenter:
            print("Morphological Segmenter ready.", flush=True)

        # 3. Vocabulary & SymSpell
        print(f"Loading 5M Lexicon into SymSpell & Hash Set: {dict_vocab_path}...", flush=True)
        self.vocab = {}
        self.symspell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)

        with open(dict_vocab_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0]
                freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
                norm_w = self._norm(w)
                self.vocab[norm_w] = max(self.vocab.get(norm_w, 0), freq)
                self.symspell.create_dictionary_entry(norm_w, freq)
                if idx % 1000000 == 0:
                    print(f"  ... loaded {idx:,} dictionary words", flush=True)

        # Precompute reverse optical substitutions
        self.reverse_sub_map = collections.defaultdict(list)
        for true_ch, obs_map in self.costs.get("sub", {}).items():
            if isinstance(obs_map, dict):
                for obs_ch, cost in obs_map.items():
                    c_flt = float(cost)
                    if c_flt <= 2.8 and true_ch != obs_ch:
                        self.reverse_sub_map[obs_ch].append((true_ch, c_flt))

        print(f"Corrector initialized successfully ({len(self.vocab):,} active words).\n", flush=True)

    def _norm(self, word: str) -> str:
        if not word or not self.normalizer:
            return word
        return self.normalizer.normalize_word_strict(word)

    @lru_cache(maxsize=131072)
    def is_known_word(self, word: str) -> bool:
        if not word:
            return False
        norm_w = self._norm(word)
        if norm_w in self.vocab:
            return True
        if self.segmenter:
            try:
                res = self.segmenter.segment_word(word)
                pos = str(res.get("pos", ""))
                return bool(pos and pos not in ("UN", "UNKNOWN", ""))
            except Exception:
                return False
        return False

    def compute_optical_cost(self, ocr_str: str, cand_str: str) -> float:
        m, n = len(ocr_str), len(cand_str)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1): dp[i][0] = dp[i - 1][0] + self.default_del
        for j in range(1, n + 1): dp[0][j] = dp[0][j - 1] + self.default_ins

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c_obs = ocr_str[i - 1]
                c_cand = cand_str[j - 1]
                sub_cost = 0.0 if c_obs == c_cand else self.costs.get("sub", {}).get(c_cand, {}).get(c_obs, self.default_sub)
                ins_cost = min(float(self.costs.get("ins", {}).get(c_obs, self.default_ins)), 2.4)
                del_cost = min(float(self.costs.get("del", {}).get(c_cand, self.default_del)), 2.4)

                dp[i][j] = min(dp[i - 1][j] + del_cost, dp[i][j - 1] + ins_cost, dp[i - 1][j - 1] + sub_cost)

        return dp[m][n]

    @lru_cache(maxsize=32768)
    def generate_candidates(self, ocr_token: str):
        if not ocr_token:
            return (("", 0.0),)

        is_verified = self.is_known_word(ocr_token)
        results = {ocr_token: 0.0}

        norm_token = self._norm(ocr_token)

        # 1. Terminal Ghost Loop check (e.g. ሶስትዮሽቨ -> ሶስትዮሽ)
        if len(ocr_token) >= 4 and not is_verified:
            stem_stripped = ocr_token[:-1]
            if self.is_known_word(stem_stripped):
                results[stem_stripped] = min(float(self.costs.get("ins", {}).get(ocr_token[-1], self.default_ins)), 2.4)

        # 2. Fast SymSpell 1-Edit lookup
        suggestions = self.symspell.lookup(norm_token, Verbosity.ALL, max_edit_distance=1)
        for s in suggestions:
            results[s.term] = self.compute_optical_cost(ocr_token, s.term)

        # 3. Direct Optical Reversal (Top Confusions e.g. አ -> እ, ፕ -> ፕሮ, ከ -> ክ)
        for i, obs_ch in enumerate(ocr_token):
            for true_ch, cost in self.reverse_sub_map.get(obs_ch, []):
                variant = ocr_token[:i] + true_ch + ocr_token[i+1:]
                if self.is_known_word(variant):
                    results[variant] = min(results.get(variant, float("inf")), cost)

        # Filter by max optical cost
        valid_candidates = []
        for cand, opt_cost in results.items():
            if opt_cost <= self.max_optical_cost:
                valid_candidates.append((cand, opt_cost))

        valid_candidates.sort(key=lambda x: x[1])
        return tuple(valid_candidates[:12])

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
            beam = new_beam[:self.beam_width]

        best_score, best_state, best_path = beam[0]

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