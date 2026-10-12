# detector.py — The main detection engine.
#
# Loads its own vocab, SymSpell index, and confusion matrix.
# Does NOT need the corrector, KenLM, or the beam decoder.
#
# Usage:
#     detector = DetectionEngine(
#         dict_vocab_path="lexicon.txt",
#         confusion_matrix_path="confusion_matrix.json",
#     )
#     result = detector.verdict("በሸታ")
#     print(result.label)  # "pass" | "gray" | "flag"
# --------------------------------------------------------------------------- #
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence

# Lazy imports for heavy deps
SymSpell = Verbosity = None


def _load_symspell():
    global SymSpell, Verbosity
    if SymSpell is None:
        from symspellpy import SymSpell as _SymSpell, Verbosity as _Verbosity
        SymSpell, Verbosity = _SymSpell, _Verbosity
    return SymSpell, Verbosity


from .normalizer import AmharicNormalizer
from .confusion import ConfusionMatrix
from .signals import (
    TokenSignals, Verdict,
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals, compute_char_lm_score,
    compute_lm_fit, compute_morphology,
    MEAN_TOKEN_LEN,
)
from .scoring import combine_signals, DEFAULT_WEIGHTS


logger = logging.getLogger("ocr_error_detection")


class DetectionEngine:
    """Standalone OCR error detector.

    Loads its own lexicon, builds its own SymSpell index, and reads the
    confusion matrix.  No dependency on the corrector or KenLM.

    Parameters
    ----------
    dict_vocab_path : str
        Path to the frequency dictionary.  Each line: "word [freq]"
    confusion_matrix_path : str
        Path to confusion_matrix.json.
    weights : dict, optional
        Logistic scorer weights.  Default: sensible starting point.
    pass_below : float
        Verdict threshold: score < pass_below → PASS.
    flag_above : float
        Verdict threshold: score ≥ flag_above → FLAG.
    char_lm : object, optional
        Character-level language model with .score(word, bos, eos).
    max_one_flip : int
        Maximum number of one-flip hits to record.
    max_lexicon_types : int, optional
        Cap the lexicon at the N most frequent types (memory bound).
    dict_min_freq : int, optional
        Drop types with frequency < N.
    symspell_max_edit_distance : int
        SymSpell edit distance (1 recommended; 2+ will OOM on large lexicons).
    segmenter : object, optional
        Morphological analyzer with .segment_word(word) → dict.
        Used only for the morphology signal, which is expensive and optional.
    morphemes_json_path : str, optional
        Path to morphemes.json for the peel guard (used by the corrector,
        optional for detection).
    verbose : bool
        Print initialization progress.
    """

    def __init__(
        self,
        dict_vocab_path: str,
        confusion_matrix_path: str,
        weights: Optional[Dict[str, float]] = None,
        pass_below: float = 0.30,
        flag_above: float = 0.70,
        char_lm=None,
        max_one_flip: int = 8,
        max_lexicon_types: int = None,
        dict_min_freq: int = None,
        symspell_max_edit_distance: int = 1,
        segmenter=None,
        morphemes_json_path: str = None,
        verbose: bool = True,
    ):
        self.normalizer = AmharicNormalizer()
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.char_lm = char_lm
        self.max_one_flip = max_one_flip
        self.verbose = verbose
        self._morph_calls = 0
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

        # -- Morphological segmenter (optional) --
        self.segmenter = segmenter

        # -- Load confusion matrix --
        if self.verbose:
            print(f"Loading optical confusion matrix: {confusion_matrix_path}", flush=True)
        self._confusion = ConfusionMatrix(confusion_matrix_path)
        self.reverse_sub_map = self._confusion.reverse_sub_map

        # -- Load vocabulary --
        if self.verbose:
            print(f"Loading lexicon: {dict_vocab_path}", flush=True)
        self.vocab = {}          # norm_word → frequency
        self.surface_form = {}   # norm_word → best surface form
        self._load_vocabulary(dict_vocab_path, max_lexicon_types, dict_min_freq)

        if self.verbose:
            print(f"  {len(self.vocab):,} active types", flush=True)

        # -- Build SymSpell index --
        _SS, _V = _load_symspell()
        self._VERBOSITY_TOP = _V.CLOSEST if hasattr(_V, 'CLOSEST') else 2
        self.symspell = _SS(
            max_dictionary_edit_distance=symspell_max_edit_distance,
            prefix_length=7,
        )
        if self.verbose:
            print(f"  Building SymSpell index for {len(self.vocab):,} types...", flush=True)
        for norm_w, freq in self.vocab.items():
            self.symspell.create_dictionary_entry(norm_w, freq)
        if self.verbose:
            print("  SymSpell index ready.", flush=True)

        # -- Morphology-related data (optional) --
        self.protected_suffixes = set()
        self.protected_prefixes = set()
        if morphemes_json_path:
            self._load_morphemes(morphemes_json_path)

    # ------------------------------------------------------------------ #
    # Vocabulary loading
    # ------------------------------------------------------------------ #

    def _load_vocabulary(self, path, max_types, min_freq):
        """Read the frequency dictionary, optionally capping at top-N types."""
        import heapq
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0]
                freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
                norm_w = self._norm(w)
                prev = self.vocab.get(norm_w, 0)
                if freq >= prev:
                    self.surface_form[norm_w] = w
                self.vocab[norm_w] = max(prev, freq)

        if min_freq and min_freq > 1:
            keep = [w for w, f in self.vocab.items() if f >= min_freq]
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w]
                                 for w in keep if w in self.surface_form}

        if max_types and len(self.vocab) > max_types:
            keep = heapq.nlargest(max_types, self.vocab,
                                  key=self.vocab.__getitem__)
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w]
                                 for w in keep if w in self.surface_form}

    # ------------------------------------------------------------------ #
    # Morpheme loading (optional)
    # ------------------------------------------------------------------ #

    def _load_morphemes(self, path):
        import json, os
        if not os.path.exists(path):
            if self.verbose:
                print(f"  [WARN] morphemes.json not found at {path}", flush=True)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if not isinstance(v, (list, set, tuple)):
                continue
            if "suffix" in str(k):
                self.protected_suffixes.update(v)
            elif "prefix" in str(k):
                self.protected_prefixes.update(v)

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #

    def _norm(self, word: str) -> str:
        if not word:
            return word
        return self.normalizer.normalize_word_strict(word)

    # ------------------------------------------------------------------ #
    # Morphology helper (optional, expensive)
    # ------------------------------------------------------------------ #

    def _is_known_word(self, word: str) -> bool:
        """Check if a word is known via vocab or segmenter."""
        norm_w = self._norm(word)
        if norm_w in self.vocab:
            return True
        if self.segmenter is not None:
            try:
                self._morph_calls += 1
                res = self.segmenter.segment_word(self._norm(word))
                pos = str(res.get("pos", "")) if isinstance(res, dict) else ""
                if pos:
                    tail = pos.rsplit(".", 1)[-1].upper()
                    if tail not in ("", "UN", "UNKNOWN", "UNK", "NONE", "NULL"):
                        return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------ #
    # Signal collection
    # ------------------------------------------------------------------ #

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        """Collect cheap signals: surface, lexical, one-flip."""
        sig = TokenSignals(token=tok)
        compute_surface_signals(tok, sig)
        compute_lexical_signals(tok, sig, self.vocab, self._norm,
                                self.symspell, getattr(self, '_VERBOSITY_TOP', 2))
        compute_one_flip_signals(tok, sig, self.vocab, self._norm,
                                 self.reverse_sub_map, self.max_one_flip)
        return sig

    def _enrich_expensive(self, sig: TokenSignals, tok: str,
                          context: Sequence[str]) -> None:
        """Add the expensive signals.  Only call in the gray zone."""
        compute_lm_fit(tok, context, sig, lm=getattr(self, '_lm', None))
        compute_morphology(tok, sig,
                           is_known_word_fn=self._is_known_word
                           if self.segmenter else None)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        """Collect cheap signals for a token.  No LM, no morphology."""
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        """Score a TokenSignals into (0, 1)."""
        return combine_signals(sig, self.w)

    def verdict(self, tok: str, context: Sequence[str] = (),
                ocr_conf: Optional[float] = None) -> Verdict:
        """Full detection verdict for a token.

        Returns a Verdict with label "pass", "gray", or "flag".
        """
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))

        s = self.score(sig)

        # Hard PASS: a lexicon word with no optical neighbour and nothing odd.
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.75)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig,
                           "clean lexicon word, no optical neighbour")

        # Hard FLAG: OOV + single optical flip reaches a real word.
        if (not sig.in_lexicon and sig.one_flip_words
                and s >= self.flag_above):
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig,
                           f"OOV, one flip reaches {sig.one_flip_words[0]!r}")

        # Gray: spend the expensive signals to decide.
        self._enrich_expensive(sig, tok, context)
        s = self.score(sig)
        if s >= self.flag_above:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, "gray resolved to flag")
        if s >= self.pass_below:
            self._stage1["gray"] += 1
            return Verdict(s, "gray", sig,
                           "uncertain; correct only with a margin")
        self._stage1["pass"] += 1
        return Verdict(s, "pass", sig, "gray resolved to pass")

    def verdict_batch(self, tokens: Sequence[str],
                      context_fn: callable = None) -> Sequence[Verdict]:
        """Batch detection over a sequence of tokens.

        context_fn: optional function(index) -> context words for token at index.
        Defaults to using up to 2 preceding tokens.
        """
        results = []
        for i, tok in enumerate(tokens):
            if context_fn is not None:
                ctx = context_fn(i)
            else:
                ctx = [t for t in tokens[max(0, i - 2):i] if t]
            results.append(self.verdict(tok, ctx))
        return results

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def detection_stats(self) -> dict:
        """Cumulative detection counters."""
        return {
            "stage1": dict(self._stage1),
            "morph_calls": self._morph_calls,
            "vocab_size": len(self.vocab),
            "confusion_reverse_map_size": len(self.reverse_sub_map),
        }

    def reset_counters(self) -> None:
        """Reset detection counters (not caches)."""
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}
        self._morph_calls = 0