# signals.py — Evidence we collect for each word.
#
# TokenSignals: all the checks we ran on one word.
# Verdict: the final answer (pass/gray/flag) plus the score.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from .utils import RARE_GLYPH_CLASSES, _confusion_alts, _is_ethiopic
except ImportError:
    try:
        from OCR_error_detection.utils import RARE_GLYPH_CLASSES, _confusion_alts, _is_ethiopic
    except ImportError:
        try:
            from utils import RARE_GLYPH_CLASSES, _confusion_alts, _is_ethiopic
        except ImportError:
            # Fallback minimal definitions
            RARE_GLYPH_CLASSES = (
                set("ሠሡሢሣሤሥሦሧ"),
                set("ጰጱጲጳጴጵጶጷ"),
                set("ፀፁፂፃፄፅፆፇ"),
                set("ዐዑዒዓዔዕዖ"),
                set("ሷሹሺሻሼሽሾሿ"),
            )
            def _is_ethiopic(ch: str) -> bool:
                o = ord(ch)
                return (0x1200 <= o < 0x1380) or (0x2D80 <= o < 0x2DE0)
            def _confusion_alts(ch: str) -> list:
                return []


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class TokenSignals:
    """Per-token evidence.  `None` means 'not computed' (too expensive)."""
    token: str
    in_lexicon: bool = False
    log_freq: float = 0.0
    n_neighbors: int = 0              # dict words within edit distance 1
    one_flip_words: List[str] = field(default_factory=list)
    char_lm: Optional[float] = None   # avg log10 prob under the char LM
    lm_fit: Optional[float] = None    # logP(w|ctx) - logP_unigram(w)
    morph_ok: Optional[bool] = None   # segmenter verdict (expensive)
    ocr_conf: Optional[float] = None  # Tesseract word confidence, normalized 0-1
    # (DetectionEngine.verdict() rescales the 0-100 conf.tsv value onto this
    # field before anything downstream reads it, so consumers must compare
    # against 0-1 thresholds. features.feature_dict() normalizes the raw value
    # the same way for the legacy ocr_conf feature column.)
    len_ratio: float = 1.0            # length / corpus mean (4.14)
    rare_glyph_ratio: float = 0.0
    mixed_script: bool = False
    has_digit: bool = False

    def as_vector(self) -> Dict[str, float]:
        """Numeric view for the scorer / later for logistic regression."""
        v = {
            "in_lexicon": 1.0 if self.in_lexicon else 0.0,
            "log_freq": self.log_freq,
            "n_neighbors": float(self.n_neighbors),
            "one_flip_hits": float(len(self.one_flip_words)),
            "len_ratio": self.len_ratio,
            "rare_glyph_ratio": self.rare_glyph_ratio,
            "mixed_script": 1.0 if self.mixed_script else 0.0,
            "has_digit": 1.0 if self.has_digit else 0.0,
        }
        if self.char_lm is not None:
            v["char_lm"] = self.char_lm
        if self.lm_fit is not None:
            v["lm_fit"] = self.lm_fit
        if self.morph_ok is not None:
            v["morph_ok"] = 1.0 if self.morph_ok else 0.0
        if self.ocr_conf is not None:
            v["ocr_conf"] = self.ocr_conf
        return v


@dataclass
class Verdict:
    suspicion: float
    label: str              # "pass" | "gray" | "flag"
    signals: TokenSignals
    reason: str = ""

    @property
    def should_correct(self) -> bool:
        return self.label in ("gray", "flag")


# --------------------------------------------------------------------------- #
# Signal computation functions
# --------------------------------------------------------------------------- #

MEAN_TOKEN_LEN = 4.14  # corpus mean; adjustable via calibration


def compute_surface_signals(tok: str, sig: TokenSignals) -> None:
    """Cheap: length ratio, mixed script, digit presence, rare glyph ratio."""
    n = len(tok)
    sig.len_ratio = n / MEAN_TOKEN_LEN
    eth = sum(1 for ch in tok if _is_ethiopic(ch))
    sig.mixed_script = 0 < eth < n
    sig.has_digit = any(ch.isdigit() for ch in tok)
    rare = sum(1 for ch in tok
               if any(ch in cls for cls in RARE_GLYPH_CLASSES))
    sig.rare_glyph_ratio = rare / max(n, 1)


def compute_lexical_signals(
    tok: str,
    sig: TokenSignals,
    vocab: dict,
    norm_fn,
    sym_spell=None,
    verbosity_top: int = 2,
) -> None:
    """Cheap: in-lexicon, log frequency, neighbourhood density."""
    norm = norm_fn(tok)
    freq = vocab.get(norm)
    if freq is not None:
        sig.in_lexicon = True
        import math
        sig.log_freq = math.log10(float(freq) + 1.0)
    # neighbourhood density
    if sym_spell is not None:
        try:
            sug = sym_spell.lookup(
                tok, verbosity_top, max_edit_distance=1, include_unknown=False)
            sig.n_neighbors = len(sug)
        except Exception:
            sig.n_neighbors = 0


def compute_one_flip_signals(
    tok: str,
    sig: TokenSignals,
    vocab: dict,
    norm_fn,
    reverse_sub_map: dict,
    max_one_flip: int = 8,
) -> None:
    """Ling's one-flip test, restricted to OPTICALLY confusable chars.

    Using the confusion matrix rather than every character is what makes
    this precise: ሀ->ህ is a plausible scan error, ሀ->ጸ is not.
    """
    hits: List[str] = []
    n = len(tok)
    tok_norm = norm_fn(tok)
    for i, ch in enumerate(tok):
        learned = list(reverse_sub_map.get(ch) or [])
        if isinstance(learned, dict):
            learned = list(learned.items())
        # Curated classes come FIRST: they are few and high-precision.
        curated = [(c, 0.0) for c in _confusion_alts(ch)]
        seen = {a for a, _ in learned}
        alts = curated + [(a, c) for a, c in learned if a not in
                          {x for x, _ in curated}]
        if not alts:
            continue
        for alt, _cost in alts[:12]:
            cand = tok[:i] + alt + tok[i + 1:]
            if cand == tok:
                continue
            cn = norm_fn(cand)
            # A flip that normalizes back to the SAME key is a no-op.
            if cn == tok_norm:
                continue
            if cn in vocab:
                hits.append(cand)
                break
        if len(hits) >= max_one_flip:
            break
    sig.one_flip_words = hits


# --------------------------------------------------------------------------- #
# Expensive signals (LM, morphology) -- only computed in the gray zone
# --------------------------------------------------------------------------- #

def compute_char_lm_score(tok: str, sig: TokenSignals, char_lm) -> None:
    """Average log10-probability per character under a character-level LM."""
    if char_lm is None or not tok:
        return
    try:
        s = char_lm.score(tok, bos=True, eos=True)
        sig.char_lm = s / max(len(tok), 1)
    except Exception:
        sig.char_lm = None


def compute_lm_fit(tok: str, context: Sequence[str],
                   sig: TokenSignals, lm=None) -> None:
    """How well the word fits its preceding context, above its unigram prior."""
    if lm is None:
        return
    try:
        ctx = " ".join(context[-2:])
        with_ctx = lm.score(f"{ctx} {tok}".strip(), bos=True, eos=False)
        without = lm.score(tok, bos=True, eos=False)
        base = lm.score(f"{ctx} <unk>".strip(), bos=True, eos=False)
        sig.lm_fit = with_ctx - max(without, base)
    except Exception:
        sig.lm_fit = None


def compute_morphology(tok: str, sig: TokenSignals,
                       is_known_word_fn=None) -> None:
    """Morphological validity (expensive: ~20ms/call without cache)."""
    if is_known_word_fn is None:
        return
    try:
        sig.morph_ok = bool(is_known_word_fn(tok))
    except Exception:
        sig.morph_ok = None