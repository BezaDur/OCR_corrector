# scoring.py — Turns the checks into one number (0 to 1).
import math
from typing import Dict, Optional

from .signals import TokenSignals


# Default weights -- replaced by training once labels exist.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": 0.10,
    # not being in the lexicon is the strongest single signal
    "in_lexicon": -2.20,
    # ...but a common word is safer than a rare one
    "log_freq": -0.14,
    # many close neighbours => plausible typo, but also plausible word
    "n_neighbors": 0.03,
    # the one-flip test: a reachable real word is strong evidence of error
    "one_flip_hits": 0.85,
    # phonotactically implausible => probably corrupted
    "char_lm": -0.55,
    # common word that does not fit its context => real-word error
    "lm_fit": -0.45,
    # morphologically well-formed => probably a legitimate rare form
    "morph_ok": -1.10,
    # Tesseract's own doubt, scaled to 0-1
    "ocr_conf": -1.60,
    # surface oddities
    "len_ratio": 0.35,
    "rare_glyph_ratio": 1.30,
    "mixed_script": 0.90,
    "has_digit": 0.30,
}


def combine_signals(sig: TokenSignals, weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted logistic combination of signal features into (0, 1).

    This is the same computation as SuspicionScorer._combine, extracted so it
    can be tested and replaced independently.
    """
    w = weights or DEFAULT_WEIGHTS
    v = sig.as_vector()
    total = w.get("bias", 0.0)
    for k, coef in w.items():
        if k == "bias" or k not in v:
            continue
        total += coef * v[k]
    # logistic squash to (0, 1)
    if total >= 0:
        return 1.0 / (1.0 + math.exp(-total))
    e = math.exp(total)
    return e / (1.0 + e)