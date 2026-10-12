# adapter.py — Lets the main corrector use this detection module.
#
# Instead of loading vocab/SymSpell twice, the adapter reuses the
# corrector's existing resources.
#
# Usage in the corrector:
#     from OCR_error_detection.adapter import CorrectorDetectionAdapter
#     self.suspicion = CorrectorDetectionAdapter(self)
#     result = self.suspicion.verdict(word, context)
from __future__ import annotations

import math
from typing import Optional, Sequence

from OCR_error_detection.signals import TokenSignals, Verdict
from OCR_error_detection.signals import (
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals, compute_char_lm_score,
    compute_lm_fit, compute_morphology,
)
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS


class CorrectorDetectionAdapter:
    """Adapter that lets DetectionEngine logic run on the corrector's resources.

    This is the bridge between the standalone detection module and the main
    corrector.  It reuses the corrector's vocab, SymSpell, confusion map,
    normalizer, LM, and segmenter — no duplication.

    Parameters
    ----------
    corrector : AmharicOCRCorrector
        The main corrector instance.
    weights : dict, optional
        Logistic scorer weights.  Default: same as DetectionEngine.
    pass_below : float
        Score threshold for PASS verdict.
    flag_above : float
        Score threshold for FLAG verdict.
    use_morphology : bool
        Whether to use the segmenter for the morphology signal.
    char_lm : object, optional
        Character-level LM.
    max_one_flip : int
        Cap on one-flip test results.
    """

    def __init__(
        self,
        corrector,
        weights=None,
        pass_below: float = 0.30,
        flag_above: float = 0.70,
        use_morphology: bool = True,
        char_lm=None,
        max_one_flip: int = 8,
    ):
        self.c = corrector
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.use_morphology = use_morphology
        self.char_lm = char_lm
        self.max_one_flip = max_one_flip
        self._morph_calls = 0
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        sig = TokenSignals(token=tok)
        compute_surface_signals(tok, sig)
        compute_lexical_signals(
            tok, sig, self.c.vocab, self.c._norm,
            getattr(self.c, 'symspell', None),
            getattr(self.c, '_VERBOSITY_TOP', 2),
        )
        compute_one_flip_signals(
            tok, sig, self.c.vocab, self.c._norm,
            getattr(self.c, 'reverse_sub_map', {}),
            self.max_one_flip,
        )
        return sig

    def _enrich_expensive(self, sig: TokenSignals, tok: str,
                          context: Sequence[str]) -> None:
        compute_lm_fit(tok, context, sig, lm=getattr(self.c, 'lm', None))
        if self.use_morphology:
            seg = getattr(self.c, 'segmenter', None)
            if seg is not None:
                compute_morphology(tok, sig, is_known_word_fn=self.c.is_known_word)

    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        return combine_signals(sig, self.w)

    def verdict(self, tok: str, context: Sequence[str] = (),
                ocr_conf: Optional[float] = None) -> Verdict:
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))

        s = self.score(sig)

        # Hard PASS
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.75)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig,
                           "clean lexicon word, no optical neighbour")

        # Hard FLAG
        if (not sig.in_lexicon and sig.one_flip_words
                and s >= self.flag_above):
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig,
                           f"OOV, one flip reaches {sig.one_flip_words[0]!r}")

        # Gray: spend expensive signals
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