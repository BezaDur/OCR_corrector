# adapter.py — Unifies corrector with authoritative DetectionEngine per roadmap Stage 3
#
# The adapter now delegates to DetectionEngine (LightGBM authoritative)
# instead of duplicating logic. It implements the same DetectorInterface
# contract: verdict(tok, context, ocr_conf, doc_freq) -> Verdict
#
# Lifecycle for doc_freq:
#   - If full document/page provided → compute doc_freq via Counter
#   - If isolated token → doc_freq = NaN (not singleton default)
#
# Usage:
#   from OCR_error_detection.adapter import CorrectorDetectionAdapter
#   self.suspicion = CorrectorDetectionAdapter(self)
#   result = self.suspicion.verdict(word, context, ocr_conf, doc_freq)
from __future__ import annotations

import os
from typing import Optional, Sequence, Dict, List
import math

from OCR_error_detection.signals import TokenSignals, Verdict
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS

# Import DetectionEngine lazily to avoid circular
DetectionEngine = None

def _load_engine():
    global DetectionEngine
    if DetectionEngine is None:
        from OCR_error_detection.detector import DetectionEngine as _DE
        DetectionEngine = _DE
    return DetectionEngine

class DetectorInterface:
    """Contract for detectors: verdict + batch"""
    def verdict(self, tok: str, context: Sequence[str] = (), ocr_conf: Optional[float] = None, doc_freq: Optional[float] = None) -> Verdict:
        raise NotImplementedError
    def verdict_batch(self, tokens: Sequence[str], **kwargs) -> List[Verdict]:
        raise NotImplementedError

class CorrectorDetectionAdapter(DetectorInterface):
    """Adapter that delegates to authoritative DetectionEngine.

    Reuses corrector's vocab, SymSpell, confusion map, normalizer if possible,
    but delegates scoring to DetectionEngine's LightGBM model.

    If DetectionEngine cannot be loaded (missing LGBM), falls back to rule-based
    SuspicionScorer logic (deprecated path).

    Parameters
    ----------
    corrector : AmharicOCRCorrector
        Main corrector instance with vocab, symspell, reverse_sub_map, _norm, etc.
    weights, pass_below, flag_above, use_morphology, char_lm, max_one_flip
        Kept for backward compat, but LGBM thresholds from model take precedence.
    lgbm_model_path : optional path to model
    """

    def __init__(
        self,
        corrector,
        weights=None,
        pass_below: float = 0.20,
        flag_above: float = 0.65,
        use_morphology: bool = True,
        char_lm=None,
        max_one_flip: int = 8,
        lgbm_model_path: Optional[str] = None,
    ):
        self.c = corrector
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.use_morphology = use_morphology
        self.char_lm = char_lm
        self.max_one_flip = max_one_flip
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

        # Try to create authoritative DetectionEngine that reuses corrector's resources
        self.engine = None
        self.lgbm_model = None
        self.lgbm_features = []
        self.calibrator = None

        try:
            DE = _load_engine()
            # Create engine but inject corrector's vocab etc. if possible
            # We instantiate with minimal loading, then override vocab/symspell
            self.engine = DE(
                dict_vocab_path=None,  # will fallback to minimal, we override below
                confusion_matrix_path=None,
                char_lm_path=None,
                lgbm_model_path=lgbm_model_path,
                pass_below=pass_below,
                flag_above=flag_above,
                max_one_flip=max_one_flip,
                verbose=False,
            )
            # Override with corrector's resources to avoid double load
            if hasattr(corrector, 'vocab') and corrector.vocab:
                self.engine.vocab = corrector.vocab
                self.engine.surface_form = getattr(corrector, 'surface_form', {})
            if hasattr(corrector, 'symspell') and corrector.symspell:
                self.engine.symspell = corrector.symspell
            if hasattr(corrector, 'reverse_sub_map') and corrector.reverse_sub_map:
                self.engine.reverse_sub_map = corrector.reverse_sub_map
            if hasattr(corrector, '_norm') and corrector._norm:
                self.engine._norm = corrector._norm
            # Reuse model and thresholds
            self.lgbm_model = self.engine.lgbm_model
            self.lgbm_features = self.engine.lgbm_features
            self.calibrator = self.engine.calibrator
            self.pass_below = self.engine.pass_below
            self.flag_above = self.engine.flag_above
        except Exception as e:
            # Fallback to rule-based
            self.engine = None
            self.lgbm_model = None

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        # If engine available, use its cheap signals (which uses canonical boundary)
        if self.engine is not None:
            return self.engine._cheap_signals(tok, context)
        # Fallback: duplicate logic
        from OCR_error_detection.signals import (
            compute_surface_signals, compute_lexical_signals, compute_one_flip_signals
        )
        sig = TokenSignals(token=tok)
        compute_surface_signals(tok, sig)
        try:
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
        except Exception:
            pass
        return sig

    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        return combine_signals(sig, self.w)

    def verdict(self, tok: str, context: Sequence[str] = (),
                ocr_conf: Optional[float] = None, doc_freq: Optional[float] = None) -> Verdict:
        # Delegate to authoritative engine if available
        if self.engine is not None:
            v = self.engine.verdict(tok, context, ocr_conf=ocr_conf, doc_freq=doc_freq)
            # Mirror counters
            self._stage1 = dict(self.engine._stage1)
            return v

        # Fallback rule-based (deprecated)
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            try:
                sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0)) if ocr_conf > 1 else max(0.0, min(1.0, ocr_conf))
            except Exception:
                pass

        s = self.score(sig)
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.70)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig, "clean lexicon word (fallback)")

        if not sig.in_lexicon and sig.one_flip_words and s >= 0.50:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, f"OOV, one flip reaches {sig.one_flip_words[0]!r} (fallback)")

        if s >= self.flag_above:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, "flag (fallback)")
        if s >= self.pass_below or not sig.in_lexicon:
            self._stage1["gray"] += 1
            return Verdict(s, "gray", sig, "gray (fallback)")

        self._stage1["pass"] += 1
        return Verdict(s, "pass", sig, "pass (fallback)")

    def verdict_batch(self, tokens: Sequence[str], context_fn=None,
                      ocr_confs: Optional[Sequence[Optional[float]]] = None,
                      doc_freqs: Optional[Sequence[Optional[float]]] = None):
        if self.engine is not None:
            return self.engine.verdict_batch(tokens, context_fn=context_fn, ocr_confs=ocr_confs, doc_freqs=doc_freqs)
        # Fallback loop
        results = []
        for i, tok in enumerate(tokens):
            ctx = context_fn(i) if context_fn else []
            conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
            df = doc_freqs[i] if doc_freqs and i < len(doc_freqs) else None
            results.append(self.verdict(tok, ctx, ocr_conf=conf, doc_freq=df))
        return results

    def detection_stats(self):
        return {"stage1": dict(self._stage1)}

    def reset_counters(self):
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}
        if self.engine is not None:
            self.engine.reset_counters()
