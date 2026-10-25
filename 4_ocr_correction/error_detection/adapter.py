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
#   from error_detection.adapter import CorrectorDetectionAdapter
#   self.suspicion = CorrectorDetectionAdapter(self)
#   result = self.suspicion.verdict(word, context, ocr_conf, doc_freq)
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Dict, List
import math

logger = logging.getLogger(__name__)

from error_detection.signals import TokenSignals, Verdict
from error_detection.scoring import combine_signals, DEFAULT_WEIGHTS

# Import DetectionEngine lazily to avoid circular
DetectionEngine = None

def _load_engine():
    global DetectionEngine
    if DetectionEngine is None:
        from error_detection.detector import DetectionEngine as _DE
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
        self._lexicon_adopted = False

        self.last_error = None
        try:
            DE = _load_engine()
            # Create engine but inject corrector's vocab etc. if possible
            # We instantiate with minimal loading, then override vocab/symspell.
            # Lexicon knobs are left at None so the model payload decides; the
            # corrector's own vocabulary replaces the engine's anyway.
            self.engine = DE(
                dict_vocab_path=None,  # resolved from the model / documented default
                confusion_matrix_path=None,
                char_lm_path=None,
                lgbm_model_path=lgbm_model_path,
                pass_below=pass_below,
                flag_above=flag_above,
                max_one_flip=max_one_flip,
                max_lexicon_types=None,
                dict_min_freq=None,
                enable_symspell=False,
                verbose=False,
            )
            # Hand over the corrector's own resources (see below); the corrector
            # loads its vocabulary *after* building stage 1, so this usually
            # does nothing here and happens on the first verdict()/signals().
            self._adopt_corrector_resources()
            # Reuse model and thresholds
            self.lgbm_model = self.engine.lgbm_model
            self.lgbm_features = self.engine.lgbm_features
            self.calibrator = self.engine.calibrator
            self.pass_below = self.engine.pass_below
            self.flag_above = self.engine.flag_above
        except Exception as e:
            # Fallback to rule-based. Keep the reason so the caller can log it
            # instead of silently degrading (the corrector prints it).
            self.last_error = e
            self.engine = None
            self.lgbm_model = None

    def _adopt_corrector_resources(self) -> None:
        """Give the engine the resources the corrector already has in memory.

        README: "adapter.py — for parent to import child (reuses vocab, no
        double load)". Two things have to be true for that to work:

        * the handover must reach the FeatureContext, not just engine.vocab
          (features.py reads ctx.vocab), and
        * it must run after the corrector has loaded its vocabulary, which
          happens *after* stage 1 is constructed — hence the lazy retry from
          verdict()/signals().

        The model's own frequency filter is applied to the adopted lexicon, so
        the features stay in the distribution the model was trained on
        (main_i7 freq>=500); symspell is deliberately NOT adopted, because the
        model is served with n_neighbors=0 (README: "SymSpell OFF, gain 0").
        """
        if self.engine is None or self._lexicon_adopted:
            return
        vocab = getattr(self.c, "vocab", None)
        if not vocab:
            return
        try:
            expected = self.engine.lexicon_info.get("types") or 0
            n_types = self.engine.adopt_lexicon(
                vocab,
                surface_form=getattr(self.c, "surface_form", None),
                norm=getattr(self.c, "_norm", None),
            )
            self._lexicon_adopted = True
            if self.engine.verbose:
                print(f"Stage 1: reusing the corrector's vocabulary "
                      f"({n_types:,} types after the model's filter)", flush=True)
            # The model was trained on its own lexicon. Serving it a very
            # different one (a domain list, a test fixture) puts the feature
            # values outside the training distribution, which shows up as
            # everything looking equally likely — say so instead of silently
            # degrading.
            if expected and not (0.5 * expected <= n_types <= 2.0 * expected):
                logger.warning(
                    "detector model was trained on %s lexicon types but the "
                    "corrector supplied %s; detection quality is only "
                    "calibrated for the model's own lexicon", f"{expected:,}",
                    f"{n_types:,}")
        except Exception as e:  # never let a handover failure break correction
            self.last_error = e

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        self._adopt_corrector_resources()
        # If engine available, use its cheap signals (which uses canonical boundary)
        if self.engine is not None:
            return self.engine._cheap_signals(tok, context)
        # Fallback: duplicate logic
        from error_detection.signals import (
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
        self._adopt_corrector_resources()
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
