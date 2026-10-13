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

    Improved v2:
    - Optionally loads LightGBM model if available (reuses corrector's vocab)
    - Frequency-aware safety floors (high-freq words never flagged on low-conf)
    - Supports expanded feature set (ocr_conf, n_neighbors, etc.)

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
    lgbm_model_path : str, optional
        Path to LightGBM model. If None, tries to auto-discover.
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
        self._morph_calls = 0
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

        # Try to load LGBM model if available
        self.lgbm_model = None
        self.lgbm_features = []
        try:
            import os, joblib
            _HERE = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                lgbm_model_path,
                os.path.join(_HERE, "Model", "detector_lgbm_v2.pkl"),
                os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
                os.path.join(_HERE, "detector_lgbm.pkl"),
            ]
            target = next((p for p in candidates if p and os.path.exists(p)), None)
            if target:
                payload = joblib.load(target)
                self.lgbm_model = payload.get("model")
                self.lgbm_features = payload.get("features", [])
                self.pass_below = payload.get("pass_thresh", self.pass_below)
                self.flag_above = payload.get("flag_thresh", self.flag_above)
        except Exception:
            self.lgbm_model = None
            self.lgbm_features = []

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

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals, doc_freq: int = 1, ocr_conf: Optional[float] = None):
        import math, re
        PUNCT_RE = re.compile(r'[^\w\s]|\d')
        RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")
        clean_raw = PUNCT_RE.sub('', tok).strip()
        t_len = max(len(clean_raw), 1)
        norm_tok = self.c._norm(tok) if hasattr(self.c, '_norm') else tok
        freq = self.c.vocab.get(norm_tok, 0) if hasattr(self.c, 'vocab') else 0
        log_freq = math.log10(freq + 1) if freq > 0 else 0.0
        in_lex = 1 if freq > 0 else 0
        rare_cnt = sum(1 for ch in clean_raw if ch in RARE_GLYPHS)
        rare_ratio = rare_cnt / float(t_len)
        one_flip_cnt = len(sig.one_flip_words)
        n_neighbors = getattr(sig, "n_neighbors", 0)
        len_ratio = getattr(sig, "len_ratio", t_len / 4.14)
        mixed_script = 1 if getattr(sig, "mixed_script", False) else 0
        has_digit = 1 if getattr(sig, "has_digit", False) else 0
        if ocr_conf is not None:
            ocr_conf_norm = max(0.0, min(1.0, ocr_conf / 100.0)) if ocr_conf > 1 else max(0.0, min(1.0, ocr_conf))
        elif sig.ocr_conf is not None:
            ocr_conf_norm = sig.ocr_conf
        else:
            ocr_conf_norm = 0.85
        feat_map = {
            "log_corpus_freq": log_freq,
            "in_lexicon": in_lex,
            "doc_freq": doc_freq,
            "is_doc_singleton": 1 if doc_freq == 1 else 0,
            "token_len": t_len,
            "has_one_flip": 1 if one_flip_cnt > 0 else 0,
            "one_flip_count": one_flip_count,
            "char_kenlm_score": getattr(sig, "char_lm", -2.5) or -2.5,
            "rare_glyph_ratio": rare_ratio,
            "ocr_conf": ocr_conf_norm,
            "n_neighbors": float(n_neighbors),
            "len_ratio": float(len_ratio),
            "mixed_script": float(mixed_script),
            "has_digit": float(has_digit),
            "log_freq": log_freq,
            "is_high_freq": 1 if log_freq > 3.0 else 0,
        }
        if not self.lgbm_features:
            return []
        return [feat_map.get(col, 0.0) for col in self.lgbm_features]

    def verdict(self, tok: str, context: Sequence[str] = (),
                ocr_conf: Optional[float] = None, doc_freq: int = 1) -> Verdict:
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0)) if ocr_conf > 1 else max(0.0, min(1.0, ocr_conf))

        # If LGBM available, use it
        if self.lgbm_model is not None and self.lgbm_features:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq, ocr_conf=ocr_conf)
            try:
                prob = float(self.lgbm_model.predict([feats])[0])
            except Exception:
                prob = combine_signals(sig, self.w)
            s = prob
            if s >= self.flag_above:
                label = "flag"
                reason = f"LGBM={s:.3f} (FLAG)"
            elif s >= self.pass_below:
                label = "gray"
                reason = f"LGBM={s:.3f} (GRAY)"
            else:
                label = "pass"
                reason = f"LGBM={s:.3f} (PASS)"

            # Frequency-aware safety floors
            is_high_freq = False
            try:
                freq = self.c.vocab.get(self.c._norm(tok), 0) if hasattr(self.c, 'vocab') else 0
                is_high_freq = freq > 1000 or sig.log_freq > 3.0
            except Exception:
                pass

            if not sig.in_lexicon and sig.ocr_conf is not None and sig.ocr_conf < 0.35:
                if sig.one_flip_words:
                    label = "flag"
                    reason += f", low-conf + one-flip ({sig.ocr_conf*100:.0f}%)"
                elif not is_high_freq:
                    label = "flag"
                    reason += f", severe low-conf OOV ({sig.ocr_conf*100:.0f}%)"

            if not sig.in_lexicon and label == "pass":
                label = "gray"
                reason += ", OOV kept in gray"

            if sig.ocr_conf is not None and sig.ocr_conf < 0.55 and label == "pass" and not is_high_freq:
                label = "gray"
                reason += f", low-conf floor ({sig.ocr_conf*100:.0f}%)"

            if is_high_freq and label == "flag" and sig.ocr_conf is not None and sig.ocr_conf > 0.70:
                if s < self.flag_above + 0.15:
                    label = "gray"
                    reason += ", high-freq downgrade"

            if label == "pass":
                self._stage1["pass"] += 1
            elif label == "flag":
                self._stage1["flag"] += 1
            else:
                self._stage1["gray"] += 1
            return Verdict(s, label, sig, reason)

        # Fallback to rule-based
        s = self.score(sig)

        # Hard PASS
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.70)
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

    def detection_stats(self):
        return {"stage1": dict(self._stage1)}

    def reset_counters(self):
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}