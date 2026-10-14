# detector.py — Production OCR Error Detection Engine with LightGBM ML Classifier
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Sequence

import joblib

# Lazy imports for heavy dependencies
SymSpell = Verbosity = None
kenlm = None


def _load_symspell():
    global SymSpell, Verbosity
    if SymSpell is None:
        from symspellpy import SymSpell as _SymSpell, Verbosity as _Verbosity
        SymSpell, Verbosity = _SymSpell, _Verbosity
    return SymSpell, Verbosity


def _load_kenlm():
    global kenlm
    if kenlm is None:
        try:
            import kenlm as _kenlm
            kenlm = _kenlm
        except ImportError:
            kenlm = False
    return kenlm


from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.signals import TokenSignals, Verdict, compute_lexical_signals
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS
from OCR_error_detection.features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureContext,
    PUNCT_RE, RARE_GLYPHS,
    apply_legacy_imputation, cheap_signals, static_features, to_vector,
    with_context,
)

logger = logging.getLogger("ocr_error_detection")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHAR_LM_PATH = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm"
CANDIDATE_MODEL_PATHS = [
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
    os.path.join(_HERE, "detector_lgbm.pkl"),
    os.path.join(os.path.dirname(_HERE), "Model", "detector_lgbm.pkl"),
    os.path.join(os.path.dirname(_HERE), "detector_lgbm.pkl"),
]


class DetectionEngine:
    """Production OCR Error Detection Engine driven by LightGBM & Optical Evidence."""

    def __init__(
        self,
        dict_vocab_path: str = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt",
        confusion_matrix_path: str = "confusion_matrix.json",
        char_lm_path: Optional[str] = DEFAULT_CHAR_LM_PATH,
        lgbm_model_path: Optional[str] = "Model/detector_lgbm.pkl",
        weights: Optional[Dict[str, float]] = None,
        pass_below: float = 0.25,
        flag_above: float = 0.70,
        max_one_flip: int = 8,
        max_lexicon_types: int = None,
        dict_min_freq: int = None,
        symspell_max_edit_distance: int = 1,
        signal_cache_size: int = 4096,
        verbose: bool = True,
    ):
        self.normalizer = AmharicNormalizer()
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.max_one_flip = max_one_flip
        self.verbose = verbose
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}
        self._legacy_model = False

        # Bounded cache: OCR documents often repeat the same word many times.
        # Keep this bounded so long-running evaluation cannot grow RAM without limit.
        self._signal_cache_maxsize = max(0, int(signal_cache_size))
        self._signal_cache = OrderedDict()
        self._signal_cache_hits = 0
        self._signal_cache_misses = 0

        # ── 1. Load LightGBM Model ──
        self.lgbm_model = None
        self.lgbm_features = []

        if lgbm_model_path and os.path.exists(lgbm_model_path):
            target_pkl = lgbm_model_path
        else:
            target_pkl = next((p for p in CANDIDATE_MODEL_PATHS if os.path.exists(p)), None)

        if target_pkl:
            if self.verbose:
                print(f"Loading LightGBM Model: {target_pkl}", flush=True)
            try:
                payload = joblib.load(target_pkl)
                self.lgbm_model = payload["model"]
                self.lgbm_features = list(payload["features"])
                # Validate names now, so a mismatch fails at load time, not per word.
                to_vector({n: 0.0 for n in FEATURE_NAMES}, self.lgbm_features)
                schema = int(payload.get("feature_schema_version", 1))
                self._legacy_model = schema < FEATURE_SCHEMA_VERSION
                if self._legacy_model:
                    msg = ("LightGBM model uses feature schema %d (current: %d). Missing "
                           "context is imputed with legacy constants. Retrain to remove this."
                           % (schema, FEATURE_SCHEMA_VERSION))
                    if self.verbose:
                        print(f"  [WARN] {msg}", flush=True)
                    else:
                        logger.info(msg)
                self.pass_below = payload.get("pass_thresh", self.pass_below)
                self.flag_above = payload.get("flag_thresh", self.flag_above)
                if self.verbose:
                    print(f"  LightGBM loaded ({len(self.lgbm_features)} features, gates: {self.pass_below}/{self.flag_above})", flush=True)
            except Exception as e:
                self.lgbm_model = None
                if self.verbose:
                    print(f"  [WARN] Failed to load LightGBM model: {e}", flush=True)
        elif self.verbose:
            print("  [WARN] detector_lgbm.pkl not found, running rule fallback.", flush=True)

        # ── 2. Load KenLM Character Model ──
        self.char_lm = None
        if char_lm_path and os.path.exists(char_lm_path):
            klm = _load_kenlm()
            if klm:
                if self.verbose:
                    print(f"Loading Character KenLM: {char_lm_path}", flush=True)
                try:
                    self.char_lm = klm.Model(char_lm_path)
                except Exception as e:
                    if self.verbose:
                        print(f"  [WARN] Failed to load KenLM model: {e}", flush=True)

        # ── 3. Load Confusion Matrix ──
        if confusion_matrix_path is None:
            confusion_matrix_path = os.path.join(_HERE, "confusion_matrix.json")
            if not os.path.exists(confusion_matrix_path):
                confusion_matrix_path = os.path.join(os.path.dirname(_HERE), "confusion_matrix.json")

        if self.verbose:
            print(f"Loading optical confusion matrix: {confusion_matrix_path}", flush=True)
        self._confusion = ConfusionMatrix(confusion_matrix_path)
        self.reverse_sub_map = self._confusion.reverse_sub_map

        # ── 4. Load Vocabulary ──
        if self.verbose:
            print(f"Loading lexicon: {dict_vocab_path}", flush=True)
        self.vocab = {}
        self.surface_form = {}
        self._load_vocabulary(dict_vocab_path, max_lexicon_types, dict_min_freq)
        if self.verbose:
            print(f"  {len(self.vocab):,} active types", flush=True)

        # ── 5. Build SymSpell Index ──
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

        # ── 6. Shared feature context (same definition as training) ──
        self._ctx = FeatureContext(
            vocab=self.vocab,
            norm=self._norm,
            reverse_sub_map=self.reverse_sub_map,
            max_one_flip=self.max_one_flip,
            char_lm=self.char_lm,
            symspell=self.symspell,
            verbosity_top=self._VERBOSITY_TOP,
        )

    def _load_vocabulary(self, path, max_types, min_freq):
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
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

        if max_types and len(self.vocab) > max_types:
            keep = heapq.nlargest(max_types, self.vocab, key=self.vocab.__getitem__)
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

    def _norm(self, word: str) -> str:
        if not word: return word
        return self.normalizer.normalize_word_strict(word)

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        """Compute lexical signals, reusing a bounded cache for repeated tokens.

        The current implementation of cheap_signals receives only ``tok`` and
        ``self._ctx`` (the ``context`` argument is not used by the original code).
        A shallow copy prevents per-call OCR confidence assignment from mutating
        the cached TokenSignals object. Cache size zero disables caching.
        """
        if not self._signal_cache_maxsize or not tok:
            return cheap_signals(tok, self._ctx)

        cached = self._signal_cache.get(tok)
        if cached is not None:
            self._signal_cache_hits += 1
            self._signal_cache.move_to_end(tok)
            return copy(cached)

        self._signal_cache_misses += 1
        sig = cheap_signals(tok, self._ctx)
        self._signal_cache[tok] = copy(sig)
        self._signal_cache.move_to_end(tok)
        if len(self._signal_cache) > self._signal_cache_maxsize:
            self._signal_cache.popitem(last=False)
        return sig

    def cache_stats(self) -> Dict[str, int]:
        """Return cache counters for profiling without exposing cached objects."""
        return {
            "hits": self._signal_cache_hits,
            "misses": self._signal_cache_misses,
            "entries": len(self._signal_cache),
            "max_entries": self._signal_cache_maxsize,
        }

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals,
                               doc_freq: Optional[int] = None) -> list:
        """Model input vector, in the order the model was trained with."""
        feats = static_features(tok, self._ctx, sig)
        feats = with_context(feats, doc_freq)
        if self._legacy_model:
            feats = apply_legacy_imputation(feats)
        return to_vector(feats, self.lgbm_features)

    def verdict(
        self,
        tok: str,
        context: Sequence[str] = (),
        ocr_conf: Optional[float] = None,
        doc_freq: Optional[int] = None,
    ) -> Verdict:
        """Full detection verdict: returns 'pass', 'gray', or 'flag'.

        doc_freq: occurrences of this word on its page. None = no page context
        (isolated word); the model then sees NaN, not a singleton.
        """
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))

        # ── PATH A: LightGBM Model Prediction (Primary) ──
        if self.lgbm_model is not None:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq)
            # Score of the model. Calibration is a Stage 2 task; see docs/lgbm_detector_review.md.
            prob = float(self.lgbm_model.predict([feats])[0])
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

            # Optical Safety Floors (removed in Stage 2 of the roadmap)
            if not sig.in_lexicon and sig.ocr_conf is not None and sig.ocr_conf < 0.50:
                label = "flag"
                reason += f", severe low-conf ({sig.ocr_conf*100:.0f}%)"
            elif not sig.in_lexicon and label == "pass":
                label = "gray"
                reason += ", OOV kept in gray"
            elif sig.ocr_conf is not None and sig.ocr_conf < 0.60 and label == "pass":
                label = "gray"
                reason += f", low-conf floor ({sig.ocr_conf*100:.0f}%)"

            if label == "pass": self._stage1["pass"] += 1
            elif label == "flag": self._stage1["flag"] += 1
            else: self._stage1["gray"] += 1

            return Verdict(s, label, sig, reason)

        # ── PATH B: Rule Fallback (if LightGBM is missing) ──
        s = combine_signals(sig, self.w)
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.75)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig, "clean lexicon word")

        if not sig.in_lexicon and sig.one_flip_words and s >= 0.50:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, f"OOV, one flip reaches {sig.one_flip_words[0]!r}")

        if not sig.in_lexicon and sig.ocr_conf is not None and sig.ocr_conf < 0.50:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, f"low-conf ({sig.ocr_conf*100:.0f}%) + OOV")

        if s >= self.flag_above:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, "flag")
        if s >= self.pass_below or not sig.in_lexicon:
            self._stage1["gray"] += 1
            return Verdict(s, "gray", sig, "gray")

        self._stage1["pass"] += 1
        return Verdict(s, "pass", sig, "pass")
