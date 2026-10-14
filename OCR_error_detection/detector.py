# detector.py — Production OCR Error Detection Engine with LightGBM (Authoritative v3)
# Roadmap: LightGBM is sole authoritative detector, rule-based is fallback only
# Stage 1 fixes: strict NFC+Amharic normalization, NaN for missing, fixed sub key
# Stage 2: no manual floors, isotonic calibration, cost-sensitive thresholds
from __future__ import annotations

import logging
import os
import re
import math
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence

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

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureContext,
    static_features,
    with_context,
)
from OCR_error_detection.signals import (
    TokenSignals, Verdict,
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals,
)
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS

logger = logging.getLogger("ocr_error_detection")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

CANDIDATE_LEXICON_PATHS = [
    os.path.join(_HERE, "Model", "main_i7_frq10.txt"),  # 737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, cleaner
    os.path.join(_HERE, "main_i7_frq10.txt"),
    os.path.join(_HERE, "Model", "local_lexicon.txt"),  # 750k types, 15MB, from 4.4M tokens
    os.path.join(_HERE, "local_lexicon.txt"),
    os.path.join(_PARENT, "1_data_preparation", "data", "amh_final_training_text.txt"),
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt",
]

CANDIDATE_CONFUSION_PATHS = [
    os.path.join(_HERE, "Model", "confusion_matrix.json"),  # single source, no symlinks
    os.path.join(_PARENT, "4_ocr_correction", "confusion_matrix.json"),
    os.path.join(_PARENT, "confusion_matrix.json"),
]

CANDIDATE_MODEL_PATHS = [
    os.path.join(_HERE, "Model", "detector_lgbm_conffree.pkl"),  # conf-free, no ocr_conf, 14 feats, catches 90.8% high-conf errors
    os.path.join(_HERE, "Model", "detector_lgbm_v3.pkl"),  # v3 calibrated, NaN for missing conf
    os.path.join(_HERE, "Model", "detector_lgbm_v4.pkl"),
    os.path.join(_HERE, "Model", "detector_lgbm_v2.pkl"),  # v2: 96.7% recall but trusts conf (56% high-conf caught)
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
    os.path.join(_HERE, "detector_lgbm.pkl"),
    os.path.join(_HERE, "..", "archive", "models", "detector_lgbm_conffree.pkl"),
    os.path.join(_HERE, "..", "archive", "models", "detector_lgbm_v3.pkl"),
]

CANDIDATE_CHAR_LM_PATHS = [
    os.path.join(_HERE, "Model", "amharic_char_model.klm"),
    "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm",
]

PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')
RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")

def _resolve_first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _canonical_form(word: str, normalizer: AmharicNormalizer) -> str:
    """Canonical Ingestion Boundary: NFC + Amharic strict normalization"""
    if not word:
        return word
    # NFC first
    try:
        nfc = unicodedata.normalize('NFC', word)
    except Exception:
        nfc = word
    # Then Amharic strict
    try:
        return normalizer.normalize_word_strict(nfc)
    except Exception:
        return nfc

class DetectionEngine:
    """Authoritative LightGBM detector — sole source of truth.

    v3 changes per roadmap:
    - Canonical boundary: NFC + normalize_word_strict before any lookup
    - NaN for missing KenLM / doc_freq / ocr_conf (LightGBM handles NaN)
    - No manual safety floors — trusts calibrated P(error)
    - Isotonic calibration support
    - doc_freq optional: NaN for isolated token checks
    - Rule fallback only when LGBM unavailable
    """

    def __init__(
        self,
        dict_vocab_path: Optional[str] = None,
        confusion_matrix_path: Optional[str] = None,
        char_lm_path: Optional[str] = None,
        lgbm_model_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        pass_below: float = 0.20,
        flag_above: float = 0.65,
        max_one_flip: int = 8,
        max_lexicon_types: Optional[int] = 50000,
        dict_min_freq: Optional[int] = 10,
        symspell_max_edit_distance: int = 1,
        enable_symspell: bool = False,
        verbose: bool = False,
    ):
        self.normalizer = AmharicNormalizer()
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.max_one_flip = max_one_flip
        self.verbose = verbose
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

        # Resolve paths
        if dict_vocab_path is None:
            dict_vocab_path = _resolve_first_existing(CANDIDATE_LEXICON_PATHS)
        if confusion_matrix_path is None:
            confusion_matrix_path = _resolve_first_existing(CANDIDATE_CONFUSION_PATHS)
            if confusion_matrix_path is None:
                confusion_matrix_path = os.path.join(_HERE, "Model", "confusion_matrix.json")
        if char_lm_path is None:
            char_lm_path = _resolve_first_existing(CANDIDATE_CHAR_LM_PATHS)
        if lgbm_model_path is None:
            lgbm_model_path = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        # Load LGBM + calibrator
        self.lgbm_model = None
        self.lgbm_features: List[str] = []
        self.calibrator = None
        self._legacy_model = False  # True for schema-1 pickles (pre features.py)

        target_pkl = lgbm_model_path
        if target_pkl and not os.path.exists(target_pkl):
            target_pkl = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        if target_pkl and joblib is not None and os.path.exists(target_pkl):
            if self.verbose:
                print(f"Loading LightGBM Model: {target_pkl}", flush=True)
            try:
                payload = joblib.load(target_pkl)
                self.lgbm_model = payload.get("model")
                self.lgbm_features = payload.get("features", [])
                self.pass_below = payload.get("pass_thresh", self.pass_below)
                self.flag_above = payload.get("flag_thresh", self.flag_above)
                self.calibrator = payload.get("calibrator") or payload.get("isotonic") or payload.get("calibration")
                schema = payload.get("feature_schema_version", payload.get("feature_schema", 1))
                self._legacy_model = int(schema) < FEATURE_SCHEMA_VERSION
                if self.verbose:
                    cal_info = "with isotonic calibration" if self.calibrator is not None else "no calibration"
                    print(f"  LightGBM loaded ({len(self.lgbm_features)} feats, gates {self.pass_below}/{self.flag_above}) {cal_info}", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Failed to load LGBM: {e}", flush=True)
        elif self.verbose:
            print(f"  [WARN] LGBM not found, rule fallback active", flush=True)

        # Char LM optional
        self.char_lm = None
        if char_lm_path and os.path.exists(char_lm_path):
            klm = _load_kenlm()
            if klm:
                try:
                    self.char_lm = klm.Model(char_lm_path)
                    if self.verbose:
                        print(f"Loading Char KenLM: {char_lm_path}", flush=True)
                except Exception as e:
                    if self.verbose:
                        print(f"  [WARN] KenLM load failed: {e}", flush=True)

        # Confusion matrix
        if confusion_matrix_path and os.path.exists(confusion_matrix_path):
            if self.verbose:
                print(f"Loading confusion: {confusion_matrix_path}", flush=True)
            self._confusion = ConfusionMatrix(confusion_matrix_path)
            self.reverse_sub_map = self._confusion.reverse_sub_map
        else:
            if self.verbose:
                print(f"  [WARN] Confusion not found, empty map", flush=True)
            self._confusion = None
            self.reverse_sub_map = {}

        # Vocab
        self.vocab: Dict[str, int] = {}
        self.surface_form: Dict[str, str] = {}
        if dict_vocab_path and os.path.exists(dict_vocab_path):
            if self.verbose:
                print(f"Loading lexicon: {dict_vocab_path}", flush=True)
            try:
                self._load_vocabulary(dict_vocab_path, max_lexicon_types, dict_min_freq)
                if self.verbose:
                    print(f"  {len(self.vocab):,} types", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Lexicon load failed: {e}", flush=True)
        else:
            if self.verbose:
                print(f"  [WARN] Lexicon not found, minimal fallback", flush=True)
            for w in ["እና", "ነው", "ላይ", "እንደ", "አንድ", "ወደ", "ሰላም", "ቤት"]:
                nw = _canonical_form(w, self.normalizer)
                self.vocab[nw] = 1000
                self.surface_form[nw] = w

        # SymSpell — optional, disabled by default for performance
        # n_neighbors feature is 0 in conf-free models, so SymSpell not needed
        # Enable only if you need neighbourhood density
        self.symspell = None
        self._VERBOSITY_TOP = 2
        if enable_symspell:
            try:
                _SS, _V = _load_symspell()
                self._VERBOSITY_TOP = _V.CLOSEST if hasattr(_V, 'CLOSEST') else 2
                self.symspell = _SS(max_dictionary_edit_distance=symspell_max_edit_distance, prefix_length=7)
                if self.verbose:
                    print(f"  Building SymSpell for {len(self.vocab):,} types...", flush=True)
                for norm_w, freq in self.vocab.items():
                    try:
                        self.symspell.create_dictionary_entry(norm_w, freq)
                    except Exception:
                        continue
                if self.verbose:
                    print("  SymSpell ready.", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] SymSpell build failed: {e}", flush=True)
                self.symspell = None
        else:
            if self.verbose:
                print(f"  SymSpell disabled (enable_symspell=False) for speed — n_neighbors=0", flush=True)

        # Shared feature context — features.py is the single feature definition
        # used by training (Model/train_LightGBM.py) and inference (here).
        self._ctx = FeatureContext(
            vocab=self.vocab,
            norm=self._norm,
            reverse_sub_map=self.reverse_sub_map,
            max_one_flip=self.max_one_flip,
            char_lm=self.char_lm,
            symspell=self.symspell,
            verbosity_top=getattr(self, "_VERBOSITY_TOP", 2),
        )

    def _load_vocabulary(self, path: str, max_types: Optional[int], min_freq: Optional[int]):
        import heapq
        # Detect raw text vs lexicon
        is_raw = False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as probe:
                first = probe.readline()
                if len(first.strip().split()) > 5:
                    is_raw = True
        except Exception:
            pass

        if is_raw:
            from collections import Counter
            counter = Counter()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for tok in line.strip().split():
                        tok = tok.strip()
                        if not tok:
                            continue
                        counter[tok] += 1
            for w, freq in counter.items():
                nw = _canonical_form(w, self.normalizer)
                if not nw:
                    continue
                prev = self.vocab.get(nw, 0)
                if freq >= prev:
                    self.surface_form[nw] = w
                self.vocab[nw] = max(prev, freq)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    w = parts[0]
                    freq = 1
                    if len(parts) > 1 and parts[-1].isdigit():
                        try:
                            freq = int(parts[-1])
                        except ValueError:
                            freq = 1
                    nw = _canonical_form(w, self.normalizer)
                    if not nw:
                        continue
                    prev = self.vocab.get(nw, 0)
                    if freq >= prev:
                        self.surface_form[nw] = w
                    self.vocab[nw] = max(prev, freq)

        if min_freq and min_freq > 1:
            keep = [w for w, f in self.vocab.items() if f >= min_freq]
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

        if max_types and len(self.vocab) > max_types:
            keep = heapq.nlargest(max_types, self.vocab, key=self.vocab.__getitem__)
            self.vocab = {w: self.vocab[w] for w in keep}
            self.surface_form = {w: self.surface_form[w] for w in keep if w in self.surface_form}

    def _norm(self, word: str) -> str:
        return _canonical_form(word, self.normalizer)

    def _score_char_lm(self, word: str) -> Optional[float]:
        if not self.char_lm:
            return None
        # Canonical
        norm_w = self._norm(word)
        clean_w = PUNCT_RE.sub('', norm_w).strip()
        clean_w = ETHIOPIC_PUNCT_RE.sub('', clean_w).strip()
        if len(clean_w) < 2:
            return None
        spaced = " ".join(clean_w)
        try:
            raw = self.char_lm.score(spaced, bos=True, eos=True)
            return raw / (len(clean_w) + 1.0)
        except Exception:
            return None

    def _cheap_signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        sig = TokenSignals(token=tok)
        compute_surface_signals(tok, sig)
        compute_lexical_signals(tok, sig, self.vocab, self._norm,
                                self.symspell, getattr(self, '_VERBOSITY_TOP', 2))
        compute_one_flip_signals(tok, sig, self.vocab, self._norm,
                                 self.reverse_sub_map, self.max_one_flip)
        return sig

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals,
                               doc_freq: Optional[float] = None,
                               ocr_conf: Optional[float] = None) -> List[float]:
        """Extract features with NaN for missing (per roadmap).

        The schema-2 columns come from features.static_features/with_context —
        the exact code path Model/train_LightGBM.py uses — so a training row
        and a serving row for the same word are identical (train/serve parity).
        The extra columns at the bottom cover schema-1 (legacy) model shapes.
        """
        # Schema-2 columns: identical to a training extract_frame row.
        feat_map = with_context(static_features(tok, self._ctx), doc_freq)

        # Legacy (schema-1) extras, taken from the caller's signals.
        n_neighbors = getattr(sig, "n_neighbors", 0)
        len_ratio = getattr(sig, "len_ratio", None)
        if len_ratio is None:
            len_ratio = max(len(PUNCT_RE.sub('', tok or '').strip()), 1) / 4.14
        mixed_script = 1 if getattr(sig, "mixed_script", False) else 0
        has_digit = 1 if getattr(sig, "has_digit", False) else 0

        # OCR conf: normalized 0-1, NaN if missing
        if ocr_conf is not None:
            if ocr_conf > 1:
                ocr_conf_norm = max(0.0, min(1.0, ocr_conf / 100.0))
            else:
                ocr_conf_norm = max(0.0, min(1.0, ocr_conf))
        elif sig.ocr_conf is not None:
            ocr_conf_norm = sig.ocr_conf
        else:
            ocr_conf_norm = float('nan')

        feat_map["ocr_conf"] = ocr_conf_norm
        feat_map["n_neighbors"] = float(n_neighbors)
        feat_map["len_ratio"] = float(len_ratio)
        feat_map["mixed_script"] = float(mixed_script)
        feat_map["has_digit"] = float(has_digit)
        feat_map["log_freq"] = feat_map["log_corpus_freq"]
        feat_map["is_high_freq"] = 1 if feat_map["log_corpus_freq"] > 3.0 else 0

        if not self.lgbm_features:
            return []

        return [feat_map.get(col, float('nan')) for col in self.lgbm_features]

    # Public API for tests
    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        return combine_signals(sig, self.w)

    def detection_stats(self):
        return {"stage1": dict(self._stage1)}

    def reset_counters(self):
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

    def verdict_batch(self, tokens: Sequence[str],
                      context_fn: Optional[Callable] = None,
                      ocr_confs: Optional[Sequence[Optional[float]]] = None,
                      doc_freqs: Optional[Sequence[Optional[float]]] = None) -> List[Verdict]:
        results = []
        for i, tok in enumerate(tokens):
            ctx = context_fn(i) if context_fn else []
            conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
            df = doc_freqs[i] if doc_freqs and i < len(doc_freqs) else None
            results.append(self.verdict(tok, ctx, ocr_conf=conf, doc_freq=df))
        return results

    def verdict(
        self,
        tok: str,
        context: Sequence[str] = (),
        ocr_conf: Optional[float] = None,
        doc_freq: Optional[float] = None,
    ) -> Verdict:
        """Authoritative verdict: calibrated P(error) -> pass/gray/flag, no manual floors."""
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            # Store normalized 0-1
            try:
                if ocr_conf > 1:
                    sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))
                else:
                    sig.ocr_conf = max(0.0, min(1.0, ocr_conf))
            except Exception:
                sig.ocr_conf = None

        # PATH A: LightGBM authoritative
        if self.lgbm_model is not None and self.lgbm_features:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq, ocr_conf=ocr_conf)
            try:
                raw_prob = float(self.lgbm_model.predict([feats])[0])
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] LGBM predict failed: {e}, fallback to rule", flush=True)
                raw_prob = combine_signals(sig, self.w)

            # Isotonic calibration if available
            prob = raw_prob
            if self.calibrator is not None:
                try:
                    # IsotonicRegression expects array
                    prob = float(self.calibrator.predict([raw_prob])[0])
                    # Clip to [0,1]
                    prob = max(0.0, min(1.0, prob))
                except Exception:
                    prob = raw_prob

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

            # No manual floors per roadmap — trust calibrated probabilities
            # Only keep minimal OOV->gray for backward compat if model says pass but OOV
            # But per roadmap we should NOT override. We keep no overrides.

            if label == "pass":
                self._stage1["pass"] += 1
            elif label == "flag":
                self._stage1["flag"] += 1
            else:
                self._stage1["gray"] += 1

            return Verdict(s, label, sig, reason)

        # PATH B: Rule fallback (deprecated, lightweight)
        s = combine_signals(sig, self.w)
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
