# detector.py — Production OCR Error Detection Engine with LightGBM ML Classifier
from __future__ import annotations

import logging
import os
import re
import math 
import joblib
from typing import Callable, Dict, Optional, Sequence

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


from normalizer import AmharicNormalizer
from confusion import ConfusionMatrix
from signals import (
    TokenSignals, Verdict,
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals,
)
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS

logger = logging.getLogger("ocr_error_detection")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHAR_LM_PATH = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm"
CANDIDATE_MODEL_PATHS = [
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
    os.path.join(_HERE, "detector_lgbm.pkl"),
    os.path.join(os.path.dirname(_HERE), "Model", "detector_lgbm.pkl"),
    os.path.join(os.path.dirname(_HERE), "detector_lgbm.pkl"),
]

PUNCT_RE = re.compile(r'[^\w\s]|\d')
RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")


class DetectionEngine:
    """Production OCR Error Detection Engine driven by LightGBM & Optical Evidence."""

    def __init__(
        self,
        dict_vocab_path: str ="/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt",
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
        verbose: bool = True,
    ):
        self.normalizer = AmharicNormalizer()
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.pass_below = pass_below
        self.flag_above = flag_above
        self.max_one_flip = max_one_flip
        self.verbose = verbose
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

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
                self.lgbm_features = payload["features"]
                self.pass_below = payload.get("pass_thresh", self.pass_below)
                self.flag_above = payload.get("flag_thresh", self.flag_above)
                if self.verbose:
                    print(f"  LightGBM loaded ({len(self.lgbm_features)} features, gates: {self.pass_below}/{self.flag_above})", flush=True)
            except Exception as e:
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

    def _score_char_lm(self, word: str) -> Optional[float]:
        if not self.char_lm: return None
        norm_w = self._norm(word)
        clean_w = PUNCT_RE.sub('', norm_w).strip()
        if len(clean_w) < 2: return None
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

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals, doc_freq: int = 1) -> list:
        clean_w = PUNCT_RE.sub('', tok).strip()
        t_len = max(len(clean_w), 1)

        # Lookup corpus frequency from master dictionary
        freq = self.vocab.get(clean_w, self.vocab.get(tok, 0))
        log_freq = math.log10(freq + 1) if freq > 0 else 0.0
        in_lex = 1 if freq > 0 else 0

        char_klm = getattr(sig, "char_lm_score", None)
        if char_klm is None:
            char_klm = self._score_char_lm(tok)
            if char_klm is not None:
                setattr(sig, "char_lm_score", char_klm)
            else:
                char_klm = -2.5

        rare_cnt = sum(1 for ch in clean_w if ch in RARE_GLYPHS)
        rare_ratio = rare_cnt / float(t_len)
        one_flip_cnt = len(sig.one_flip_words)

        feat_map = {
            "log_corpus_freq": log_freq,
            "in_lexicon": in_lex,
            "doc_freq": doc_freq,
            "is_doc_singleton": 1 if doc_freq == 1 else 0,
            "token_len": t_len,
            "has_one_flip": 1 if one_flip_cnt > 0 else 0,
            "one_flip_count": one_flip_cnt,
            "char_kenlm_score": char_klm,
            "rare_glyph_ratio": rare_ratio,
        }
        return [feat_map.get(col, 0.0) for col in self.lgbm_features]

    def verdict(
        self,
        tok: str,
        context: Sequence[str] = (),
        ocr_conf: Optional[float] = None,
        doc_freq: int = 1,
    ) -> Verdict:
        """Full detection verdict: returns 'pass', 'gray', or 'flag'."""
        sig = self._cheap_signals(tok, context)
        if ocr_conf is not None:
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0))

        # ── PATH A: LightGBM Model Prediction (Primary) ──
        if self.lgbm_model is not None:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq)
            # Calibrated probability P(error) in [0, 1]
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

            # Optical Safety Floors
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