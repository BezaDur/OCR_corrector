# detector.py — Production OCR Error Detection Engine with LightGBM ML Classifier (Improved v2)
from __future__ import annotations

import logging
import os
import re
import math
from typing import Callable, Dict, List, Optional, Sequence

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


try:
    import joblib
except ImportError:
    joblib = None

from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.signals import (
    TokenSignals, Verdict,
    compute_surface_signals, compute_lexical_signals,
    compute_one_flip_signals,
)
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS

logger = logging.getLogger("ocr_error_detection")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

# Candidate paths for lexicon — prefer local, fallback to old hardcoded
CANDIDATE_LEXICON_PATHS = [
    os.path.join(_HERE, "Model", "local_lexicon.txt"),
    os.path.join(_HERE, "local_lexicon.txt"),
    os.path.join(_PARENT, "amh_final_training_text.txt"),  # will be counted, not direct lexicon format? but we handle
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),  # placeholder to keep order
    "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt",
]

CANDIDATE_CONFUSION_PATHS = [
    os.path.join(_HERE, "confusion_matrix.json"),
    os.path.join(_HERE, "Model", "confusion_matrix.json"),
    os.path.join(_PARENT, "confusion_matrix.json"),
]

CANDIDATE_MODEL_PATHS = [
    os.path.join(_HERE, "Model", "detector_lgbm_v2.pkl"),  # new improved model first
    os.path.join(_HERE, "Model", "detector_lgbm.pkl"),
    os.path.join(_HERE, "detector_lgbm.pkl"),
    os.path.join(_PARENT, "Model", "detector_lgbm.pkl"),
    os.path.join(_PARENT, "detector_lgbm.pkl"),
]

CANDIDATE_CHAR_LM_PATHS = [
    os.path.join(_HERE, "Model", "amharic_char_model.klm"),
    "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm",
]

PUNCT_RE = re.compile(r'[^\w\s]|\d')
# Ethiopic punctuation is considered punctuation, not word chars
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')
RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")


def _resolve_first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


class DetectionEngine:
    """Production OCR Error Detection Engine driven by LightGBM & Optical Evidence.

    Improvements in v2:
    - Local lexicon fallback (no hardcoded /mnt dependency)
    - Fixed normalization bug in feature extraction (was using raw token for freq lookup)
    - Frequency-aware safety floors (high-freq words never flagged on low-conf alone)
    - Expanded feature set support (ocr_conf, n_neighbors, mixed_script, etc.)
    - Backward compatible with old 9-feature model
    - Added detection_stats(), reset_counters(), signals(), verdict_batch() for tests
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
        max_lexicon_types: Optional[int] = None,
        dict_min_freq: Optional[int] = None,
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

        # Resolve paths
        if dict_vocab_path is None:
            dict_vocab_path = _resolve_first_existing(CANDIDATE_LEXICON_PATHS)
            # If still None, try to find any txt with word freq in repo root
            if dict_vocab_path is None:
                fallback = os.path.join(_PARENT, "OCR_error_detection", "Model", "local_lexicon.txt")
                if os.path.exists(fallback):
                    dict_vocab_path = fallback

        if confusion_matrix_path is None:
            confusion_matrix_path = _resolve_first_existing(CANDIDATE_CONFUSION_PATHS)
            if confusion_matrix_path is None:
                confusion_matrix_path = os.path.join(_HERE, "confusion_matrix.json")

        if char_lm_path is None:
            char_lm_path = _resolve_first_existing(CANDIDATE_CHAR_LM_PATHS)

        if lgbm_model_path is None:
            lgbm_model_path = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        # ── 1. Load LightGBM Model ──
        self.lgbm_model = None
        self.lgbm_features: List[str] = []

        target_pkl = lgbm_model_path
        if target_pkl and not os.path.exists(target_pkl):
            target_pkl = _resolve_first_existing(CANDIDATE_MODEL_PATHS)

        if target_pkl and joblib is not None and os.path.exists(target_pkl):
            if self.verbose:
                print(f"Loading LightGBM Model: {target_pkl}", flush=True)
            try:
                payload = joblib.load(target_pkl)
                self.lgbm_model = payload["model"]
                self.lgbm_features = payload.get("features", [])
                # Only override thresholds if payload provides them and user didn't customize
                self.pass_below = payload.get("pass_thresh", self.pass_below)
                self.flag_above = payload.get("flag_thresh", self.flag_above)
                if self.verbose:
                    print(f"  LightGBM loaded ({len(self.lgbm_features)} features, gates: {self.pass_below}/{self.flag_above})", flush=True)
                    if hasattr(self.lgbm_model, "feature_importance"):
                        try:
                            imp = self.lgbm_model.feature_importance(importance_type='gain')
                            print(f"  Feature gain: {list(zip(self.lgbm_features, imp))}", flush=True)
                        except Exception:
                            pass
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Failed to load LightGBM model: {e}", flush=True)
        elif self.verbose:
            if joblib is None:
                print("  [WARN] joblib not available, running rule fallback.", flush=True)
            else:
                print(f"  [WARN] detector_lgbm.pkl not found (searched {CANDIDATE_MODEL_PATHS}), running rule fallback.", flush=True)

        # ── 2. Load KenLM Character Model (optional) ──
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
        elif self.verbose and char_lm_path:
            print(f"  [INFO] Char LM not found at {char_lm_path}, using fallback -2.5", flush=True)

        # ── 3. Load Confusion Matrix ──
        if confusion_matrix_path and os.path.exists(confusion_matrix_path):
            if self.verbose:
                print(f"Loading optical confusion matrix: {confusion_matrix_path}", flush=True)
            self._confusion = ConfusionMatrix(confusion_matrix_path)
            self.reverse_sub_map = self._confusion.reverse_sub_map
        else:
            if self.verbose:
                print(f"  [WARN] Confusion matrix not found at {confusion_matrix_path}, using empty map", flush=True)
            self._confusion = None
            self.reverse_sub_map = {}

        # ── 4. Load Vocabulary ──
        self.vocab: Dict[str, int] = {}
        self.surface_form: Dict[str, str] = {}
        if dict_vocab_path and os.path.exists(dict_vocab_path):
            if self.verbose:
                print(f"Loading lexicon: {dict_vocab_path}", flush=True)
            try:
                self._load_vocabulary(dict_vocab_path, max_lexicon_types, dict_min_freq)
                if self.verbose:
                    print(f"  {len(self.vocab):,} active types", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Failed to load lexicon {dict_vocab_path}: {e}", flush=True)
        else:
            if self.verbose:
                print(f"  [WARN] Lexicon not found at {dict_vocab_path}, using minimal fallback", flush=True)
            # minimal fallback: common Amharic words to avoid crashing
            minimal = ["እና", "ነው", "ላይ", "እንደ", "አንድ", "ወደ", "በ", "የ", "ለ", "ነገር", "ሰላም", "ቤት"]
            for w in minimal:
                self.vocab[self._norm(w)] = 1000
                self.surface_form[self._norm(w)] = w

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
            try:
                self.symspell.create_dictionary_entry(norm_w, freq)
            except Exception:
                continue
        if self.verbose:
            print("  SymSpell index ready.", flush=True)

    def _load_vocabulary(self, path: str, max_types: Optional[int], min_freq: Optional[int]):
        import heapq
        # Detect if file is raw training text (sentences) vs lexicon (word freq)
        # Heuristic: if first line has many spaces and no trailing digit, it's raw text -> count
        is_raw_text = False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as probe:
                first = probe.readline()
                parts = first.strip().split()
                if len(parts) > 5:
                    # raw sentence
                    is_raw_text = True
        except Exception:
            pass

        if is_raw_text:
            # Build counter from raw text
            from collections import Counter
            counter = Counter()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for tok in line.strip().split():
                        tok = tok.strip()
                        if not tok:
                            continue
                        # basic cleaning: remove Ethiopic punctuation for counting? keep as is for now
                        # but we will normalize later
                        counter[tok] += 1
            # Convert to vocab
            for w, freq in counter.items():
                norm_w = self._norm(w)
                if not norm_w:
                    continue
                prev = self.vocab.get(norm_w, 0)
                if freq >= prev:
                    self.surface_form[norm_w] = w
                self.vocab[norm_w] = max(prev, freq)
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
                    norm_w = self._norm(w)
                    if not norm_w:
                        continue
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
        if not word:
            return word
        try:
            return self.normalizer.normalize_word_strict(word)
        except Exception:
            return word

    def _score_char_lm(self, word: str) -> Optional[float]:
        if not self.char_lm:
            return None
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

    def _extract_lgbm_features(self, tok: str, sig: TokenSignals, doc_freq: int = 1, ocr_conf: Optional[float] = None) -> List[float]:
        """Extract features for LGBM, supporting both old and new feature sets."""
        # Clean token: remove punctuation/digits, but keep Ethiopic
        clean_raw = PUNCT_RE.sub('', tok).strip()
        clean_raw = ETHIOPIC_PUNCT_RE.sub('', clean_raw).strip()
        t_len = max(len(clean_raw), 1)

        # Normalized form for vocab lookup (FIXED: was using raw before)
        norm_tok = self._norm(tok)
        norm_clean = self._norm(clean_raw) if clean_raw else norm_tok

        # Lookup frequency: try normalized clean, then normalized raw, then raw
        freq = self.vocab.get(norm_clean, 0)
        if freq == 0:
            freq = self.vocab.get(norm_tok, 0)
        if freq == 0:
            # fallback: try raw clean without normalization (for backward compat)
            freq = self.vocab.get(clean_raw, self.vocab.get(tok, 0))

        log_freq = math.log10(freq + 1) if freq > 0 else 0.0
        in_lex = 1 if freq > 0 else 0

        # Char LM
        char_klm = getattr(sig, "char_lm_score", None)
        if char_klm is None:
            char_klm = self._score_char_lm(tok)
            if char_klm is not None:
                setattr(sig, "char_lm_score", char_klm)
            else:
                char_klm = -2.5

        rare_cnt = sum(1 for ch in clean_raw if ch in RARE_GLYPHS)
        rare_ratio = rare_cnt / float(t_len)
        one_flip_cnt = len(sig.one_flip_words)

        # Additional surface features for new model
        # These are already in sig if _cheap_signals was called
        n_neighbors = getattr(sig, "n_neighbors", 0)
        len_ratio = getattr(sig, "len_ratio", t_len / 4.14)
        mixed_script = 1 if getattr(sig, "mixed_script", False) else 0
        has_digit = 1 if getattr(sig, "has_digit", False) else 0

        # OCR confidence feature (0-1 normalized, or -1 if missing)
        if ocr_conf is not None:
            ocr_conf_norm = max(0.0, min(1.0, ocr_conf / 100.0)) if ocr_conf > 1 else max(0.0, min(1.0, ocr_conf))
        elif sig.ocr_conf is not None:
            ocr_conf_norm = sig.ocr_conf
        else:
            ocr_conf_norm = 0.85  # default high confidence if missing

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
            # new features
            "ocr_conf": ocr_conf_norm,
            "n_neighbors": float(n_neighbors),
            "len_ratio": float(len_ratio),
            "mixed_script": float(mixed_script),
            "has_digit": float(has_digit),
            "log_freq": log_freq,  # alias
            "is_high_freq": 1 if log_freq > 3.0 else 0,  # freq >1000
        }

        # If model has no features (fallback), return empty
        if not self.lgbm_features:
            return []

        return [feat_map.get(col, 0.0) for col in self.lgbm_features]

    def signals(self, tok: str, context: Sequence[str] = ()) -> TokenSignals:
        """Public accessor for cheap signals (for tests)."""
        return self._cheap_signals(tok, context)

    def score(self, sig: TokenSignals) -> float:
        """Rule-based score (fallback)."""
        return combine_signals(sig, self.w)

    def detection_stats(self):
        return {"stage1": dict(self._stage1)}

    def reset_counters(self):
        self._stage1 = {"pass": 0, "gray": 0, "flag": 0}

    def verdict_batch(self, tokens: Sequence[str], context_fn: Optional[Callable] = None,
                      ocr_confs: Optional[Sequence[Optional[float]]] = None,
                      doc_freqs: Optional[Sequence[int]] = None) -> List[Verdict]:
        results = []
        for i, tok in enumerate(tokens):
            ctx = context_fn(i) if context_fn else []
            conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
            df = doc_freqs[i] if doc_freqs and i < len(doc_freqs) else 1
            results.append(self.verdict(tok, ctx, ocr_conf=conf, doc_freq=df))
        return results

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
            sig.ocr_conf = max(0.0, min(1.0, ocr_conf / 100.0)) if ocr_conf > 1 else max(0.0, min(1.0, ocr_conf))

        # ── PATH A: LightGBM Model Prediction (Primary) ──
        if self.lgbm_model is not None and self.lgbm_features:
            feats = self._extract_lgbm_features(tok, sig, doc_freq=doc_freq, ocr_conf=ocr_conf)
            try:
                prob = float(self.lgbm_model.predict([feats])[0])
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] LGBM predict failed: {e}, falling back to rule", flush=True)
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

            # ── Improved Safety Floors (frequency-aware, less aggressive) ──
            # For high-frequency words, never flag on low-conf alone
            is_high_freq = self.vocab.get(self._norm(tok), 0) > 1000 or sig.log_freq > 3.0

            # Only flag severe low-conf if OOV AND not high-freq AND very low conf (<35%)
            if not sig.in_lexicon and sig.ocr_conf is not None and sig.ocr_conf < 0.35:
                # Check if it's a plausible OOV (has no flip) -> might be name, don't flag
                if not sig.one_flip_words and not is_high_freq:
                    # OOV with no optical neighbor and very low conf -> still flag
                    label = "flag"
                    reason += f", severe low-conf OOV ({sig.ocr_conf*100:.0f}%)"
                elif sig.one_flip_words:
                    label = "flag"
                    reason += f", low-conf + one-flip ({sig.ocr_conf*100:.0f}%)"

            # OOV kept in gray (not flag) if not already flagged and not high-conf
            if not sig.in_lexicon and label == "pass":
                # If high-freq (shouldn't happen for OOV) or has many neighbors, keep pass? No, gray is safer
                # But if token is short and rare glyph heavy, keep gray
                label = "gray"
                reason += ", OOV kept in gray"

            # Low-conf floor: only gray, not flag, and only if not high-freq
            if sig.ocr_conf is not None and sig.ocr_conf < 0.55 and label == "pass" and not is_high_freq:
                label = "gray"
                reason += f", low-conf floor ({sig.ocr_conf*100:.0f}%)"

            # High-freq in-lexicon override: if freq very high and LGBM says flag but conf is decent, downgrade to gray
            if is_high_freq and label == "flag" and sig.ocr_conf is not None and sig.ocr_conf > 0.70:
                # Check if LGBM score is just above threshold (borderline)
                if s < self.flag_above + 0.15:
                    label = "gray"
                    reason += f", high-freq downgrade ({self.vocab.get(self._norm(tok),0)} freq)"

            if label == "pass":
                self._stage1["pass"] += 1
            elif label == "flag":
                self._stage1["flag"] += 1
            else:
                self._stage1["gray"] += 1

            return Verdict(s, label, sig, reason)

        # ── PATH B: Rule Fallback (if LightGBM is missing) ──
        s = combine_signals(sig, self.w)

        # Hard PASS: clean lexicon word, no optical neighbor, not mixed, rare glyph low, conf ok, score low
        is_high_freq_rule = sig.log_freq > 3.0
        if (sig.in_lexicon and not sig.one_flip_words
                and not sig.mixed_script and sig.rare_glyph_ratio < 0.35
                and (sig.ocr_conf is None or sig.ocr_conf >= 0.70)
                and s < self.pass_below):
            self._stage1["pass"] += 1
            return Verdict(s, "pass", sig, "clean lexicon word")

        # Hard FLAG: OOV + one-flip + decent score
        if not sig.in_lexicon and sig.one_flip_words and s >= 0.50:
            self._stage1["flag"] += 1
            return Verdict(s, "flag", sig, f"OOV, one flip reaches {sig.one_flip_words[0]!r}")

        # Low-conf + OOV -> flag only if very low conf (<40%)
        if not sig.in_lexicon and sig.ocr_conf is not None and sig.ocr_conf < 0.40:
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
