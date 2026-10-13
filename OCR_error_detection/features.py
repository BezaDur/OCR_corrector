# features.py — the single definition of the LightGBM feature vector.
#
# Training (Model/train_LightGBM.py) and inference (detector.py) both call
# this module. Keeping one definition is what prevents train/serve skew.
#
# Canonical ingestion rules (applied to every token before any lookup):
#   1. strip punctuation and digits  -> `clean_surface`
#   2. lexicon / frequency lookups use the NORMALIZED key (`ctx.norm`),
#      because the vocabulary is keyed by normalized forms
#   3. the char LM and rare-glyph ratio use the RAW (punctuation-free)
#      surface, because H-family / AIN collapses are the optical signal
#      these features exist to capture
#   4. missing context is NaN, never a magic number:
#      doc_freq / is_doc_singleton when no page is given,
#      char_kenlm_score when no KenLM model is loaded or the word has < 2 chars
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from OCR_error_detection.signals import (
    TokenSignals,
    compute_surface_signals,
    compute_lexical_signals,
    compute_one_flip_signals,
)

FEATURE_SCHEMA_VERSION = 2

FEATURE_NAMES = (
    "log_corpus_freq",
    "in_lexicon",
    "doc_freq",
    "is_doc_singleton",
    "token_len",
    "has_one_flip",
    "one_flip_count",
    "char_kenlm_score",
    "rare_glyph_ratio",
)

# Schema-1 models (trained before this module existed) were fit on
# NaN-free data. LightGBM treats an unseen NaN as 0, so legacy models need
# explicit imputation with the values they were trained under.
LEGACY_IMPUTE = {
    "doc_freq": 1.0,
    "is_doc_singleton": 1.0,
    "char_kenlm_score": -2.5,
}

PUNCT_RE = re.compile(r"[^\w\s]|\d")
RARE_GLYPHS = frozenset("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")

NAN = float("nan")


def clean_surface(tok: str) -> str:
    """Punctuation- and digit-free surface form. Empty string if nothing is left."""
    return PUNCT_RE.sub("", tok or "").strip()


@dataclass
class FeatureContext:
    """Shared resources a feature computation needs.

    vocab        normalized key -> frequency (max over raw surface forms)
    norm         the canonical normalizer (AmharicNormalizer.normalize_word_strict)
    reverse_sub_map  observed char -> [(true char, cost)], from ConfusionMatrix
    char_lm      KenLM model or None
    symspell     optional; only used by the rule path's neighbour count
    """

    vocab: Mapping[str, int]
    norm: Callable[[str], str]
    reverse_sub_map: Mapping[str, Any]
    max_one_flip: int = 8
    char_lm: Any = None
    symspell: Any = None
    verbosity_top: int = 2

    def char_lm_score(self, clean: str) -> Optional[float]:
        """Average log10 probability per character on the RAW clean surface.

        Returns None (-> NaN feature) when no LM is loaded or the word is too
        short to score.
        """
        if self.char_lm is None or len(clean) < 2:
            return None
        try:
            raw = self.char_lm.score(" ".join(clean), bos=True, eos=True)
        except Exception:
            return None
        return raw / (len(clean) + 1.0)


def cheap_signals(tok: str, ctx: FeatureContext) -> TokenSignals:
    """Signals computed on the canonical (punctuation-free) surface.

    Both the detector and the training pipeline use this, so membership and
    one-flip results are identical in both places.
    """
    clean = clean_surface(tok)
    sig = TokenSignals(token=tok)
    compute_surface_signals(clean, sig)
    sig.has_digit = any(ch.isdigit() for ch in (tok or ""))
    compute_lexical_signals(clean, sig, ctx.vocab, ctx.norm,
                            ctx.symspell, ctx.verbosity_top)
    compute_one_flip_signals(clean, sig, ctx.vocab, ctx.norm,
                             ctx.reverse_sub_map, ctx.max_one_flip)
    return sig


def static_features(tok: str, ctx: FeatureContext,
                    sig: Optional[TokenSignals] = None) -> Dict[str, float]:
    """Features that depend only on the word (no page context)."""
    clean = clean_surface(tok)
    if sig is None:
        sig = cheap_signals(tok, ctx)

    key = ctx.norm(clean) if clean else ""
    in_vocab = bool(key) and key in ctx.vocab
    freq = ctx.vocab[key] if in_vocab else 0

    n_flip = len(sig.one_flip_words)
    char_klm = ctx.char_lm_score(clean)
    rare = sum(1 for ch in clean if ch in RARE_GLYPHS)

    return {
        "log_corpus_freq": math.log10(freq + 1.0) if in_vocab else 0.0,
        "in_lexicon": 1.0 if in_vocab else 0.0,
        "token_len": float(len(clean)),
        "has_one_flip": 1.0 if n_flip > 0 else 0.0,
        "one_flip_count": float(n_flip),
        "char_kenlm_score": NAN if char_klm is None else float(char_klm),
        "rare_glyph_ratio": rare / float(len(clean)) if clean else 0.0,
    }


def with_context(feats: Mapping[str, float], doc_freq: Optional[int]) -> Dict[str, float]:
    """Add page-context features. doc_freq=None means 'no page context' -> NaN."""
    out = dict(feats)
    if doc_freq is None:
        out["doc_freq"] = NAN
        out["is_doc_singleton"] = NAN
    else:
        out["doc_freq"] = float(doc_freq)
        out["is_doc_singleton"] = 1.0 if doc_freq == 1 else 0.0
    return out


def apply_legacy_imputation(feats: Mapping[str, float]) -> Dict[str, float]:
    """For schema-1 models only: replace NaN with the values they were trained under."""
    out = dict(feats)
    for name, value in LEGACY_IMPUTE.items():
        if name in out and math.isnan(out[name]):
            out[name] = value
    return out


def to_vector(feats: Mapping[str, float], names: Sequence[str]) -> List[float]:
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"model expects features this code does not define: {unknown}")
    return [float(feats[n]) for n in names]


def extract_frame(words_and_doc_freq, ctx: FeatureContext,
                  names: Sequence[str] = FEATURE_NAMES) -> List[List[float]]:
    """Batch version for training: (word, doc_freq) pairs -> feature rows.

    Word-level features are cached per distinct surface form.
    """
    cache: Dict[str, Dict[str, float]] = {}
    rows: List[List[float]] = []
    for word, doc_freq in words_and_doc_freq:
        base = cache.get(word)
        if base is None:
            base = static_features(word, ctx)
            cache[word] = base
        rows.append(to_vector(with_context(base, doc_freq), names))
    return rows
