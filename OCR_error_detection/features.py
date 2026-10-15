# features.py — the single definition of the feature vector.
#
# Training (Model/train_LightGBM.py) and inference (detector.py) both call
# this module, for every column the README's model has: the 14 conf-free
# features. Keeping one definition is what prevents train/serve skew.
#
# Canonical ingestion rules (applied to every token before any lookup):
#   1. strip punctuation and digits  -> `clean_surface`
#   2. lexicon / frequency lookups use the NORMALIZED key (`ctx.norm`) of the
#      clean surface, because the vocabulary is keyed by normalized forms
#   3. token_len / len_ratio / rare_glyph_ratio / mixed_script are measured on
#      the clean surface (the same string the lookups use, so a punctuated
#      token and its bare form get identical features)
#   4. has_digit is measured on the CLEAN surface, like every other column.
#      It used to read the raw token, on the reasoning that cleaning removes
#      digits so the clean surface "can never answer" the question. But
#      clean_surface only removes ASCII digits: Ethiopic numerals U+1369-U+1371
#      are category No, so neither [^\w\s] nor \d strips them. The training
#      corpus (Model/harmonized_corpus_jiwer.tsv) contains no ASCII digits and
#      no punctuation at all — clean_surface(w) == w for 946,057/946,057 rows —
#      so its only has_digit=1 rows (812, 0.086%) are Ethiopic-numeral words
#      such as "መሆ፭ም" and "፬ዎቹ", where the numeral is part of the surface the
#      other columns describe. Serving saw a different population: real OCR
#      tokens carry ASCII digits ("ከ20", "በ5ሺህ", "“3ኛውን"), which cleaning
#      removes from every other column while has_digit stayed 1, dropping them
#      into a branch the trees learned from Ethiopic numerals. Result:
#      P(error) 0.012 -> 0.828 for "ከ20" vs its clean form "ከ", i.e. FLAG on a
#      clean dictionary word. 420 of the 62,479 held-out Evaluation/ tokens
#      (0.67%) carry an ASCII digit, and they accounted for essentially all of
#      the excess false flags: with a local_lexicon f>=10 model, FF 0.730% ->
#      0.081% (424 -> 47 flags); with the shipped model, 0.72% -> 0.10%
#      (419 -> 59). Reading the clean surface is bit-identical on all training
#      rows, so no retraining is needed and no shipped model changes behaviour
#      on any input it was trained on.
#   5. missing context is NaN, never a magic number:
#      doc_freq / is_doc_singleton when no page is given,
#      char_kenlm_score when no KenLM model is loaded or the word has < 2 chars,
#      n_neighbors when SymSpell is off (the documented default: gain 0)
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from OCR_error_detection.signals import (
    MEAN_TOKEN_LEN,
    TokenSignals,
    compute_lexical_signals,
    compute_one_flip_signals,
    compute_surface_signals,
)
from OCR_error_detection.utils import MODEL_RARE_GLYPHS

# Schema history
#   1  pre-features.py pickles: 14 loose columns, missing values filled with
#      magic numbers before the model saw them (superseded, still servable)
#   2  features.py first cut: 9 shared columns, NaN for missing values
#   3  this file: the full 14-column conf-free vector, one definition for
#      training and inference
FEATURE_SCHEMA_VERSION = 3

# The first schema that uses NaN for missing values. Anything older needs
# explicit imputation (apply_legacy_imputation) because LightGBM treats an
# unseen NaN as a value the tree never trained on.
NAN_SCHEMA_VERSION = 2

# Column order of the shipped final model (Model/detector_lgbm_conffree.pkl)
# and of every model this repo trains from now on.
FEATURE_NAMES = (
    "log_corpus_freq",
    "in_lexicon",
    "doc_freq",
    "is_doc_singleton",
    "token_len",
    "len_ratio",
    "has_one_flip",
    "one_flip_count",
    "char_kenlm_score",
    "rare_glyph_ratio",
    "n_neighbors",
    "mixed_script",
    "has_digit",
    "is_high_freq",
)

# The 9 columns schema 2 defined. Kept so a schema-2 pickle can still be
# described and served without a code change.
SCHEMA2_FEATURE_NAMES = (
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

# Every column this module can produce, for any schema. ocr_conf is legacy:
# the model is conf-free (README: confidence stays out of training), but
# older pickles that still carry a confidence column stay servable.
KNOWN_FEATURE_NAMES = tuple(dict.fromkeys(FEATURE_NAMES + SCHEMA2_FEATURE_NAMES + ("ocr_conf",)))

# Schema-1 models (trained before this module existed) were fit on NaN-free
# data. LightGBM treats an unseen NaN as 0, so legacy models need explicit
# imputation with the values they were trained under. The detector applies
# this only when the payload says the model is older than NAN_SCHEMA_VERSION.
LEGACY_IMPUTE = {
    "doc_freq": 1.0,
    "is_doc_singleton": 1.0,
    "char_kenlm_score": -2.5,
}

PUNCT_RE = re.compile(r"[^\w\s]|\d")

NAN = float("nan")

# is_high_freq is a coarse bucket of log10(freq+1); 3.0 means freq >= 1000.
HIGH_FREQ_LOG = 3.0


class FeatureError(ValueError):
    """A model asked for a column this code cannot compute."""


def clean_surface(tok: str) -> str:
    """Punctuation- and digit-free surface form. Empty string if nothing is left."""
    return PUNCT_RE.sub("", tok or "").strip()


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


@dataclass
class FeatureContext:
    """Shared resources a feature computation needs.

    vocab        normalized key -> frequency (max over raw surface forms)
    norm         the canonical normalizer (AmharicNormalizer.normalize_word_strict)
    reverse_sub_map  observed char -> [(true char, cost)], from ConfusionMatrix
    char_lm      KenLM model or None
    symspell     optional; supplies n_neighbors when enabled (off by default)
    """

    vocab: Mapping[str, int]
    norm: Callable[[str], str]
    reverse_sub_map: Mapping[str, Any]
    max_one_flip: int = 8
    char_lm: Any = None
    symspell: Any = None
    verbosity_top: int = 2

    def char_lm_score(self, clean: str) -> Optional[float]:
        """Average log10 probability per character on the clean surface.

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

    def neighbor_count(self, clean: str) -> Optional[float]:
        """Dictionary neighbours within edit distance 1.

        SymSpell is off by default (README: "SymSpell OFF (gain 0)"), and the
        model is trained and served with this column at 0 in that mode. So
        "off" returns 0.0 — the value the model learned — not NaN, which would
        be a branch the trees never saw. NaN is reserved for a lookup that was
        attempted and failed.
        """
        if self.symspell is None:
            return 0.0
        try:
            sugg = self.symspell.lookup(clean, self.verbosity_top,
                                        max_edit_distance=1, include_unknown=False)
            return float(len(sugg))
        except Exception:
            return None


def cheap_signals(tok: str, ctx: FeatureContext) -> TokenSignals:
    """Signals computed on the canonical (punctuation-free) surface.

    Both the detector and the training pipeline use this, so membership and
    one-flip results are identical in both places. Every column, `has_digit`
    included, is measured on the clean surface — see rule 4 in the module
    docstring for why reading the raw token there was a train/serve skew.
    """
    clean = clean_surface(tok)
    sig = TokenSignals(token=tok)
    # compute_surface_signals already derives has_digit from `clean`.
    compute_surface_signals(clean, sig)
    compute_lexical_signals(clean, sig, ctx.vocab, ctx.norm,
                            ctx.symspell, ctx.verbosity_top)
    compute_one_flip_signals(clean, sig, ctx.vocab, ctx.norm,
                             ctx.reverse_sub_map, ctx.max_one_flip)
    return sig


def static_features(tok: str, ctx: FeatureContext,
                    sig: Optional[TokenSignals] = None) -> Dict[str, float]:
    """Every column that does not depend on page context (12 of the 14)."""
    clean = clean_surface(tok)
    if sig is None:
        sig = cheap_signals(tok, ctx)

    key = ctx.norm(clean) if clean else ""
    in_vocab = bool(key) and key in ctx.vocab
    freq = ctx.vocab[key] if in_vocab else 0
    log_freq = math.log10(freq + 1.0) if in_vocab else 0.0

    n_flip = len(sig.one_flip_words)
    char_klm = ctx.char_lm_score(clean)
    rare = sum(1 for ch in clean if ch in MODEL_RARE_GLYPHS)
    ethiopic = sum(1 for ch in clean if 0x1200 <= ord(ch) < 0x1380)
    neighbors = ctx.neighbor_count(clean)
    length = len(clean)

    return {
        "log_corpus_freq": log_freq,
        "in_lexicon": 1.0 if in_vocab else 0.0,
        "token_len": float(length),
        "len_ratio": length / MEAN_TOKEN_LEN,
        "has_one_flip": 1.0 if n_flip > 0 else 0.0,
        "one_flip_count": float(n_flip),
        "char_kenlm_score": NAN if char_klm is None else float(char_klm),
        "rare_glyph_ratio": rare / float(length) if length else 0.0,
        "n_neighbors": NAN if neighbors is None else float(neighbors),
        "mixed_script": 1.0 if 0 < ethiopic < length else 0.0,
        "has_digit": 1.0 if getattr(sig, "has_digit", False) else 0.0,
        "is_high_freq": 1.0 if log_freq > HIGH_FREQ_LOG else 0.0,
    }


def with_context(feats: Mapping[str, float], doc_freq: Optional[float]) -> Dict[str, float]:
    """Add page-context features.

    `doc_freq=None` (or NaN, which is how the corrector passes "no page") means
    "unknown" -> NaN. Both context columns move together so the model never
    sees a word that is simultaneously "not a singleton" and "unknown".
    """
    out = dict(feats)
    if doc_freq is None or _is_nan(doc_freq):
        out["doc_freq"] = NAN
        out["is_doc_singleton"] = NAN
    else:
        out["doc_freq"] = float(doc_freq)
        out["is_doc_singleton"] = 1.0 if float(doc_freq) == 1 else 0.0
    return out


def apply_legacy_imputation(feats: Mapping[str, float]) -> Dict[str, float]:
    """For schema-1 models only: replace NaN with the values they were trained under."""
    out = dict(feats)
    for name, value in LEGACY_IMPUTE.items():
        if name in out and _is_nan(out[name]):
            out[name] = value
    return out


def missing_features(names: Sequence[str]) -> List[str]:
    """Columns a model asks for that this module cannot compute."""
    return [n for n in names if n not in KNOWN_FEATURE_NAMES]


def to_vector(feats: Mapping[str, float], names: Sequence[str]) -> List[float]:
    unknown = missing_features(names)
    if unknown:
        raise FeatureError(
            "model expects features this code does not define: %s" % ", ".join(unknown))
    return [float(feats[n]) for n in names]


def feature_dict(tok: str, ctx: FeatureContext,
                 doc_freq: Optional[float] = None,
                 ocr_conf: Optional[float] = None) -> Dict[str, float]:
    """Full 14-column feature map for one token — training and serving path."""
    feats = with_context(static_features(tok, ctx), doc_freq)
    # Legacy confidence column, 0-1. Kept only for older pickles that still
    # carry it; the shipped model is conf-free so this is not in FEATURE_NAMES.
    if ocr_conf is not None and not _is_nan(ocr_conf):
        conf = float(ocr_conf)
        if conf > 1.0:
            conf = conf / 100.0
        feats["ocr_conf"] = max(0.0, min(1.0, conf))
    else:
        feats["ocr_conf"] = NAN
    return feats


def extract_vector(tok: str, ctx: FeatureContext, names: Sequence[str],
                   doc_freq: Optional[float] = None,
                   ocr_conf: Optional[float] = None,
                   legacy_impute: bool = False) -> List[float]:
    """One token -> the vector the model expects, in the model's own order."""
    feats = feature_dict(tok, ctx, doc_freq=doc_freq, ocr_conf=ocr_conf)
    if legacy_impute:
        feats = apply_legacy_imputation(feats)
    return to_vector(feats, names)


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
