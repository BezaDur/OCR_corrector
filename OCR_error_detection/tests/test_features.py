#!/usr/bin/env python3
"""
Step 1 contract tests: one feature definition for training and inference.

Run:  python -m unittest OCR_error_detection.tests.test_features -v
"""
import contextlib
import importlib.util
import io
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parents[1]
CONFUSION = str(_HERE / "Model" / "confusion_matrix.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.features import (
    FEATURE_NAMES, FeatureContext, LEGACY_IMPUTE, apply_legacy_imputation,
    cheap_signals, clean_surface, extract_frame, static_features,
    to_vector, with_context,
)
from OCR_error_detection.normalizer import AmharicNormalizer

NORM = AmharicNormalizer.normalize_word_strict
LEXICON = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000), ("ዓለም", 9000), ("ሀገር", 8000)]


def _lexicon_file(words):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "words.txt")
    with open(p, "w", encoding="utf-8") as fh:
        for w, f in words:
            fh.write(f"{w} {f}\n")
    return d, p


def _engine(words=LEXICON, char_lm_path=None):
    from OCR_error_detection.detector import DetectionEngine
    d, lex = _lexicon_file(words)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            eng = DetectionEngine(dict_vocab_path=lex, confusion_matrix_path=CONFUSION,
                                  char_lm_path=char_lm_path, lgbm_model_path=None,
                                  verbose=False)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return eng


def _schema2(eng):
    """Give an engine the current (schema 2) feature set, independent of any pickle on disk."""
    eng._legacy_model = False
    eng.lgbm_features = list(FEATURE_NAMES)
    return eng


def _context_from_vocab(vocab):
    confusion = ConfusionMatrix(CONFUSION)
    return FeatureContext(vocab=vocab, norm=NORM,
                          reverse_sub_map=confusion.reverse_sub_map, char_lm=None)


class TestConfusionKey(unittest.TestCase):
    def test_reverse_map_is_built_from_sub_table(self):
        """Regression: training read 'substitutions' and got an empty map."""
        rev = ConfusionMatrix(CONFUSION).reverse_sub_map
        self.assertGreater(len(rev), 0)

    def test_one_flip_fires_on_a_real_neighbour(self):
        # 'ህገር' differs from 'ሀገር' by one optical flip (ህ <-> ሀ family).
        eng = _engine()
        sig = cheap_signals("ህገር", eng._ctx)
        self.assertTrue(sig.one_flip_words, "expected at least one one-flip hit")
        # the hit may be spelled with a variant glyph; it must reach the lexicon key
        self.assertTrue(any(NORM(c) == "ሀገር" for c in sig.one_flip_words), sig.one_flip_words)


class TestCanonicalLookups(unittest.TestCase):
    def test_surface_with_variant_glyph_is_in_lexicon(self):
        """Regression: 'ዓለም' is stored as 'አለም' in the vocabulary."""
        eng = _engine()
        f = static_features("ዓለም", eng._ctx)
        self.assertEqual(f["in_lexicon"], 1.0)
        self.assertGreater(f["log_corpus_freq"], 0.0)

    def test_h_family_variant_is_in_lexicon(self):
        eng = _engine()
        self.assertEqual(static_features("ሐገር", eng._ctx)["in_lexicon"], 1.0)

    def test_trailing_punctuation_does_not_remove_lexicon_membership(self):
        """Regression: 'ቤት፡' was out-of-lexicon, which triggered the OOV floor."""
        eng = _engine()
        for tok in ["ቤት፡", "ቤት።", "ቤት,", "ቤት፣"]:
            sig = cheap_signals(tok, eng._ctx)
            self.assertTrue(sig.in_lexicon, tok)
            self.assertEqual(static_features(tok, eng._ctx, sig),
                             static_features("ቤት", eng._ctx), tok)

    def test_clean_surface(self):
        self.assertEqual(clean_surface("ቤት፡"), "ቤት")
        self.assertEqual(clean_surface("12"), "")


class TestMissingContextIsNaN(unittest.TestCase):
    def test_no_char_lm_gives_nan_not_magic_number(self):
        eng = _engine()
        self.assertIsNone(eng.char_lm)
        f = static_features("ቤት", eng._ctx)
        self.assertTrue(math.isnan(f["char_kenlm_score"]))

    def test_isolated_word_has_nan_doc_freq(self):
        f = with_context(static_features("ቤት", _engine()._ctx), None)
        self.assertTrue(math.isnan(f["doc_freq"]))
        self.assertTrue(math.isnan(f["is_doc_singleton"]))

    def test_page_word_has_real_doc_freq(self):
        f = with_context({}, 1)
        self.assertEqual(f["doc_freq"], 1.0)
        self.assertEqual(f["is_doc_singleton"], 1.0)
        f = with_context({}, 5)
        self.assertEqual(f["is_doc_singleton"], 0.0)

    def test_verdict_without_context_matches_explicit_none(self):
        eng = _schema2(_engine())
        vec_none = eng._extract_lgbm_features("ቤት", eng._cheap_signals("ቤት"), doc_freq=None)
        names = eng.lgbm_features
        self.assertTrue(math.isnan(vec_none[names.index("doc_freq")]))

    def test_to_vector_rejects_unknown_feature(self):
        with self.assertRaises(ValueError):
            to_vector({"x": 1.0}, ["x"])


class TestLegacyModels(unittest.TestCase):
    def test_legacy_imputation_uses_training_constants(self):
        feats = {"doc_freq": math.nan, "is_doc_singleton": math.nan, "char_kenlm_score": math.nan,
                 "in_lexicon": 1.0}
        out = apply_legacy_imputation(feats)
        self.assertEqual(out["doc_freq"], LEGACY_IMPUTE["doc_freq"])
        self.assertEqual(out["is_doc_singleton"], 1.0)
        self.assertEqual(out["char_kenlm_score"], -2.5)
        self.assertEqual(out["in_lexicon"], 1.0)

    def test_legacy_flag_set_for_schema_1_pickle(self):
        """The shipped pickle is schema 1; the engine must mark it legacy."""
        from OCR_error_detection.detector import CANDIDATE_MODEL_PATHS
        pkl = next((p for p in CANDIDATE_MODEL_PATHS if os.path.exists(p)), None)
        if pkl is None:
            self.skipTest("no shipped model")
        try:
            import joblib  # noqa: F401
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm/joblib not installed")
        from OCR_error_detection.detector import DetectionEngine
        d, lex = _lexicon_file(LEXICON)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                eng = DetectionEngine(dict_vocab_path=lex, confusion_matrix_path=CONFUSION,
                                      char_lm_path=None, lgbm_model_path=pkl, verbose=False)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if eng.lgbm_model is None:
            self.skipTest("model did not load")
        self.assertTrue(eng._legacy_model)


class TestTrainServeParity(unittest.TestCase):
    """The same word and page context must give the same vector in training and inference."""

    WORDS = ["ሰላም", "ቤት፡", "ዓለም", "ሐገር", "ህገር", "ነገር", "xyz", "ሰላም፣", "1234", ""]

    # has_digit is the one column that reads the raw token (cleaning removes
    # digits, so the clean surface can never answer it). For a punctuated or
    # digit-bearing token it is therefore expected to differ from the clean
    # form's vector; every other column must match exactly.
    RAW_ONLY_DIFF = {"has_digit"}

    def test_extract_frame_matches_detector(self):
        eng = _schema2(_engine())
        names = list(eng.lgbm_features)
        pairs = [(w, df) for w in self.WORDS for df in (None, 1, 4)]
        # training path: words are clean in the corpus, so give it the clean form
        training_pairs = [(clean_surface(w), df) for w, df in pairs]
        train_rows = extract_frame(training_pairs, eng._ctx)
        for (w, df), train_row in zip(pairs, train_rows):
            serve_row = eng._extract_lgbm_features(w, eng._cheap_signals(w), doc_freq=df)
            dirty = clean_surface(w) != w
            for name, a, b in zip(names, train_row, serve_row):
                if dirty and name in self.RAW_ONLY_DIFF:
                    continue
                if math.isnan(a):
                    self.assertTrue(math.isnan(b), (w, df, name))
                else:
                    self.assertAlmostEqual(a, b, places=9, msg=(w, df, name))

    def test_has_digit_reads_the_raw_token(self):
        """'12' cleans to '' but must still report a digit; 'ቤት፡' must not."""
        eng = _schema2(_engine())
        names = list(eng.lgbm_features)
        digit = dict(zip(names, eng._extract_lgbm_features("12", None, doc_freq=1)))
        self.assertEqual(digit["has_digit"], 1.0)
        self.assertEqual(digit["token_len"], 0.0)  # stripped by clean_surface
        bare = dict(zip(names, eng._extract_lgbm_features("ቤት", None, doc_freq=1)))
        punct = dict(zip(names, eng._extract_lgbm_features("ቤት፡", None, doc_freq=1)))
        self.assertEqual(bare, punct)

    def test_training_vocab_normalization_matches_detector(self):
        """The training script's normalize_vocab must equal the detector's vocabulary."""
        spec = importlib.util.find_spec("lightgbm")
        if spec is None:
            self.skipTest("lightgbm not installed")
        script = ROOT / "OCR_error_detection" / "Model" / "train_LightGBM.py"
        mod_spec = importlib.util.spec_from_file_location("train_lgbm_under_test", script)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)

        raw = {"ዓለም": 9000, "አለም": 12, "ሀገር": 8000, "ሐገር": 30, "ቤት": 4000}
        d = tempfile.mkdtemp()
        p = os.path.join(d, "raw.txt")
        with open(p, "w", encoding="utf-8") as fh:
            for w, f in raw.items():
                fh.write(f"{w} {f}\n")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                eng = _engine(words=[(w, f) for w, f in raw.items()])
                train_vocab = mod.normalize_vocab(mod.load_master_dict(p), NORM)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(train_vocab, eng.vocab)


if __name__ == "__main__":
    unittest.main()
