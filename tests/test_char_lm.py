"""Character-level LM wiring: spacing, loading, and the Stage 1 signal.

Runs with a fake LM, so it needs neither kenlm nor the real model file.
Run: python -m unittest discover -s tests -p 'test_char_lm.py'
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # repo root

from test_head_contract import CONFUSION, FakeLM, State  # noqa: E402  (tests/ on sys.path)
from amharic_ocr_corrector import AmharicOCRCorrector  # noqa: E402


class SpyCharLM:
    """Records every string scored; returns a fixed log10 score per call."""

    def __init__(self, value=-2.0):
        self.value = value
        self.seen = []

    def score(self, sentence, bos=True, eos=True):
        self.seen.append(sentence)
        return self.value


class CharLMBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lex = os.path.join(self.tmp, "lex.txt")
        Path(self.lex).write_text("ሰላም 900\nቤት 800\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, **kw):
        opts = dict(kenlm_path="unused", confusion_matrix_path=CONFUSION,
                    dict_vocab_path=self.lex, lm=FakeLM(), state_factory=State,
                    segmenter=None, verbose=False, stage1_detection=True)
        opts.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return AmharicOCRCorrector(**opts)


class TestSpacing(CharLMBase):
    """The char-LM column: spaced characters, score per character.

    The conf-free model takes the char LM as a FEATURE column
    (features.FeatureContext.char_lm_score -> char_kenlm_score) that the
    training script and the detector share. The Stage-1 *signal* attribute
    (sig.char_lm) only exists in the deprecated rule-based fallback, so it is
    not what these tests can pin down any more. KenLM is off for the shipped
    model ("KenLM OFF (gain 0)"), which is why this column is usually NaN.
    """

    def _ctx(self, spy):
        from OCR_error_detection.features import FeatureContext
        return FeatureContext(vocab={}, norm=lambda w: w, reverse_sub_map={},
                              char_lm=spy, symspell=None)

    def test_characters_are_spaced_before_scoring(self):
        from OCR_error_detection.features import static_features
        spy = SpyCharLM()
        static_features("ትምህርት", self._ctx(spy))
        self.assertIn("ት ም ህ ር ት", spy.seen)

    def test_score_is_per_character(self):
        from OCR_error_detection.features import static_features
        spy = SpyCharLM(value=-6.0)
        feats = static_features("ትምህርት", self._ctx(spy))
        # bos + eos are scored as well, so the total is divided by len + 1.
        self.assertAlmostEqual(feats["char_kenlm_score"],
                               -6.0 / (len("ትምህርት") + 1))

    def test_missing_char_lm_column_is_nan(self):
        from OCR_error_detection.features import static_features
        feats = static_features("ትምህርት", self._ctx(None))
        self.assertNotEqual(feats["char_kenlm_score"], feats["char_kenlm_score"])


class TestLoading(CharLMBase):
    def test_injected_model_takes_precedence_over_path(self):
        spy = SpyCharLM()
        # The path does not exist; if it were loaded this would raise.
        c = self.make(char_lm=spy, char_lm_path="/no/such/model.klm")
        self.assertIs(c.char_lm, spy)

    def test_path_is_loaded_when_no_model_injected(self):
        import sys
        import types
        loaded = []

        class FakeKenLM:
            class Model:
                def __init__(self, path):
                    loaded.append(path)

                def score(self, s, bos=True, eos=True):
                    return -1.0

                def BeginSentenceWrite(self, st):
                    pass

                def BaseScore(self, a, w, b):
                    return 0.0

        fake = types.ModuleType("kenlm")
        fake.Model = FakeKenLM.Model
        fake.State = State
        import amharic_ocr_corrector as mod
        with unittest.mock.patch.object(mod, "_load_kenlm", return_value=fake):
            c = self.make(lm=FakeLM(), char_lm_path="/x/amharic_char_model.klm")
        self.assertEqual(loaded, ["/x/amharic_char_model.klm"])
        self.assertIsNotNone(c.char_lm)

    def test_no_path_and_no_model_means_no_char_lm(self):
        c = self.make()
        self.assertIsNone(c.char_lm)
        sig = c.suspicion.signals("ሰላም", [])
        self.assertIsNone(sig.char_lm)

    def test_missing_kenlm_with_a_path_fails_loudly(self):
        import amharic_ocr_corrector as mod
        def boom():
            raise ImportError("kenlm not installed")
        with unittest.mock.patch.object(mod, "_load_kenlm", side_effect=boom):
            with self.assertRaises(ImportError):
                self.make(lm=FakeLM(), char_lm_path="/x/model.klm")


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
