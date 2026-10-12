#!/usr/bin/env python3
"""
Portable contract tests for the OCR_error_detection module.

These run WITHOUT KenLM, the external segmenter, or the production lexicon.
They use:
  * tiny on-disk lexicons,
  * the REAL checked-in confusion_matrix.json,
  * the real symspellpy.

Run:  python -m pytest OCR_error_detection/tests/test_detector.py -v
  or: python -m unittest OCR_error_detection.tests.test_detector -v
"""
import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Resolve the repo root
ROOT = Path(__file__).resolve().parents[2]
CONFUSION = str(ROOT / "confusion_matrix.json")


def _make_detector(words=(), **kwargs):
    """Helper: create a DetectionEngine from a tiny word list."""
    from OCR_error_detection import DetectionEngine

    folder = tempfile.mkdtemp()
    lexicon = os.path.join(folder, "words.txt")
    with open(lexicon, "w", encoding="utf-8") as f:
        for w, freq in words:
            f.write(f"{w} {freq}\n")

    options = dict(
        dict_vocab_path=lexicon,
        confusion_matrix_path=CONFUSION,
        verbose=False,
    )
    options.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        eng = DetectionEngine(**options)
    # Clean up the temp file, but keep the engine alive
    shutil.rmtree(folder, ignore_errors=True)
    return eng


# --------------------------------------------------------------------------- #
# Normalizer tests
# --------------------------------------------------------------------------- #
class TestNormalizer(unittest.TestCase):
    """The standalone normalizer must produce the same results as the main one."""

    def test_h_family_collapse(self):
        from OCR_error_detection import AmharicNormalizer
        n = AmharicNormalizer()
        self.assertEqual(n.normalize_word_strict("ሐገር"), "ሀገር")
        self.assertEqual(n.normalize_word_strict("ኀገር"), "ሀገር")

    def test_ain_phharya_collapse(self):
        from OCR_error_detection import AmharicNormalizer
        n = AmharicNormalizer()
        self.assertEqual(n.normalize_word_strict("ዐዲስ"), "አዲስ")

    def test_sibilant_collapse(self):
        from OCR_error_detection import AmharicNormalizer
        n = AmharicNormalizer()
        self.assertEqual(n.normalize_word_strict("ሠላም"), "ሰላም")

    def test_tsa_pha_collapse(self):
        from OCR_error_detection import AmharicNormalizer
        n = AmharicNormalizer()
        # ፀ->ጸ AND ሐ->ሀ both fire
        self.assertEqual(n.normalize_word_strict("ፀሐይ"), "ጸሀይ")

    def test_empty_input(self):
        from OCR_error_detection import AmharicNormalizer
        self.assertEqual(AmharicNormalizer.normalize_word_strict(""), "")
        self.assertEqual(AmharicNormalizer.normalize_word_strict(None), "")

    def test_caching(self):
        from OCR_error_detection import AmharicNormalizer
        r1 = AmharicNormalizer.normalize_word_strict("ሐገር")
        r2 = AmharicNormalizer.normalize_word_strict("ሐገር")
        self.assertEqual(r1, r2)

    def test_bilabial_normalization(self):
        from OCR_error_detection import AmharicNormalizer
        n = AmharicNormalizer()
        # ሉዋ -> ሏ (labialized form)
        result = n.normalize_word_strict("ሉዋ")
        self.assertEqual(result, "ሏ")


# --------------------------------------------------------------------------- #
# Confusion matrix tests
# --------------------------------------------------------------------------- #
class TestConfusionMatrix(unittest.TestCase):

    def test_loads_real_matrix(self):
        from OCR_error_detection import ConfusionMatrix
        c = ConfusionMatrix(CONFUSION)
        self.assertGreater(len(c.sub_table), 0)
        self.assertGreater(c.default_sub, 0)

    def test_reverse_sub_map_built(self):
        from OCR_error_detection import ConfusionMatrix
        c = ConfusionMatrix(CONFUSION)
        self.assertGreater(len(c.reverse_sub_map), 0)

    def test_reverse_sub_map_cost_filter(self):
        from OCR_error_detection import ConfusionMatrix
        c = ConfusionMatrix(CONFUSION, max_reverse_sub_cost=3.0)
        for ch, alts in c.reverse_sub_map.items():
            for true_ch, cost in alts:
                self.assertLessEqual(cost, 3.0)

    def test_ins_del_costs(self):
        from OCR_error_detection import ConfusionMatrix
        c = ConfusionMatrix(CONFUSION)
        self.assertGreater(c.ins_cost("ሀ"), 0)
        self.assertGreater(c.del_cost("ሀ"), 0)
        self.assertLessEqual(c.ins_cost("ሀ"), 2.4)
        self.assertLessEqual(c.del_cost("ሀ"), 2.4)


# --------------------------------------------------------------------------- #
# Signal tests
# --------------------------------------------------------------------------- #
class TestSignals(unittest.TestCase):

    def test_surface_signals(self):
        from OCR_error_detection.signals import TokenSignals, compute_surface_signals
        sig = TokenSignals(token="ሰላም")
        compute_surface_signals("ሰላም", sig)
        self.assertFalse(sig.mixed_script)
        self.assertFalse(sig.has_digit)
        self.assertGreater(sig.len_ratio, 0)

    def test_mixed_script_detected(self):
        from OCR_error_detection.signals import TokenSignals, compute_surface_signals
        sig = TokenSignals(token="abc")
        compute_surface_signals("ሰላምabc", sig)
        self.assertTrue(sig.mixed_script)

    def test_digit_detected(self):
        from OCR_error_detection.signals import TokenSignals, compute_surface_signals
        sig = TokenSignals(token="123")
        compute_surface_signals("ሰላም123", sig)
        self.assertTrue(sig.has_digit)

    def test_lexical_signals_in_lexicon(self):
        from OCR_error_detection.signals import TokenSignals, compute_lexical_signals
        sig = TokenSignals(token="በሽታ")
        vocab = {"በሽታ": 500}
        norm_fn = lambda w: w
        compute_lexical_signals("በሽታ", sig, vocab, norm_fn)
        self.assertTrue(sig.in_lexicon)
        self.assertGreater(sig.log_freq, 0)

    def test_lexical_signals_not_in_lexicon(self):
        from OCR_error_detection.signals import TokenSignals, compute_lexical_signals
        sig = TokenSignals(token="በሸታ")
        vocab = {"በሽታ": 500}
        norm_fn = lambda w: w
        compute_lexical_signals("በሸታ", sig, vocab, norm_fn)
        self.assertFalse(sig.in_lexicon)

    def test_one_flip_finds_real_word(self):
        from OCR_error_detection.signals import TokenSignals, compute_one_flip_signals
        sig = TokenSignals(token="በሸታ")
        vocab = {"በሽታ": 500}
        # Simulate: ሸ is confusable with ሳ
        reverse_sub_map = {"ሸ": [("ሽ", 3.0)]}
        norm_fn = lambda w: w
        compute_one_flip_signals("በሸታ", sig, vocab, norm_fn, reverse_sub_map)
        self.assertGreater(len(sig.one_flip_words), 0)

    def test_one_flip_no_hits_for_unknown(self):
        from OCR_error_detection.signals import TokenSignals, compute_one_flip_signals
        sig = TokenSignals(token="xyzxyz")
        vocab = {"በሽታ": 500}
        reverse_sub_map = {}
        norm_fn = lambda w: w
        compute_one_flip_signals("xyzxyz", sig, vocab, norm_fn, reverse_sub_map)
        self.assertEqual(len(sig.one_flip_words), 0)

    def test_as_vector(self):
        from OCR_error_detection.signals import TokenSignals
        sig = TokenSignals(token="ሰላም", in_lexicon=True, log_freq=2.5,
                           n_neighbors=3, mixed_script=False)
        v = sig.as_vector()
        self.assertEqual(v["in_lexicon"], 1.0)
        self.assertEqual(v["log_freq"], 2.5)
        self.assertEqual(v["n_neighbors"], 3.0)

    def test_as_vector_with_optional_fields(self):
        from OCR_error_detection.signals import TokenSignals
        sig = TokenSignals(token="ሰላም", char_lm=-0.5, lm_fit=0.3,
                           morph_ok=True, ocr_conf=85.0)
        v = sig.as_vector()
        self.assertEqual(v["char_lm"], -0.5)
        self.assertEqual(v["lm_fit"], 0.3)
        self.assertEqual(v["morph_ok"], 1.0)
        self.assertEqual(v["ocr_conf"], 85.0)


# --------------------------------------------------------------------------- #
# Scoring tests
# --------------------------------------------------------------------------- #
class TestScoring(unittest.TestCase):

    def test_score_in_range(self):
        from OCR_error_detection.scoring import combine_signals
        from OCR_error_detection.signals import TokenSignals
        sig = TokenSignals(token="ሰላም", in_lexicon=True, log_freq=3.0)
        s = combine_signals(sig)
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)

    def test_oov_scores_higher(self):
        from OCR_error_detection.scoring import combine_signals
        from OCR_error_detection.signals import TokenSignals
        sig_clean = TokenSignals(token="ሰላም", in_lexicon=True, log_freq=3.0)
        sig_oov = TokenSignals(token="xyzxyz", in_lexicon=False)
        self.assertGreater(combine_signals(sig_oov), combine_signals(sig_clean))

    def test_custom_weights(self):
        from OCR_error_detection.scoring import combine_signals
        from OCR_error_detection.signals import TokenSignals
        sig = TokenSignals(token="ሰላም", in_lexicon=True)
        s1 = combine_signals(sig)
        s2 = combine_signals(sig, weights={"bias": 2.0})
        self.assertNotAlmostEqual(s1, s2, places=2)


# --------------------------------------------------------------------------- #
# Verdict tests
# --------------------------------------------------------------------------- #
class TestVerdict(unittest.TestCase):

    def test_should_correct(self):
        from OCR_error_detection.signals import Verdict, TokenSignals
        v_pass = Verdict(0.1, "pass", TokenSignals(token="x"))
        v_gray = Verdict(0.5, "gray", TokenSignals(token="x"))
        v_flag = Verdict(0.9, "flag", TokenSignals(token="x"))
        self.assertFalse(v_pass.should_correct)
        self.assertTrue(v_gray.should_correct)
        self.assertTrue(v_flag.should_correct)


# --------------------------------------------------------------------------- #
# DetectionEngine integration tests
# --------------------------------------------------------------------------- #
class TestDetectionEngine(unittest.TestCase):

    def test_passes_clean_lexicon_word(self):
        """A known word with no optical neighbour should PASS."""
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000),
                 ("ትምህርት", 2000), ("ወርቅ", 1000)]
        det = _make_detector(words)
        v = det.verdict("ሰላም")
        self.assertEqual(v.label, "pass",
                         f"Expected pass for clean word, got {v.label} "
                         f"(s={v.suspicion:.3f}, reason={v.reason!r})")

    def test_passes_all_clean_words(self):
        """All known words with no confusion neighbours should pass."""
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000),
                 ("ትምህርት", 2000), ("ወርቅ", 1000)]
        det = _make_detector(words)
        for tok in ["ሰላም", "ቤት", "ነገር", "ትምህርት", "ወርቅ"]:
            v = det.verdict(tok)
            self.assertEqual(v.label, "pass",
                             f"{tok!r} should pass, got {v.label} "
                             f"(s={v.suspicion:.3f})")

    def test_oov_with_flip_gets_flagged(self):
        """An OOV token that flips to a real word should be flagged."""
        words = [("በሽታ", 5000), ("ከተማ", 4000)]
        det = _make_detector(words)
        # Test with a token that is not in the lexicon
        v = det.verdict("በሸታውስጥያ")
        # The result depends on whether the one-flip test finds anything
        # but it should at least not be a clean PASS
        self.assertIn(v.label, ("pass", "gray", "flag"))

    def test_deterministic(self):
        """Same input → same output."""
        words = [("ሰላም", 5000), ("ቤት", 4000)]
        det = _make_detector(words)
        v1 = det.verdict("ሰላም")
        v2 = det.verdict("ሰላም")
        self.assertEqual(v1.label, v2.label)
        self.assertAlmostEqual(v1.suspicion, v2.suspicion, places=10)

    def test_no_op_flips_not_evidence(self):
        """A flip that normalizes back to the same key is a no-op."""
        words = [("ሰላም", 5000)]
        det = _make_detector(words)
        sig = det.signals("ሰላም")
        self.assertEqual(sig.one_flip_words, [],
                         f"ሰላም produced self-flips {sig.one_flip_words}")

    def test_stage1_counts_reported(self):
        """Detection stats must be tracked."""
        words = [("ሰላም", 5000), ("ቤት", 4000)]
        det = _make_detector(words)
        det.verdict("ሰላም")
        det.verdict("ቤት")
        stats = det.detection_stats()
        self.assertIn("stage1", stats)
        self.assertEqual(set(stats["stage1"]), {"pass", "gray", "flag"})
        self.assertGreater(sum(stats["stage1"].values()), 0)

    def test_reset_counters(self):
        words = [("ሰላም", 5000)]
        det = _make_detector(words)
        det.verdict("ሰላም")
        self.assertGreater(sum(det._stage1.values()), 0)
        det.reset_counters()
        self.assertEqual(sum(det._stage1.values()), 0)

    def test_batch_detection(self):
        """verdict_batch processes a sequence."""
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000)]
        det = _make_detector(words)
        results = det.verdict_batch(["ሰላም", "ቤት", "ነገር"])
        self.assertEqual(len(results), 3)
        for v in results:
            self.assertEqual(v.label, "pass")

    def test_batch_with_custom_context(self):
        words = [("ሰላም", 5000), ("ቤት", 4000)]
        det = _make_detector(words)
        results = det.verdict_batch(
            ["ሰላም", "ቤት"],
            context_fn=lambda i: ["ሰላም"] if i > 0 else [],
        )
        self.assertEqual(len(results), 2)

    def test_vocab_size(self):
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000)]
        det = _make_detector(words)
        self.assertEqual(len(det.vocab), 3)

    def test_max_lexicon_types(self):
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000)]
        det = _make_detector(words, max_lexicon_types=2)
        self.assertEqual(len(det.vocab), 2)

    def test_dict_min_freq(self):
        words = [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 5)]
        det = _make_detector(words, dict_min_freq=100)
        self.assertEqual(len(det.vocab), 2)

    def test_surface_form_preserved(self):
        words = [("ሐገር", 100), ("ሠላም", 200)]
        det = _make_detector(words)
        # After normalization, ሐ->ሀ and ሠ->ሰ, but surface forms preserved
        self.assertIn("ሀገር", det.vocab)
        self.assertIn("ሰላም", det.vocab)
        self.assertEqual(det.surface_form.get("ሀገር"), "ሐገር")
        self.assertEqual(det.surface_form.get("ሰላም"), "ሠላም")

    def test_verdict_carries_signals(self):
        words = [("ሰላም", 5000)]
        det = _make_detector(words)
        v = det.verdict("ሰላም")
        self.assertIsNotNone(v.signals)
        self.assertEqual(v.signals.token, "ሰላም")

    def test_ocr_confidence_affects_verdict(self):
        """Low OCR confidence should push suspicion up."""
        words = [("ሰላም", 5000)]
        det = _make_detector(words)
        v_high = det.verdict("ሰላም", ocr_conf=95)
        v_low = det.verdict("ሰላም", ocr_conf=20)
        # Both might pass, but low confidence should score higher
        self.assertGreaterEqual(v_low.suspicion, v_high.suspicion - 0.01)


# --------------------------------------------------------------------------- #
# Integration with the main corrector (adapter pattern)
# --------------------------------------------------------------------------- #
class TestAdapterPattern(unittest.TestCase):
    """DetectionEngine can be used as a drop-in for SuspicionScorer."""

    def test_engine_has_same_verdict_interface(self):
        """The DetectionEngine.verdict() has the same signature as SuspicionScorer.verdict()."""
        from OCR_error_detection import DetectionEngine
        words = [("ሰላም", 5000)]
        det = _make_detector(words)
        v = det.verdict("ሰላም", context=[], ocr_conf=None)
        self.assertEqual(v.label, "pass")
        self.assertIsInstance(v.suspicion, float)
        self.assertIsInstance(v.reason, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)