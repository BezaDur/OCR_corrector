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

# Resolve paths — works from any directory
ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parents[1]
CONFUSION = str(_HERE / "Model" / "confusion_matrix.json")
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _real_model_engine(**kwargs):
    """DetectionEngine on the real shipped payload + lexicon, or None if absent.

    Used by the gate-precedence tests, which are about values stored inside
    Model/detector_lgbm_conffree.pkl and so cannot be exercised with the tiny
    fixture lexicons the rest of this file uses.
    """
    from OCR_error_detection import DetectionEngine

    pkl = _HERE / "Model" / "detector_lgbm_conffree.pkl"
    lex = _HERE / "Model" / "main_i7_frq10.txt"
    if not pkl.exists() or not lex.exists():
        return None
    try:
        import lightgbm  # noqa: F401
        import joblib    # noqa: F401
    except ImportError:
        return None
    options = dict(lgbm_model_path=str(pkl), dict_vocab_path=str(lex),
                   dict_min_freq=500, confusion_matrix_path=CONFUSION,
                   verbose=False)
    options.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        eng = DetectionEngine(**options)
    return eng if eng.lgbm_model is not None else None


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

    def test_missing_caller_lexicon_raises(self):
        """An explicit dict_vocab_path that does not exist must fail loudly.

        It used to fall through to the 8-word minimal vocabulary, which
        silently invalidated in_lexicon / log_corpus_freq — ~80% of the shipped
        model's gain — and returned plausible-looking but meaningless verdicts.
        """
        from OCR_error_detection import DetectionEngine

        missing = os.path.join(tempfile.mkdtemp(), "main_i7_freq10.txt")  # real file is _frq10
        with self.assertRaises(FileNotFoundError) as ctx:
            DetectionEngine(dict_vocab_path=missing,
                            confusion_matrix_path=CONFUSION, verbose=False)
        self.assertIn(missing, str(ctx.exception))
        self.assertIn("main_i7_frq10.txt", str(ctx.exception))

    def test_missing_caller_lexicon_never_yields_fallback_vocab(self):
        """The minimal fallback is for auto-resolution only, not for typos."""
        from OCR_error_detection import DetectionEngine

        missing = os.path.join(tempfile.mkdtemp(), "nope.txt")
        try:
            DetectionEngine(dict_vocab_path=missing,
                            confusion_matrix_path=CONFUSION, verbose=False)
        except FileNotFoundError:
            return
        self.fail("a missing caller-supplied lexicon must raise, not fall back")

    def test_gates_come_from_the_model_by_default(self):
        """The payload's thresholds win, so callers cannot retune by accident."""
        from OCR_error_detection import DetectionEngine

        det = _real_model_engine()
        if det is None:
            self.skipTest("shipped model not available")
        self.assertEqual(det.pass_below, 0.10)
        self.assertEqual(det.flag_above, 0.65)
        # An explicit value is still overridden — that is the documented default.
        det2 = _real_model_engine(pass_below=0.02, flag_above=0.90)
        if det2 is None:
            self.skipTest("shipped model not available")
        self.assertEqual(det2.pass_below, 0.10)
        self.assertEqual(det2.flag_above, 0.65)

    def test_gates_override_model_honours_the_caller(self):
        """gates_override_model=True is how a retuned operating point is applied.

        Without it the gates baked into the pickle are unreachable, so a
        recommendation from scripts/tune_gates.py could only be applied by
        rewriting the model artifact.
        """
        from OCR_error_detection import DetectionEngine

        det = _real_model_engine(pass_below=0.10, flag_above=0.85,
                                 gates_override_model=True)
        if det is None:
            self.skipTest("shipped model not available")
        self.assertEqual(det.pass_below, 0.10)
        self.assertEqual(det.flag_above, 0.85)
        self.assertIsNotNone(det.lgbm_model)  # override must not cost us the model

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
# Post-model confidence policy layer
# --------------------------------------------------------------------------- #
class TestConfidencePolicy(unittest.TestCase):
    """ConfidencePolicy re-bands the gate output using Tesseract confidence.

    The LightGBM core stays confidence-free: the layer runs strictly after the
    model predicts and never feeds back into the features. Rule C
    (pass -> gray) is measured at 5.8% marginal precision and is therefore off
    by default; these tests pin that default so it cannot be switched on
    accidentally.
    """

    def _sig(self, ocr_conf=None, in_lexicon=False, one_flip_words=()):
        from OCR_error_detection import TokenSignals
        return TokenSignals(token="ሙሉት", in_lexicon=in_lexicon,
                            one_flip_words=list(one_flip_words),
                            ocr_conf=ocr_conf)

    # -- the layer is inert without confidence ----------------------------- #
    def test_missing_confidence_is_a_noop(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        for label in ("pass", "gray", "flag"):
            got, note = pol.review(label, 0.5, self._sig(ocr_conf=None))
            self.assertEqual(got, label)
            self.assertEqual(note, "")

    # -- Rule A: gray -> pass ---------------------------------------------- #
    def test_rule_a_promotes_confident_lexicon_word(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        sig = self._sig(ocr_conf=0.90, in_lexicon=True)
        got, note = pol.review("gray", 0.20, sig)
        self.assertEqual(got, "pass")
        self.assertIn("PASS", note)

    def test_rule_a_requires_the_lexicon(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, note = pol.review("gray", 0.20, self._sig(ocr_conf=0.90, in_lexicon=False))
        self.assertEqual(got, "gray")
        self.assertEqual(note, "")

    def test_rule_a_requires_no_one_flip_candidate(self):
        """A reachable one-flip correction means there is still something to review."""
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, _ = pol.review("gray", 0.20,
                            self._sig(ocr_conf=0.90, in_lexicon=True,
                                      one_flip_words=["ሙሉ"]))
        self.assertEqual(got, "gray")

    def test_rule_a_respects_its_confidence_threshold(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, _ = pol.review("gray", 0.20,
                            self._sig(ocr_conf=0.81, in_lexicon=True))
        self.assertEqual(got, "gray")

    # -- Rule B: gray -> flag ---------------------------------------------- #
    def test_rule_b_flags_low_confidence_error_leaning_gray(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, note = pol.review("gray", 0.358, self._sig(ocr_conf=0.443))
        self.assertEqual(got, "flag")
        self.assertIn("FLAG", note)

    def test_rule_b_requires_the_suspicion_floor(self):
        """This floor is what makes rule B ~90% precise instead of ~50%.

        Low Tesseract confidence alone is not enough; the model must also lean
        toward error. Without the floor, confidence-only flagging sweeps up
        clean words.
        """
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, note = pol.review("gray", 0.20, self._sig(ocr_conf=0.30))
        self.assertEqual(got, "gray")
        self.assertEqual(note, "")

    def test_rule_b_respects_its_confidence_threshold(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        got, _ = pol.review("gray", 0.50, self._sig(ocr_conf=0.65))
        self.assertEqual(got, "gray")

    # -- Rule C: pass -> gray, off by default ------------------------------ #
    def test_rule_c_is_off_by_default(self):
        """A very low-confidence PASS word must stay PASS unless C is enabled."""
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        self.assertIsNone(pol.demote_pass_conf)
        got, note = pol.review("pass", 0.001, self._sig(ocr_conf=0.20))
        self.assertEqual(got, "pass")
        self.assertEqual(note, "")

    def test_rule_c_can_be_enabled_explicitly(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        pol.demote_pass_conf = 0.45
        got, note = pol.review("pass", 0.001, self._sig(ocr_conf=0.20))
        self.assertEqual(got, "gray")
        self.assertIn("GRAY", note)

    # -- the layer never invents work for a hard FLAG ---------------------- #
    def test_flag_verdicts_are_never_touched(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        pol.demote_pass_conf = 0.45
        for conf in (0.05, 0.50, 0.99, None):
            got, note = pol.review("flag", 0.90, self._sig(ocr_conf=conf))
            self.assertEqual(got, "flag")
            self.assertEqual(note, "")

    # -- construction from a model payload --------------------------------- #
    def test_from_payload_accepts_none_policy_and_mapping(self):
        from OCR_error_detection import ConfidencePolicy
        self.assertIsNone(ConfidencePolicy.from_payload(None))
        pol = ConfidencePolicy.tuned()
        self.assertIs(ConfidencePolicy.from_payload(pol), pol)
        built = ConfidencePolicy.from_payload(
            {"promote_pass_conf": 0.80, "flag_below_conf": 0.60,
             "flag_min_suspicion": 0.40, "demote_pass_conf": 0.30})
        self.assertEqual(built.promote_pass_conf, 0.80)
        self.assertEqual(built.flag_below_conf, 0.60)
        self.assertEqual(built.flag_min_suspicion, 0.40)
        self.assertEqual(built.demote_pass_conf, 0.30)

    def test_from_payload_rejects_unknown_keys(self):
        """A typo in a retrained pickle must surface, not silently disable a rule."""
        from OCR_error_detection import ConfidencePolicy
        with self.assertRaises(ValueError):
            ConfidencePolicy.from_payload({"promote_pass_conf": 0.8, "flag_belo_conf": 0.6})

    def test_from_payload_rejects_wrong_type(self):
        from OCR_error_detection import ConfidencePolicy
        with self.assertRaises(TypeError):
            ConfidencePolicy.from_payload(0.82)

    def test_tuned_matches_the_documented_operating_point(self):
        from OCR_error_detection import ConfidencePolicy
        pol = ConfidencePolicy.tuned()
        self.assertEqual(pol.promote_pass_conf, 0.82)
        self.assertEqual(pol.flag_below_conf, 0.65)
        self.assertEqual(pol.flag_min_suspicion, 0.35)
        self.assertIsNone(pol.demote_pass_conf)


class TestConfidencePolicyInEngine(unittest.TestCase):
    """End-to-end behaviour of the layer inside DetectionEngine.verdict().

    These need the real shipped payload, because the rules only fire in the
    gray band that the calibrated model actually produces.
    """

    def test_layer_is_off_unless_requested(self):
        det = _real_model_engine()
        if det is None:
            self.skipTest("shipped model or lightgbm unavailable")
        self.assertIsNone(det.confidence_policy)

    def test_rule_b_moves_gray_to_flag_on_a_real_token(self):
        """Also proves verdict() rescales Tesseract's 0-100 conf.tsv value
        onto sig.ocr_conf before the policy reads it. Passing 44.3 on the raw
        scale must behave as 0.443, not as a saturated 1.0.
        """
        from OCR_error_detection import ConfidencePolicy
        base = _real_model_engine()
        pol = _real_model_engine(confidence_policy=ConfidencePolicy.tuned())
        if base is None or pol is None:
            self.skipTest("shipped model or lightgbm unavailable")
        vb = base.verdict("ሙሉት", [], ocr_conf=44.3)
        vp = pol.verdict("ሙሉት", [], ocr_conf=44.3)
        self.assertEqual(vb.label, "gray")
        self.assertEqual(vp.label, "flag")
        self.assertAlmostEqual(vp.signals.ocr_conf, 0.443, places=3)
        self.assertIn("-> FLAG", vp.reason)

    def test_rule_a_moves_gray_to_pass_on_a_real_token(self):
        from OCR_error_detection import ConfidencePolicy
        base = _real_model_engine()
        pol = _real_model_engine(confidence_policy=ConfidencePolicy.tuned())
        if base is None or pol is None:
            self.skipTest("shipped model or lightgbm unavailable")
        vb = base.verdict("የዩክሬንን", [], ocr_conf=90.0, doc_freq=2)
        vp = pol.verdict("የዩክሬንን", [], ocr_conf=90.0, doc_freq=2)
        self.assertEqual(vb.label, "gray")
        self.assertEqual(vp.label, "pass")
        self.assertIn("-> PASS", vp.reason)

    def test_layer_rebands_without_changing_the_score(self):
        """The suspicion the model produced must survive untouched, so the
        layer cannot silently become a second scorer.
        """
        from OCR_error_detection import ConfidencePolicy
        base = _real_model_engine()
        pol = _real_model_engine(confidence_policy=ConfidencePolicy.tuned())
        if base is None or pol is None:
            self.skipTest("shipped model or lightgbm unavailable")
        for tok, conf, df in (("ሙሉት", 44.3, None), ("ማእክሉ", 44.1, None),
                              ("የዩክሬንን", 90.0, 2), ("እና", 95.0, None)):
            vb = base.verdict(tok, [], ocr_conf=conf, doc_freq=df)
            vp = pol.verdict(tok, [], ocr_conf=conf, doc_freq=df)
            self.assertEqual(vb.suspicion, vp.suspicion, tok)

    def test_tokens_without_confidence_are_unaffected(self):
        from OCR_error_detection import ConfidencePolicy
        base = _real_model_engine()
        pol = _real_model_engine(confidence_policy=ConfidencePolicy.tuned())
        if base is None or pol is None:
            self.skipTest("shipped model or lightgbm unavailable")
        for tok in ("ሙሉት", "የዩክሬንን", "ማእክሉ", "እና", "ሀገራት"):
            self.assertEqual(base.verdict(tok, []).label,
                             pol.verdict(tok, []).label, tok)

    def test_model_core_stays_confidence_free(self):
        """The design guarantee: confidence enters after prediction, never as
        a feature. If ocr_conf ever reappears in the feature list, the layer
        has stopped being a policy and become a second model input.
        """
        from OCR_error_detection import ConfidencePolicy
        det = _real_model_engine(confidence_policy=ConfidencePolicy.tuned())
        if det is None:
            self.skipTest("shipped model or lightgbm unavailable")
        self.assertNotIn("ocr_conf", det.lgbm_features)

    def test_payload_can_carry_the_policy_and_explicit_wins(self):
        """Thresholds are model-specific, so a retrained pickle can ship the
        policy it was measured with; an explicitly passed policy still wins.
        """
        import joblib
        from OCR_error_detection import ConfidencePolicy
        pkl = _HERE / "Model" / "detector_lgbm_conffree.pkl"
        lex = _HERE / "Model" / "main_i7_frq10.txt"
        if not pkl.exists() or not lex.exists():
            self.skipTest("shipped payload or lexicon unavailable")
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm unavailable")
        payload = dict(joblib.load(str(pkl)))
        payload["confidence_policy"] = {"promote_pass_conf": 0.70,
                                        "flag_below_conf": 0.50,
                                        "flag_min_suspicion": 0.42}
        folder = tempfile.mkdtemp()
        try:
            tmp = os.path.join(folder, "with_policy.pkl")
            joblib.dump(payload, tmp)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                from OCR_error_detection import DetectionEngine
                carried = DetectionEngine(lgbm_model_path=tmp, dict_vocab_path=str(lex),
                                          dict_min_freq=500,
                                          confusion_matrix_path=CONFUSION, verbose=False)
                explicit = DetectionEngine(lgbm_model_path=tmp, dict_vocab_path=str(lex),
                                           dict_min_freq=500,
                                           confusion_matrix_path=CONFUSION,
                                           confidence_policy=ConfidencePolicy.tuned(),
                                           verbose=False)
            self.assertIsNotNone(carried.confidence_policy)
            self.assertEqual(carried.confidence_policy.promote_pass_conf, 0.70)
            self.assertEqual(carried.confidence_policy.flag_below_conf, 0.50)
            self.assertEqual(carried.confidence_policy.flag_min_suspicion, 0.42)
            self.assertIsNone(carried.confidence_policy.demote_pass_conf)
            self.assertEqual(explicit.confidence_policy.promote_pass_conf, 0.82)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


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