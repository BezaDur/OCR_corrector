"""
Regression tests for the framework principles (see docs/principles_review.md).

  1. Stage 1 gates: PASS tokens still feed the LM (LM context must not be lost).
  2. Do no harm: an OOV original is kept unless the best alternative beats it
     by a margin, even when the lexicon word carries a morphology bonus.
  3. One confusion table: curated visually-similar pairs reach Stage 2.
  4. Stage 4: merge two OOV tokens; split one OOV token into two words.
  5. Stage 1 can use Tesseract confidences; mismatched counts are dropped.

Runs without kenlm, the external segmenter, or a real model (fake LM).
Run:  python -m unittest discover -s tests -p 'test_principles.py'
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from test_head_contract import CONFUSION, FakeLM, State  # noqa: E402
from amharic_ocr_corrector import AmharicOCRCorrector  # noqa: E402


class PrinciplesBase(unittest.TestCase):
    WORDS = [("ሰላም", 900), ("ቤት", 800), ("ኽብር", 500)]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lex = os.path.join(self.tmp, "lex.txt")
        Path(self.lex).write_text(
            "\n".join(f"{w} {f}" for w, f in self.WORDS), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, lm=None, **kw):
        opts = dict(
            kenlm_path="unused", confusion_matrix_path=CONFUSION,
            dict_vocab_path=self.lex, lm=lm or FakeLM(), state_factory=State,
            segmenter=None, verbose=False,
        )
        opts.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return AmharicOCRCorrector(**opts)


class TestStage1KeepsLMContext(PrinciplesBase):
    def test_pass_tokens_are_scored_by_the_lm(self):
        lm = FakeLM()
        c = self.make(lm=lm, stage1_detection=True)
        out = c.correct_sentence("ሰላም ቤት")
        self.assertEqual(out, "ሰላም ቤት")
        scored = [w for _, w in lm.calls]
        self.assertIn("ሰላም", scored)
        self.assertIn("ቤት", scored)
        # the second word must be scored in the context of the first
        self.assertIn(("ሰላም", "ቤት"), lm.calls)


class TestDoNoHarm(PrinciplesBase):
    OOV = "ኸብር"   # not in the lexicon; one optical flip reaches ኽብር

    def test_oov_original_survives_small_lm_gap_with_morph_bonus(self):
        # LM mildly prefers the lexicon word (0.5 nats). The morphology bonus
        # (lambda_morph) used to be enough to flip it. The margin must hold it.
        lm = FakeLM({("<s>", self.OOV): -4.0, ("<s>", "ኽብር"): -3.5})
        c = self.make(lm=lm, stage1_detection=True, lambda_morph=1.5)
        self.assertEqual(c.correct_sentence(self.OOV), self.OOV)

    def test_oov_original_survives_without_stage1(self):
        lm = FakeLM({("<s>", self.OOV): -4.0, ("<s>", "ኽብር"): -3.5})
        c = self.make(lm=lm, stage1_detection=False, lambda_morph=1.5)
        self.assertEqual(c.correct_sentence(self.OOV), self.OOV)

    def test_large_lm_gap_still_corrects(self):
        lm = FakeLM({("<s>", self.OOV): -8.0, ("<s>", "ኽብር"): -2.0})
        c = self.make(lm=lm, stage1_detection=True)
        self.assertEqual(c.correct_sentence(self.OOV), "ኽብር")


class TestSingleConfusionTable(PrinciplesBase):
    def test_curated_pair_is_in_cost_table_and_reverse_map(self):
        c = self.make()
        # ኘ (U+1258) and ነ (U+1290) sit in a curated class; the learned matrix
        # has no entry for them, so they must now be present at the 0.5 tier.
        self.assertEqual(c.costs["sub"]["ነ"]["ኘ"], 0.5)
        self.assertIn(("ነ", 0.5), c.reverse_sub_map["ኘ"])

    def test_learned_costs_take_precedence(self):
        """Curated injection must never overwrite a learned matrix entry."""
        import json
        with open(CONFUSION, encoding="utf-8") as f:
            learned = json.load(f)["sub"]
        c = self.make()
        checked = 0
        for true_ch, row in learned.items():
            for obs_ch, cost in row.items():
                if float(cost) != 0.5:
                    self.assertEqual(c.costs["sub"][true_ch][obs_ch], float(cost))
                    checked += 1
        self.assertGreater(checked, 100)

    def test_stage1_one_flip_is_reachable_by_stage2(self):
        """A flip Stage 1 uses as evidence must be a candidate Stage 2 can propose."""
        c = self.make(stage1_detection=True)
        sig = c.suspicion.verdict("ኸብር", []).signals
        self.assertIn("ኽብር", sig.one_flip_words)
        cands = {w for w, _ in c.generate_candidates("ኸብር")}
        self.assertIn("ኽብር", cands)


class TestMergeSplit(PrinciplesBase):
    # Merge: 'ትምህርት' and 'ቤት' are both OOV here; their concatenation is a word.
    MERGE_WORDS = [("ትምህርትቤት", 300), ("ሰላም", 900)]
    # Split: 'ሰላምቤት' is OOV; both halves are words.
    SPLIT_WORDS = [("ሰላም", 900), ("ቤት", 800)]

    def use_lexicon(self, words):
        Path(self.lex).write_text("\n".join(f"{w} {f}" for w, f in words), encoding="utf-8")

    def test_merge_two_oov_tokens_into_one_lexicon_word(self):
        self.use_lexicon(self.MERGE_WORDS)
        # Separate reading is penalised by the LM; the compound reads naturally.
        lm = FakeLM({("<s>", "ትምህርት"): -6.0, ("<s>", "ትምህርትቤት"): -1.0})
        c = self.make(lm=lm, stage1_detection=False, enable_merge_split=True)
        self.assertEqual(c.correct_sentence("ትምህርት ቤት"), "ትምህርትቤት")

    def test_no_merge_when_disabled(self):
        self.use_lexicon(self.MERGE_WORDS)
        lm = FakeLM({("<s>", "ትምህርት"): -6.0, ("<s>", "ትምህርትቤት"): -1.0})
        c = self.make(lm=lm, stage1_detection=False, enable_merge_split=False)
        self.assertEqual(c.correct_sentence("ትምህርት ቤት"), "ትምህርት ቤት")

    def test_merge_change_is_counted(self):
        self.use_lexicon(self.MERGE_WORDS)
        lm = FakeLM({("<s>", "ትምህርት"): -6.0, ("<s>", "ትምህርትቤት"): -1.0})
        c = self.make(lm=lm, stage1_detection=False)
        c.correct_sentence("ትምህርት ቤት")
        self.assertGreaterEqual(c._morph_stats["tokens_changed"], 1)

    def test_split_oov_token_into_two_lexicon_words(self):
        self.use_lexicon(self.SPLIT_WORDS)
        # The unsplit original keeps the OOV margin; the LM must clearly prefer the halves.
        lm = FakeLM({("<s>", "ሰላምቤት"): -6.0, ("<s>", "ሰላም"): 0.0, ("ሰላም", "ቤት"): 0.0})
        c = self.make(lm=lm, stage1_detection=False, enable_merge_split=True)
        self.assertEqual(c.correct_sentence("ሰላምቤት"), "ሰላም ቤት")

    def test_split_is_not_applied_when_lm_does_not_prefer_it(self):
        self.use_lexicon(self.SPLIT_WORDS)
        lm = FakeLM({("<s>", "ሰላምቤት"): 0.0})
        c = self.make(lm=lm, stage1_detection=False, enable_merge_split=True)
        self.assertEqual(c.correct_sentence("ሰላምቤት"), "ሰላምቤት")


class TestConfidences(PrinciplesBase):
    def test_mismatched_confidence_count_is_dropped_not_fatal(self):
        c = self.make(stage1_detection=True)
        out = c.correct_sentence("ሰላም ቤት", confidences=[90])
        self.assertEqual(out, "ሰላም ቤት")
        self.assertEqual(c._conf_dropped, 1)

    def test_matched_confidences_reach_stage1(self):
        c = self.make(stage1_detection=True)
        seen = []
        real = c.suspicion.verdict

        def spy(tok, context=(), ocr_conf=None):
            seen.append(ocr_conf)
            return real(tok, context, ocr_conf=ocr_conf)

        c.suspicion.verdict = spy
        c.correct_sentence("ሰላም ቤት", confidences=[95, 40])
        self.assertEqual(seen, [95.0, 40.0])


if __name__ == "__main__":
    unittest.main()
