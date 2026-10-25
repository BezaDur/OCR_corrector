"""Tests for scripts/evaluate_out_of_sample.py — metric plumbing only.

The script itself scores the whole corpus, which is too slow for a unit test;
these check the pieces that decide the reported numbers, so a silent change in
the metric definitions cannot slip through.
"""
import os
import sys
import unittest

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for _p in (_ROOT, _PKG, os.path.join(_PKG, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate_out_of_sample as oos  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["ocr_word", "gt_word", "page", "is_error", "p"])


class TestMetrics(unittest.TestCase):
    def test_recall_counts_gray_as_caught(self):
        # one error at 0.2 -> gray (caught), one at 0.02 -> pass (missed)
        df = frame([("a", "b", "p1", 1, 0.2), ("c", "d", "p1", 1, 0.02),
                    ("e", "e", "p1", 0, 0.0), ("f", "f", "p1", 0, 0.9)])
        m = oos.metrics(df, pass_below=0.10, flag_above=0.65)
        self.assertEqual(m["errors"], 2)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["recall_flag_only"], 0.0)   # 0.2 is gray, not flag
        self.assertEqual(m["false_flag"], 0.5)         # 1 of the 2 clean rows flagged

    def test_gates_are_half_open(self):
        # exactly pass_below -> gray (caught); exactly flag_above -> flag
        df = frame([("a", "b", "p1", 1, 0.10), ("c", "d", "p1", 1, 0.65),
                    ("e", "e", "p1", 0, 0.10), ("f", "f", "p1", 0, 0.65)])
        m = oos.metrics(df, pass_below=0.10, flag_above=0.65)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["recall_flag_only"], 0.5)
        self.assertEqual(m["false_flag"], 0.5)
        self.assertEqual(m["gray_on_clean"], 0.5)   # the clean row sitting exactly on the gate

    def test_score_is_recall_times_one_minus_false_flag(self):
        df = frame([("a", "b", "p1", 1, 0.7), ("c", "c", "p1", 0, 0.7),
                    ("d", "e", "p1", 1, 0.0), ("f", "f", "p1", 0, 0.0)])
        m = oos.metrics(df, 0.10, 0.65)
        self.assertAlmostEqual(m["score"], m["recall"] * (1.0 - m["false_flag"]))


class TestCalibration(unittest.TestCase):
    def test_ece_is_zero_for_a_perfect_predictor(self):
        # 50 rows at p=0.5 with exactly 25 errors, 50 at p=0.1 with 5 errors
        rows = ([("w", "g", "p1", 1, 0.5)] * 25 + [("w", "g", "p1", 0, 0.5)] * 25 +
                [("w", "g", "p1", 1, 0.1)] * 5 + [("w", "g", "p1", 0, 0.1)] * 45)
        cal = oos.calibration(frame(rows), n_bins=2)
        self.assertLess(cal["ece"], 0.01)
        self.assertEqual(sum(b["n"] for b in cal["bins"]), 100)

    def test_ece_flags_a_confident_wrong_model(self):
        rows = [("w", "g", "p1", 1, 0.01)] * 50 + [("w", "g", "p1", 0, 0.01)] * 50
        cal = oos.calibration(frame(rows), n_bins=2)
        self.assertGreater(cal["ece"], 0.4)


class TestOverlap(unittest.TestCase):
    def test_shared_text_scores_one_and_unseen_scores_zero(self):
        seen = " ".join(f"ወ{i}" for i in range(12))
        new = " ".join(f"አ{i}" for i in range(12))
        dev = pd.DataFrame({"page": ["page_0001", "page_0002"],
                            "gt_word": [seen, new]})
        prod = pd.DataFrame({"page": ["page_0101"], "gt_word": [seen]})
        share = oos.text_overlap(dev, prod)
        self.assertEqual(share["page_0001"], 1.0)
        self.assertEqual(share["page_0002"], 0.0)

    def test_tiers_partition_the_pages(self):
        for share, expected in ((0.0, "disjoint"), (0.4, "partial"),
                                (0.9, "near_duplicate"), (1.0, "near_duplicate")):
            hit = [name for name, test in oos.TIERS if test(share)]
            self.assertEqual(hit, [expected], share)


if __name__ == "__main__":
    unittest.main()
