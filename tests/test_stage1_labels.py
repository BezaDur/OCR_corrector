"""Stage 1 verdict labelling against ground truth."""
import unittest

from stage1_labels import label_verdicts


def v(word, label="pass", s=0.1):
    return {"word": word, "label": label, "suspicion": s, "ocr_conf": None}


class TestLabelVerdicts(unittest.TestCase):
    def test_correct_token_is_clean_and_wrong_token_is_error(self):
        gold = "ሰላም ቤት"
        raw = "ሰላም ቤት"
        res = label_verdicts(gold, raw, [v("ሰላም"), v("ቤት")])
        self.assertEqual(res["status"], "ok")
        self.assertEqual([r["status"] for r in res["rows"]], ["clean", "clean"])

        raw_bad = "ሰላም ቤቱ"
        res = label_verdicts(gold, raw_bad, [v("ሰላም"), v("ቤቱ")])
        self.assertEqual([r["status"] for r in res["rows"]], ["clean", "error"])

    def test_punctuation_is_ignored_when_matching(self):
        res = label_verdicts("ሰላም።", "ሰላም", [v("ሰላም")])
        self.assertEqual(res["rows"][0]["status"], "clean")

    def test_verdict_count_mismatch_is_reported_not_scored(self):
        res = label_verdicts("ሰላም ቤት", "ሰላም ቤት", [v("ሰላም")])
        self.assertEqual(res["status"], "mismatch")
        self.assertEqual(res["rows"], [])

    def test_verdict_word_mismatch_is_reported(self):
        res = label_verdicts("ሰላም", "ሰላም", [v("ቤት")])
        self.assertEqual(res["status"], "mismatch")

    def test_label_and_score_are_kept(self):
        res = label_verdicts("ሰላም", "ሰላም", [v("ሰላም", label="flag", s=0.9)])
        self.assertEqual(res["rows"][0]["label"], "flag")
        self.assertEqual(res["rows"][0]["suspicion"], 0.9)


if __name__ == "__main__":
    unittest.main()


class TestThresholdSweep(unittest.TestCase):
    def test_precision_and_recall_at_threshold(self):
        from stage1_labels import threshold_sweep
        rows = [
            {"suspicion": 0.9, "status": "error"},
            {"suspicion": 0.8, "status": "clean"},
            {"suspicion": 0.4, "status": "error"},
            {"suspicion": 0.2, "status": "clean"},
            {"suspicion": 0.1, "status": "ambiguous"},
        ]
        res = {s["threshold"]: s for s in threshold_sweep(rows, [0.5, 0.0])}
        # at 0.5: flagged {0.9 error, 0.8 clean} -> precision 1/2, recall 1/2
        self.assertEqual(res[0.5]["flagged"], 2)
        self.assertAlmostEqual(res[0.5]["precision"], 0.5)
        self.assertAlmostEqual(res[0.5]["error_recall"], 0.5)
        # at 0.0 everything is flagged; ambiguous is excluded from precision
        self.assertEqual(res[0.0]["ambiguous"], 1)
        self.assertAlmostEqual(res[0.0]["precision"], 2 / 4)
        self.assertAlmostEqual(res[0.0]["error_recall"], 1.0)
