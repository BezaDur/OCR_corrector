import json
import os
import random
import tempfile
import unittest

from detection.detector import ErrorDetector
from detection.train import load_rows, sweep


def synthetic(n_pages=3, n=200, seed=0):
    rnd = random.Random(seed)
    rows = []
    for p in range(n_pages):
        for _ in range(n):
            err = rnd.random() < 0.2
            x = rnd.gauss(1.0 if err else 0.0, 1.0)
            rows.append({"page": f"p{p}", "status": "error" if err else "clean",
                         "features": {"x": x, "noise": rnd.random()}})
    return rows


class TestDetector(unittest.TestCase):
    def test_fit_score_and_flag(self):
        rows = synthetic()
        feats = [r["features"] for r in rows]
        labels = [1 if r["status"] == "error" else 0 for r in rows]
        det = ErrorDetector.fit(feats, labels, ["x", "noise"], threshold=0.5)
        scores = det.score(feats)
        self.assertEqual(len(scores), len(rows))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        flags = det.flag(feats)
        err_flagged = sum(f for f, y in zip(flags, labels) if y)
        cln_flagged = sum(f for f, y in zip(flags, labels) if not y)
        self.assertGreater(err_flagged, cln_flagged)

    def test_save_and_load_round_trip(self):
        det = ErrorDetector(["x"], [0.0], [1.0], [2.0], -1.0, threshold=0.4)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            det.save(path)
            again = ErrorDetector.load(path)
        finally:
            os.remove(path)
        self.assertEqual(again.to_dict(), det.to_dict())
        self.assertEqual(again.score([{"x": 1.0}]), det.score([{"x": 1.0}]))

    def test_load_rows_keeps_only_labelled_rows_with_features(self):
        d = tempfile.mkdtemp()
        rows = [{"status": "clean", "features": {"x": 0.0}, "page": "a"},
                {"status": "ambiguous", "features": {"x": 0.0}, "page": "a"},
                {"status": "error"}]
        with open(os.path.join(d, "a.verdicts.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        self.assertEqual([r["status"] for r in load_rows(d)], ["clean"])

    def test_sweep_precision_and_recall(self):
        rows = [{"status": "error"}, {"status": "clean"}, {"status": "error"}]
        res = {s["threshold"]: s for s in sweep([0.9, 0.8, 0.1], rows, [0.5])}
        self.assertEqual(res[0.5]["flagged"], 2)
        self.assertAlmostEqual(res[0.5]["precision"], 0.5)
        self.assertAlmostEqual(res[0.5]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
