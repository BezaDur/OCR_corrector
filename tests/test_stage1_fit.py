import unittest

from stage1_fit import auc, fit_logreg, predict, leave_one_page_out, feature_names


class TestStage1Fit(unittest.TestCase):
    def test_auc_perfect_and_chance(self):
        self.assertAlmostEqual(auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(auc([0.5, 0.5], [1, 0]), 0.5)

    def test_logreg_learns_a_separable_feature(self):
        X = [[0.0], [0.1], [0.9], [1.0]] * 5
        y = [0, 0, 1, 1] * 5
        m = fit_logreg(X, y, l2=0.01, iters=300)
        p = predict(m, [[0.0], [1.0]])
        self.assertLess(p[0], 0.5)
        self.assertGreater(p[1], 0.5)

    def test_leave_one_page_out_covers_every_row(self):
        rows = []
        for page in ("a", "b", "c"):
            for k, (x, st) in enumerate([(0.0, "clean"), (1.0, "error"), (0.1, "clean"), (0.9, "error")]):
                rows.append({"page": page, "status": st, "features": {"x": x}})
        self.assertEqual(feature_names(rows), ["x"])
        scores = leave_one_page_out(rows, ["x"])
        self.assertTrue(all(s is not None for s in scores))
        errs = [s for s, r in zip(scores, rows) if r["status"] == "error"]
        cln = [s for s, r in zip(scores, rows) if r["status"] == "clean"]
        self.assertGreater(min(errs), max(cln))


if __name__ == "__main__":
    unittest.main()
