import json
import os
import random
import shutil
import tempfile
import unittest

from detection.detector import ErrorDetector
from detection.evaluate import evaluate, sweep
from detection.examples import check, read_examples, scored, write_examples
from detection.train import train
from detection.adapters import plain_words, tesseract_tsv, verdicts

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def synthetic(n_docs=3, n=200, seed=0):
    rnd = random.Random(seed)
    out = []
    for d in range(n_docs):
        for i in range(n):
            err = rnd.random() < 0.2
            x = rnd.gauss(1.0 if err else 0.0, 1.0)
            out.append(dict(doc=f"d{d}", index=i, text="w", label="error" if err else "clean",
                            features={"x": x, "noise": rnd.random()}))
    return out


class TestExamples(unittest.TestCase):
    def test_round_trip_and_checks(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "e.jsonl")
            ex = synthetic(n_docs=1, n=5)
            self.assertEqual(write_examples(p, ex), 5)
            self.assertEqual(read_examples(p)[0]["features"], ex[0]["features"])
        finally:
            shutil.rmtree(d)

    def test_non_numeric_feature_rejected(self):
        with self.assertRaises(ValueError):
            check(dict(doc="a", text="x", label=None, features={"f": "high"}))

    def test_bad_label_rejected(self):
        with self.assertRaises(ValueError):
            check(dict(doc="a", text="x", label="maybe", features={}))

    def test_scored_keeps_only_clean_and_error(self):
        ex = [dict(doc="a", text="x", label=l, features={}) for l in ("clean", "error", "ambiguous", None)]
        self.assertEqual([e["label"] for e in scored(ex)], ["clean", "error"])


class TestAdapters(unittest.TestCase):
    def test_plain_words_gives_conf_feature_only_when_present(self):
        out = plain_words.to_examples([dict(doc="a", text="ሰላም", conf=90.0),
                                       dict(doc="a", text="ቤት")])
        self.assertIn("ocr_conf", out[0]["features"])
        self.assertNotIn("ocr_conf", out[1]["features"])

    def test_tesseract_adapter_skips_containers_and_minus_one(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "page_0001.tsv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(HEADER + "\n")
                fh.write("2\t1\t1\t0\t0\t0\t0\t0\t10\t10\t-1\t\n")
                fh.write("5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tሰላም\n")
                fh.write("5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t20\tቤቱ\n")
                fh.write("5\t1\t1\t1\t1\t3\t0\t0\t10\t10\t-1\tጉዳይ\n")
            ex = tesseract_tsv.to_examples(p, gt_text="ሰላም ቤት")
        finally:
            shutil.rmtree(d)
        self.assertEqual([e["text"] for e in ex], ["ሰላም", "ቤቱ"])
        self.assertEqual([e["label"] for e in ex], ["clean", "error"])
        self.assertEqual(ex[1]["features"]["ocr_conf"], 20.0)
        self.assertEqual(ex[0]["doc"], "page_0001")

    def test_tesseract_adapter_without_gt_is_unlabelled(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "p.tsv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(HEADER + "\n5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tሰላም\n")
            ex = tesseract_tsv.to_examples(p)
        finally:
            shutil.rmtree(d)
        self.assertIsNone(ex[0]["label"])

    def test_verdict_adapter(self):
        d = tempfile.mkdtemp()
        try:
            rows = [{"page": "p1", "word": "ሰላም", "status": "clean", "features": {"x": 0.5}},
                    {"page": "p1", "word": "ቤቱ", "status": "error", "features": {"x": 2.0}}]
            with open(os.path.join(d, "p1.verdicts.jsonl"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
            ex = verdicts.to_examples(d)
        finally:
            shutil.rmtree(d)
        self.assertEqual([e["label"] for e in ex], ["clean", "error"])
        self.assertEqual(ex[1]["features"], {"x": 2.0})


class TestEvaluate(unittest.TestCase):
    def test_sweep_counts(self):
        res = {r["threshold"]: r for r in sweep([0.9, 0.8, 0.1], [1, 0, 1], [0.5])}
        self.assertEqual(res[0.5]["flagged"], 2)
        self.assertAlmostEqual(res[0.5]["precision"], 0.5)
        self.assertAlmostEqual(res[0.5]["recall"], 0.5)

    def test_evaluate_reports_per_document(self):
        ex = synthetic(n_docs=2, n=50)
        det = ErrorDetector.fit([e["features"] for e in ex],
                                [1 if e["label"] == "error" else 0 for e in ex], ["x", "noise"])
        rep = evaluate(ex, det)
        self.assertEqual(set(rep["per_doc"]), {"d0", "d1"})
        self.assertEqual(rep["words"], len(ex))
        self.assertGreater(rep["auc"], 0.5)

    def test_evaluate_needs_labels(self):
        det = ErrorDetector(["x"], [0.0], [1.0], [1.0], 0.0)
        with self.assertRaises(ValueError):
            evaluate([dict(doc="a", text="x", label=None, features={"x": 1.0})], det)


class TestTrain(unittest.TestCase):
    def test_train_reports_out_of_sample_and_drops_conf_when_asked(self):
        ex = synthetic(n_docs=3, n=150)
        for e in ex:
            e["features"]["ocr_conf"] = 50.0
        det, rep = train(ex, target=0.5, drop=["ocr_conf"])
        self.assertNotIn("ocr_conf", det.names)
        self.assertEqual(rep["note"], "out-of-sample (leave-one-document-out)")
        self.assertGreater(rep["auc"], 0.5)

    def test_train_uses_conf_when_present(self):
        ex = synthetic(n_docs=2, n=60)
        for e in ex:
            e["features"]["ocr_conf"] = 50.0
        det, _ = train(ex)
        self.assertIn("ocr_conf", det.names)


class TestDetectorIO(unittest.TestCase):
    def test_save_load_and_score(self):
        det = ErrorDetector(["x"], [0.0], [1.0], [2.0], -1.0, threshold=0.4)
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "m.json")
            det.save(p)
            again = ErrorDetector.load(p)
        finally:
            shutil.rmtree(d)
        self.assertEqual(again.score([{"x": 1.0}]), det.score([{"x": 1.0}]))


if __name__ == "__main__":
    unittest.main()


class TestBuildExamples(unittest.TestCase):
    def test_tsv_folder_with_gt_labels_each_page(self):
        from detection.build_examples import from_tsv_dir
        d = tempfile.mkdtemp()
        try:
            tdir = os.path.join(d, "tsv"); gdir = os.path.join(d, "gt")
            os.makedirs(tdir); os.makedirs(gdir)
            with open(os.path.join(tdir, "page_0001.tsv"), "w", encoding="utf-8") as fh:
                fh.write(HEADER + "\n5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tሰላም\n5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t20\tቤቱ\n")
            with open(os.path.join(gdir, "page_0001.gt.txt"), "w", encoding="utf-8") as fh:
                fh.write("ሰላም ቤት\n")
            ex = from_tsv_dir(tdir, gdir)
        finally:
            shutil.rmtree(d)
        self.assertEqual([e["doc"] for e in ex], ["page_0001", "page_0001"])
        self.assertEqual([e["label"] for e in ex], ["clean", "error"])
