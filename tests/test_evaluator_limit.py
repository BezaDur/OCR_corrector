"""Smoke-run option for 04_evaluate_full_corrector.py: --limit-pages.

Uses a stub corrector, so it runs without kenlm, symspellpy or the dataset.
Run: python -m unittest discover -s tests -p 'test_evaluator_limit.py'
"""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluator_limit", ROOT / "04_evaluate_full_corrector.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class _Cache:
    def cache_info(self):
        return "stub"


class StubCorrector:
    def correct_text_block(self, text):
        return text

    def morphology_stats(self):
        return {}

    _generate_candidates_cached = _Cache()
    _validate_morphology_cached = _Cache()


class TestLimitPages(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.gt = self.tmp / "gt"
        self.raw = self.tmp / "raw"
        self.gt.mkdir()
        self.raw.mkdir()
        for i in range(5):
            name = f"page{i:02d}"
            (self.gt / f"{name}.gt.txt").write_text(f"ሰላም {i}\n", encoding="utf-8")
            (self.raw / f"{name}.ocr.txt").write_text(f"ሰላም {i}\n", encoding="utf-8")
        self.saved = {k: getattr(evaluator, k) for k in (
            "GT_DIR", "RAW_OCR_DIR", "CORRECTED_OUT_DIR", "LIMIT_PAGES",
            "EXPECTED_PAGES", "JSON_REPORT", "MAX_DAMAGE_RATE", "USE_MP")}
        evaluator.GT_DIR = str(self.gt)
        evaluator.RAW_OCR_DIR = str(self.raw)
        evaluator.CORRECTED_OUT_DIR = str(self.tmp / "out")
        evaluator.EXPECTED_PAGES = None
        evaluator.JSON_REPORT = None
        evaluator.MAX_DAMAGE_RATE = None
        evaluator.USE_MP = False

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(evaluator, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, limit):
        evaluator.LIMIT_PAGES = limit
        out = io.StringIO()
        with patch.object(evaluator, "build_corrector", return_value=StubCorrector()), \
                contextlib.redirect_stdout(out):
            report = evaluator.main()
        return report

    def test_limit_takes_first_n_pages_in_sorted_order(self):
        report = self.run_main(2)
        self.assertEqual(report["completed_pages"], 2)
        self.assertEqual([p["page"] for p in report["pages"]], ["page00", "page01"])
        self.assertEqual(report["limit_pages"], 2)

    def test_no_limit_runs_every_page(self):
        report = self.run_main(None)
        self.assertEqual(report["completed_pages"], 5)
        self.assertIsNone(report["limit_pages"])

    def test_limit_larger_than_dataset_runs_all(self):
        report = self.run_main(99)
        self.assertEqual(report["completed_pages"], 5)

    def test_expected_pages_still_checks_full_dataset(self):
        evaluator.EXPECTED_PAGES = 7
        with self.assertRaises(ValueError):
            self.run_main(2)

    def test_cli_rejects_non_positive_limit(self):
        argv = ["prog", "--limit-pages", "0"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluator.configure_from_cli()

    def test_cli_sets_limit(self):
        argv = ["prog", "--limit-pages", "3"]
        with patch.object(sys, "argv", argv):
            evaluator.configure_from_cli()
        self.assertEqual(evaluator.LIMIT_PAGES, 3)


if __name__ == "__main__":
    unittest.main()
