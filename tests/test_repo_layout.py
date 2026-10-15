#!/usr/bin/env python3
"""Structural invariants of the repository layout.

These are the checks that a reorganization silently breaks and that the rest of
the suite cannot see:

* `4_ocr_correction/tests/test_corrector.py` cannot even be collected without a
  KenLM model, so nothing in the normal run ever exercises the corrector's
  default data paths. When the detector moved into stage 4, those defaults kept
  resolving against the repository root and quietly lost the `4_ocr_correction`
  path component -- a wrong-but-plausible path that only fails at construction
  time, in a file the suite skips. `test_corrector_default_paths_exist` is the
  guard that would have caught it, and it needs no KenLM.
* Stage folders start with a digit, so they can never be Python packages. Every
  cross-stage import depends on `tools/stage_paths.py` having put those folders
  on sys.path. If that registrar stops matching the folders on disk, imports
  fail in confusing ways, so the two lists are checked against each other.
* The root used to carry thirteen shim modules that forwarded imports into the
  stages. They are gone; if one comes back it is a step backwards, so their
  absence is asserted rather than assumed.

Run:  python -m pytest tests/test_repo_layout.py -v
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import repo_paths      # noqa: E402
import stage_paths     # noqa: E402


class TestRootDiscovery(unittest.TestCase):
    """stage_paths must agree with where this test file actually is."""

    def test_find_root_matches_the_test_file_location(self):
        self.assertEqual(Path(stage_paths.find_root()).resolve(), ROOT.resolve())

    def test_marker_file_exists(self):
        self.assertTrue((ROOT / "tools" / "stage_paths.py").is_file())

    def test_repo_paths_root_is_the_repo_root(self):
        self.assertEqual(Path(repo_paths.REPO_ROOT).resolve(), ROOT.resolve())


class TestStageDirsAreReal(unittest.TestCase):
    """The registrar's list must match the folders on disk.

    A stage folder that is renamed or removed without updating STAGE_DIRS drops
    every module in it out of the flat import namespace, and the failure shows
    up as an unrelated ModuleNotFoundError somewhere else.
    """

    def test_every_declared_stage_dir_exists(self):
        for name in stage_paths.STAGE_DIRS:
            with self.subTest(stage=name):
                self.assertTrue((ROOT / name).is_dir(),
                                f"declared stage folder is missing: {name}")

    def test_every_numbered_top_level_dir_is_declared(self):
        on_disk = sorted(p.name for p in ROOT.iterdir()
                         if p.is_dir() and p.name[:1].isdigit())
        self.assertEqual(on_disk, sorted(stage_paths.STAGE_DIRS))

    def test_register_is_idempotent(self):
        first = stage_paths.register(str(ROOT))
        second = stage_paths.register(str(ROOT))
        self.assertEqual(second, [])
        self.assertIsInstance(first, list)


class TestDetectorLivesInsideStageFour(unittest.TestCase):
    """The detector is stage 4's first half, so it sits inside stage 4."""

    def test_package_location(self):
        pkg = ROOT / "4_ocr_correction" / "error_detection"
        self.assertTrue((pkg / "__init__.py").is_file())
        self.assertTrue((pkg / "detector.py").is_file())

    def test_old_top_level_location_is_gone(self):
        self.assertFalse((ROOT / "OCR_error_detection").exists())

    def test_evaluation_corpus_is_shared_at_the_root(self):
        """Stage 3 and the detector both read these pages, so the corpus lives
        above both rather than inside either."""
        for sub in ("GroundTruth", "OCRed", "images"):
            with self.subTest(sub=sub):
                self.assertTrue((ROOT / "Evaluation" / sub).is_dir())
        self.assertFalse((ROOT / "4_ocr_correction" / "error_detection"
                          / "Evaluation").exists())


class TestCorrectorDefaultPaths(unittest.TestCase):
    """The regression test for the bug the reorganization introduced.

    Only checks that the documented defaults point at files that exist; it does
    not construct the corrector, so it needs no KenLM model.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from amharic_ocr_corrector import AmharicOCRCorrector
        except Exception as exc:            # pragma: no cover - import guard
            raise unittest.SkipTest(f"corrector module unavailable: {exc}")
        cls.corrector_cls = AmharicOCRCorrector

    def test_default_paths_exist(self):
        for attr in ("DEFAULT_CONFUSION_MATRIX", "DEFAULT_DICT_VOCAB"):
            path = getattr(self.corrector_cls, attr)
            with self.subTest(attr=attr):
                self.assertTrue(os.path.isabs(path), f"{attr} is not absolute: {path}")
                self.assertTrue(os.path.exists(path),
                                f"{attr} does not exist: {path}")

    def test_default_paths_are_inside_stage_four(self):
        """Guards the specific failure mode: resolving against the repo root and
        silently dropping the 4_ocr_correction component."""
        stage4 = str(ROOT / "4_ocr_correction")
        for attr in ("DEFAULT_CONFUSION_MATRIX", "DEFAULT_DICT_VOCAB"):
            with self.subTest(attr=attr):
                self.assertTrue(getattr(self.corrector_cls, attr).startswith(stage4 + os.sep),
                                f"{attr} is not under 4_ocr_correction/: "
                                f"{getattr(self.corrector_cls, attr)}")

    def test_repo_paths_model_dir_exists(self):
        self.assertTrue(os.path.isdir(repo_paths.REPO_MODEL_DIR),
                        repo_paths.REPO_MODEL_DIR)
        self.assertTrue(os.path.isfile(repo_paths.main_i7_dict()),
                        repo_paths.main_i7_dict())
        self.assertTrue(os.path.isfile(repo_paths.confusion_matrix()),
                        repo_paths.confusion_matrix())


class TestRootShimsAreGone(unittest.TestCase):
    """Nothing importable belongs at the repo root except conftest.py.

    The thirteen shims that used to live here forwarded bare imports into the
    stage folders; tools/stage_paths.py replaced all of them by registering the
    folders instead. A shim reappearing at the root means the indirection came
    back with it.
    """

    REMOVED_SHIMS = (
        "amharic_canonical_normalizer.py", "amharic_ocr_corrector.py",
        "generate_eval_dataset.py", "generate_eval_pages.py", "normalizer.py",
        "ocr_decision_audit.py", "ocr_metrics.py", "stage1_fit.py",
        "stage1_labels.py", "suspicion.py", "tsv_confidence.py", "tsv_gt.py",
        "tsv_gt_stats.py", "repo_paths.py",
    )

    def test_no_shim_modules_at_the_root(self):
        for name in self.REMOVED_SHIMS:
            with self.subTest(module=name):
                self.assertFalse((ROOT / name).exists(),
                                 f"{name} is back at the repo root; it belongs in "
                                 "a stage folder or in tools/")

    def test_root_python_files_are_only_conftest(self):
        stray = sorted(p.name for p in ROOT.glob("*.py"))
        self.assertEqual(stray, ["conftest.py"])


class TestNumberingIsPerStage(unittest.TestCase):
    """A number means 'pipeline step, in this stage, in order'.

    The old global counter spread 01..15 across the five folders and left two
    different files both named 09_, so a filename said nothing about where it
    belonged. These checks keep the per-stage convention from drifting back.
    """

    def test_no_duplicate_step_numbers_within_a_stage(self):
        for stage in stage_paths.STAGE_DIRS:
            seen = {}
            for p in sorted((ROOT / stage).iterdir()):
                if not p.is_file():
                    continue
                head = p.name.split("_", 1)[0]
                if head.isdigit():
                    seen.setdefault(head, []).append(p.name)
            for num, names in seen.items():
                with self.subTest(stage=stage, number=num):
                    self.assertEqual(len(names), 1,
                                     f"{stage} has two step {num} scripts: {names}")

    def test_step_numbers_start_at_one_and_are_contiguous(self):
        for stage in stage_paths.STAGE_DIRS:
            nums = sorted(int(p.name.split("_", 1)[0])
                          for p in (ROOT / stage).iterdir()
                          if p.is_file() and p.name[:1].isdigit()
                          and p.name.split("_", 1)[0].isdigit())
            if nums:
                with self.subTest(stage=stage):
                    self.assertEqual(nums, list(range(1, len(nums) + 1)),
                                     f"{stage} step numbers are not 1..N: {nums}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
