#!/usr/bin/env python3
"""test_child.py — run the child project's test suite.

Discovers and runs every test under OCR_error_detection/tests/ (the
self-contained child module's contract tests). Works from any working
directory.

Usage:
    python3 OCR_error_detection/scripts/test_child.py [-v|--verbose]

Exit code 0 iff every test passes.
"""
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]  # repo root
TESTS_DIR = _HERE.parents[1] / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv):
    verbosity = 2 if any(a in ("-v", "--verbose") for a in argv) else 1
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TESTS_DIR), pattern="test_*.py", top_level_dir=str(ROOT)
    )
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    print()
    print(
        f"test_child: ran {result.testsRun} tests — "
        f"{len(result.failures)} failures, {len(result.errors)} errors, "
        f"{len(result.skipped)} skipped"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
