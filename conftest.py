# conftest.py — make the stage folders importable for the whole test suite.
#
# tests/ imports modules from every stage by bare name (`from ocr_metrics
# import align_tokens`, `from stage1_labels import label_verdicts`, ...), and
# the detector's own tests import `error_detection`. Stage folders start with a
# digit, so they can never be Python packages; the only way those bare imports
# resolve is for each stage folder to be on sys.path.
#
# pytest loads this file automatically before collecting anything, which is why
# the individual test modules need no path setup of their own. Running a test
# module directly (`python tests/test_ocr_metrics.py`) does not load conftest,
# so tools/stage_paths.py documents the three-line bootstrap for that case.

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import stage_paths  # noqa: E402

stage_paths.register(_ROOT)
