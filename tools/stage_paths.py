# stage_paths.py — put every pipeline stage on sys.path, in one place.
#
# WHY THIS EXISTS
# ---------------
# The stages import each other by bare module name: stage 5 pulls
# `ocr_metrics` from stage 3 and `amharic_canonical_normalizer` from stage 4,
# the detector's scripts pull `ocr_metrics` from stage 3, and the root test
# suite pulls from all of them. Stage folder names start with a digit
# (`4_ocr_correction`), so they can never be Python packages — `import
# 4_ocr_correction.x` is a syntax error. The only way to keep those bare
# imports working is to put each stage folder on sys.path directly.
#
# That used to be done with thirteen root-level shim modules, one per shared
# module (`ocr_metrics.py`, `suspicion.py`, `stage1_fit.py`, ...), each doing
# an importlib dance to forward to the real file inside a stage folder. They
# were pure indirection: nothing at the root except a pointer, and every new
# shared module needed a new shim written by hand.
#
# This module replaces all thirteen. Registering the stage *folders* makes
# every module inside them importable by name, so adding a shared module no
# longer requires touching anything at the root.
#
# USAGE
# -----
# From any script, at the top, before the cross-stage imports:
#
#   import os, sys
#   _r = os.path.dirname(os.path.abspath(__file__))
#   while _r != os.path.dirname(_r) and not os.path.exists(
#           os.path.join(_r, "tools", "stage_paths.py")):
#       _r = os.path.dirname(_r)
#   sys.path.insert(0, os.path.join(_r, "tools"))
#   import stage_paths
#   stage_paths.register()          # now `import repo_paths` and every
#                                   # stage module resolve
#
# `register()` is idempotent and cheap, so calling it more than once in a
# process is harmless. The root `conftest.py` calls it too, which is what lets
# `tests/` keep importing stage modules with no path setup of its own.
#
# `python tools/stage_paths.py` prints what it would register.

import os
import sys

# The stage folders, in pipeline order. A stage folder is on sys.path so that
# the modules inside it are importable by bare name from any other stage.
STAGE_DIRS = (
    "1_data_preparation",
    "2_training",
    "3_training_evaluation",
    "4_ocr_correction",
    "5_correction_evaluation",
)

# Folders that hold importable code but are not pipeline stages.
#
# The detector's scripts/ folder is here because stage 5 imports
# `audit_decision` from it: 05_collect_error_cases.py audits the detector's
# pass/gray/flag decisions, so that one module is genuinely cross-stage rather
# than private to the detector. Everything else in scripts/ is a CLI entry
# point and is meant to be run, not imported.
EXTRA_DIRS = ("tools", "4_ocr_correction/error_detection/scripts")

# This file's own marker: the repo root is the ancestor that contains it.
_MARKER = os.path.join("tools", "stage_paths.py")


def find_root(start=None):
    """Walk up from `start` (default: this file) to the repository root.

    Returns None rather than raising if the marker is never found, so a caller
    can degrade gracefully — e.g. when this module is imported from an
    installed copy or a partial checkout.
    """
    if start is None:
        # two levels up from tools/stage_paths.py
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.dirname(here)
        if os.path.exists(os.path.join(candidate, _MARKER)):
            return candidate
        start = here
    cur = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(cur, _MARKER)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def paths_to_register(root=None):
    """The directories `register()` will add, in the order they are added.

    Only existing directories are returned; a stage folder that is absent in a
    partial checkout is skipped rather than polluting sys.path.
    """
    root = root or find_root()
    if root is None:
        return []
    # Stage folders first so that a stage's own modules win over anything of
    # the same name elsewhere, then the root and the extra code folders.
    out = [os.path.join(root, s) for s in STAGE_DIRS]
    out += [os.path.join(root, d) for d in EXTRA_DIRS]
    out.append(root)
    return [p for p in out if os.path.isdir(p)]


def register(root=None, quiet=True):
    """Add the repo root, tools/, and every stage folder to sys.path.

    Returns the list of directories added (empty if they were all already
    present). Idempotent: repeated calls do not duplicate entries.
    """
    added = []
    for p in paths_to_register(root):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    if added and not quiet:
        for p in added:
            print(f"[stage_paths] + {p}")
    return added


if __name__ == "__main__":
    root = find_root()
    print(f"repo root: {root}")
    if root is None:
        raise SystemExit(1)
    print("would register:")
    for p in paths_to_register(root):
        print(f"  {p}")
