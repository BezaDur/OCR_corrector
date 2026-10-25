#!/usr/bin/env python3
# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "tools", "stage_paths.py")):
    _root = _os.path.dirname(_root)
# tools/ first so stage_paths and repo_paths resolve, then register every stage
# folder so cross-stage imports work by bare module name. This replaces the
# thirteen root-level shims that used to forward those imports by hand.
_sys.path.insert(0, _os.path.join(_root, "tools"))
import stage_paths  # noqa: E402  (imported after the path bootstrap)
stage_paths.register(_root)
import repo_paths  # noqa: E402
# ----------------------------------------------------------------------------

import os
import glob
from fontTools.ttLib import TTFont

FONTS_DIR = repo_paths.fonts()

header_file = "FILE NAME"
header_pango = "EXACT PANGO NAME"
print(f"{header_file:<32} | {header_pango}")
print("-" * 70)

for p in sorted(glob.glob(os.path.join(FONTS_DIR, "*.[to]tf"))):
    fname = os.path.basename(p)
    try:
        tt = TTFont(p)
        names = {}
        for r in tt["name"].names:
            if r.nameID in (1, 4):
                try:
                    names[r.nameID] = r.toUnicode()
                except Exception:
                    pass
        # ID 4 is Full Font Name, ID 1 is Font Family
        pango_name = names.get(4) or names.get(1) or "Unknown"
        print(f"{fname:<32} | {pango_name}")
    except Exception as e:
        print(f"{fname:<32} | Error: {e}")