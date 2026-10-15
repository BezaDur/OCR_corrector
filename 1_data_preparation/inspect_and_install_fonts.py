#!/usr/bin/env python3
"""
Diagnostic & Installer Tool:
1. Audits all .ttf/.otf fonts in the fonts folder.
2. Copies them to ~/.local/share/fonts/amharic_fonts/.
3. Rebuilds the system Fontconfig cache.
4. Tests each font directly with text2image to verify Pango registration.
"""

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
import shutil
import subprocess
from fontTools.ttLib import TTFont

SOURCE_FONTS_DIR = repo_paths.workspace("fonts")
USER_FONTS_DIR = os.path.expanduser("~/.local/share/fonts/amharic_fonts")


def get_font_metadata(font_path):
    """Extracts internal Family Name (ID 1) and Full Name (ID 4) from font tables."""
    try:
        tt = TTFont(font_path)
        family = ""
        full = ""
        style = ""
        for r in tt["name"].names:
            try:
                name_str = r.toUnicode()
                if r.nameID == 1 and not family:
                    family = name_str
                elif r.nameID == 2 and not style:
                    style = name_str
                elif r.nameID == 4 and not full:
                    full = name_str
            except Exception:
                pass
        return family or "Unknown", style or "Regular", full or family
    except Exception as e:
        return "Corrupted", "Unknown", str(e)


def main():
    if not os.path.exists(SOURCE_FONTS_DIR):
        print(f"Error: Fonts directory not found at {SOURCE_FONTS_DIR}")
        return

    os.makedirs(USER_FONTS_DIR, exist_ok=True)
    font_files = sorted(glob.glob(os.path.join(SOURCE_FONTS_DIR, "*.[to]tf")))

    if not font_files:
        print(f"No font files found in {SOURCE_FONTS_DIR}")
        return

    print(f"Found {len(font_files)} font files in {SOURCE_FONTS_DIR}.")
    print("Installing fonts into user directory and refreshing Fontconfig cache...\n")

    # 1. Copy fonts into ~/.local/share/fonts/amharic_fonts
    for f in font_files:
        dest = os.path.join(USER_FONTS_DIR, os.path.basename(f))
        shutil.copy2(f, dest)

    # 2. Rebuild Fontconfig cache
    subprocess.run(["fc-cache", "-f", USER_FONTS_DIR], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Test text2image rendering on a sample Ethiopic sentence
    test_line = "ኢትዮጵያ ሀገሬ ሡራፌል ቯይታሚን 1234 ።"
    tmp_gt = "/tmp/font_install_test.txt"
    with open(tmp_gt, "w", encoding="utf-8") as tf:
        tf.write(test_line + "\n")

    print("=" * 85)
    print(f"{'FILE NAME':<30} | {'FAMILY NAME':<22} | {'PANGO STATUS':<10} | {'PANG0 FONT STRING'}")
    print("=" * 85)

    verified_fonts = []

    for fpath in font_files:
        fname = os.path.basename(fpath)
        family, style, full = get_font_metadata(fpath)

        # Test candidate strings with text2image
        candidate_strings = [full, f"{family} {style}", family]
        working_pango_name = None

        for cand in candidate_strings:
            res = subprocess.run([
                "text2image",
                f"--font={cand}",
                f"--text={tmp_gt}",
                "--outputbase=/tmp/font_test_out",
                "--max_pages=1",
                "--ptsize=14"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if res.returncode == 0:
                working_pango_name = cand
                break

        if working_pango_name:
            status = "✅ PASS"
            verified_fonts.append(working_pango_name)
            print(f"{fname:<30} | {family:<22} | {status:<10} | '{working_pango_name}'")
        else:
            status = "❌ FAIL"
            print(f"{fname:<30} | {family:<22} | {status:<10} | (Could not render via Pango)")

    print("=" * 85)
    print(f"\nAudit & Installation Complete!")
    print(f"Total Working Fonts for Tesseract: {len(verified_fonts)} / {len(font_files)}")
    print(f"Installed to: {USER_FONTS_DIR}")


if __name__ == "__main__":
    main()
