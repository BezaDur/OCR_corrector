#!/usr/bin/env python3
"""
Purges all redundant, condensed, and unsupported font files from:
1. /media/des/Tess/Tesseract2/fonts/
2. ~/.local/share/fonts/amharic_fonts/
Leaving strictly the 20 curated, verified production fonts.
"""

import os
import glob
import subprocess

APPROVED_FONT_FILES = {
    "AbyssinicaSIL-R.ttf",
    "nyala.ttf",
    "Brana-Regular.ttf",
    "FreeSerif.ttf",
    "NotoSerifEthiopic-Regular.ttf",
    "NotoSerifEthiopic-Bold.ttf",
    "NotoSerifEthiopic-Light.ttf",
    "NotoSansEthiopic-Regular.ttf",
    "NotoSansEthiopic-Bold.ttf",
    "NotoSansEthiopic-Light.ttf",
    "jiret.ttf",
    "jiretsl.ttf",
    "washra-Slant.ttf",
    "washra-BS.ttf",
    "EthiopicAbay-Regular.ttf",
    "EthiopicLessan-Regular.ttf",
    "EthiopicSadiss-Regular.ttf",
    "hiwua.ttf",
    "wookianos.ttf",
    "yebse.ttf"
}

TARGET_DIRS = [
    "/media/des/Tess/Tesseract2/fonts",
    os.path.expanduser("~/.local/share/fonts/amharic_fonts")
]

def clean_directory(folder_path):
    if not os.path.exists(folder_path):
        return

    print(f"Cleaning: {folder_path}...")
    all_files = glob.glob(os.path.join(folder_path, "*.[to]tf"))
    removed_count = 0

    for fpath in all_files:
        fname = os.path.basename(fpath)
        if fname not in APPROVED_FONT_FILES:
            try:
                os.remove(fpath)
                removed_count += 1
            except OSError as e:
                print(f"  Error removing {fname}: {e}")

    remaining = glob.glob(os.path.join(folder_path, "*.[to]tf"))
    print(f"  Removed {removed_count} redundant files.")
    print(f"  Preserved {len(remaining)} approved font files.")


def main():
    print("=" * 65)
    print("      AMHARIC PRODUCTION FONT REPOSITORY CLEANUP      ")
    print("=" * 65)

    for target in TARGET_DIRS:
        clean_directory(target)

    # Rebuild system font cache
    print("\nRefreshing Fontconfig system cache...")
    subprocess.run(["fc-cache", "-f", os.path.expanduser("~/.local/share/fonts/amharic_fonts")], stdout=subprocess.DEVNULL)
    print("Font cache refreshed cleanly.")
    print("=" * 65)


if __name__ == "__main__":
    main()
