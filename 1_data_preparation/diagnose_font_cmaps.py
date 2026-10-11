#!/usr/bin/env python3
"""
Inspects the binary CMAP tables of all fonts to find which ones lack
Arabic digits, quotes, colons, or Ethiopic punctuation.
"""

import os
import glob
from fontTools.ttLib import TTFont

FONTS_DIR = "/media/des/Tess/Tesseract/TessTraining/Fonts"

TEST_SETS = {
    "Digits (0-9)": "0123456789",
    "ASCII Punctuation (: . , -)": ":.,-",
    "Curly Quotes (“ ”)": "“”",
    "Ethiopic Punctuation (። ፡ ፤)": "።፡፤፥፦",
    "Ethiopic Numerals (፩ ፪ ፫)": "፩፪፫፬፭፮፯፰፱፲"
}

def get_font_supported_chars(font_path):
    """Extracts all mapped Unicode codepoints from the font's CMAP table."""
    font = TTFont(font_path)
    codepoints = set()
    for table in font['cmap'].tables:
        if table.isUnicode():
            codepoints.update(table.cmap.keys())
    return codepoints

def main():
    font_paths = sorted(glob.glob(os.path.join(FONTS_DIR, "*.[to]tf")))
    print(f"Scanning {len(font_paths)} font files in {FONTS_DIR}...\n")

    for path in font_paths:
        name = os.path.basename(path)
        try:
            supported = get_font_supported_chars(path)
            failures = []

            for label, chars in TEST_SETS.items():
                missing = [c for c in chars if ord(c) not in supported]
                if missing:
                    failures.append(f"{label} missing: {''.join(missing)}")

            if not failures:
                print(f"✅ {name:<35} : FULL SUPPORT for all tests")
            else:
                print(f"❌ {name:<35} : {'; '.join(failures)}")

        except Exception as e:
            print(f"⚠️ {name:<35} : Error reading font ({e})")

if __name__ == "__main__":
    main()