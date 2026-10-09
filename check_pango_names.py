#!/usr/bin/env python3
import os
import glob
from fontTools.ttLib import TTFont

FONTS_DIR = "/media/des/Tess/Tesseract/TessTraining/Fonts"

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