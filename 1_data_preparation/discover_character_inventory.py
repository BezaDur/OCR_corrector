#!/usr/bin/env python3
"""
Step 1: Character Inventory Discovery Tool.
Scans the non-MC4 dataset to extract the true, empirical Amharic character set.
Categorizes every discovered character to define the final, clean target unicharset.
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
import unicodedata
import collections

DATASET_DIR = repo_paths.WORD_DATASET
OUTPUT_REPORT = repo_paths.workspace("discovered_character_inventory.txt")

def categorize_character(ch):
    cp = ord(ch)
    # Ethiopic Unicode Block: U+1200 - U+137F
    # Ethiopic Supplement: U+1380 - U+139F
    # Ethiopic Extended:   U+2D80 - U+2DDF
    if 0x1200 <= cp <= 0x137C:
        if 0x1360 <= cp <= 0x1368:
            return "ETHIOPIC_PUNCTUATION"
        elif 0x1369 <= cp <= 0x137C:
            return "ETHIOPIC_NUMERAL"
        else:
            return "ETHIOPIC_SYLLABARY"
    elif 0x1380 <= cp <= 0x139F or 0x2D80 <= cp <= 0x2DDF:
        return "ETHIOPIC_EXTENDED"
    elif ch in "0123456789":
        return "ARABIC_DIGIT"
    elif ch in ".,:;!?\"'()[]{}/-–—%":
        return "STANDARD_PUNCTUATION"
    elif ch in "“”‘’«»‹›":
        return "CURLY_QUOTES"
    elif 0x0020 <= cp <= 0x007E:
        return "ASCII_OTHER"
    else:
        return "FOREIGN_OR_SYMBOL"

def main():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    # EXCLUDE MC4 files (starting with 048_)
    active_files = [f for f in files if not os.path.basename(f).startswith("048_")]

    print(f"Total files discovered:  {len(files)}")
    print(f"Active non-MC4 files:    {len(active_files)}")
    print("Beginning character inventory scan (sampling up to first 25,000 lines per file)...\n")

    char_counts = collections.defaultdict(int)
    total_lines = 0

    for file_path in active_files:
        base_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= 25000:  # Deep statistical sample across every file
                        break
                    total_lines += 1
                    for ch in line:
                        if ch not in ('\n', '\r', ' '):
                            char_counts[ch] += 1
        except Exception as e:
            print(f"Error reading {base_name}: {e}")

    # Categorize all discovered characters
    categories = collections.defaultdict(list)
    for ch, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        cat = categorize_character(ch)
        categories[cat].append((ch, count))

    # Export Human-Readable Audit Report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as out:
        out.write("=====================================================\n")
        out.write("     EMPIRICAL AMHARIC CHARACTER INVENTORY AUDIT     \n")
        out.write("=====================================================\n")
        out.write(f"Total Lines Sampled:        {total_lines:,}\n")
        out.write(f"Total Unique Characters:    {len(char_counts):,}\n\n")

        for cat_name, char_list in categories.items():
            out.write("-" * 55 + "\n")
            out.write(f"CATEGORY: {cat_name} ({len(char_list)} unique characters)\n")
            out.write("-" * 55 + "\n")
            out.write(f"Character List: {' '.join(c for c, _ in char_list)}\n\n")
            out.write("Top / Rare breakdown:\n")
            for c, cnt in char_list[:15]:
                out.write(f"  '{c}' (U+{ord(c):04X}) : {cnt:,} occurrences\n")
            if len(char_list) > 15:
                out.write("  ...\n")
                for c, cnt in char_list[-10:]:
                    out.write(f"  '{c}' (U+{ord(c):04X}) : {cnt:,} occurrences (RARE)\n")
            out.write("\n")

    print(f"Inventory Audit Finished! Check results: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()