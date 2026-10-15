#!/usr/bin/env python3
"""
Audits the exact frequencies of all target rare characters, labiovelars,
numerals, and publication symbols in amh.training_text.
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
import collections

TRAINING_TEXT = repo_paths.workspace("amh.training_text")

# Target Rare Characters Grouped by Category
CATEGORIES = {
    "1. Rare Syllables (ሡ, ሤ, ቯ, ዤ, etc.)": [
        'ሡ', 'ሤ', 'ቯ', 'ዤ', 'ጬ', 'ጿ', 'ጰ', 'ጱ', 'ጲ', 'ጴ', 'ጶ', 'ኂ', 'ኄ', 'ኆ', 'ኍ', 'ዧ'
    ],
    "2. Dikala Labiovelars (ቈ, ኈ, ኰ, ጐ series)": [
        'ቈ', 'ቊ', 'ቌ', 'ቍ', 'ኈ', 'ኊ', 'ኌ', 'ኰ', 'ኲ', 'ኴ', 'ኵ', 'ጐ', 'ጒ', 'ጔ', 'ጕ',
        'ጯ', 'ዷ', 'ጧ', 'ጇ', 'ሿ', 'ሟ', 'ሯ', 'ፏ', 'ኋ', 'ኳ', 'ጓ'
    ],
    "3. Ethiopic Numerals (፩ to ፼)": [
        '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻', '፼'
    ],
    "4. Ethiopic Punctuation": [
        '።', '፡', '፣', '፤', '፥', '፦', '፧', '፨', '፠'
    ],
    "5. Publication & Typography Symbols": [
        '…', '•', '※', '†', '―', '–', '—', '“', '”', '‘', '’', '«', '»', '‹', '›', '„', '‟', '/', '.'
    ]
}


def main():
    if not os.path.exists(TRAINING_TEXT):
        print(f"Error: {TRAINING_TEXT} not found.")
        return

    char_counts = collections.defaultdict(int)
    total_lines = 0

    with open(TRAINING_TEXT, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            for ch in line:
                char_counts[ch] += 1

    print("=" * 75)
    print(f"RARE GLYPH CENSUS IN amh.training_text ({total_lines:,} lines)")
    print("=" * 75)

    underrepresented = []

    for cat_name, chars in CATEGORIES.items():
        print(f"\n{cat_name}:")
        print("-" * 75)
        print(f"{'CHAR':<6} | {'UNICODE':<8} | {'COUNT':<10} | {'STATUS'}")
        print("-" * 75)
        for ch in chars:
            cnt = char_counts[ch]
            if cnt >= 500:
                status = "✅ Abundant"
            elif cnt >= 100:
                status = "⚠️ Moderate"
            elif cnt > 0:
                status = "❌ Low (<100)"
                underrepresented.append((ch, cnt))
            else:
                status = "🚫 ZERO (Missing!)"
                underrepresented.append((ch, 0))

            print(f"'{ch}'    | U+{ord(ch):04X}  | {cnt:8,d}   | {status}")

    print("\n" + "=" * 75)
    print(f"AUDIT SUMMARY:")
    print(f"Total monitored rare glyphs: {sum(len(v) for v in CATEGORIES.values())}")
    print(f"Critically under-represented glyphs (<100 instances): {len(underrepresented)}")
    if underrepresented:
        print("Under-represented list: " + " ".join(f"'{c}'({n})" for c, n in underrepresented))
    print("=" * 75)


if __name__ == "__main__":
    main()
