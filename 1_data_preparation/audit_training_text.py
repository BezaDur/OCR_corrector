#!/usr/bin/env python3
"""
Audits langdata/amh.training_text against langdata/amh.unicharset:
1. Detects characters in unicharset that NEVER appear in training_text (Zero-shot blind spots).
2. Counts occurrences of critical rare characters (like ሡ).
3. Audits representation of the 15 most confused Sadis/Ge'ez pairs.
4. Checks line length distributions.
"""

# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "repo_paths.py")):
    _root = _os.path.dirname(_root)
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import repo_paths  # noqa: E402  (imported after the path bootstrap)
# ----------------------------------------------------------------------------

import os
import collections

TRAINING_TEXT = repo_paths.langdata("amh.training_text")
UNICHARSET_FILE = repo_paths.langdata("amh.unicharset")

CONFUSED_PAIRS = [
    ('ስ', 'ሰ'), ('ሽ', 'ሸ'), ('ክ', 'ከ'), ('ህ', 'ሀ'), ('ት', 'ተ'),
    ('ብ', 'በ'), ('ል', 'ለ'), ('ም', 'መ'), ('ር', 'ረ'), ('ን', 'ነ'),
    ('ው', 'ወ'), ('ዝ', 'ዘ'), ('ድ', 'ደ'), ('ግ', 'ገ'), ('ፍ', 'ፈ')
]

def load_unicharset(path):
    chars = set()
    with open(path, "r", encoding="utf-8") as f:
        # Skip header count
        next(f)
        for line in f:
            parts = line.split()
            if parts:
                char = parts[0]
                if char not in ("NULL", "Joined", "|Broken|0|1"):
                    chars.add(char)
    return chars

def main():
    print("Auditing unicharset vs training text...")
    target_chars = load_unicharset(UNICHARSET_FILE)
    print(f"Target unicharset classes: {len(target_chars)}")

    char_counts = collections.defaultdict(int)
    line_lengths = []
    line_count = 0

    with open(TRAINING_TEXT, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_count += 1
            line_str = line.strip()
            line_lengths.append(len(line_str))
            for ch in line_str:
                char_counts[ch] += 1

    # 1. Check missing characters
    missing_in_text = target_chars - set(char_counts.keys())
    
    print("\n" + "=" * 65)
    print("      TRAINING TEXT AUDIT REPORT      ")
    print("=" * 65)
    print(f"Total Lines:                  {line_count:,}")
    print(f"Average Line Length:          {sum(line_lengths)/max(1, len(line_lengths)):.1f} chars")
    print(f"Max Line Length:              {max(line_lengths) if line_lengths else 0} chars")
    print(f"Unique Characters in Text:    {len(char_counts):,}")

    if missing_in_text:
        print(f"\n⚠️ CRITICAL WARNING: {len(missing_in_text)} characters in unicharset are MISSING in training text!")
        print(f"Missing Characters: {' '.join(sorted(missing_in_text))}")
    else:
        print("\n✅ FULL COVERAGE: All unicharset characters exist in the training text.")

    # 2. Inspect specifically 'ሡ' and related rare variants
    print("\n" + "-" * 65)
    print("CRITICAL RARE CHARACTERS CHECK:")
    print("-" * 65)
    rare_watch = ['ሡ', 'ሤ', 'ሧ', 'ቯ', 'ዤ', 'ጯ', 'ጿ', 'ፘ', 'ፙ', 'ፚ']
    for rc in rare_watch:
        cnt = char_counts.get(rc, 0)
        status = "✅ Sufficient" if cnt >= 100 else ("⚠️ Low" if cnt > 0 else "❌ ZERO")
        print(f"  Character '{rc}' : {cnt:6,d} occurrences ({status})")

    # 3. Check the 15 Sadis vs Ge'ez Confused Pairs
    print("\n" + "-" * 65)
    print("SADIS (6th) vs GEEZ (1st) ORDER BALANCE IN TRAINING TEXT:")
    print("-" * 65)
    geez_col_header = "Ge'ez (1st) Count"
    print(f"{'Pair':<10} | {'Sadis (6th) Count':<20} | {geez_col_header:<20} | {'Ratio'}")
    print("-" * 65)
    for sadis, geez in CONFUSED_PAIRS:
        s_cnt = char_counts.get(sadis, 0)
        g_cnt = char_counts.get(geez, 0)
        ratio = f"{g_cnt / max(1, s_cnt):.2f}:1"
        print(f"'{sadis}' vs '{geez}' | {s_cnt:10,d}           | {g_cnt:10,d}           | {ratio}")

    print("=" * 65)

if __name__ == "__main__":
    main()