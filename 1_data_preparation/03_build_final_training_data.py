#!/usr/bin/env python3
"""
Step 1: Empirical Dataset Builder & Target Unicharset Generator.
Rules:
  - Complete Ethiopic Syllabary (all base consonants + labiovelars + ሡ/ሤ/ቯ/ጿ).
  - All Ethiopic Numerals (፩-፼) and Ethiopic Punctuation (። ፡ ፣ ፤ ፥ ፦ ፧ ፨ ፠).
  - Arabic Digits (0-9) and Standard Punctuation (including / and . for abbreviations).
  - High-frequency publication symbols (…, •, ÷, ‼, ※, ‐, ―, ·, †, £).
  - ZERO Latin alphabet letters (A-Z, a-z) to prevent T vs ፕ collisions.
  - Skip all MC4 files (starting with 048_).
  - Extracts 150,000 clean lines into /media/des/Tess/Tesseract2/amh.training_text.
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
import collections

DATASET_DIR = repo_paths.WORD_DATASET
OUTPUT_DIR = repo_paths.TESSERACT_WORKSPACE
TARGET_LINE_COUNT = 150000

# Minimum occurrence threshold from census
MIN_COUNT_THRESHOLD = 100


def is_latin_letter(ch):
    cp = ord(ch)
    return (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)


def build_whitelist_from_census():
    print(f"Auditing census dataset for approved Ethiopic characters (count >= {MIN_COUNT_THRESHOLD})...")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    active_files = [f for f in files if not os.path.basename(f).startswith("048_")]

    char_counts = collections.defaultdict(int)

    # Statistical census sampling across all clean files
    for fpath in active_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    if idx >= 20000:
                        break
                    for ch in line:
                        if ch not in ('\n', '\r', '\t', ' '):
                            char_counts[ch] += 1
        except Exception:
            continue

    approved_chars = set()
    rejected_extended = 0
    rejected_latin = 0
    rejected_low_freq = 0

    for ch, count in char_counts.items():
        cp = ord(ch)

        # 1. Reject ETHIOPIC_EXTENDED (U+1380 - U+139F, U+2D80 - U+2DDF)
        if 0x1380 <= cp <= 0x139F or 0x2D80 <= cp <= 0x2DDF:
            rejected_extended += 1
            continue

        # 2. Reject Latin Alphabet Letters (A-Z, a-z)
        if is_latin_letter(ch):
            rejected_latin += 1
            continue

        # 3. Reject counts below threshold
        if count < MIN_COUNT_THRESHOLD:
            rejected_low_freq += 1
            continue

        # 4. Reject foreign non-punctuation scripts (Arabic, Cyrillic, Chinese, Hebrew)
        if cp > 0x0250 and not (0x1200 <= cp <= 0x137C or 0x2000 <= cp <= 0x206F or ch in "…•÷‼※‐―·†→‟←„£′"):
            continue

        approved_chars.add(ch)

    print(f"Approved Character Whitelist: {len(approved_chars)} unique characters.")
    print(f"Excluded Latin Letters:       {rejected_latin}")
    print(f"Excluded Extended Set:        {rejected_extended}")
    print(f"Excluded Low Frequency:       {rejected_low_freq}")
    return approved_chars


def write_unicharset_file(approved_chars, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        # Header: total classes = approved chars + 3 Tesseract special tokens
        total_classes = len(approved_chars) + 3
        f.write(f"{total_classes}\n")
        f.write("NULL 0 Common 0\n")
        f.write("Joined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n")
        f.write("|Broken|0|1 15 0,255,0,255,0,0,0,0,0,0 Common 2 10 2 |Broken|0|1\n")

        for idx, ch in enumerate(sorted(approved_chars), start=3):
            script = "Ethiopic" if 0x1200 <= ord(ch) <= 0x137F else "Common"
            f.write(f"{ch} 1 0,255,0,255,0,0,0,0,0,0 {script} {idx} 0 {idx} {ch}\n")

    print(f"\nSaved Target Unicharset: {output_path} ({total_classes} total classes)")


def sample_training_lines(approved_chars, out_text_path):
    print(f"\nSampling {TARGET_LINE_COUNT:,} clean lines into {out_text_path}...")
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    active_files = [f for f in files if not os.path.basename(f).startswith("048_")]

    selected_lines = []
    seen_hashes = set()
    char_counts = collections.defaultdict(int)

    for fpath in active_files:
        if len(selected_lines) >= TARGET_LINE_COUNT:
            break
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as infile:
                for line in infile:
                    text = line.strip()
                    # Keep ideal box-strip length: 45 to 85 characters
                    if not (45 <= len(text) <= 85):
                        continue

                    # Deduplicate identical sentences
                    line_hash = hash(text)
                    if line_hash in seen_hashes:
                        continue

                    # Strict Whitelist Validation: Every character must be approved
                    if any(c not in approved_chars and c != ' ' for c in text):
                        continue

                    seen_hashes.add(line_hash)
                    selected_lines.append(text)
                    for c in text:
                        if c != ' ':
                            char_counts[c] += 1

                    if len(selected_lines) >= TARGET_LINE_COUNT:
                        break
        except Exception:
            continue

    with open(out_text_path, "w", encoding="utf-8") as out:
        for l in selected_lines:
            out.write(l + "\n")

    print(f"Sampled {len(selected_lines):,} clean lines successfully.")
    print(f"Target rare 'ሡ' occurrences: {char_counts.get('ሡ', 0):,}")
    print(f"Target rare 'ቯ' occurrences: {char_counts.get('ቯ', 0):,}")
    print(f"Target rare 'ጿ' occurrences: {char_counts.get('ጿ', 0):,}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unicharset_path = os.path.join(OUTPUT_DIR, "amh.unicharset")
    training_text_path = os.path.join(OUTPUT_DIR, "amh.training_text")

    approved = build_whitelist_from_census()
    write_unicharset_file(approved, unicharset_path)
    sample_training_lines(approved, training_text_path)


if __name__ == "__main__":
    main()