#!/usr/bin/env python3
"""
Step 2: Combines the mined rare sentences with standard general prose
to produce a 70,000-line training set with balanced character representation.
Compiles the final, zero-defect amh.unicharset.
"""

import os
import random
import collections

MINED_RARE_FILE = "/media/des/Tess/Tesseract2/mined_rare_sentences.txt"
GENERAL_TEXT_FILE = "/media/des/Tess/Tesseract2/amh.training_text"

FINAL_TRAINING_TEXT = "/media/des/Tess/Tesseract2/amh_final_training_text.txt"
FINAL_UNICHARSET = "/media/des/Tess/Tesseract2/amh.unicharset"

TOTAL_TARGET_LINES = 70000
BASE_SEED = 42


def main():
    print("Loading mined rare sentences...")
    rare_lines = []
    if os.path.exists(MINED_RARE_FILE):
        with open(MINED_RARE_FILE, "r", encoding="utf-8") as f:
            rare_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded {len(rare_lines):,} unique rare-character lines.")

    print("Loading general training lines...")
    general_lines = []
    if os.path.exists(GENERAL_TEXT_FILE):
        with open(GENERAL_TEXT_FILE, "r", encoding="utf-8") as f:
            general_lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded {len(general_lines):,} general lines.")

    # Blend: Prioritize all unique rare lines, then backfill with general prose
    combined_lines = []
    seen_hashes = set()

    for l in rare_lines:
        h = hash(l)
        if h not in seen_hashes:
            seen_hashes.add(h)
            combined_lines.append(l)

    rng = random.Random(BASE_SEED)
    rng.shuffle(general_lines)

    for l in general_lines:
        if len(combined_lines) >= TOTAL_TARGET_LINES:
            break
        h = hash(l)
        if h not in seen_hashes:
            seen_hashes.add(h)
            combined_lines.append(l)

    rng.shuffle(combined_lines)

    # Save final blended training text
    with open(FINAL_TRAINING_TEXT, "w", encoding="utf-8") as out:
        for l in combined_lines:
            out.write(l + "\n")

    print(f"\nFinal Blended Training Text: {FINAL_TRAINING_TEXT} ({len(combined_lines):,} lines)")

    # Extract exact, empirical unicharset from the final 70,000 lines
    char_counts = collections.defaultdict(int)
    for l in combined_lines:
        for ch in l:
            if ch != ' ':
                char_counts[ch] += 1

    approved_chars = sorted(char_counts.keys())
    total_classes = len(approved_chars) + 3

    with open(FINAL_UNICHARSET, "w", encoding="utf-8") as uf:
        uf.write(f"{total_classes}\n")
        uf.write("NULL 0 Common 0\n")
        uf.write("Joined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n")
        uf.write("|Broken|0|1 15 0,255,0,255,0,0,0,0,0,0 Common 2 10 2 |Broken|0|1\n")

        for idx, ch in enumerate(approved_chars, start=3):
            script = "Ethiopic" if 0x1200 <= ord(ch) <= 0x137F else "Common"
            uf.write(f"{ch} 1 0,255,0,255,0,0,0,0,0,0 {script} {idx} 0 {idx} {ch}\n")

    print(f"Final amh.unicharset:         {FINAL_UNICHARSET} ({total_classes} total classes)")
    print("-" * 65)
    print("VERIFYING CRITICAL RARE TARGETS IN THE FINAL 70,000 LINES:")
    print("-" * 65)
    for rc in ['ሡ', 'ሤ', 'ቯ', 'ዤ', 'ጬ', 'ጿ', 'ጰ', 'ጲ', 'ጴ', '፩', '፪', '፲', '፻', '፥', '፧', '፨']:
        print(f"  '{rc}' (U+{ord(rc):04X}) : {char_counts.get(rc, 0):6,d} occurrences in final training lines")
    print("=" * 65)


if __name__ == "__main__":
    main()
