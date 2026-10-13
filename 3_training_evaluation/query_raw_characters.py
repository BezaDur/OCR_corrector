#!/usr/bin/env python3
"""
High-Speed Raw Dataset Auditor:
Scans /mnt/mintdata/Users/Des/Mint/1_Dataset_raw/ recursively.
Finds exact counts and parent-folder origins for:
["ቌ", "ኈ", "ኊ", "ኌ", "ጔ", "፼"]
"""

import os
import glob
import collections

RAW_DIR = "/mnt/mintdata/Users/Des/Mint/1_Dataset_raw"
TARGET_CHARS = set(["ቌ", "ኈ", "ኊ", "ኌ", "ጔ", "፼"])


def main():
    if not os.path.exists(RAW_DIR):
        print(f"Error: Raw dataset path not found at {RAW_DIR}")
        return

    print(f"Scanning raw dataset recursively at {RAW_DIR} for {list(TARGET_CHARS)}...\n")

    char_counts = collections.defaultdict(int)
    folder_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    samples = collections.defaultdict(list)

    all_files = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True)
    print(f"Found {len(all_files):,} raw files to scan.")

    total_scanned_lines = 0

    for fpath in all_files:
        # Get immediate parent folder name (e.g., Geez, News, Books, etc.)
        rel_path = os.path.relpath(fpath, RAW_DIR)
        parent_folder = rel_path.split(os.sep)[0] if os.sep in rel_path else "root"

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    total_scanned_lines += 1
                    found_in_line = set(line) & TARGET_CHARS
                    if found_in_line:
                        for ch in found_in_line:
                            cnt = line.count(ch)
                            char_counts[ch] += cnt
                            folder_counts[ch][parent_folder] += cnt
                            if len(samples[ch]) < 3:
                                samples[ch].append((parent_folder, line.strip()))
        except Exception:
            continue

    print("=" * 75)
    print(f"RAW DATASET CHARACTER CENSUS ({total_scanned_lines:,} lines scanned)")
    print("=" * 75)

    for ch in sorted(TARGET_CHARS):
        total = char_counts[ch]
        print(f"\nCharacter: '{ch}' (U+{ord(ch):04X}) | Total Occurrences: {total:,}")
        print("-" * 75)
        if total == 0:
            print("  (Zero instances found in raw dataset)")
        else:
            print("  Parent Folder Distribution:")
            for folder, count in sorted(folder_counts[ch].items(), key=lambda x: x[1], reverse=True):
                print(f"    • {folder:<30} : {count:6,d} instances")

            print("\n  Sample Raw Sentences:")
            for folder, sample_text in samples[ch]:
                print(f"    [{folder}] {sample_text[:80]}...")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
