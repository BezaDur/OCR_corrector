#!/usr/bin/env python3
"""
Corpus Miner: Searches the entire 4.9 GB corpus to collect EVERY authentic
sentence containing under-represented rare characters, Dikala labiovelars,
Ethiopic numerals, and publication symbols.
"""

import os
import glob
import collections

DATASET_DIR = "/mnt/mintdata/Users/Des/Mint/2_Dataset_text/04_OCR_tess_word_defrag"
UNICHARSET_FILE = "/media/des/Tess/Tesseract2/amh.unicharset"
OUTPUT_MINED_FILE = "/media/des/Tess/Tesseract2/mined_rare_sentences.txt"

# The 59 Under-Represented Target Characters from our census
TARGET_RARE_CHARS = set(
    "ሡሤቯዤጬጿጰጱጲጴጶኂኄኆኍዧ"          # Rare Syllables
    "ቈቊቌቍኈኊኌኰኲኴኵጐጒጔጕጯሿ"          # Dikala Labiovelars
    "፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼"          # Ethiopic Numerals
    "፥፧፨፠"                              # Rare Punctuation
    "•※†―—„‟"                           # Publication Symbols
)


def load_approved_unicharset(path):
    approved = set()
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # Skip header count
        for line in f:
            parts = line.split()
            if parts and parts[0] not in ("NULL", "Joined", "|Broken|0|1"):
                approved.add(parts[0])
    return approved


def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Corpus directory not found at {DATASET_DIR}")
        return

    approved_chars = load_approved_unicharset(UNICHARSET_FILE)
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    active_files = [f for f in files if not os.path.basename(f).startswith("048_")]

    print(f"Scanning {len(active_files)} non-MC4 corpus files for {len(TARGET_RARE_CHARS)} rare characters...")

    mined_sentences = []
    seen_hashes = set()
    hit_counts = collections.defaultdict(int)
    total_lines_scanned = 0

    for fpath in active_files:
        base_name = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    total_lines_scanned += 1
                    text = line.strip()

                    # Check length suitability for text2image line strip
                    if not (35 <= len(text) <= 90):
                        continue

                    # Does this line contain any of our target rare characters?
                    found_rare = set(text) & TARGET_RARE_CHARS
                    if not found_rare:
                        continue

                    # Strict Whitelist Gate: Ensure all characters in the line are approved
                    if any(c not in approved_chars and c != ' ' for c in text):
                        continue

                    line_hash = hash(text)
                    if line_hash in seen_hashes:
                        continue

                    seen_hashes.add(line_hash)
                    mined_sentences.append(text)

                    for rc in found_rare:
                        hit_counts[rc] += text.count(rc)

        except Exception as e:
            print(f"Error reading {base_name}: {e}")

    # Write all mined authentic sentences to disk
    with open(OUTPUT_MINED_FILE, "w", encoding="utf-8") as out:
        for s in mined_sentences:
            out.write(s + "\n")

    print("\n" + "=" * 65)
    print("      AUTHENTIC RARE CORPUS MINING REPORT      ")
    print("=" * 65)
    print(f"Total Lines Scanned Across Corpus: {total_lines_scanned:,}")
    print(f"Total Unique Rare Lines Mined:    {len(mined_sentences):,}")
    print(f"Output Mined File:                {OUTPUT_MINED_FILE}")
    print("-" * 65)
    print(f"{'CHAR':<6} | {'UNICODE':<8} | {'MINED INSTANCES'}")
    print("-" * 65)
    for rc in sorted(TARGET_RARE_CHARS):
        cnt = hit_counts[rc]
        status = f"{cnt:6,d} instances" if cnt > 0 else "0 (None in corpus)"
        print(f"'{rc}'    | U+{ord(rc):04X}  | {status}")
    print("=" * 65)


if __name__ == "__main__":
    main()
