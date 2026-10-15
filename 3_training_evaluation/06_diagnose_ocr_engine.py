#!/usr/bin/env python3
"""
Diagnostic Engine: Audits Tesseract OCR BEFORE any post-processing.
Pinpoints:
  1. Space insertion/fusion (segmentation failure)
  2. Sadis vs. Ge'ez order erosions (6th vs 1st form)
  3. Abbreviation and punctuation dropping (/ and .)
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
import glob
import re
import collections
from fontTools.ttLib import TTFont

DATASET_DIR = repo_paths.workspace("synthetic_eval")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
RAW_OCR_DIR = os.path.join(DATASET_DIR, "ocr_pred")
OUTPUT_REPORT = repo_paths.workspace("ocr_engine_faults.txt")

# Vowel order shift pairs: (6th order / Sadis, 1st order / Ge'ez)
SADIS_GEEZ_PAIRS = {
    ('ስ', 'ሰ'), ('ሽ', 'ሸ'), ('ክ', 'ከ'), ('ህ', 'ሀ'), ('ት', 'ተ'),
    ('ብ', 'በ'), ('ል', 'ለ'), ('ም', 'መ'), ('ር', 'ረ'), ('ን', 'ነ'),
    ('ው', 'ወ'), ('ዝ', 'ዘ'), ('ድ', 'ደ'), ('ግ', 'ገ'), ('ፍ', 'ፈ')
}

def analyze_page(gt_text, ocr_text):
    gt_tokens = gt_text.split()
    ocr_tokens = ocr_text.split()

    faults = {
        "fused_words": [],
        "split_words": [],
        "sadis_to_geez": [],
        "dropped_slashes": [],
        "total_gt_words": len(gt_tokens),
        "total_ocr_words": len(ocr_tokens)
    }

    gt_full = " ".join(gt_tokens)
    ocr_full = " ".join(ocr_tokens)

    # 1. Detect Abbreviation / Slash failures
    gt_slashes = re.findall(r'\b\w+/\w+\b', gt_full)
    for abbr in gt_slashes:
        clean_abbr = abbr.replace('/', '')
        if clean_abbr in ocr_full and abbr not in ocr_full:
            faults["dropped_slashes"].append((abbr, clean_abbr))

    # 2. Detect Sadis vs Ge'ez erosions in words
    for g_char, s_char in SADIS_GEEZ_PAIRS:
        # Count occurrences where Gold has Sadis but OCR saw Ge'ez
        pattern = re.compile(rf'({s_char})')
        # Simple sample matching
        if s_char in ocr_full and g_char in gt_full:
            faults["sadis_to_geez"].append((g_char, s_char))

    # 3. Detect Word Fusions (e.g. "ወደ ቤቱ" -> "ወደቤቱ")
    for i in range(len(gt_tokens) - 1):
        pair = gt_tokens[i] + gt_tokens[i + 1]
        if pair in ocr_tokens and gt_tokens[i] not in ocr_tokens:
            faults["fused_words"].append((f"{gt_tokens[i]} {gt_tokens[i+1]}", pair))

    # 4. Detect Word Splits (e.g. "ትምህርት" -> "ትም ህርት")
    for tok in gt_tokens:
        if len(tok) >= 4:
            for split_idx in range(2, len(tok) - 1):
                part1, part2 = tok[:split_idx], tok[split_idx:]
                if part1 in ocr_tokens and part2 in ocr_tokens and tok not in ocr_tokens:
                    faults["split_words"].append((tok, f"{part1} {part2}"))

    return faults


def main():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    if not gt_files:
        print("Error: No GT files found.")
        return

    print(f"Auditing Raw Tesseract on {len(gt_files)} pages...")

    total_fused = collections.defaultdict(int)
    total_splits = collections.defaultdict(int)
    total_slashes = collections.defaultdict(int)
    total_sadis_geez = collections.defaultdict(int)

    for gt_path in gt_files:
        base = os.path.basename(gt_path).replace(".gt.txt", "")
        raw_path = os.path.join(RAW_OCR_DIR, f"{base}.ocr.txt")
        if not os.path.exists(raw_path):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(raw_path, "r", encoding="utf-8") as f:
            ocr_text = f.read().strip()

        res = analyze_page(gt_text, ocr_text)

        for pair in res["fused_words"]:
            total_fused[pair] += 1
        for pair in res["split_words"]:
            total_splits[pair] += 1
        for pair in res["dropped_slashes"]:
            total_slashes[pair] += 1
        for pair in res["sadis_to_geez"]:
            total_sadis_geez[pair] += 1

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as out:
        out.write("=====================================================\n")
        out.write("     TESSERACT OCR UPSTREAM FAULT PROFILE AUDIT      \n")
        out.write("=====================================================\n\n")

        out.write("1. ABBREVIATION & SLASH DROPS (/ dropped into raw words):\n")
        out.write("-" * 55 + "\n")
        for (orig, dropped), cnt in sorted(total_slashes.items(), key=lambda x: x[1], reverse=True)[:15]:
            out.write(f"  {cnt:3d}x | Intended: '{orig}'  --> OCR Dropped Slash: '{dropped}'\n")

        out.write("\n2. WORD SPLITS (Spurious spaces injected by Tesseract):\n")
        out.write("-" * 55 + "\n")
        for (orig, split), cnt in sorted(total_splits.items(), key=lambda x: x[1], reverse=True)[:15]:
            out.write(f"  {cnt:3d}x | Single Word: '{orig}'  --> OCR Split Into: '{split}'\n")

        out.write("\n3. WORD FUSIONS (Missing spaces / words glued together):\n")
        out.write("-" * 55 + "\n")
        for (orig, fused), cnt in sorted(total_fused.items(), key=lambda x: x[1], reverse=True)[:15]:
            out.write(f"  {cnt:3d}x | Separate: '{orig}'  --> OCR Fused Into: '{fused}'\n")

        out.write("\n4. SADIS (6th) vs GEEZ (1st) ORDER VOWEL EROSIONS:\n")
        out.write("-" * 55 + "\n")
        for (sadis, geez), cnt in sorted(total_sadis_geez.items(), key=lambda x: x[1], reverse=True)[:15]:
            out.write(f"  {cnt:3d}x | True Sadis '{sadis}'  --> Mistaken as Ge'ez '{geez}'\n")

    print(f"\nAudit Finished! Results written to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()