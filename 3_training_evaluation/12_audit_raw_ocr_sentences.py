#!/usr/bin/env python3
"""
Step 12: Robust Word-Level Alignment & Failure Case Extractor.
Uses canonical punctuation equivalence and global DP alignment to guarantee
that line wraps and blank lines NEVER cause off-by-one desynchronization.
"""

# Make the repo-root shared modules importable from this subfolder.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import glob
import difflib
from amharic_canonical_normalizer import canonicalize_amharic_punctuation

DATASET_DIR = "/media/des/Tess/Tesseract2/synthetic_eval"
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
RAW_OCR_DIR = os.path.join(DATASET_DIR, "ocr_new_model")
OUTPUT_REPORT = "/media/des/Tess/Tesseract2/sentence_ocr_failures.txt"


def align_page_words(gt_words, ocr_words):
    """
    Globally aligns page words using SequenceMatcher.
    Returns: list of (gold_chunk, ocr_chunk, context_window)
    """
    sm = difflib.SequenceMatcher(None, gt_words, ocr_words)
    diffs = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        gold_chunk = " ".join(gt_words[i1:i2])
        ocr_chunk = " ".join(ocr_words[j1:j2])

        # 3-word surrounding context window
        ctx_before = " ".join(gt_words[max(0, i1 - 3):i1])
        ctx_after = " ".join(gt_words[i2:min(len(gt_words), i2 + 3)])

        diffs.append((gold_chunk, ocr_chunk, ctx_before, ctx_after))

    return diffs


def main():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    if not gt_files:
        print("Error: No Ground Truth files found.")
        return

    print(f"Auditing true word mismatches across {len(gt_files)} pages...")

    total_words = 0
    mismatches = []

    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace(".gt.txt", "")
        ocr_path = os.path.join(RAW_OCR_DIR, f"{base_name}.ocr.txt")

        if not os.path.exists(ocr_path):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            raw_gt_text = f.read()

        with open(ocr_path, "r", encoding="utf-8") as f:
            raw_ocr_text = f.read()

        # Step 1: Apply Canonical Punctuation Equivalence
        clean_gt = canonicalize_amharic_punctuation(raw_gt_text)
        clean_ocr = canonicalize_amharic_punctuation(raw_ocr_text)

        gt_words = clean_gt.split()
        ocr_words = clean_ocr.split()

        total_words += len(gt_words)

        page_diffs = align_page_words(gt_words, ocr_words)
        for g_tok, o_tok, before, after in page_diffs:
            mismatches.append((base_name, g_tok, o_tok, before, after))

    # Write the Failure Audit Report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as out:
        out.write("======================================================================\n")
        out.write("             AUTHENTIC RAW OCR MISMATCH AUDIT REPORT                  \n")
        out.write("======================================================================\n")
        out.write(f"Total Words Audited:         {total_words:,}\n")
        out.write(f"Total Discrepancies Found:   {len(mismatches):,} ({(len(mismatches)/max(1, total_words))*100:.2f}% of words)\n\n")

        for idx, (page, gold, ocr, before, after) in enumerate(mismatches[:150], 1):
            out.write(f"--- [Mismatch #{idx}] Page: {page} ---\n")
            out.write(f"CONTEXT : ... {before} >>> [ {gold} ] <<< {after} ...\n")
            out.write(f"GOLD    : {gold}\n")
            out.write(f"OCR     : {ocr}\n\n")

    print(f"\nAudit complete!")
    print(f"Total Discrepancies Logged: {len(mismatches):,}")
    print(f"Saved Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()