#!/usr/bin/env python3
"""
Quick check: are the GT and OCR files aligned properly?

Shows the first page side by side and counts mismatches.

Usage:
    python OCR_error_detection/check_alignment.py \
        --gt-dir OCR_error_detection/Evaluation/GroundTruth \
        --ocr-dir OCR_error_detection/Evaluation/OCRed
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--ocr-dir", required=True)
    parser.add_argument("--pages", type=int, default=3, help="How many pages to show")
    args = parser.parse_args()

    gt_files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".txt"))
    if not gt_files:
        print(f"No .txt files in {args.gt_dir}")
        return

    for fname in gt_files[:args.pages]:
        base = fname.replace(".gt.txt", "").replace(".txt", "")

        # Find OCR file
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            candidate = os.path.join(args.ocr_dir, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            print(f"[SKIP] {base}: no OCR file found")
            continue

        gt_text = open(os.path.join(args.gt_dir, fname), encoding="utf-8").read().strip()
        ocr_text = open(ocr_path, encoding="utf-8").read().strip()

        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()

        print(f"\n{'='*60}")
        print(f"Page: {base}")
        print(f"  GT lines:   {gt_text.count(chr(10)) + 1}")
        print(f"  OCR lines:  {ocr_text.count(chr(10)) + 1}")
        print(f"  GT tokens:  {len(gt_tokens)}")
        print(f"  OCR tokens: {len(ocr_tokens)}")
        print(f"  Token count match: {'YES' if len(gt_tokens) == len(ocr_tokens) else 'NO'}")

        n = min(len(gt_tokens), len(ocr_tokens))
        mismatches = sum(1 for i in range(n) if gt_tokens[i] != ocr_tokens[i])
        print(f"  Mismatches (first {n}): {mismatches} ({100*mismatches/max(n,1):.1f}%)")

        print(f"\n  First 20 tokens side by side:")
        print(f"  {'#':>4} {'GT':<25} {'OCR':<25} {'Match'}")
        print(f"  {'-'*4} {'-'*25} {'-'*25} {'-'*5}")
        for i in range(min(20, max(len(gt_tokens), len(ocr_tokens)))):
            gt = gt_tokens[i] if i < len(gt_tokens) else "(missing)"
            ocr = ocr_tokens[i] if i < len(ocr_tokens) else "(missing)"
            match = "✓" if gt == ocr else "✗"
            print(f"  {i:4d} {gt:<25} {ocr:<25} {match}")

    # Summary across all pages
    print(f"\n{'='*60}")
    print(f"Summary across all pages:")
    total_gt = 0
    total_ocr = 0
    total_mismatch = 0
    count_mismatch_pages = 0
    for fname in gt_files:
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            candidate = os.path.join(args.ocr_dir, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            continue
        gt = open(os.path.join(args.gt_dir, fname), encoding="utf-8").read().split()
        ocr = open(ocr_path, encoding="utf-8").read().split()
        total_gt += len(gt)
        total_ocr += len(ocr)
        if len(gt) != len(ocr):
            count_mismatch_pages += 1
        n = min(len(gt), len(ocr))
        total_mismatch += sum(1 for i in range(n) if gt[i] != ocr[i])

    print(f"  Total GT tokens:  {total_gt:,}")
    print(f"  Total OCR tokens: {total_ocr:,}")
    print(f"  Pages with different token counts: {count_mismatch_pages}")
    aligned = min(total_gt, total_ocr)
    print(f"  Mismatches (aligned): {total_mismatch:,} ({100*total_mismatch/max(aligned,1):.1f}%)")


if __name__ == "__main__":
    main()