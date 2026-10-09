#!/usr/bin/env python3
"""
Diagnostic Tool: Extracts, aligns, and categorizes all remaining OCR errors.
Compares: Ground Truth vs. Raw Tesseract vs. Post-Corrector Output.
"""

import os
import glob
import difflib
import collections

DATASET_DIR = "/media/des/Tess/Tesseract2/synthetic_eval"
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
RAW_OCR_DIR = os.path.join(DATASET_DIR, "ocr_pred")
CORRECTED_DIR = os.path.join(DATASET_DIR, "ocr_corrected")

OUTPUT_REPORT = "/media/des/Tess/Tesseract2/error_summary_report.txt"
OUTPUT_TSV = "/media/des/Tess/Tesseract2/failed_cases.tsv"


def align_word_sequences(gt_words, raw_words, corr_words):
    """
    Aligns three word sequences using SequenceMatcher to detect exact
    word replacements, uncorrected errors, and over-corrections.
    """
    sm = difflib.SequenceMatcher(None, gt_words, corr_words)
    matches = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        gt_chunk = " ".join(gt_words[i1:i2])
        corr_chunk = " ".join(corr_words[j1:j2])
        # Find corresponding raw OCR slice using proportional index mapping
        r_start = int((i1 / max(1, len(gt_words))) * len(raw_words))
        r_end = max(r_start + 1, int((i2 / max(1, len(gt_words))) * len(raw_words)))
        raw_chunk = " ".join(raw_words[r_start:r_end])

        matches.append((gt_chunk, raw_chunk, corr_chunk))

    return matches


def main():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    if not gt_files:
        print("Error: No Ground Truth files found.")
        return

    print(f"Analyzing errors across {len(gt_files)} pages...")

    # Categories
    unrepaired_errors = collections.defaultdict(int)  # OCR was wrong, corrector left it wrong
    over_corrections = collections.defaultdict(int)   # OCR was right, corrector broke it
    suboptimal_repairs = collections.defaultdict(int) # Corrector changed OCR, but to the wrong word
    segmentation_errors = collections.defaultdict(int) # Spaces / word boundaries

    total_error_instances = 0

    tsv_out = open(OUTPUT_TSV, "w", encoding="utf-8")
    tsv_out.write("PAGE\tTYPE\tGROUND_TRUTH\tRAW_OCR\tCORRECTED\n")

    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace(".gt.txt", "")
        raw_path = os.path.join(RAW_OCR_DIR, f"{base_name}.ocr.txt")
        corr_path = os.path.join(CORRECTED_DIR, f"{base_name}.corr.txt")

        if not (os.path.exists(raw_path) and os.path.exists(corr_path)):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_words = f.read().strip().split()
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_words = f.read().strip().split()
        with open(corr_path, "r", encoding="utf-8") as f:
            corr_words = f.read().strip().split()

        diffs = align_word_sequences(gt_words, raw_words, corr_words)

        for gt_tok, raw_tok, corr_tok in diffs:
            total_error_instances += 1

            # Detect space segmentation errors
            if " " in gt_tok or " " in raw_tok or " " in corr_tok:
                err_type = "SEGMENTATION_SPACE"
                segmentation_errors[(gt_tok, raw_tok, corr_tok)] += 1
            # Detect over-corrections (OCR was already correct, but corrector changed it)
            elif gt_tok == raw_tok and corr_tok != gt_tok:
                err_type = "OVER_CORRECTION"
                over_corrections[(gt_tok, corr_tok)] += 1
            # Detect uncorrected slips (OCR was wrong, corrector changed nothing)
            elif raw_tok == corr_tok and corr_tok != gt_tok:
                err_type = "UNREPAIRED_TYPO"
                unrepaired_errors[(gt_tok, raw_tok)] += 1
            # Sub-optimal repair (corrector changed OCR, but didn't reach Gold)
            else:
                err_type = "SUBOPTIMAL_REPAIR"
                suboptimal_repairs[(gt_tok, raw_tok, corr_tok)] += 1

            tsv_out.write(f"{base_name}\t{err_type}\t{gt_tok}\t{raw_tok}\t{corr_tok}\n")

    tsv_out.close()

    # Generate Human-Readable Diagnostic Summary
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("      AMHARIC OCR POST-CORRECTION FAILURE AUDIT       \n")
        f.write("=====================================================\n")
        f.write(f"Total Failed Word Alignments: {total_error_instances:,}\n")
        f.write(f"  • Unrepaired Typos:         {sum(unrepaired_errors.values()):,}\n")
        f.write(f"  • Space/Segmentation Errors: {sum(segmentation_errors.values()):,}\n")
        f.write(f"  • Suboptimal Repairs:       {sum(suboptimal_repairs.values()):,}\n")
        f.write(f"  • Over-Corrections:         {sum(over_corrections.values()):,}\n\n")

        f.write("-" * 55 + "\n")
        f.write("TOP 25 UNREPAIRED TYPOS (Gold -> OCR Saw / Left Intact)\n")
        f.write("-" * 55 + "\n")
        for (g, r), cnt in sorted(unrepaired_errors.items(), key=lambda x: x[1], reverse=True)[:25]:
            f.write(f"{cnt:4d}x | Gold: '{g}'  <--  Left As: '{r}'\n")

        f.write("\n" + "-" * 55 + "\n")
        f.write("TOP 25 WORD BOUNDARY & SPACE ERRORS\n")
        f.write("-" * 55 + "\n")
        for (g, r, c), cnt in sorted(segmentation_errors.items(), key=lambda x: x[1], reverse=True)[:25]:
            f.write(f"{cnt:4d}x | Gold: '{g}' | Raw: '{r}' | Got: '{c}'\n")

        f.write("\n" + "-" * 55 + "\n")
        f.write("TOP 20 OVER-CORRECTIONS (Raw OCR Was Correct, Corrector Broke It)\n")
        f.write("-" * 55 + "\n")
        for (g, c), cnt in sorted(over_corrections.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{cnt:4d}x | Gold: '{g}'  -->  Turned Into: '{c}'\n")

    print("\nAudit Complete!")
    print(f"Summary Report: {OUTPUT_REPORT}")
    print(f"Detailed TSV:   {OUTPUT_TSV}")


if __name__ == "__main__":
    main()