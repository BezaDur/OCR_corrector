#!/usr/bin/env python3
"""
Step 3: Robust Two-Tier Alignment & Optical Confusion Matrix Extraction.
Computes CER/WER and exports optical substitution costs to confusion_matrix.json.
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
import json
import math
import collections
import difflib

# ----------------- Configuration -----------------
DATASET_DIR = repo_paths.workspace("synthetic_eval")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
# Defaults to ocr_new_model for historical reasons, but that directory holds
# the output of amh_epoch1_fast -- measured at 26.65% CER against the stock
# engine's 2.05% on the same pages. A confusion matrix learned there encodes
# that model's errors (3.75 wrong chars per bad word), not your engine's
# (1.06). Rebuild from the engine you actually run:
#
#   python 01_build_confusion_matrix.py --ocr-dir .../ocr_pred \
#          --out /media/des/Tess/Tesseract2/confusion_matrix_pred.json
OCR_DIR = os.path.join(DATASET_DIR, "ocr_new_model")

OUTPUT_MATRIX_JSON = repo_paths.confusion_matrix()
OUTPUT_REPORT_TXT = repo_paths.workspace("confusion_report.txt")
# -------------------------------------------------


def align_characters_dp(true_str, ocr_str):
    """
    Standard dynamic programming Needleman-Wunsch string alignment.
    Returns: list of tuples (true_char, ocr_char), where '-' denotes gap.
    """
    m, n = len(true_str), len(ocr_str)
    
    # Pre-allocate DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if true_str[i - 1] == ocr_str[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Match/Substitution
            )

    # Backtrace
    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if true_str[i - 1] == ocr_str[j - 1] else 1):
            aligned.append((true_str[i - 1], ocr_str[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((true_str[i - 1], "-"))
            i -= 1
        else:
            aligned.append(("-", ocr_str[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def align_page_lines(gt_lines, ocr_lines):
    """
    Tier-1 Line Matcher: Uses SequenceMatcher to align lines robustly
    even if OCR dropped or fused lines.
    """
    sm = difflib.SequenceMatcher(None, gt_lines, ocr_lines)
    paired_lines = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for gt_l, oc_l in zip(gt_lines[i1:i2], ocr_lines[j1:j2]):
                paired_lines.append((gt_l, oc_l))
        elif tag == "replace":
            # Best effort zip of modified block
            len_gt = i2 - i1
            len_oc = j2 - j1
            max_len = max(len_gt, len_oc)
            gts = gt_lines[i1:i2] + [""] * (max_len - len_gt)
            ocs = ocr_lines[j1:j2] + [""] * (max_len - len_oc)
            for g, o in zip(gts, ocs):
                paired_lines.append((g, o))
        elif tag == "delete":
            for g in gt_lines[i1:i2]:
                paired_lines.append((g, ""))
        elif tag == "insert":
            for o in ocr_lines[j1:j2]:
                paired_lines.append(("", o))

    return paired_lines


def main():
    import argparse
    global GT_DIR, OCR_DIR, OUTPUT_MATRIX_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=GT_DIR)
    ap.add_argument("--ocr-dir", default=OCR_DIR)
    ap.add_argument("--out", default=OUTPUT_MATRIX_JSON)
    a = ap.parse_args()
    GT_DIR, OCR_DIR, OUTPUT_MATRIX_JSON = a.gt_dir, a.ocr_dir, a.out
    print(f"GT  : {GT_DIR}")
    print(f"OCR : {OCR_DIR}")
    print(f"OUT : {OUTPUT_MATRIX_JSON}")

    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    if not gt_files:
        print(f"Error: No GT files found in {GT_DIR}")
        return

    substitutions = collections.defaultdict(lambda: collections.defaultdict(int))
    insertions = collections.defaultdict(int)
    deletions = collections.defaultdict(int)
    char_counts = collections.defaultdict(int)

    total_ref_chars = 0
    total_char_errors = 0

    print(f"Aligning {len(gt_files)} pages of ground truth vs OCR predictions...")

    for gt_path in gt_files:
        base_name = os.path.basename(gt_path).replace(".gt.txt", "")
        ocr_path = os.path.join(OCR_DIR, f"{base_name}.ocr.txt")

        if not os.path.exists(ocr_path):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_lines = [l.strip() for l in f if l.strip()]

        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_lines = [l.strip() for l in f if l.strip()]

        # Tier 1: Align lines
        line_pairs = align_page_lines(gt_lines, ocr_lines)

        # Tier 2: Align characters
        for g_line, o_line in line_pairs:
            total_ref_chars += len(g_line)
            aligned_chars = align_characters_dp(g_line, o_line)

            for t_char, o_char in aligned_chars:
                if t_char != "-":
                    char_counts[t_char] += 1

                if t_char != "-" and o_char != "-":
                    if t_char != o_char:
                        substitutions[t_char][o_char] += 1
                        total_char_errors += 1
                elif t_char == "-" and o_char != "-":
                    insertions[o_char] += 1
                    total_char_errors += 1
                elif t_char != "-" and o_char == "-":
                    deletions[t_char] += 1
                    total_char_errors += 1

    # Accuracy Stats
    cer = (total_char_errors / max(1, total_ref_chars)) * 100.0
    print(f"\n==========================================")
    print(f"Overall Character Error Rate (CER): {cer:.2f}%")
    print(f"Character Accuracy:                 {100.0 - cer:.2f}%")
    print(f"Total Reference Characters:        {total_ref_chars:,}")
    print(f"Total Optical Errors Encountered:  {total_char_errors:,}")
    print(f"==========================================")

    # Convert counts to log costs: -log(P(ocr | true))
    # Default penalties for unseen operations:
    cost_data = {
        "sub": {},
        "del": {},
        "ins": {},
        "default_sub": 4.5,
        "default_del": 5.0,
        "default_ins": 5.0
    }

    # Substitution costs
    for t_char, obs_map in substitutions.items():
        cost_data["sub"][t_char] = {}
        total = char_counts[t_char]
        for o_char, cnt in obs_map.items():
            prob = max(cnt / total, 1e-5)
            cost_data["sub"][t_char][o_char] = round(-math.log(prob), 4)

    # Deletion costs
    for t_char, cnt in deletions.items():
        prob = max(cnt / char_counts[t_char], 1e-5)
        cost_data["del"][t_char] = round(-math.log(prob), 4)

    # Insertion costs
    total_chars_all = sum(char_counts.values()) or 1
    for o_char, cnt in insertions.items():
        prob = max(cnt / total_chars_all, 1e-5)
        cost_data["ins"][o_char] = round(-math.log(prob), 4)

    # Export Matrix
    with open(OUTPUT_MATRIX_JSON, "w", encoding="utf-8") as f:
        json.dump(cost_data, f, ensure_ascii=False, indent=2)

    # Generate Human-Readable Top Confusions Report
    top_substitutions = []
    for t_char, obs_map in substitutions.items():
        for o_char, cnt in obs_map.items():
            top_substitutions.append((t_char, o_char, cnt))
    top_substitutions.sort(key=lambda x: x[2], reverse=True)

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== AMHARIC OCR ERROR PROFILE REPORT ===\n")
        f.write(f"Character Error Rate (CER): {cer:.2f}%\n")
        f.write(f"Character Accuracy:         {100.0 - cer:.2f}%\n\n")
        f.write("TOP 50 MOST FREQUENT CHARACTER SUBSTITUTIONS:\n")
        f.write("--------------------------------------------\n")
        f.write("Rank | True Char -> OCR Saw | Count | Optical Cost\n")
        f.write("--------------------------------------------\n")
        for rank, (t, o, cnt) in enumerate(top_substitutions[:50], 1):
            cost = cost_data["sub"][t][o]
            f.write(f"{rank:4d} |    '{t}'    ->    '{o}'    | {cnt:5d} | {cost:.4f}\n")

    print(f"\n1. Machine-readable matrix saved to: {OUTPUT_MATRIX_JSON}")
    print(f"2. Human-readable diagnostic report: {OUTPUT_REPORT_TXT}")


if __name__ == "__main__":
    main()