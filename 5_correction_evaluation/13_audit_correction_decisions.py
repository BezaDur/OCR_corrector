#!/usr/bin/env python3
"""
Step 13: Three-Way Decision Auditor (Ground Truth vs. Raw OCR vs. Corrected).
Categorizes every post-processing decision into:
  - TRUE_POSITIVE   (Legitimate OCR repair)
  - FALSE_POSITIVE  (Clean-token damage / over-correction)
  - FALSE_NEGATIVE  (Unrepaired OCR typo)
  - SUBOPTIMAL_FLIP (Changed OCR, but to the wrong word)
Exports full side-by-side TSV and prints top failure patterns.
"""

# Make the repo-root shared modules importable from this subfolder.
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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import glob
import difflib
import collections
from amharic_canonical_normalizer import canonicalize_amharic_punctuation

DATASET_DIR = repo_paths.workspace("synthetic_eval")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
RAW_OCR_DIR = os.path.join(DATASET_DIR, "ocr_new_model")
CORR_DIR = os.path.join(DATASET_DIR, "ocr_corrected")

OUTPUT_TSV = repo_paths.workspace("correction_decisions.tsv")
OUTPUT_SUMMARY = repo_paths.workspace("correction_patterns_report.txt")


def align_three_way(gt_tokens, raw_tokens, corr_tokens):
    """
    Simultaneously aligns Ground Truth, Raw OCR, and Corrected tokens.
    """
    # Align GT vs Raw OCR
    sm_raw = difflib.SequenceMatcher(None, gt_tokens, raw_tokens)
    # Align GT vs Corrected
    sm_corr = difflib.SequenceMatcher(None, gt_tokens, corr_tokens)

    # Use 1-to-1 matching where tokens align cleanly
    aligned_events = []

    # Fast token-by-token comparison when lengths match closely
    min_len = min(len(gt_tokens), len(raw_tokens), len(corr_tokens))
    for i in range(min_len):
        g = gt_tokens[i]
        r = raw_tokens[i]
        c = corr_tokens[i]

        ctx_before = " ".join(gt_tokens[max(0, i - 2):i])
        ctx_after = " ".join(gt_tokens[i + 1:min(len(gt_tokens), i + 3)])
        context = f"... {ctx_before} > [{g}] < {ctx_after} ..."

        if r == g and c == g:
            # True Negative: Clean token untouched (skip to keep TSV focused on changes/errors)
            continue
        elif r != g and c == g:
            status = "TRUE_POSITIVE"     # Successful repair
        elif r == g and c != g:
            status = "FALSE_POSITIVE"    # Over-correction (damage!)
        elif r != g and c == r:
            status = "FALSE_NEGATIVE"    # Unrepaired typo
        else:
            status = "SUBOPTIMAL_FLIP"   # Changed, but not to Ground Truth

        aligned_events.append((status, g, r, c, context))

    return aligned_events


def main():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    if not gt_files:
        print("Error: No GT files found.")
        return

    print(f"Auditing decisions across {len(gt_files)} pages...")

    decisions = []
    stats = collections.defaultdict(int)

    tp_patterns = collections.defaultdict(int)
    fp_patterns = collections.defaultdict(int)
    fn_patterns = collections.defaultdict(int)

    with open(OUTPUT_TSV, "w", encoding="utf-8") as tsv_out:
        tsv_out.write("PAGE\tSTATUS\tGROUND_TRUTH\tRAW_OCR\tCORRECTED\tCONTEXT\n")

        for gt_path in gt_files:
            base = os.path.basename(gt_path).replace(".gt.txt", "")
            raw_path = os.path.join(RAW_OCR_DIR, f"{base}.ocr.txt")
            corr_path = os.path.join(CORR_DIR, f"{base}.corr.txt")

            if not (os.path.exists(raw_path) and os.path.exists(corr_path)):
                continue

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = canonicalize_amharic_punctuation(f.read())
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_text = canonicalize_amharic_punctuation(f.read())
            with open(corr_path, "r", encoding="utf-8") as f:
                corr_text = canonicalize_amharic_punctuation(f.read())

            gt_tokens = gt_text.split()
            raw_tokens = raw_text.split()
            corr_tokens = corr_text.split()

            events = align_three_way(gt_tokens, raw_tokens, corr_tokens)

            for status, g, r, c, ctx in events:
                stats[status] += 1
                decisions.append((base, status, g, r, c, ctx))
                tsv_out.write(f"{base}\t{status}\t{g}\t{r}\t{c}\t{ctx}\n")

                if status == "TRUE_POSITIVE":
                    tp_patterns[(r, g)] += 1
                elif status == "FALSE_POSITIVE":
                    fp_patterns[(g, c)] += 1
                elif status == "FALSE_NEGATIVE":
                    fn_patterns[(g, r)] += 1

    total_audited = sum(stats.values())

    # Write summary report
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as rep:
        rep.write("======================================================================\n")
        rep.write("          AMHARIC OCR POST-CORRECTION DECISION AUDIT                  \n")
        rep.write("======================================================================\n")
        rep.write(f"Total Evaluated Event Tokens : {total_audited:,}\n")
        rep.write(f"  • TRUE_POSITIVE  (Repaired) : {stats['TRUE_POSITIVE']:,}\n")
        rep.write(f"  • FALSE_POSITIVE (Damaged)  : {stats['FALSE_POSITIVE']:,}\n")
        rep.write(f"  • FALSE_NEGATIVE (Missed)   : {stats['FALSE_NEGATIVE']:,}\n")
        rep.write(f"  • SUBOPTIMAL_FLIP           : {stats['SUBOPTIMAL_FLIP']:,}\n\n")

        rep.write("-" * 65 + "\n")
        rep.write("TOP 20 CLEAN-TOKEN DAMAGES (FALSE POSITIVES: Gold was right -> Corrector broke it)\n")
        rep.write("-" * 65 + "\n")
        for (g, c), count in sorted(fp_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
            rep.write(f"{count:4d}x | Clean: '{g}'  --> Falsely Changed To: '{c}'\n")

        rep.write("\n" + "-" * 65 + "\n")
        rep.write("TOP 20 SUCCESSFUL REPAIRS (TRUE POSITIVES: Raw OCR was wrong -> Corrector fixed it)\n")
        rep.write("-" * 65 + "\n")
        for (r, g), count in sorted(tp_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
            rep.write(f"{count:4d}x | OCR Error: '{r}'  --> Successfully Repaired To: '{g}'\n")

        rep.write("\n" + "-" * 65 + "\n")
        rep.write("TOP 20 MISSED OCR TYPOS (FALSE NEGATIVES: OCR was wrong -> Left uncorrected)\n")
        rep.write("-" * 65 + "\n")
        for (g, r), count in sorted(fn_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
            rep.write(f"{count:4d}x | Gold: '{g}'  <-- Left As OCR Typo: '{r}'\n")

    print(f"\nAudit Complete!")
    print(f"Summary Report : {OUTPUT_SUMMARY}")
    print(f"Full TSV Log   : {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
