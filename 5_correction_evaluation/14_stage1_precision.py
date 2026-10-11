"""Step 14: precision of Stage 1's flagged set, from per-token verdict files.

Reads the <page>.verdicts.jsonl files written by 04_evaluate_full_corrector.py
(--verdict-dir). Each row is one token with its Stage 1 label, its suspicion
score, and whether it is a true error, a clean token, or ambiguous.

Prints:
  - counts per Stage 1 label, split by true status
  - a threshold sweep on suspicion: precision, error recall, clean tokens flagged
  - the lowest threshold whose precision reaches the target

Usage: python3 14_stage1_precision.py VERDICT_DIR [--target 0.85]
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stage1_labels import threshold_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict_dir")
    ap.add_argument("--target", type=float, default=0.85)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.verdict_dir, "*.verdicts.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise SystemExit(f"No verdict rows found in {args.verdict_dir}")

    print(f"Tokens: {len(rows)}")
    by_label = Counter((r["label"], r["status"]) for r in rows)
    print("\nlabel   clean  error  ambiguous")
    for label in ("pass", "gray", "flag"):
        print(f"{label:<7} {by_label[(label, 'clean')]:>5}  {by_label[(label, 'error')]:>5}  "
              f"{by_label[(label, 'ambiguous')]:>9}")

    thresholds = [round(x / 100, 2) for x in range(0, 101, 5)]
    sweep = threshold_sweep(rows, thresholds)
    print("\nthreshold  flagged  errors  clean  precision  error_recall")
    for s in sweep:
        p = "  -  " if s["precision"] is None else f"{s['precision']:.3f}"
        r = "  -  " if s["error_recall"] is None else f"{s['error_recall']:.3f}"
        print(f"{s['threshold']:>9.2f}  {s['flagged']:>7}  {s['errors']:>6}  {s['clean']:>5}  "
              f"{p:>9}  {r:>12}")

    hit = [s for s in sweep if s["precision"] is not None and s["precision"] >= args.target]
    if hit:
        best = hit[0]
        print(f"\nLowest threshold with precision >= {args.target}: {best['threshold']} "
              f"(flagged {best['flagged']}, error recall {best['error_recall']:.3f})")
    else:
        print(f"\nNo threshold reaches precision >= {args.target}.")


if __name__ == "__main__":
    main()
