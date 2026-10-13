"""Confidence vs ground truth across pages.

Counts, for words with a clean or error status:
  - clean words below the low threshold (default 50)
  - error words at or above the high threshold (default 80), and at or above the low threshold

Usage: python3 tsv_gt_stats.py TSV_DIR GT_DIR [--low 50] [--high 80]
TSV_DIR holds <page>.tsv files; GT_DIR holds <page>.gt.txt files.
"""
import argparse
import csv
import glob
import os

from tsv_gt import annotate_rows


def collect(tsv_dir: str, gt_dir: str):
    words = []  # (page, conf, status)
    pages = 0
    for tsv in sorted(glob.glob(os.path.join(tsv_dir, "*.tsv"))):
        base = os.path.basename(tsv)[:-len(".tsv")]
        gt_path = os.path.join(gt_dir, base + ".gt.txt")
        if not os.path.exists(gt_path):
            print(f"skip {base}: no ground truth at {gt_path}")
            continue
        with open(gt_path, encoding="utf-8") as fh:
            gt = fh.read()
        with open(tsv, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
        for r in annotate_rows(rows, gt):
            if r["gt_status"] in ("clean", "error"):
                words.append((base, float(r["conf"]), r["gt_status"]))
        pages += 1
    return words, pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv_dir")
    ap.add_argument("gt_dir")
    ap.add_argument("--low", type=float, default=50)
    ap.add_argument("--high", type=float, default=80)
    a = ap.parse_args()
    words, pages = collect(a.tsv_dir, a.gt_dir)
    clean = [c for _, c, s in words if s == "clean"]
    errs = [c for _, c, s in words if s == "error"]
    print(f"pages: {pages}   judged words: {len(words)}   clean: {len(clean)}   error: {len(errs)}")
    print(f"clean words with conf < {a.low:g}:            {sum(c < a.low for c in clean)}"
          f"  ({100 * sum(c < a.low for c in clean) / max(len(clean), 1):.1f}% of clean)")
    print(f"error words with conf >= {a.low:g}:           {sum(c >= a.low for c in errs)}"
          f"  ({100 * sum(c >= a.low for c in errs) / max(len(errs), 1):.1f}% of errors)")
    print(f"error words with conf >= {a.high:g}:           {sum(c >= a.high for c in errs)}"
          f"  ({100 * sum(c >= a.high for c in errs) / max(len(errs), 1):.1f}% of errors)")
    print(f"error words with conf < {a.low:g}  (caught):   {sum(c < a.low for c in errs)}")


if __name__ == "__main__":
    main()
