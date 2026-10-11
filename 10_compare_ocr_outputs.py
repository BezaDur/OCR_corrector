#!/usr/bin/env python3
"""
Step 10: Compare OCR outputs against ground truth. No model, no GPU, no images.

Answers, in seconds: "which of my OCR outputs is actually good?"

CER/WER is computed from TEXT pairs alone (ground_truth/*.gt.txt vs
<ocr_dir>/*.ocr.txt), so this runs anywhere the text files live -- it does
not need Tesseract, the page images, the fonts, or kenlm.

Motivation
----------
The correction pipeline is only ever evaluated against one OCR output
(`ocr_new_model`). If that output came from an undertrained model while a
better model exists, every "the corrector is net-negative" conclusion is
really "we are measuring against bad upstream input". Compare before you
conclude.

Usage
-----
    # the question that matters
    python 10_compare_ocr_outputs.py \\
        --gt /media/des/Tess/Tesseract2/synthetic_eval/ground_truth \\
        --ocr pred=/media/des/Tess/Tesseract2/synthetic_eval/ocr_pred \\
        --ocr new=/media/des/Tess/Tesseract2/synthetic_eval/ocr_new_model

    # any number of --ocr name=DIR pairs
"""
import argparse
import glob
import os
import sys


def edit_distance(a, b):
    """Levenshtein distance with O(min(len)) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def page_cer_wer(gt_text, ocr_text):
    c_err = edit_distance(gt_text, ocr_text)
    c_ref = len(gt_text)
    g = gt_text.split()
    o = ocr_text.split()
    w_err = edit_distance(g, o)
    w_ref = len(g)
    return c_err, c_ref, w_err, w_ref


def evaluate(gt_dir, ocr_dir, limit=None):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.gt.txt")))
    if limit:
        gt_files = gt_files[:limit]
    if not gt_files:
        raise SystemExit(f"No GT files in {gt_dir}")
    c_err = c_ref = w_err = w_ref = 0
    matched = 0
    for gt_path in gt_files:
        base = os.path.basename(gt_path).replace(".gt.txt", "")
        ocr_path = os.path.join(ocr_dir, f"{base}.ocr.txt")
        if not os.path.exists(ocr_path):
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = f.read()
        with open(ocr_path, encoding="utf-8") as f:
            ocr = f.read()
        a, b, c, d = page_cer_wer(gt, ocr)
        c_err += a; c_ref += b; w_err += c; w_ref += d
        matched += 1
    if not matched:
        raise SystemExit(f"No matching pages in {ocr_dir} "
                         f"(expected <base>.ocr.txt)")
    cer = 100.0 * c_err / max(c_ref, 1)
    wer = 100.0 * w_err / max(w_ref, 1)
    return dict(pages=matched, cer=cer, wer=wer,
                acc=100.0 - cer, c_ref=c_ref, w_ref=w_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ocr", action="append", default=[],
                    metavar="NAME=DIR",
                    help="repeatable: --ocr pred=/path/to/ocr_pred")
    ap.add_argument("--limit", type=int, help="only first N pages (fast check)")
    args = ap.parse_args()

    if not args.ocr:
        raise SystemExit("give at least one --ocr NAME=DIR")

    rows = []
    for spec in args.ocr:
        if "=" not in spec:
            raise SystemExit(f"--ocr expects NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        sys.stderr.write(f"scoring {name} <- {d}\n")
        rows.append((name, d, evaluate(args.gt, d, args.limit)))

    print()
    print("=" * 74)
    print(f"{'output':<14}{'pages':>7}{'CER %':>10}{'WER %':>10}"
          f"{'char acc %':>12}{'CER/WER':>10}")
    print("-" * 74)
    for name, d, r in rows:
        print(f"{name:<14}{r['pages']:>7}{r['cer']:>10.4f}{r['wer']:>10.4f}"
              f"{r['acc']:>12.4f}{r['cer']/max(r['wer'],1e-9):>10.3f}")
    print("=" * 74)
    print()
    print("CER/WER is the tell. chars wrong per erroneous word = "
          "(CER/WER) x 4.14")
    for name, _, r in rows:
        ratio = r["cer"] / max(r["wer"], 1e-9)
        verdict = ("errors are 1-2 chars -> correctable"
                   if ratio < 0.5 else
                   "erroneous words largely destroyed -> NOT correctable")
        print(f"  {name:<14} {ratio:.3f} -> {ratio*4.14:.1f}/4.1 chars wrong; {verdict}")

    best = min(rows, key=lambda x: x[2]["cer"])
    print(f"\nbest: {best[0]} at {best[2]['cer']:.4f}% CER "
          f"({best[2]['acc']:.2f}% character accuracy)")


if __name__ == "__main__":
    main()
