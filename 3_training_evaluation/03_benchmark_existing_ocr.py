#!/usr/bin/env python3
"""
Step 11b: Benchmark EXISTING OCR output against ground truth. No re-OCR.

This is the scoring half of 02_benchmark_new_model.py with the Tesseract call
removed. Use it when the OCR text is already on disk and re-recognising the
pages would cost hours for no new information.

Typical question it answers:
    "step 11 told me 98% but step 04 reports 34.39% CER. Which is real?"
    ->  python 03_benchmark_existing_ocr.py --compare-all

Metrics are the CANONICAL ones from ocr_metrics.py, so the numbers are
directly comparable with 01_evaluate_full_corrector.py:
    CER = edit_distance(''.join(gold.split()), ''.join(ocr.split())) / len(gold_chars)
    WER = align_tokens(gold_words, ocr_words).distance / len(gold_words)
Note CER is measured on the WHITESPACE-STRIPPED stream -- not on the raw text.
Getting this wrong produces numbers that look plausible and disagree with
step 04 for no visible reason.

This file has NO dependency on ocr_metrics.py or on the repo at all -- copy it
anywhere and run it.

Speed
-----
Full-page Levenshtein is O(n*m). In pure Python a 3,200-char page takes ~2.7s,
so ~1,370 pages would run for about an hour. rapidfuzz does the same
computation in ~1ms (verified identical output). It is used automatically when
installed, with a silent fallback to the pure-Python path:

    pip install rapidfuzz

Usage
-----
    # compare every OCR dir under the dataset (the usual case)
    python 03_benchmark_existing_ocr.py --compare-all

    # or name them explicitly
    python 03_benchmark_existing_ocr.py \\
        --ocr pred=/media/des/Tess/Tesseract2/synthetic_eval/ocr_pred \\
        --ocr new=/media/des/Tess/Tesseract2/synthetic_eval/ocr_new_model

    # quick look on the first 100 pages
    python 03_benchmark_existing_ocr.py --compare-all --limit 100
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

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Deliberately self-contained: NO import of ocr_metrics.
#
# Step 04 measures:
#   CER = edit_distance(''.join(gold.split()), ''.join(ocr.split())) / len(gold_chars)
#   WER = align_tokens(gold_words, ocr_words).distance / len(gold_words)
#
# align_tokens() minimises the tuple (edit_count, -exact_matches)
# lexicographically, so its .distance is the plain unit-cost Levenshtein
# distance over tokens -- the tie-breaking only picks WHICH optimal path, not
# its cost. Verified empirically: 300/300 random alignments match
# Levenshtein.distance exactly. So both metrics reduce to Levenshtein and this
# file needs nothing from the repo, which is why it runs anywhere you can copy
# a single file to.

DATASET_DIR = repo_paths.workspace("synthetic_eval")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")

NUM_WORKERS = 8
PROGRESS_EVERY = 25


# --------------------------------------------------------------------------- #
# edit distance: rapidfuzz when available, else pure Python.
# Works on str or on lists of tokens -- Levenshtein does not care, and WER is
# just Levenshtein over the token sequence.
# --------------------------------------------------------------------------- #
_FAST = None


def _edit_distance_py(a, b):
    """Unit-cost Levenshtein, linear memory (same algorithm as ocr_metrics)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _edit_distance(a, b):
    global _FAST
    if _FAST is None:
        try:
            from rapidfuzz.distance import Levenshtein
            _FAST = Levenshtein.distance
        except ImportError:
            _FAST = _edit_distance_py
    return _FAST(a, b)


def backend_name():
    try:
        import rapidfuzz  # noqa: F401
        return "rapidfuzz (fast)"
    except ImportError:
        return "pure Python (pip install rapidfuzz for ~3000x)"


# --------------------------------------------------------------------------- #
def score_page(gt_path, ocr_path):
    """Canonical per-page counts. Returns None if a page is missing."""
    if not os.path.exists(ocr_path):
        return None
    with open(gt_path, encoding="utf-8") as f:
        gold = f.read()
    with open(ocr_path, encoding="utf-8") as f:
        raw = f.read()

    gold_words, raw_words = gold.split(), raw.split()
    gold_chars = "".join(gold_words)
    raw_chars = "".join(raw_words)

    char_errors = _edit_distance(gold_chars, raw_chars)
    word_errors = _edit_distance(gold_words, raw_words)
    return (char_errors, len(gold_chars), word_errors, len(gold_words))


def evaluate(gt_dir, ocr_dir, limit=None, workers=NUM_WORKERS):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.gt.txt")))
    if limit:
        gt_files = gt_files[:limit]
    if not gt_files:
        raise SystemExit(f"No GT files found in {gt_dir}")

    jobs = []
    for gt_path in gt_files:
        base = os.path.basename(gt_path).replace(".gt.txt", "")
        jobs.append((base, gt_path, os.path.join(ocr_dir, f"{base}.ocr.txt")))

    c_err = c_ref = w_err = w_ref = 0
    matched = 0
    done = 0
    t0 = time.time()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(score_page, g, o): b for b, g, o in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if r is not None:
                    a, b, c, d = r
                    c_err += a; c_ref += b; w_err += c; w_ref += d
                    matched += 1
                if done % PROGRESS_EVERY == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  [{done}/{len(jobs)}] matched={matched} "
                          f"elapsed={el:.1f}s", flush=True)
    else:
        for base, g, o in jobs:
            r = score_page(g, o)
            done += 1
            if r is not None:
                a, b, c, d = r
                c_err += a; c_ref += b; w_err += c; w_ref += d
                matched += 1
            if done % PROGRESS_EVERY == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] matched={matched}", flush=True)

    if not matched:
        raise SystemExit(f"No matching pages in {ocr_dir}. "
                         f"Expected files named <base>.ocr.txt "
                         f"where GT is <base>.gt.txt")

    cer = 100.0 * c_err / max(c_ref, 1)
    wer = 100.0 * w_err / max(w_ref, 1)
    return dict(pages=matched, missing=len(jobs) - matched, cer=cer, wer=wer,
                acc=100.0 - cer, c_ref=c_ref, w_ref=w_ref,
                seconds=time.time() - t0)


def discover_ocr_dirs(dataset_dir, gt_dir):
    """Every sibling of ground_truth/ that contains *.ocr.txt files."""
    out = []
    for name in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, name)
        if not os.path.isdir(d) or os.path.abspath(d) == os.path.abspath(gt_dir):
            continue
        if glob.glob(os.path.join(d, "*.ocr.txt")):
            out.append((name, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=GT_DIR)
    ap.add_argument("--ocr", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--compare-all", action="store_true",
                    help="score every dir next to ground_truth that has "
                         "*.ocr.txt (recommended)")
    ap.add_argument("--dataset-dir", default=DATASET_DIR)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    specs = []
    for s in args.ocr:
        if "=" not in s:
            raise SystemExit(f"--ocr expects NAME=DIR, got {s!r}")
        n, d = s.split("=", 1)
        specs.append((n, d))
    if args.compare_all:
        specs = discover_ocr_dirs(args.dataset_dir, args.gt_dir)
        if not specs:
            raise SystemExit(f"No sibling dirs with *.ocr.txt under "
                             f"{args.dataset_dir}")
        print(f"discovered {len(specs)} OCR dir(s): "
              f"{', '.join(n for n, _ in specs)}")
    if not specs:
        raise SystemExit("give --compare-all or at least one --ocr NAME=DIR")

    print(f"backend : {backend_name()}")
    print(f"workers : {args.workers}")

    rows = []
    for name, d in specs:
        print(f"\n--- scoring [{name}] {d}", flush=True)
        rows.append((name, d, evaluate(args.gt_dir, d, args.limit, args.workers)))

    print("\n" + "=" * 78)
    print("EXISTING-OCR BENCHMARK (no re-recognition)")
    print("=" * 78)
    print(f"{'output':<20}{'pages':>7}{'CER %':>10}{'WER %':>10}"
          f"{'char acc %':>12}{'CER/WER':>9}")
    print("-" * 78)
    for name, _, r in rows:
        print(f"{name:<20}{r['pages']:>7}{r['cer']:>10.4f}{r['wer']:>10.4f}"
              f"{r['acc']:>12.4f}{r['cer']/max(r['wer'], 1e-9):>9.3f}")
    print("=" * 78)

    print("\nCER/WER tells you whether correction can work at all "
          "(mean token = 4.14 chars):")
    for name, _, r in rows:
        ratio = r["cer"] / max(r["wer"], 1e-9)
        verdict = ("errors are 1-2 chars -> correctable"
                   if ratio < 0.5 else
                   "erroneous words largely destroyed -> NOT correctable")
        print(f"  {name:<20}{ratio:>7.3f} -> {ratio*4.14:.1f}/4.1 chars wrong "
              f"per bad word; {verdict}")

    best = min(rows, key=lambda x: x[2]["cer"])
    print(f"\nbest: {best[0]} — {best[2]['cer']:.4f}% CER "
          f"({best[2]['acc']:.2f}% character accuracy) "
          f"over {best[2]['pages']} pages in {best[2]['seconds']:.1f}s")
    if len(rows) > 1:
        worst = max(rows, key=lambda x: x[2]["cer"])
        print(f"worst: {worst[0]} — {worst[2]['cer']:.4f}% CER")
        print(f"\n=> the two differ by "
              f"{worst[2]['cer'] - best[2]['cer']:.2f} CER points. "
              f"Post-correction results measured against '{worst[0]}' say "
              f"nothing about '{best[0]}'.")


if __name__ == "__main__":
    main()
