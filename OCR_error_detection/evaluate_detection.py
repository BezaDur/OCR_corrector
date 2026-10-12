#!/usr/bin/env python3
"""
Run the detector on paired (ground truth, OCR) data and print results.

Shows: precision, recall, speed, and per-verdict breakdown.

Usage:
    # Basic (no confidence scores):
    python OCR_error_detection/evaluate_detection.py \\
        --gt-dir eval_data/ground_truth \\
        --ocr-dir eval_data/ocr_output \\
        --confusion confusion_matrix.json

    # With Tesseract confidence scores:
    python OCR_error_detection/evaluate_detection.py \\
        --gt-dir OCR_error_detection/Evaluation/GroundTruth \\
        --ocr-dir OCR_error_detection/Evaluation/OCRed \\
        --confusion confusion_matrix.json \\
        --conf-dir OCR_error_detection/Evaluation/OCRed
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCR_error_detection import DetectionEngine


def load_pages(gt_dir, ocr_dir, conf_dir=None, max_pages=None):
    """Load aligned ground truth, OCR pages, and optional confidence scores.

    Returns list of (page_name, gt_text, ocr_text, conf_map) tuples.
    conf_map: dict of {word_index: confidence} or None.
    """
    pages = []
    # Find GT files — try both .gt.txt and .txt patterns
    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".txt"))
    for fname in gt_files:
        if max_pages and len(pages) >= max_pages:
            break

        base = fname.replace(".gt.txt", "").replace(".txt", "")

        # Find matching OCR file
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            candidate = os.path.join(ocr_dir, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            continue

        gt_path = os.path.join(gt_dir, fname)
        with open(gt_path, encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(ocr_path, encoding="utf-8") as f:
            ocr_text = f.read().strip()

        # Load confidence if available
        conf_map = None
        if conf_dir:
            conf_path = os.path.join(conf_dir, base + ".conf.tsv")
            if os.path.exists(conf_path):
                conf_map = {}
                with open(conf_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("word"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            try:
                                conf_map[len(conf_map)] = float(parts[1])
                            except ValueError:
                                pass

        pages.append((base, gt_text, ocr_text, conf_map))
    return pages


def build_lexicon_from_gt(pages):
    """Build a frequency lexicon from ground truth text."""
    vocab = Counter()
    for _, gt_text, _, _ in pages:
        vocab.update(gt_text.split())
    return dict(vocab)


def evaluate_detection(detector, pages):
    """Run detection on OCR text and compare to ground truth alignment.

    For each OCR token:
      - Align with ground truth (position-based)
      - Mark as error (gt_token != ocr_token) or correct
      - Record the detection verdict (with confidence if available)
    """
    stats = {
        "total_tokens": 0,
        "total_errors": 0,
        "total_correct": 0,
        "verdicts": defaultdict(lambda: {"correct": 0, "error": 0}),
        "pages_processed": 0,
        "pages_with_confidence": 0,
    }

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()

        n = min(len(gt_tokens), len(ocr_tokens))
        if n == 0:
            continue

        has_conf = conf_map is not None and len(conf_map) > 0
        if has_conf:
            stats["pages_with_confidence"] += 1

        for i in range(n):
            gt_tok = gt_tokens[i]
            ocr_tok = ocr_tokens[i]
            is_error = (gt_tok != ocr_tok)

            ctx = [t for t in ocr_tokens[max(0, i - 2):i] if t]

            # Pass confidence if available
            ocr_conf = conf_map.get(i) if has_conf else None
            verdict = detector.verdict(ocr_tok, ctx, ocr_conf=ocr_conf)
            label = verdict.label

            stats["total_tokens"] += 1
            if is_error:
                stats["total_errors"] += 1
                stats["verdicts"][label]["error"] += 1
            else:
                stats["total_correct"] += 1
                stats["verdicts"][label]["correct"] += 1

        stats["pages_processed"] += 1

    return stats


def print_results(stats, elapsed):
    """Print evaluation results."""
    total = stats["total_tokens"]
    errors = stats["total_errors"]
    correct = stats["total_correct"]
    error_rate = 100.0 * errors / max(total, 1)

    print(f"\n{'='*60}")
    print(f"OCR Error Detection Evaluation Results")
    print(f"{'='*60}")
    print(f"Pages processed:    {stats['pages_processed']}")
    if stats["pages_with_confidence"] > 0:
        print(f"Pages w/ confidence: {stats['pages_with_confidence']}")
    print(f"Total tokens:       {total:,}")
    print(f"Error tokens:       {errors:,} ({error_rate:.2f}%)")
    print(f"Correct tokens:     {correct:,}")

    print(f"\n--- Verdict breakdown ---")
    print(f"{'Verdict':<8} {'Correct':>10} {'Error':>10} {'Total':>10} "
          f"{'Prec%':>8} {'Recall%':>8}")

    for label in ["pass", "gray", "flag"]:
        d = stats["verdicts"][label]
        t = d["correct"] + d["error"]
        prec = 100.0 * d["error"] / max(t, 1) if t > 0 else 0
        recall = 100.0 * d["error"] / max(errors, 1) if errors > 0 else 0
        print(f"{label:<8} {d['correct']:>10,} {d['error']:>10,} {t:>10,} "
              f"{prec:>8.1f} {recall:>8.1f}")

    pass_d = stats["verdicts"]["pass"]
    flagged_d = stats["verdicts"]["gray"]
    flag_d = stats["verdicts"]["flag"]

    pass_total = pass_d["correct"] + pass_d["error"]
    pass_precision = 100.0 * pass_d["correct"] / max(pass_total, 1)

    caught_errors = flagged_d["error"] + flag_d["error"]
    recall = 100.0 * caught_errors / max(errors, 1)

    flagged_total = (flagged_d["correct"] + flagged_d["error"] +
                     flag_d["correct"] + flag_d["error"])
    flag_precision = 100.0 * caught_errors / max(flagged_total, 1)

    f1 = (2 * flag_precision * recall / max(flag_precision + recall, 0.01))

    print(f"\n--- Summary ---")
    print(f"PASS precision (correct among passed): {pass_precision:.1f}%")
    print(f"Flag+Gray recall (errors caught):      {recall:.1f}%")
    print(f"Flag+Gray precision (errors among flagged): {flag_precision:.1f}%")
    print(f"F1 (flag+gray):                        {f1:.1f}")

    if elapsed > 0:
        tps = total / elapsed
        print(f"\nThroughput: {tps:,.0f} tokens/sec ({elapsed:.2f}s total)")

    return {
        "pass_precision": pass_precision,
        "recall": recall,
        "flag_precision": flag_precision,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gt-dir", required=True,
                        help="Directory of ground truth text files")
    parser.add_argument("--ocr-dir", required=True,
                        help="Directory of OCR output text files")
    parser.add_argument("--confusion", required=True,
                        help="Path to confusion_matrix.json")
    parser.add_argument("--conf-dir", default=None,
                        help="Directory with *.conf.tsv files (Tesseract confidence)")
    parser.add_argument("--lexicon", default=None,
                        help="Path to external lexicon (default: build from GT)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit number of pages")
    parser.add_argument("--pass-below", type=float, default=0.30,
                        help="PASS threshold")
    parser.add_argument("--flag-above", type=float, default=0.70,
                        help="FLAG threshold")
    parser.add_argument("--symspell-distance", type=int, default=1,
                        help="SymSpell edit distance")
    parser.add_argument("--max-lexicon-types", type=int, default=None,
                        help="Cap lexicon size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load pages
    pages = load_pages(args.gt_dir, args.ocr_dir, args.conf_dir, args.max_pages)
    print(f"Loaded {len(pages)} pages")
    n_conf = sum(1 for _, _, _, c in pages if c)
    if n_conf:
        print(f"  {n_conf} pages have confidence scores")

    if not pages:
        print("No pages found!")
        sys.exit(1)

    # Build or load lexicon
    if args.lexicon:
        detector = DetectionEngine(
            dict_vocab_path=args.lexicon,
            confusion_matrix_path=args.confusion,
            pass_below=args.pass_below,
            flag_above=args.flag_above,
            symspell_max_edit_distance=args.symspell_distance,
            max_lexicon_types=args.max_lexicon_types,
            verbose=args.verbose,
        )
    else:
        import tempfile
        vocab = build_lexicon_from_gt(pages)
        tmpdir = tempfile.mkdtemp()
        lex_path = os.path.join(tmpdir, "eval_lexicon.txt")
        with open(lex_path, "w", encoding="utf-8") as f:
            for word, freq in sorted(vocab.items(), key=lambda x: -x[1]):
                f.write(f"{word} {freq}\n")
        print(f"Built lexicon from GT: {len(vocab):,} types")

        detector = DetectionEngine(
            dict_vocab_path=lex_path,
            confusion_matrix_path=args.confusion,
            pass_below=args.pass_below,
            flag_above=args.flag_above,
            symspell_max_edit_distance=args.symspell_distance,
            max_lexicon_types=args.max_lexicon_types,
            verbose=args.verbose,
        )

    # Evaluate
    t0 = time.perf_counter()
    stats = evaluate_detection(detector, pages)
    elapsed = time.perf_counter() - t0

    results = print_results(stats, elapsed)

    ds = detector.detection_stats()
    print(f"\nDetection engine stats: {ds}")


if __name__ == "__main__":
    main()