#!/usr/bin/env python3
"""
Run the detector on the Evaluation data and print results.

Usage:
    python 4_ocr_correction/error_detection/evaluate_detection.py
"""
import argparse
import difflib
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_detection import DetectionEngine

# ── Paths (relative to repo root) ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONF_DIR = OCR_DIR  # confidence TSVs live next to the OCR text
CONFUSION = os.path.join(_HERE, "confusion_matrix.json")


def load_pages():
    pages = []
    gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".txt"))
    for fname in gt_files:
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            candidate = os.path.join(OCR_DIR, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            continue
        with open(os.path.join(GT_DIR, fname), encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(ocr_path, encoding="utf-8") as f:
            ocr_text = f.read().strip()

        # load confidence
        conf_map = None
        conf_path = os.path.join(CONF_DIR, base + ".conf.tsv")
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


def build_lexicon(pages):
    vocab = Counter()
    for _, gt_text, _, _ in pages:
        vocab.update(gt_text.split())
    return dict(vocab)


def align_words(gt_tokens, ocr_tokens):
    sm = difflib.SequenceMatcher(None, gt_tokens, ocr_tokens, autojunk=False)
    results = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                results.append((j1 + k, i1 + k, False))
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                results.append((j1 + k, i1 + k, True))
            for k in range(n, j2 - j1):
                results.append((j1 + k, None, True))
        elif tag == "insert":
            for k in range(j2 - j1):
                results.append((j1 + k, None, True))
        elif tag == "delete":
            pass
    return results


def evaluate(detector, pages):
    stats = {
        "total_aligned": 0, "total_errors": 0, "total_correct": 0,
        "total_insertions": 0,
        "verdicts": defaultdict(lambda: {"correct": 0, "error": 0}),
        "pages_processed": 0, "pages_with_confidence": 0,
    }
    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()
        if not gt_tokens or not ocr_tokens:
            continue
        alignment = align_words(gt_tokens, ocr_tokens)
        has_conf = conf_map is not None and len(conf_map) > 0
        if has_conf:
            stats["pages_with_confidence"] += 1
        for ocr_idx, gt_idx, is_error in alignment:
            ocr_tok = ocr_tokens[ocr_idx]
            ctx = [t for t in ocr_tokens[max(0, ocr_idx - 2):ocr_idx] if t]
            ocr_conf = conf_map.get(ocr_idx) if has_conf else None
            verdict = detector.verdict(ocr_tok, ctx, ocr_conf=ocr_conf)
            label = verdict.label
            if gt_idx is None:
                stats["total_insertions"] += 1
            elif is_error:
                stats["total_aligned"] += 1
                stats["total_errors"] += 1
                stats["verdicts"][label]["error"] += 1
            else:
                stats["total_aligned"] += 1
                stats["total_correct"] += 1
                stats["verdicts"][label]["correct"] += 1
        stats["pages_processed"] += 1
    return stats


def print_results(stats, elapsed):
    total = stats["total_aligned"]
    errors = stats["total_errors"]
    correct = stats["total_correct"]
    insertions = stats["total_insertions"]
    error_rate = 100.0 * errors / max(total, 1)

    print(f"\n{'='*60}")
    print(f"OCR Error Detection Evaluation Results")
    print(f"{'='*60}")
    print(f"Pages processed:     {stats['pages_processed']}")
    if stats["pages_with_confidence"] > 0:
        print(f"Pages w/ confidence: {stats['pages_with_confidence']}")
    print(f"Aligned tokens:      {total:,}")
    print(f"  Error tokens:      {errors:,} ({error_rate:.2f}%)")
    print(f"  Correct tokens:    {correct:,}")
    print(f"Insertions:          {insertions:,} (OCR words with no GT match)")

    print(f"\n{'Verdict':<8} {'Correct':>10} {'Error':>10} {'Total':>10} "
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

    print(f"\nPASS precision:  {pass_precision:.1f}%")
    print(f"Recall:          {recall:.1f}%")
    print(f"Flag precision:  {flag_precision:.1f}%")
    print(f"F1:              {f1:.1f}")
    if elapsed > 0:
        print(f"Throughput:      {(total + insertions) / elapsed:,.0f} tokens/sec")


def main():
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")
    if not pages:
        print("No pages found!")
        sys.exit(1)

    vocab = build_lexicon(pages)
    import tempfile
    tmpdir = tempfile.mkdtemp()
    lex_path = os.path.join(tmpdir, "lex.txt")
    with open(lex_path, "w", encoding="utf-8") as f:
        for w, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{w} {freq}\n")
    print(f"Lexicon: {len(vocab):,} types")

    detector = DetectionEngine(
        dict_vocab_path=lex_path,
        confusion_matrix_path=CONFUSION,
        verbose=False,
    )

    t0 = time.perf_counter()
    stats = evaluate(detector, pages)
    elapsed = time.perf_counter() - t0
    print_results(stats, elapsed)


if __name__ == "__main__":
    main()