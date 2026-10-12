#!/usr/bin/env python3
"""
Break down which detection methods catch the errors.

Usage:
    python OCR_error_detection/analyze_methods.py
"""
import difflib
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCR_error_detection import DetectionEngine

_HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONF_DIR = OCR_DIR
CONFUSION = os.path.join(_HERE, "confusion_matrix.json")


def load_pages():
    pages = []
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".txt"):
            continue
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            candidate = os.path.join(OCR_DIR, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            continue
        gt_text = open(os.path.join(GT_DIR, fname), encoding="utf-8").read().strip()
        ocr_text = open(ocr_path, encoding="utf-8").read().strip()
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
    return results


def classify_detection(sig, ocr_tok, vocab, norm_fn):
    """Figure out WHY this token was flagged/grayed.

    Returns a short reason string.
    """
    reasons = []

    # Is the OCR word in the lexicon?
    if not sig.in_lexicon:
        reasons.append("OOV")

    # Did one-flip find a real word?
    if sig.one_flip_words:
        reasons.append(f"one-flip({len(sig.one_flip_words)})")

    # Is OCR confidence low?
    if sig.ocr_conf is not None and sig.ocr_conf < 0.75:
        reasons.append(f"low-conf({sig.ocr_conf*100:.0f}%)")

    # Rare glyphs?
    if sig.rare_glyph_ratio > 0.3:
        reasons.append("rare-glyphs")

    # Mixed script?
    if sig.mixed_script:
        reasons.append("mixed-script")

    if not reasons:
        reasons.append("score-only")

    return ", ".join(reasons)


def main():
    pages = load_pages()
    if not pages:
        print("No pages found!")
        return

    vocab = Counter()
    for _, gt_text, _, _ in pages:
        vocab.update(gt_text.split())

    import tempfile
    tmpdir = tempfile.mkdtemp()
    lex_path = os.path.join(tmpdir, "lex.txt")
    with open(lex_path, "w", encoding="utf-8") as f:
        for w, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{w} {freq}\n")

    detector = DetectionEngine(
        dict_vocab_path=lex_path,
        confusion_matrix_path=CONFUSION,
        verbose=False,
    )

    # Track what catches errors
    caught_reasons = Counter()      # reason -> count of errors caught
    missed_examples = []            # (gt, ocr, conf) for missed errors
    caught_examples = defaultdict(list)  # reason -> [(gt, ocr)]

    total_errors = 0
    caught_errors = 0

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()
        if not gt_tokens or not ocr_tokens:
            continue
        alignment = align_words(gt_tokens, ocr_tokens)
        has_conf = conf_map is not None

        for ocr_idx, gt_idx, is_error in alignment:
            if gt_idx is None or not is_error:
                continue
            total_errors += 1

            ocr_tok = ocr_tokens[ocr_idx]
            gt_tok = gt_tokens[gt_idx]
            ctx = [t for t in ocr_tokens[max(0, ocr_idx - 2):ocr_idx] if t]
            ocr_conf = conf_map.get(ocr_idx) if has_conf else None

            verdict = detector.verdict(ocr_tok, ctx, ocr_conf=ocr_conf)

            if verdict.label in ("flag", "gray"):
                caught_errors += 1
                reason = classify_detection(
                    verdict.signals, ocr_tok, vocab, detector._norm)
                caught_reasons[reason] += 1
                if len(caught_examples[reason]) < 3:
                    caught_examples[reason].append((gt_tok, ocr_tok))
            else:
                if len(missed_examples) < 20:
                    missed_examples.append((gt_tok, ocr_tok, ocr_conf))

    # Print results
    print(f"{'='*60}")
    print(f"Detection Method Analysis")
    print(f"{'='*60}")
    print(f"Total errors:  {total_errors:,}")
    print(f"Caught:        {caught_errors:,} ({100*caught_errors/max(total_errors,1):.1f}%)")
    print(f"Missed:        {total_errors - caught_errors:,} ({100*(total_errors-caught_errors)/max(total_errors,1):.1f}%)")

    print(f"\n--- What caught the errors ---")
    print(f"{'Method':<30} {'Count':>8} {'% of caught':>12} {'% of all':>10}")
    print(f"{'-'*30} {'-'*8} {'-'*12} {'-'*10}")
    for reason, count in caught_reasons.most_common():
        pct_caught = 100 * count / max(caught_errors, 1)
        pct_all = 100 * count / max(total_errors, 1)
        print(f"{reason:<30} {count:>8,} {pct_caught:>11.1f}% {pct_all:>9.1f}%")

    # Show examples for each method
    print(f"\n--- Examples per method ---")
    for reason, examples in sorted(caught_examples.items(),
                                     key=lambda x: -caught_reasons[x[0]]):
        print(f"\n  {reason}:")
        for gt, ocr in examples:
            print(f"    GT={gt!r:25s} OCR={ocr!r}")

    # Show missed errors
    print(f"\n--- Missed errors (passed when wrong) ---")
    for gt, ocr, conf in missed_examples:
        in_lex = detector._norm(ocr) in detector.vocab
        conf_str = f"conf={conf:.0f}%" if conf is not None else "no-conf"
        print(f"  GT={gt!r:25s} OCR={ocr!r:25s} in_lex={in_lex} {conf_str}")


if __name__ == "__main__":
    main()