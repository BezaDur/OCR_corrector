#!/usr/bin/env python3
"""
Analyze detection results. Writes a TSV and prints Tesseract accuracy.

Outputs: OCR_error_detection/Evaluation/analysis.tsv

Usage:
    python OCR_error_detection/analyze_methods.py
"""
import csv
import difflib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OCR_error_detection import DetectionEngine

_HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONF_DIR = OCR_DIR
CONFUSION = os.path.join(_HERE, "confusion_matrix.json")
OUTPUT_TSV = os.path.join(_HERE, "Evaluation", "analysis.tsv")


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


def edit_distance(a, b):
    """Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def build_reason(sig, in_lex, label):
    parts = []
    if not in_lex:
        parts.append("OOV")
    if sig.one_flip_words:
        parts.append(f"one-flip({len(sig.one_flip_words)})")
    if sig.ocr_conf is not None and sig.ocr_conf < 0.75:
        parts.append(f"low-conf({sig.ocr_conf*100:.0f}%)")
    if sig.rare_glyph_ratio > 0.3:
        parts.append("rare-glyphs")
    if sig.mixed_script:
        parts.append("mixed-script")
    if not parts:
        parts.append(label)
    return ", ".join(parts)


def main():
    pages = load_pages()
    if not pages:
        print("No pages found!")
        return

    # Build lexicon
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

    # Collect rows
    rows = []
    stats = {
        "total": 0, "errors": 0, "caught": 0,
        "insertions": 0,
        # For WER/CER
        "gt_chars": 0, "char_errors": 0,
        "gt_words": 0, "word_errors": 0, "word_substitutions": 0,
        "word_insertions": 0, "word_deletions": 0,
        # Per-page
        "pages": 0,
    }

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()
        if not gt_tokens or not ocr_tokens:
            continue

        alignment = align_words(gt_tokens, ocr_tokens)
        has_conf = conf_map is not None
        stats["pages"] += 1

        # Page-level WER: substitutions + insertions + deletions
        page_subs = 0
        page_ins = 0
        page_dels = 0

        for ocr_idx, gt_idx, is_error in alignment:
            ocr_tok = ocr_tokens[ocr_idx]
            gt_tok = gt_tokens[gt_idx] if gt_idx is not None else ""
            ctx = [t for t in ocr_tokens[max(0, ocr_idx - 2):ocr_idx] if t]
            ocr_conf = conf_map.get(ocr_idx) if has_conf else None

            verdict = detector.verdict(ocr_tok, ctx, ocr_conf=ocr_conf)
            in_lex = detector._norm(ocr_tok) in detector.vocab
            reason = build_reason(verdict.signals, in_lex, verdict.label)

            if gt_idx is None:
                # Insertion
                stats["insertions"] += 1
                stats["word_insertions"] += 1
                is_error_val = 2
            elif is_error:
                stats["errors"] += 1
                stats["word_substitutions"] += 1
                stats["char_errors"] += edit_distance(gt_tok, ocr_tok)
                stats["gt_chars"] += len(gt_tok)
                stats["gt_words"] += 1
                if verdict.label in ("flag", "gray"):
                    stats["caught"] += 1
                is_error_val = 1
            else:
                stats["gt_chars"] += len(gt_tok)
                stats["gt_words"] += 1
                is_error_val = 0

            rows.append({
                "page": page_name,
                "ocr_idx": ocr_idx,
                "gt_idx": gt_idx if gt_idx is not None else "",
                "ocr_word": ocr_tok,
                "gt_word": gt_tok,
                "is_error": is_error_val,
                "verdict": verdict.label,
                "suspicion": f"{verdict.suspicion:.3f}",
                "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                "in_lexicon": 1 if in_lex else 0,
                "one_flip": len(verdict.signals.one_flip_words),
                "reason": reason,
            })

        # Count deletions from the alignment
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, gt_tokens, ocr_tokens, autojunk=False).get_opcodes():
            if tag == "delete":
                stats["word_deletions"] += (i2 - i1)

    # Write TSV
    fieldnames = [
        "page", "ocr_idx", "gt_idx", "ocr_word", "gt_word",
        "is_error", "verdict", "suspicion", "tesseract_conf",
        "in_lexicon", "one_flip", "reason",
    ]
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {OUTPUT_TSV}")

    # ── Tesseract accuracy ──────────────────────────────────────────
    total_words = stats["gt_words"] + stats["word_insertions"]
    wer = 100.0 * (stats["word_substitutions"] + stats["word_insertions"] + stats["word_deletions"]) / max(stats["gt_words"], 1)
    cer = 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1)

    print(f"\n{'='*60}")
    print(f"Tesseract Accuracy")
    print(f"{'='*60}")
    print(f"Pages:                    {stats['pages']}")
    print(f"GT words:                 {stats['gt_words']:,}")
    print(f"Substitutions:            {stats['word_substitutions']:,}")
    print(f"Insertions:               {stats['word_insertions']:,}")
    print(f"Deletions:                {stats['word_deletions']:,}")
    print(f"WER (word error rate):    {wer:.2f}%")
    print(f"CER (char error rate):    {cer:.2f}%")
    print(f"Word accuracy:            {100 - wer:.2f}%")
    print(f"Char accuracy:            {100 - cer:.2f}%")

    # ── Detection accuracy ──────────────────────────────────────────
    if stats["errors"] > 0:
        print(f"\n{'='*60}")
    print(f"Detection Performance")
    print(f"{'='*60}")
    print(f"Error tokens:             {stats['errors']:,}")
    print(f"Caught (flag+gray):       {stats['caught']:,} ({100*stats['caught']/max(stats['errors'],1):.1f}%)")
    print(f"Missed (passed):          {stats['errors'] - stats['caught']:,}")

    # Confidence stats
    confs = [float(r["tesseract_conf"]) for r in rows
             if r["tesseract_conf"] and r["is_error"] != "2"]
    correct_confs = [float(r["tesseract_conf"]) for r in rows
                     if r["tesseract_conf"] and r["is_error"] == "0"]
    error_confs = [float(r["tesseract_conf"]) for r in rows
                   if r["tesseract_conf"] and r["is_error"] == "1"]
    if correct_confs:
        print(f"\nAvg Tesseract confidence (correct words): {sum(correct_confs)/len(correct_confs):.1f}%")
    if error_confs:
        print(f"Avg Tesseract confidence (error words):   {sum(error_confs)/len(error_confs):.1f}%")


if __name__ == "__main__":
    main()