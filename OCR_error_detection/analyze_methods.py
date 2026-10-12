#!/usr/bin/env python3
"""
Analyze detection results and write a comprehensive TSV.

Outputs: OCR_error_detection/Evaluation/analysis.tsv

Columns:
  page          — page name
  ocr_idx       — word position in OCR output
  gt_idx        — aligned position in ground truth (empty if insertion)
  ocr_word      — what Tesseract output
  gt_word       — what it should be
  is_error      — 1 if OCR != GT, 0 if correct
  verdict       — pass / gray / flag
  suspicion     — detector score (0-1)
  tesseract_conf — Tesseract confidence (0-100, empty if no conf file)
  in_lexicon    — 1 if OCR word is in dictionary
  one_flip      — number of one-flip dictionary hits
  reason        — why the detector gave this verdict

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


def build_reason(sig, in_lex, label):
    """Short text explaining why the detector gave this verdict."""
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
    stats = {"total": 0, "errors": 0, "caught": 0}

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens = gt_text.split()
        ocr_tokens = ocr_text.split()
        if not gt_tokens or not ocr_tokens:
            continue

        alignment = align_words(gt_tokens, ocr_tokens)
        has_conf = conf_map is not None

        # Also handle insertions (OCR words with no GT) — run detection on them
        aligned_ocr_idxs = {ocr_idx for ocr_idx, _, _ in alignment}

        for ocr_idx, gt_idx, is_error in alignment:
            ocr_tok = ocr_tokens[ocr_idx]
            gt_tok = gt_tokens[gt_idx] if gt_idx is not None else ""
            ctx = [t for t in ocr_tokens[max(0, ocr_idx - 2):ocr_idx] if t]
            ocr_conf = conf_map.get(ocr_idx) if has_conf else None

            verdict = detector.verdict(ocr_tok, ctx, ocr_conf=ocr_conf)
            in_lex = detector._norm(ocr_tok) in detector.vocab

            reason = build_reason(verdict.signals, in_lex, verdict.label)

            stats["total"] += 1
            if gt_idx is not None and is_error:
                stats["errors"] += 1
                if verdict.label in ("flag", "gray"):
                    stats["caught"] += 1

            rows.append({
                "page": page_name,
                "ocr_idx": ocr_idx,
                "gt_idx": gt_idx if gt_idx is not None else "",
                "ocr_word": ocr_tok,
                "gt_word": gt_tok,
                "is_error": 1 if (gt_idx is not None and is_error) else (2 if gt_idx is None else 0),
                "verdict": verdict.label,
                "suspicion": f"{verdict.suspicion:.3f}",
                "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                "in_lexicon": 1 if in_lex else 0,
                "one_flip": len(verdict.signals.one_flip_words),
                "reason": reason,
            })

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
    print(f"\nColumns:")
    print(f"  page          — page name")
    print(f"  ocr_idx       — word position in OCR")
    print(f"  gt_idx        — aligned GT position (empty = insertion)")
    print(f"  ocr_word      — what Tesseract output")
    print(f"  gt_word       — what it should be")
    print(f"  is_error      — 0=correct, 1=error, 2=insertion (no GT match)")
    print(f"  verdict       — pass / gray / flag")
    print(f"  suspicion     — detector score (0-1)")
    print(f"  tesseract_conf — Tesseract confidence (0-100)")
    print(f"  in_lexicon    — 1 if OCR word is in dictionary")
    print(f"  one_flip      — number of one-flip dictionary hits")
    print(f"  reason        — why the detector gave this verdict")

    # Quick summary
    if stats["errors"] > 0:
        print(f"\nErrors: {stats['errors']:,} / {stats['total']:,} ({100*stats['errors']/stats['total']:.1f}%)")
        print(f"Caught: {stats['caught']:,} ({100*stats['caught']/stats['errors']:.1f}%)")


if __name__ == "__main__":
    main()