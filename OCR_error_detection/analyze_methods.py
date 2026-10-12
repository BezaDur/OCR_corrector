#!/usr/bin/env python3
"""
Analyze detection results with proper DP alignment.

Uses edit-distance-based word alignment instead of SequenceMatcher.
This correctly handles word splits, merges, insertions, and deletions.

Outputs: Evaluation/analysis.tsv

Usage:
    python analyze_methods.py
"""
import csv
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from alignment import align_tokens
from OCR_error_detection import DetectionEngine

GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONF_DIR = OCR_DIR
CONFUSION = os.path.join(_HERE, "confusion_matrix.json")
OUTPUT_TSV = os.path.join(_HERE, "Evaluation", "analysis.tsv")

# Matches any punctuation (Ethiopic + ASCII + symbols) and digits
PUNCT_RE = re.compile(r'[^\w\s]|\d')


def clean_token(token):
    """Remove punctuation from a token for comparison."""
    return PUNCT_RE.sub('', token).strip()


def is_word(token):
    """True if token is an actual word (not punctuation or number)."""
    return bool(clean_token(token))


def char_edit_distance(a, b):
    """Character-level Levenshtein distance."""
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


def load_pages():
    """Load all GT/OCR page pairs and confidence data."""
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

        # Load confidence keyed by raw OCR word index
        conf_map = None
        conf_path = os.path.join(CONF_DIR, base + ".conf.tsv")
        if os.path.exists(conf_path):
            conf_map = {}
            idx = 0
            with open(conf_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("word"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            conf_map[idx] = float(parts[1])
                        except ValueError:
                            pass
                        idx += 1

        pages.append((base, gt_text, ocr_text, conf_map))
    return pages


def main():
    pages = load_pages()
    if not pages:
        print("No pages found!")
        return

    # Build lexicon from cleaned GT words
    vocab = Counter()
    for _, gt_text, _, _ in pages:
        for tok in gt_text.split():
            cleaned = clean_token(tok)
            if cleaned:
                vocab[cleaned] += 1

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

    # Collect rows and stats
    rows = []
    stats = {
        "total_aligned": 0, "errors": 0, "caught": 0,
        "gt_chars": 0, "char_errors": 0,
        "gt_words": 0, "ocr_words": 0,
        "substitutions": 0, "insertions": 0, "deletions": 0,
        "pages": 0,
        "correct_confs": [], "error_confs": [],
    }

    for page_name, gt_text, ocr_text, conf_map in pages:
        # Split into word tokens, filter punctuation-only
        gt_raw = gt_text.split()
        ocr_raw = ocr_text.split()

        gt_clean = [clean_token(t) for t in gt_raw]
        ocr_clean = [clean_token(t) for t in ocr_raw]

        # Filter to word-only tokens (keep indices back to original)
        gt_words = [(i, gt_raw[i], gt_clean[i]) for i in range(len(gt_raw)) if gt_clean[i]]
        ocr_words = [(j, ocr_raw[j], ocr_clean[j]) for j in range(len(ocr_raw)) if ocr_clean[j]]

        if not gt_words or not ocr_words:
            continue

        stats["pages"] += 1
        stats["gt_words"] += len(gt_words)
        stats["ocr_words"] += len(ocr_words)

        # Use DP alignment on cleaned word sequences
        gt_seq = [w[2] for w in gt_words]
        ocr_seq = [w[2] for w in ocr_words]
        steps, distance, n_insertions, n_deletions, match_set = align_tokens(gt_seq, ocr_seq)

        stats["insertions"] += n_insertions
        stats["deletions"] += n_deletions

        has_conf = conf_map is not None

        for ref_idx, hyp_idx in steps:
            if ref_idx is not None and hyp_idx is not None:
                # Aligned pair (match or substitution)
                gt_orig_idx, gt_orig, gt_c = gt_words[ref_idx]
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                is_match = (gt_c == ocr_c)

                # Character errors for this pair
                if not is_match:
                    stats["char_errors"] += char_edit_distance(gt_c, ocr_c)
                    stats["substitutions"] += 1
                stats["gt_chars"] += len(gt_c)

                # Confidence
                ocr_conf = conf_map.get(ocr_orig_idx) if has_conf else None

                # Context: previous 2 OCR words
                ctx = []
                for j in range(max(0, hyp_idx - 2), hyp_idx):
                    ctx.append(ocr_words[j][1])  # original OCR word

                verdict = detector.verdict(ocr_orig, ctx, ocr_conf=ocr_conf)
                in_lex = detector._norm(ocr_c) in detector.vocab
                reason = build_reason(verdict.signals, in_lex, verdict.label)

                if not is_match:
                    stats["errors"] += 1
                    if verdict.label in ("flag", "gray"):
                        stats["caught"] += 1
                    if ocr_conf is not None:
                        stats["error_confs"].append(ocr_conf)
                    is_error_val = 1
                else:
                    if ocr_conf is not None:
                        stats["correct_confs"].append(ocr_conf)
                    is_error_val = 0

                stats["total_aligned"] += 1

                rows.append({
                    "page": page_name,
                    "ocr_word": ocr_c,
                    "gt_word": gt_c,
                    "is_error": is_error_val,
                    "detector_verdict": verdict.label,
                    "detector_score": f"{verdict.suspicion:.3f}",
                    "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                    "in_lexicon": 1 if in_lex else 0,
                    "one_flip": len(verdict.signals.one_flip_words),
                    "reason": reason,
                })

            elif ref_idx is not None:
                # Deletion: GT word has no OCR match
                gt_orig_idx, gt_orig, gt_c = gt_words[ref_idx]
                stats["gt_chars"] += len(gt_c)

                rows.append({
                    "page": page_name,
                    "ocr_word": "",
                    "gt_word": gt_c,
                    "is_error": 3,  # deletion
                    "detector_verdict": "",
                    "detector_score": "",
                    "tesseract_conf": "",
                    "in_lexicon": "",
                    "one_flip": "",
                    "reason": "DELETION",
                })

            elif hyp_idx is not None:
                # Insertion: OCR word has no GT match
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                ocr_conf = conf_map.get(ocr_orig_idx) if has_conf else None

                rows.append({
                    "page": page_name,
                    "ocr_word": ocr_c,
                    "gt_word": "",
                    "is_error": 2,  # insertion
                    "detector_verdict": "",
                    "detector_score": "",
                    "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                    "in_lexicon": "",
                    "one_flip": "",
                    "reason": "INSERTION",
                })

    # Write TSV
    fieldnames = [
        "page", "ocr_word", "gt_word", "is_error",
        "detector_verdict", "detector_score", "tesseract_conf",
        "in_lexicon", "one_flip", "reason",
    ]
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {OUTPUT_TSV}")

    # ── Tesseract accuracy ──────────────────────────────────────────
    total_errors = stats["substitutions"] + stats["insertions"] + stats["deletions"]
    wer = 100.0 * total_errors / max(stats["gt_words"], 1)
    cer = 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1)

    print(f"\n{'='*60}")
    print(f"Tesseract Accuracy (punctuation stripped)")
    print(f"{'='*60}")
    print(f"Pages:                    {stats['pages']}")
    print(f"GT words:                 {stats['gt_words']:,}")
    print(f"OCR words:                {stats['ocr_words']:,}")
    print(f"Aligned pairs:            {stats['total_aligned']:,}")
    print(f"Substitutions:            {stats['substitutions']:,}")
    print(f"Insertions (OCR-only):    {stats['insertions']:,}")
    print(f"Deletions (GT-only):      {stats['deletions']:,}")
    print(f"WER:                      {wer:.2f}%")
    print(f"CER:                      {cer:.2f}%")
    print(f"Word accuracy:            {100 - wer:.2f}%")
    print(f"Char accuracy:            {100 - cer:.2f}%")

    if stats["correct_confs"]:
        avg_c = sum(stats["correct_confs"]) / len(stats["correct_confs"])
        print(f"Avg confidence (correct): {avg_c:.1f}%")
    if stats["error_confs"]:
        avg_e = sum(stats["error_confs"]) / len(stats["error_confs"])
        print(f"Avg confidence (errors):  {avg_e:.1f}%")

    # ── Detection accuracy ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Detection Performance")
    print(f"{'='*60}")
    print(f"Error words (subst):      {stats['errors']:,}")
    print(f"Caught (flag+gray):       {stats['caught']:,} ({100*stats['caught']/max(stats['errors'],1):.1f}%)")
    print(f"Missed (passed):          {stats['errors'] - stats['caught']:,}")


if __name__ == "__main__":
    main()