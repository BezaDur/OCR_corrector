#!/usr/bin/env python3
"""
Analyze detection results. Writes a TSV and prints Tesseract accuracy.

Punctuation and whitespace are stripped before comparison.
Only actual words are compared.

Outputs: OCR_error_detection/Evaluation/analysis.tsv

Usage:
    python OCR_error_detection/analyze_methods.py
"""
import csv
import difflib
import os
import re
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

# Matches any punctuation (Ethiopic + ASCII + symbols)
PUNCT_RE = re.compile(r'[^\w\s]|\d')


def is_word(token):
    """True if token is an actual word (not punctuation or number)."""
    if not token:
        return False
    # Remove all punctuation, check if anything alphabetic remains
    cleaned = PUNCT_RE.sub('', token).strip()
    return len(cleaned) > 0


def clean_token(token):
    """Remove punctuation from a token for comparison."""
    return PUNCT_RE.sub('', token).strip()


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

        # Load confidence — keyed by OCR word position (including punct)
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


def extract_words(text):
    """Split text into tokens, return (original_tokens, word_tokens_with_indices).

    word_tokens: list of (original_token, cleaned_token, original_index)
    """
    tokens = text.split()
    words = []
    for i, tok in enumerate(tokens):
        cleaned = clean_token(tok)
        if cleaned:
            words.append((tok, cleaned, i))
    return tokens, words


def align_words(gt_words, ocr_words):
    """Align cleaned word tokens between GT and OCR.

    gt_words, ocr_words: list of (original_token, cleaned_token, original_index)
    Returns: list of (ocr_word_entry, gt_word_entry, is_error)
    where entries are (original_token, cleaned_token, original_index) or None.
    """
    gt_clean = [w[1] for w in gt_words]
    ocr_clean = [w[1] for w in ocr_words]

    sm = difflib.SequenceMatcher(None, gt_clean, ocr_clean, autojunk=False)
    results = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                results.append((ocr_words[j1 + k], gt_words[i1 + k], False))
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                results.append((ocr_words[j1 + k], gt_words[i1 + k], True))
            for k in range(n, j2 - j1):
                results.append((ocr_words[j1 + k], None, True))
        elif tag == "insert":
            for k in range(j2 - j1):
                results.append((ocr_words[j1 + k], None, True))
    return results


def edit_distance(a, b):
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

    # Collect rows
    rows = []
    stats = {
        "total": 0, "errors": 0, "caught": 0,
        "insertions": 0,
        "gt_chars": 0, "char_errors": 0,
        "gt_words": 0,
        "word_substitutions": 0, "word_insertions": 0, "word_deletions": 0,
        "pages": 0,
        # Confidence tracking
        "correct_confs": [], "error_confs": [],
    }

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_tokens, gt_words = extract_words(gt_text)
        ocr_tokens, ocr_words = extract_words(ocr_text)
        if not gt_words or not ocr_words:
            continue

        alignment = align_words(gt_words, ocr_words)
        has_conf = conf_map is not None
        stats["pages"] += 1

        for ocr_entry, gt_entry, is_error in alignment:
            ocr_orig, ocr_clean, ocr_orig_idx = ocr_entry
            if gt_entry is not None:
                gt_orig, gt_clean, gt_orig_idx = gt_entry
            else:
                gt_orig, gt_clean, gt_orig_idx = "", "", None

            # Get Tesseract confidence for this OCR word
            ocr_conf = conf_map.get(ocr_orig_idx) if has_conf else None

            # Context: previous 2 OCR words (cleaned, words only)
            ctx_indices = [i for i in range(ocr_orig_idx - 2, ocr_orig_idx) if i >= 0]
            ctx = []
            for ci in ctx_indices:
                for orig, clean, idx in ocr_words:
                    if idx == ci:
                        ctx.append(orig)
                        break

            verdict = detector.verdict(ocr_orig, ctx, ocr_conf=ocr_conf)
            in_lex = detector._norm(ocr_clean) in detector.vocab
            reason = build_reason(verdict.signals, in_lex, verdict.label)

            if gt_entry is None:
                stats["insertions"] += 1
                stats["word_insertions"] += 1
                is_error_val = 2
            elif is_error:
                stats["errors"] += 1
                stats["word_substitutions"] += 1
                stats["char_errors"] += edit_distance(gt_clean, ocr_clean)
                stats["gt_chars"] += len(gt_clean)
                stats["gt_words"] += 1
                if verdict.label in ("flag", "gray"):
                    stats["caught"] += 1
                is_error_val = 1
                if ocr_conf is not None:
                    stats["error_confs"].append(ocr_conf)
            else:
                stats["gt_chars"] += len(gt_clean)
                stats["gt_words"] += 1
                is_error_val = 0
                if ocr_conf is not None:
                    stats["correct_confs"].append(ocr_conf)

            rows.append({
                "page": page_name,
                "ocr_word": ocr_clean,
                "gt_word": gt_clean,
                "is_error": is_error_val,
                "detector_verdict": verdict.label,
                "detector_score": f"{verdict.suspicion:.3f}",
                "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                "in_lexicon": 1 if in_lex else 0,
                "one_flip": len(verdict.signals.one_flip_words),
                "reason": reason,
            })

        # Count deletions
        gt_clean_all = [w[1] for w in gt_words]
        ocr_clean_all = [w[1] for w in ocr_words]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, gt_clean_all, ocr_clean_all, autojunk=False).get_opcodes():
            if tag == "delete":
                stats["word_deletions"] += (i2 - i1)

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
    wer = 100.0 * (stats["word_substitutions"] + stats["word_insertions"] + stats["word_deletions"]) / max(stats["gt_words"], 1)
    cer = 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1)

    print(f"\n{'='*60}")
    print(f"Tesseract Accuracy (punctuation stripped)")
    print(f"{'='*60}")
    print(f"Pages:                    {stats['pages']}")
    print(f"GT words:                 {stats['gt_words']:,}")
    print(f"Substitutions:            {stats['word_substitutions']:,}")
    print(f"Insertions:               {stats['word_insertions']:,}")
    print(f"Deletions:                {stats['word_deletions']:,}")
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
    print(f"Error words:              {stats['errors']:,}")
    print(f"Caught (flag+gray):       {stats['caught']:,} ({100*stats['caught']/max(stats['errors'],1):.1f}%)")
    print(f"Missed (passed):          {stats['errors'] - stats['caught']:,}")


if __name__ == "__main__":
    main()