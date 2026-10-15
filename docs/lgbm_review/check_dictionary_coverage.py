"""Dictionary coverage of ground-truth words for an evaluation set.

Answers one question: what share of the CORRECT words in a page set are found in the
dictionary? Ground truth is (almost entirely) correct text, so no word alignment with the
OCR output is needed.

Run this on any evaluation folder laid out like Evaluation/:
    GroundTruth/page_NNNN.gt.txt   hand-transcribed truth
    OCRed/page_NNNN.ocr.txt        OCR output (optional, reported for reference)

Usage (on the machine that has the dictionary):
    python docs/lgbm_review/check_dictionary_coverage.py \
        --dict /path/to/main_i7.txt \
        --set  4_ocr_correction/error_detection/Evaluation

Compare the result with the in-corpus evaluation set (about 94% of correct words in the
dictionary). A clearly lower number on fresh pages means the dictionary feature is
optimistic on new text (see docs/lgbm_step1_changes.md, section 5.3).

Dictionary format: one entry per line, "word" or "word<whitespace>frequency". Same parser
as DetectionEngine._load_vocabulary, so the result matches what the detector sees.
"""
import argparse
import glob
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from error_detection.normalizer import AmharicNormalizer  # noqa: E402
from error_detection.features import clean_surface  # noqa: E402


def load_vocab(path, norm):
    vocab = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            freq = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 1
            key = norm(parts[0])
            vocab[key] = max(vocab.get(key, 0), freq)
    return vocab


def tokens(folder, pattern):
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            yield os.path.basename(path), f.read().split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--set", required=True, help="folder with GroundTruth/ and OCRed/")
    args = ap.parse_args()

    normalizer = AmharicNormalizer()
    norm = normalizer.normalize_word_strict
    vocab = load_vocab(args.dict, norm)
    print(f"dictionary keys: {len(vocab):,}")

    gt_total = gt_hit = 0
    per_page = Counter()
    for name, words in tokens(os.path.join(args.set, "GroundTruth"), "*.gt.txt"):
        for w in words:
            key = clean_surface(w)
            if not key:
                continue
            gt_total += 1
            if norm(key) in vocab:
                gt_hit += 1
            else:
                per_page[name] += 1
    if gt_total == 0:
        print("no ground-truth words found")
        return
    print(f"ground-truth words: {gt_total:,}")
    print(f"found in dictionary: {gt_hit:,} ({100 * gt_hit / gt_total:.1f}%)")
    print("pages with the most dictionary misses:")
    for name, n in per_page.most_common(5):
        print(f"  {name}: {n}")

    ocr_total = ocr_hit = 0
    for _, words in tokens(os.path.join(args.set, "OCRed"), "*.ocr.txt"):
        for w in words:
            key = clean_surface(w)
            if not key:
                continue
            ocr_total += 1
            if norm(key) in vocab:
                ocr_hit += 1
    if ocr_total:
        print(f"OCR tokens found in dictionary (reference, includes errors): "
              f"{ocr_hit:,}/{ocr_total:,} ({100 * ocr_hit / ocr_total:.1f}%)")


if __name__ == "__main__":
    main()
