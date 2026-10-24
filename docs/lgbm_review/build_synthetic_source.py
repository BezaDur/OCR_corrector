"""Select source text for the synthetic test set from a new text file.

Steps:
  1. Read the new source (one post or sentence per line).
  2. Keep lines that contain Ethiopic text and have 3-60 words; drop exact duplicates.
  3. Remove any line that shares a word 5-gram (punctuation stripped) with the existing
     corpora: the harmonized training ground truth, amh_final_training_text.txt,
     mined_rare_sentences.txt, amh.training_text, and the evaluation ground truth.
     This is strict: one shared 5-gram removes the line.
  4. Cut the remaining lines into blocks of consecutive lines, sample blocks at random
     (fixed seed), and write one block per page, separated by '---'.

Output:
  <out>/source_pages.txt    input for make_synthetic_set.py
  <out>/selection.csv       page, first and last source line number in the new text

Usage:
  python docs/lgbm_review/build_synthetic_source.py --source telegram_c4.txt \
      --out synthetic_eval --pages 100 --lines-per-page 35 --seed 7
"""
import argparse
import csv
import glob
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUNCT = re.compile(r"[^\w\s]|\d")
ETHIOPIC = re.compile(r"[\u1200-\u137F]")

# Paths follow the 2026-10 repository reorganisation (files moved into their
# stage folders). They used to point at the repository root, where the corpora
# no longer live; the `os.path.exists` guards below then silently skipped them,
# which would have weakened the "no shared 5-grams" guarantee without saying so.
CORPUS_FILES = [
    os.path.join(ROOT, "1_data_preparation", "data", "amh_final_training_text.txt"),
    os.path.join(ROOT, "1_data_preparation", "data", "mined_rare_sentences.txt"),
    os.path.join(ROOT, "1_data_preparation", "data", "amh.training_text"),
]
TRAIN_TSV = os.path.join(ROOT, "OCR_error_detection", "Model", "harmonized_corpus_jiwer.tsv")
EVAL_GT = os.path.join(ROOT, "OCR_error_detection", "Evaluation", "GroundTruth")


def clean(text):
    return " ".join(w for w in (PUNCT.sub("", t).strip() for t in text.split()) if w)


def grams(text, n=5):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def corpus_grams():
    """All word 5-grams (cleaned) of the existing corpora, as hashes to save memory."""
    seen = set()
    for path in CORPUS_FILES:
        if not os.path.exists(path):
            raise SystemExit(f"corpus file missing, cannot guarantee decontamination: {path}")
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                seen.update(hash(g) for g in grams(clean(line)))
    if not os.path.exists(TRAIN_TSV):
        raise SystemExit(f"training corpus missing, cannot guarantee decontamination: {TRAIN_TSV}")
    pages = {}
    with open(TRAIN_TSV, encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # noqa: F841
        for row in f:
            parts = row.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                pages.setdefault(parts[0], []).append(parts[2])
    for words in pages.values():
        seen.update(hash(g) for g in grams(clean(" ".join(words))))
    for path in glob.glob(os.path.join(EVAL_GT, "*.txt")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            seen.update(hash(g) for g in grams(clean(f.read())))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument("--lines-per-page", type=int, default=35)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(args.source, encoding="utf-8", errors="ignore") as f:
        raw = [l.rstrip("\n") for l in f]

    kept, seen_lines = [], set()
    for no, line in enumerate(raw, 1):
        if not line.strip() or not ETHIOPIC.search(line):
            continue
        n = len(line.split())
        if not (3 <= n <= 60):
            continue
        key = clean(line)
        if not key or key in seen_lines:
            continue
        seen_lines.add(key)
        kept.append((no, line.strip(), key))
    print(f"lines {len(raw):,}; kept after Ethiopic/length/duplicate filter: {len(kept):,}")

    existing = corpus_grams()
    print(f"existing 5-gram hashes: {len(existing):,}")
    clean_lines = [(no, line) for no, line, key in kept
                   if not any(hash(g) in existing for g in grams(key))]
    removed = len(kept) - len(clean_lines)
    print(f"removed for sharing a 5-gram with existing text: {removed:,}; left: {len(clean_lines):,}")

    k = args.lines_per_page
    blocks = [clean_lines[i:i + k] for i in range(0, len(clean_lines) - k + 1, k)]
    if len(blocks) < args.pages:
        raise SystemExit(f"only {len(blocks)} blocks of {k} lines; need {args.pages}")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(blocks)), args.pages))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "source_pages.txt"), "w", encoding="utf-8") as f:
        for j, b in enumerate(chosen):
            if j:
                f.write("---\n")
            for _, line in blocks[b]:
                f.write(line + "\n")
    with open(os.path.join(args.out, "selection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "first_source_line", "last_source_line"])
        for j, b in enumerate(chosen, 1):
            first, last = blocks[b][0][0], blocks[b][-1][0]
            w.writerow([f"page_{j:04d}", first, last])
    print(f"wrote {args.pages} pages ({args.pages * k} lines) to {args.out}")


if __name__ == "__main__":
    main()
