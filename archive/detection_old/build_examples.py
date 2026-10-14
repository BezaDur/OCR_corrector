"""Build an examples file from one of the supported sources.

  verdicts  corrector verdict folder (04_evaluate_full_corrector.py --verdict-dir)
  tsv       folder of Tesseract TSV files, labelled from a ground-truth folder (optional)
  words     generic word JSONL (see adapters/plain_words.py)

Usage:
  python3 -m detection.build_examples verdicts VERDICT_DIR --out examples.jsonl
  python3 -m detection.build_examples tsv TSV_DIR [--gt-dir GT_DIR] --out examples.jsonl
  python3 -m detection.build_examples words WORDS.jsonl --out examples.jsonl
"""
import argparse
import glob
import os

from detection.adapters import plain_words, tesseract_tsv, verdicts
from detection.examples import write_examples


def from_tsv_dir(tsv_dir: str, gt_dir=None):
    out = []
    for path in sorted(glob.glob(os.path.join(tsv_dir, "*.tsv"))):
        base = os.path.basename(path)[:-len(".tsv")]
        gt_text = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, base + ".gt.txt")
            if os.path.exists(gt_path):
                with open(gt_path, encoding="utf-8") as fh:
                    gt_text = fh.read()
        examples = tesseract_tsv.to_examples(path, gt_text=gt_text, doc=base)
        out.extend(examples)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["verdicts", "tsv", "words"])
    ap.add_argument("path")
    ap.add_argument("--gt-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.source == "verdicts":
        ex = verdicts.to_examples(a.path)
    elif a.source == "tsv":
        ex = from_tsv_dir(a.path, a.gt_dir)
    else:
        ex = plain_words.to_examples(plain_words.read_words(a.path))
    n = write_examples(a.out, ex)
    labelled = sum(1 for e in ex if e.get("label") in ("clean", "error"))
    print(f"wrote {n} examples ({labelled} labelled clean/error) -> {a.out}")


if __name__ == "__main__":
    main()
