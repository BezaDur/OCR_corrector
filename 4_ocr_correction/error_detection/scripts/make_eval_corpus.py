#!/usr/bin/env python3
"""make_eval_corpus.py — turn a folder of OCR/ground-truth text into a scorable corpus.

The detector is evaluated on a tab-separated corpus with one row per OCR token
(``page``, ``ocr_word``, ``gt_word``, ``is_error``, ``match_type``,
``tesseract_conf``, ``doc_freq``).  ``Model/harmonized_corpus_jiwer.tsv`` is such
a file, but there is no script in the repository that builds one, so a new test
set could not be scored without inventing the format by hand.  This script is
that missing step: point it at a folder laid out like
``Evaluation/`` and it writes a compatible TSV.

The labelling recipe is not a guess.  It reproduces the shipped corpus exactly:

    python 4_ocr_correction/error_detection/scripts/make_eval_corpus.py --eval-dir \\
        Evaluation --out /tmp/rebuilt.tsv
    # 62,479 rows, 4,361 errors, identical to harmonized_corpus_jiwer.tsv

Details of the recipe, all inherited from the shipped corpus:

* Tokens are whitespace-separated; punctuation and digits are stripped and
  tokens that become empty are dropped (``PUNCT_RE``, same rule the trainer and
  the evaluator use).
* Ground truth and OCR text are aligned with ``align_tokens`` (word-level DP).
* One row per OCR token.  A ground-truth word with no OCR counterpart is a
  *deletion*; it has no OCR word to judge, so no row is written for it and the
  count is reported.  This is the same blind spot the shipped corpus has.
* ``doc_freq`` is the number of times the OCR word occurs **on its own page**
  (verified against the shipped corpus: እና occurs 13 times on page_0001 and has
  doc_freq 13 there).
* ``tesseract_conf`` is read from ``OCRed/<page>.conf.tsv`` when present.  The
  shipped corpus has the constant 75.0 in that column and the shipped model
  does not use it (it is "conf-free"), so a missing file is not a problem.

Before the corpus is written, the text is checked against the corpus the model
was trained on, because a test set built from training text measures nothing:

* how many of the new pages share no word 5-gram with the training pages,
* how many of the new *words* never occur in the training text.

That second number is the one the existing corpus cannot report: every word in
it is also a training word, so the current evaluation says nothing about
unseen vocabulary.

Usage
-----
    # build and check
    python 4_ocr_correction/error_detection/scripts/make_eval_corpus.py \\
        --eval-dir fresh_eval --out /tmp/fresh.tsv

    # then score it with the production path
    python 4_ocr_correction/error_detection/scripts/evaluate_out_of_sample.py \\
        --corpus /tmp/fresh.tsv --eval-all --report /tmp/fresh.json

The expected layout is the one ``docs/new_test_set_plan.md`` prescribes::

    fresh_eval/
      GroundTruth/page_0001.gt.txt
      OCRed/page_0001.ocr.txt
      OCRed/page_0001.conf.tsv     # optional
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # 4_ocr_correction/error_detection/
_STAGE = os.path.dirname(_PKG)                   # 4_ocr_correction/
_ROOT = os.path.dirname(_STAGE)                  # repo root
for _p in (_ROOT, _STAGE, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from error_detection.alignment import align_tokens  # noqa: E402

CORPUS = os.path.join(_PKG, "Model", "harmonized_corpus_jiwer.tsv")

PUNCT_RE = re.compile(r"[^\w\s]|\d")
COLUMNS = ["page", "ocr_word", "gt_word", "is_error", "match_type",
           "tesseract_conf", "doc_freq"]
CONF_PLACEHOLDER = 75.0   # the constant the shipped corpus carries


def tokens(text: str) -> list[str]:
    """Whitespace split, punctuation/digits stripped, empties dropped."""
    return [c for c in (PUNCT_RE.sub("", w).strip() for w in text.split()) if c]


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def read_conf(path: str) -> dict[str, float]:
    """``<page>.conf.tsv`` -> {word: confidence}. Missing/short files give {}."""
    out: dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            wi = header.index("word")
        except ValueError:
            wi = 0
        ci = 1 if len(header) > 1 else None
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[wi]:
                continue
            key = PUNCT_RE.sub("", parts[wi]).strip()
            if not key:
                continue
            try:
                out[key] = float(parts[ci]) if ci is not None and len(parts) > ci else CONF_PLACEHOLDER
            except ValueError:
                out[key] = CONF_PLACEHOLDER
    return out


def align_page(gt_text: str, ocr_text: str) -> tuple[list[tuple], dict]:
    """Return rows for one page plus a small stats dict.

    Rows are ``(ocr_word, gt_word|None, is_error, match_type)`` in OCR order.
    """
    gt, ocr = tokens(gt_text), tokens(ocr_text)
    steps, distance, insertions, deletions, _matches = align_tokens(gt, ocr)
    rows, dropped = [], 0
    for ref_idx, hyp_idx in steps:
        if hyp_idx is None:          # ground-truth word the OCR missed entirely
            dropped += 1
            continue
        word = ocr[hyp_idx]
        if ref_idx is None:
            rows.append((word, None, 1, "INSERTION"))
        elif gt[ref_idx] == word:
            rows.append((word, word, 0, "MATCH"))
        else:
            rows.append((word, gt[ref_idx], 1, "SUBSTITUTION"))
    stats = {"gt_tokens": len(gt), "ocr_tokens": len(ocr), "deletions": dropped,
             "edit_distance": distance, "insertions": insertions}
    return rows, stats


def build(eval_dir: str, quiet: bool = False) -> tuple[pd.DataFrame, dict]:
    gt_dir = os.path.join(eval_dir, "GroundTruth")
    ocr_dir = os.path.join(eval_dir, "OCRed")
    if not os.path.isdir(gt_dir):
        raise SystemExit(f"no GroundTruth/ under {eval_dir}")
    if not os.path.isdir(ocr_dir):
        raise SystemExit(f"no OCRed/ under {eval_dir}")

    gt_files = {os.path.basename(p).replace(".gt.txt", ""): p
                for p in glob.glob(os.path.join(gt_dir, "*.gt.txt"))}
    ocr_files = {os.path.basename(p).replace(".ocr.txt", ""): p
                 for p in glob.glob(os.path.join(ocr_dir, "*.ocr.txt"))}
    if not gt_files:
        raise SystemExit(f"no *.gt.txt files in {gt_dir}")

    paired = sorted(set(gt_files) & set(ocr_files))
    unmatched = sorted(set(gt_files) ^ set(ocr_files))
    if not paired:
        raise SystemExit("no page has both a .gt.txt and an .ocr.txt")

    records, per_page, totals = [], [], Counter()
    for stem in paired:
        page = stem if stem.startswith("page_") else f"page_{len(per_page) + 1:04d}"
        rows, stats = align_page(read_text(gt_files[stem]), read_text(ocr_files[stem]))
        conf = read_conf(os.path.join(ocr_dir, f"{stem}.conf.tsv"))
        counts = Counter(r[0] for r in rows)      # doc_freq = within-page frequency
        for word, gt_word, is_error, match_type in rows:
            records.append({
                "page": page,
                "ocr_word": word,
                "gt_word": "" if gt_word is None else gt_word,
                "is_error": is_error,
                "match_type": match_type,
                "tesseract_conf": conf.get(word, CONF_PLACEHOLDER),
                "doc_freq": counts[word],
            })
        per_page.append({"page": page, "source": stem, **stats, "rows": len(rows)})
        totals["deletions"] += stats["deletions"]
        totals["errors"] += sum(1 for r in rows if r[2])

    df = pd.DataFrame(records, columns=COLUMNS)
    info = {"eval_dir": os.path.abspath(eval_dir), "pages": len(paired),
            "rows": len(df), "errors": int(df.is_error.sum()),
            "unmatched_pages": unmatched, "deletions_not_scored": totals["deletions"],
            "per_page": per_page}
    if not quiet:
        print(f"pages {len(paired)}  rows {len(df):,}  errors {info['errors']:,} "
              f"({100 * info['errors'] / max(1, len(df)):.2f} %)")
        print(f"ground-truth words with no OCR word (not scored): {totals['deletions']:,}")
        if unmatched:
            print(f"warning: {len(unmatched)} file(s) without a pair: {unmatched[:5]}")
    return df, info


def novelty(df: pd.DataFrame, reference: str = CORPUS, n: int = 5,
            dev_pages: int = 100, quiet: bool = False) -> dict:
    """How new is this text relative to the corpus the model was trained on?"""
    if not os.path.exists(reference):
        raise SystemExit(f"reference corpus not found: {reference}")
    ref = pd.read_csv(reference, sep="\t")
    ref["page_no"] = ref["page"].str.replace("page_", "", regex=False).astype(int)
    train = ref[ref.page_no > dev_pages]          # the pages the model was fit on

    def clean_text(values) -> str:
        return " ".join(w for w in (PUNCT_RE.sub("", t).strip()
                                    for t in values if isinstance(t, str)) if w)

    train_grams: set[str] = set()
    for _page, words in train.groupby("page")["gt_word"]:
        text = clean_text(words.tolist()).split()
        train_grams |= {" ".join(text[i:i + n]) for i in range(len(text) - n + 1)}
    train_words = set(train.gt_word.dropna().astype(str))

    shares = {}
    for page, words in df.groupby("page")["gt_word"]:
        text = clean_text(words.tolist()).split()
        grams = {" ".join(text[i:i + n]) for i in range(len(text) - n + 1)}
        shares[page] = (len(grams & train_grams) / len(grams)) if grams else 0.0

    tiers = Counter()
    for value in shares.values():
        tiers["disjoint" if value == 0.0 else
              "near_duplicate" if value >= 0.90 else "partial"] += 1

    words = df.ocr_word.astype(str)
    unseen = ~words.isin(train_words)
    errors = df.is_error == 1
    out = {
        "reference_rows": int(len(train)),
        "pages_by_text_tier": dict(tiers),
        "page_shares": {p: round(v, 4) for p, v in sorted(shares.items())},
        "ocr_tokens": int(len(df)),
        "tokens_unseen_in_training": int(unseen.sum()),
        "share_tokens_unseen": round(float(unseen.mean()), 4),
        "errors": int(errors.sum()),
        "errors_on_unseen_tokens": int((errors & unseen).sum()),
        "share_errors_on_unseen": round(float((errors & unseen).sum() / max(1, errors.sum())), 4),
    }
    if not quiet:
        t = out["pages_by_text_tier"]
        print(f"text novelty vs training pages: {t.get('disjoint', 0)}/{len(shares)} pages "
              f"share no {n}-gram, {t.get('partial', 0)} partial, "
              f"{t.get('near_duplicate', 0)} near-duplicate")
        print(f"vocabulary: {out['share_tokens_unseen']:.1%} of OCR tokens never occur in "
              f"training text; {out['share_errors_on_unseen']:.1%} of errors are on them")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True,
                    help="folder with GroundTruth/*.gt.txt and OCRed/*.ocr.txt")
    ap.add_argument("--out", default=None, help="write the corpus TSV here")
    ap.add_argument("--report", default=None, help="write build + novelty JSON here")
    ap.add_argument("--reference", default=CORPUS,
                    help="corpus to measure novelty against (default: the shipped one)")
    ap.add_argument("--reference-dev-pages", type=int, default=100,
                    help="pages of the reference corpus to treat as training (default 100)")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--no-check", action="store_true", help="skip the novelty check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df, info = build(args.eval_dir, quiet=args.quiet)
    if not args.no_check:
        info["novelty"] = novelty(df, args.reference, args.ngram,
                                  args.reference_dev_pages, args.quiet)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False, na_rep="")
        if not args.quiet:
            print(f"wrote {args.out}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"wrote {args.report}")
    if not args.out and not args.report and not args.quiet:
        print("(nothing written; pass --out and/or --report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
