"""Tesseract-specific input. The only module in detection/ that knows the TSV format.

Tesseract TSV: level 5 rows are words (text, conf 0-100). Other levels are layout
containers. Any row with conf -1 is not a word and is skipped.
Labels come from the ground-truth alignment in tsv_gt.py.
"""
import csv
import os
from typing import Dict, List, Optional

from tsv_gt import annotate_rows


def read_words(tsv_path: str) -> List[Dict]:
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    out = []
    for r in rows:
        if r.get("level") != "5" or float(r["conf"]) < 0 or not r["text"].strip():
            continue
        out.append(dict(text=r["text"].strip(), conf=float(r["conf"])))
    return out


def to_examples(tsv_path: str, gt_text: Optional[str] = None, doc: Optional[str] = None) -> List[Dict]:
    """Examples for one page. With gt_text, each word gets a clean/error/ambiguous label."""
    doc = doc or os.path.splitext(os.path.basename(tsv_path))[0]
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    labelled = annotate_rows(rows, gt_text) if gt_text is not None else None
    out = []
    for i, r in enumerate(rows):
        if r.get("level") != "5" or float(r["conf"]) < 0 or not r["text"].strip():
            continue
        status = (labelled[i]["gt_status"] or None) if labelled else None
        out.append(dict(doc=doc, index=len(out), text=r["text"].strip(), label=status,
                        features={"ocr_conf": float(r["conf"]),
                                  "word_length": float(len(r["text"].strip()))}))
    return out
