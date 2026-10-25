"""Add ground truth to a Tesseract TSV: one row per word, three extra columns.

  core       the word as compared (normalised, punctuation and diacritics removed)
  gt_word    the ground-truth word it matched in the alignment ("" if none)
  gt_status  clean | error | ambiguous, as in stage1_labels

gt_word comes from one optimal alignment (the traceback): the gold word a token
was matched or substituted for. It is blank when the token was an insertion.
gt_status comes from all optimal alignments, so a token can be ambiguous even
when gt_word is set.
Non-word rows and words with no core keep blank values.
"""
import csv
import os, sys
_root = os.path.dirname(os.path.dirname(__file__))
_corr = os.path.join(_root, '4_ocr_correction')
for p in [_root, _corr]:
    if p not in sys.path:
        sys.path.insert(0, p)
from typing import Dict, List, Optional

from ocr_metrics import align_tokens
from amharic_ocr_corrector import EthiopicPunctuationManager as P

EXTRA = ["core", "gt_word", "gt_status"]


# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 3_training_evaluation/tsv_gt.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "tsv_in": os.path.join(os.environ.get("TESSERACT_WORKSPACE", "/media/des/Tess/Tesseract2"),
            "synthetic_eval", "tsv", "page_0001.tsv"),  # a Tesseract TSV for one page
    "gt_txt": os.path.join(_root, "Evaluation", "GroundTruth", "page_0001.gt.txt"),
    "tsv_out": "/tmp/page_0001.gt.tsv",
}

def _core(token: str) -> str:
    norm = P.normalize_sentence_text(token).split()
    return P.decompose_token(norm[0])[1] if norm else ""


def _gold_cores(gold_text: str) -> List[str]:
    out = []
    for line in gold_text.splitlines():
        for tok in P.normalize_sentence_text(line).split():
            c = P.decompose_token(tok)[1]
            if c:
                out.append(c)
    return out


def annotate_rows(rows: List[Dict], gold_text: str) -> List[Dict]:
    """rows: TSV rows as dicts. Returns the same rows with EXTRA filled in."""
    gold = _gold_cores(gold_text)
    word_idx: List[int] = []
    cores: List[str] = []
    for i, r in enumerate(rows):
        r.update({k: "" for k in EXTRA})
        if r.get("level") != "5" or float(r["conf"]) < 0 or not r["text"].strip():
            continue
        c = _core(r["text"].strip())
        if not c:
            continue
        r["core"] = c
        word_idx.append(i)
        cores.append(c)

    # The OCR words are the reference here, so certain/possible indices and
    # the traceback steps (ocr index, gold index) all refer to the OCR words.
    alignment = align_tokens(cores, gold, include_details=True)
    matched_gold: Dict[int, Optional[int]] = {}
    for o, g in alignment.details.steps:
        if o is not None and g is not None:
            matched_gold[o] = g
    for j, i in enumerate(word_idx):
        g = matched_gold.get(j)
        rows[i]["gt_word"] = gold[g] if g is not None else ""
        if j in alignment.certain_reference:
            rows[i]["gt_status"] = "clean"
        elif j in alignment.possible_reference:
            rows[i]["gt_status"] = "ambiguous"
        else:
            rows[i]["gt_status"] = "error"
    return rows


def annotate_tsv(tsv_in: str, gold_text: str, tsv_out: str) -> int:
    """Write tsv_out with the extra columns. Returns the number of annotated words."""
    with open(tsv_in, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        fields = list(reader.fieldnames)
        rows = list(reader)
    rows = annotate_rows(rows, gold_text)
    with open(tsv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields + EXTRA, delimiter="\t",
                                quoting=csv.QUOTE_NONE, escapechar="\\", extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sum(1 for r in rows if r["gt_status"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Add ground truth columns to a Tesseract TSV")
    ap.add_argument("tsv_in", nargs="?", default=CONFIG["tsv_in"])
    ap.add_argument("gt_txt", nargs="?", default=CONFIG["gt_txt"])
    ap.add_argument("tsv_out", nargs="?", default=CONFIG["tsv_out"])
    a = ap.parse_args()
    with open(a.gt_txt, encoding="utf-8") as fh:
        n = annotate_tsv(a.tsv_in, fh.read(), a.tsv_out)
    print(f"annotated {n} words -> {a.tsv_out}")
