"""Per-word Tesseract confidences from a TSV file, aligned to text lines.

Tesseract TSV rows: level 5 = word rows (carry text and conf 0-100);
levels 1-4 = page/block/paragraph/line containers (conf -1, ignored).

Confidences are kept on Tesseract's 0-100 scale; the corrector's
SuspicionScorer divides by 100 itself.

Alignment: TSV word groups and the .txt lines come from the same run, in the
same order. Each text line is matched to the next TSV line group with the same
word count. If the counts disagree, the line gets None (no confidence) and the
mismatch is counted. Nothing is guessed.
"""
import csv
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class TsvWord:
    text: str
    conf: float          # Tesseract word confidence, 0-100
    left: int
    top: int
    width: int
    height: int


def read_tsv_lines(tsv_path: str) -> List[List[TsvWord]]:
    """Group level-5 word rows into lines, in file order. Empty tokens are dropped."""
    groups: List[List[TsvWord]] = []
    last_key = None
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("level") != "5":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            key = (row["block_num"], row["par_num"], row["line_num"])
            word = TsvWord(
                text=text,
                conf=max(0.0, float(row["conf"])),
                left=int(row["left"]), top=int(row["top"]),
                width=int(row["width"]), height=int(row["height"]),
            )
            if key != last_key:
                groups.append([])
                last_key = key
            groups[-1].append(word)
    return groups


def align_confidences(
    tsv_path: str, text_lines: Sequence[str]
) -> Tuple[List[Optional[List[float]]], dict]:
    """Return one entry per text line: a list of word confidences (0-100), or None.

    Stats: matched, mismatched, empty (text lines with no words).
    """
    groups = read_tsv_lines(tsv_path)
    out: List[Optional[List[float]]] = []
    stats = {"matched": 0, "mismatched": 0, "empty": 0}
    g = 0
    for line in text_lines:
        n = len(line.split())
        if n == 0:
            out.append(None)
            stats["empty"] += 1
            continue
        if g < len(groups) and len(groups[g]) == n:
            out.append([w.conf for w in groups[g]])
            stats["matched"] += 1
            g += 1
        else:
            out.append(None)
            stats["mismatched"] += 1
            g += 1  # assume this group belongs to this line and move on
    return out, stats
