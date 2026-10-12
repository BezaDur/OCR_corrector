"""The engine-neutral example format. Every adapter writes it; training and evaluation read it.

One JSON object per line:
  {"doc": "page_0001", "index": 12, "text": "ሰላም",
   "label": "clean" | "error" | "ambiguous" | null,
   "features": {"log_freq": -3.2, "ocr_conf": 91.5, ...}}

- doc      the document (page) the word came from. Used to split train and test.
- label    ground truth: clean, error, or ambiguous (excluded from scoring). null = unlabelled.
- features numeric values only. A feature is "ocr_conf" when the engine gave a confidence
  (0-100); it is absent when the engine did not.
"""
import json
from typing import Dict, Iterable, List

LABELS = ("clean", "error", "ambiguous")


def write_examples(path: str, examples: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            check(ex)
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_examples(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ex = json.loads(line)
                check(ex)
                out.append(ex)
    return out


def check(ex: Dict) -> None:
    for key in ("doc", "text", "features"):
        if key not in ex:
            raise ValueError(f"example missing '{key}': {ex}")
    if ex.get("label") not in (None,) + LABELS:
        raise ValueError(f"bad label {ex.get('label')!r}")
    for k, v in ex["features"].items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"feature {k} must be numeric, got {v!r}")


def scored(examples: List[Dict]) -> List[Dict]:
    """Only examples that can be scored: labelled clean or error."""
    return [e for e in examples if e.get("label") in ("clean", "error")]
