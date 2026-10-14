"""Generic word list: one JSON object per line, from any OCR engine.

  {"doc": "page_0001", "index": 0, "text": "ሰላም", "conf": 91.2}

`conf` is optional (0-100). Labels come from a separate ground-truth alignment,
so this adapter writes unlabelled examples; use `label` only if already known.
Features produced here: ocr_conf (when conf is given), word_length (characters).
"""
import json
from typing import Dict, Iterable, List


def to_examples(words: Iterable[Dict]) -> List[Dict]:
    out = []
    for w in words:
        feats = {"word_length": float(len(w["text"]))}
        if w.get("conf") is not None:
            feats["ocr_conf"] = float(w["conf"])
        out.append(dict(doc=w["doc"], index=w.get("index", len(out)), text=w["text"],
                        label=w.get("label"), features=feats))
    return out


def read_words(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
