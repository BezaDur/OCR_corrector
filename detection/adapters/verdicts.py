"""Corrector verdict files (from 04_evaluate_full_corrector.py --verdict-dir) to examples.

Each verdict row already has its ground-truth status and the corrector's feature
vector. This adapter only renames fields into the common format.
"""
import glob
import json
import os
from typing import Dict, List


def to_examples(verdict_dir: str) -> List[Dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(verdict_dir, "*.verdicts.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                feats = dict(r.get("features", {}))
                out.append(dict(doc=r.get("page", os.path.basename(path)), index=len(out),
                                text=r.get("word", ""), label=r.get("status"), features=feats))
    return out
