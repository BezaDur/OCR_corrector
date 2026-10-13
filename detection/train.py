"""Train a detector from examples, with leave-one-document-out checks.

Steps:
 1. Leave-one-document-out: each document is scored by a model that never saw it.
    This is the honest estimate; the report uses only these scores.
 2. The threshold is the lowest one whose out-of-sample precision reaches --target.
 3. The final model is fitted on all labelled examples and saved with that threshold.

Usage:
  python3 -m detection.train EXAMPLES.jsonl --out detection/model.json [--target 0.85] [--l2 1.0] [--report report.json]
  Add --no-ocr-conf to train without the engine confidence (for engines that give none).
"""
import argparse
import json
from typing import Dict, List

from detection.detector import ErrorDetector
from detection.evaluate import DEFAULT_THRESHOLDS, first_at_target, print_report, sweep
from detection.examples import read_examples, scored
from stage1_fit import auc


def feature_names(examples: List[Dict], drop: List[str] = ()) -> List[str]:
    """Features present in every scored example, minus any dropped."""
    names = None
    for e in examples:
        keys = set(e["features"])
        names = keys if names is None else names & keys
    return sorted(k for k in (names or []) if k not in drop)


def leave_one_doc_out(examples: List[Dict], names: List[str], l2: float) -> List[float]:
    docs = sorted({e["doc"] for e in examples})
    scores = [None] * len(examples)
    for held in docs:
        train = [e for e in examples if e["doc"] != held]
        det = ErrorDetector.fit([e["features"] for e in train],
                                [1 if e["label"] == "error" else 0 for e in train], names, l2=l2)
        idx = [i for i, e in enumerate(examples) if e["doc"] == held]
        for i, s in zip(idx, det.score([examples[i]["features"] for i in idx])):
            scores[i] = s
    return scores


def train(examples: List[Dict], target: float = 0.85, l2: float = 1.0, drop: List[str] = ()):
    rows = scored(examples)
    if not rows:
        raise ValueError("no labelled clean/error examples")
    names = feature_names(rows, drop=list(drop))
    if not names:
        raise ValueError("no features shared by all labelled examples")
    labels = [1 if r["label"] == "error" else 0 for r in rows]

    oof = leave_one_doc_out(rows, names, l2)
    table = sweep(oof, labels, DEFAULT_THRESHOLDS)
    chosen = first_at_target(table, target)
    threshold = chosen["threshold"] if chosen else 0.5

    det = ErrorDetector.fit([r["features"] for r in rows], labels, names, l2=l2, threshold=threshold)
    # Report the out-of-sample numbers, not the in-sample ones.
    report = dict(
        words=len(rows), errors=sum(labels), base_error_rate=sum(labels) / len(rows),
        auc=auc(oof, labels), threshold=threshold, at_threshold=chosen,
        target=target, first_threshold_at_target=chosen, sweep=table,
        per_doc={}, features=names, note="out-of-sample (leave-one-document-out)",
        target_reached=chosen is not None,
    )
    return det, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("--out", default="detection/model.json")
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--no-ocr-conf", action="store_true", help="do not use the engine confidence feature")
    ap.add_argument("--report")
    a = ap.parse_args()

    examples = read_examples(a.examples)
    det, report = train(examples, target=a.target, l2=a.l2,
                        drop=["ocr_conf"] if a.no_ocr_conf else [])
    print_report(report)
    det.save(a.out)
    print(f"\nmodel written to {a.out}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"report written to {a.report}")


if __name__ == "__main__":
    main()
