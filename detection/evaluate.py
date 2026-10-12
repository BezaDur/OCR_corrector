"""Evaluation of a detector on examples. Independent of any correction run.

Metrics, all on labelled clean/error examples (ambiguous and unlabelled are excluded):
  - AUC: how well the score ranks errors above clean words (0.5 = chance)
  - per threshold: flagged count, precision, recall, clean words flagged
  - per document: error and flag counts, so one bad page is visible

Usage:
  python3 -m detection.evaluate EXAMPLES.jsonl MODEL.json [--target 0.85] [--out report.json]
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Sequence

from detection.detector import ErrorDetector
from detection.examples import read_examples, scored
from stage1_fit import auc

DEFAULT_THRESHOLDS = [round(x / 100, 2) for x in range(1, 100)]


def sweep(scores: Sequence[float], labels: Sequence[int], thresholds: Sequence[float]) -> List[Dict]:
    total_err = sum(labels)
    out = []
    for t in thresholds:
        flagged = [y for s, y in zip(scores, labels) if s >= t]
        err = sum(flagged)
        out.append(dict(
            threshold=t,
            flagged=len(flagged),
            errors_flagged=err,
            clean_flagged=len(flagged) - err,
            precision=(err / len(flagged)) if flagged else None,
            recall=(err / total_err) if total_err else None,
        ))
    return out


def first_at_target(table: List[Dict], target: float):
    for row in table:
        if row["precision"] is not None and row["precision"] >= target:
            return row
    return None


def evaluate(examples: List[Dict], detector: ErrorDetector, target: float = 0.85) -> Dict:
    rows = scored(examples)
    if not rows:
        raise ValueError("no labelled clean/error examples to evaluate")
    labels = [1 if r["label"] == "error" else 0 for r in rows]
    scores = detector.score([r["features"] for r in rows])
    table = sweep(scores, labels, DEFAULT_THRESHOLDS)

    per_doc = defaultdict(lambda: dict(words=0, errors=0, flagged=0, errors_flagged=0))
    for r, y, s in zip(rows, labels, scores):
        d = per_doc[r["doc"]]
        d["words"] += 1
        d["errors"] += y
        if s >= detector.threshold:
            d["flagged"] += 1
            d["errors_flagged"] += y

    at_thr = [row for row in table if abs(row["threshold"] - detector.threshold) < 1e-9]
    return dict(
        words=len(rows),
        errors=sum(labels),
        base_error_rate=sum(labels) / len(rows),
        auc=auc(scores, labels),
        threshold=detector.threshold,
        at_threshold=at_thr[0] if at_thr else None,
        target=target,
        first_threshold_at_target=first_at_target(table, target),
        sweep=table,
        per_doc=dict(per_doc),
        features=detector.names,
    )


def print_report(rep: Dict) -> None:
    print(f"words {rep['words']}  errors {rep['errors']}  base error rate {rep['base_error_rate']:.3f}")
    print(f"AUC {rep['auc']:.3f}   features: {', '.join(rep['features'])}")
    at = rep["at_threshold"]
    if at:
        print(f"at model threshold {rep['threshold']}: flagged {at['flagged']}, "
              f"precision {at['precision']}, recall {at['recall']:.3f}")
    ft = rep["first_threshold_at_target"]
    print(f"lowest threshold with precision >= {rep['target']}: "
          + (f"{ft['threshold']} (recall {ft['recall']:.3f})" if ft else "none"))
    print("\nthreshold  flagged  precision  recall  clean_flagged")
    for r in rep["sweep"]:
        if round(r["threshold"] * 10) % 2 == 0 or r["threshold"] in (0.05, 0.95):
            p = "    -   " if r["precision"] is None else f"{r['precision']:.3f}"
            print(f"{r['threshold']:>9.2f}  {r['flagged']:>7}  {p:>9}  {r['recall']:.3f}  {r['clean_flagged']:>13}")
    print("\nper document (words, errors, flagged, errors flagged)")
    for doc, d in sorted(rep["per_doc"].items()):
        print(f"  {doc:<16} {d['words']:>5} {d['errors']:>5} {d['flagged']:>5} {d['errors_flagged']:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("model")
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = evaluate(read_examples(a.examples), ErrorDetector.load(a.model), target=a.target)
    print_report(rep)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\nreport written to {a.out}")


if __name__ == "__main__":
    main()
