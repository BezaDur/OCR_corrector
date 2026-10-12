"""Train the detector from labelled verdict files and report honest numbers.

Input: a folder of <page>.verdicts.jsonl files (from 04_evaluate_full_corrector.py
--verdict-dir). Rows need status clean/error and a `features` dict.

Reports leave-one-page-out AUC and precision/recall at each threshold, then
writes the model trained on all pages.

Usage: python3 -m detection.train VERDICT_DIR --out detection/model.json [--target 0.85]
"""
import argparse
import glob
import json
import os

from stage1_fit import auc, feature_names, leave_one_page_out
from detection.detector import ErrorDetector


def load_rows(verdict_dir: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(verdict_dir, "*.verdicts.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return [r for r in rows if r["status"] in ("clean", "error") and "features" in r]


def sweep(scores, rows, thresholds):
    out = []
    total_err = sum(1 for r in rows if r["status"] == "error")
    for t in thresholds:
        flagged = [r for r, s in zip(rows, scores) if s >= t]
        err = sum(1 for r in flagged if r["status"] == "error")
        cln = len(flagged) - err
        out.append(dict(threshold=t, flagged=len(flagged), precision=(err / len(flagged)) if flagged else None,
                        recall=(err / total_err) if total_err else None, clean_flagged=cln))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict_dir")
    ap.add_argument("--out", default="detection/model.json")
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()

    rows = load_rows(a.verdict_dir)
    if not rows:
        raise SystemExit("No labelled rows with features found.")
    names = feature_names(rows)
    labels = [1 if r["status"] == "error" else 0 for r in rows]
    print(f"words: {len(rows)}  errors: {sum(labels)}  features: {', '.join(names)}")

    oof = leave_one_page_out(rows, names, l2=a.l2)
    print(f"leave-one-page-out AUC: {auc(oof, labels):.3f}")
    print("\nthreshold  flagged  precision  recall  clean_flagged")
    for s in sweep(oof, rows, [round(x / 20, 2) for x in range(1, 20)]):
        p = "   -  " if s["precision"] is None else f"{s['precision']:.3f}"
        print(f"{s['threshold']:>9.2f}  {s['flagged']:>7}  {p:>9}  {s['recall']:.3f}  {s['clean_flagged']:>13}")

    feats = [r["features"] for r in rows]
    det = ErrorDetector.fit(feats, labels, names, l2=a.l2)
    hit = [s for s in sweep(oof, rows, [round(x / 100, 2) for x in range(1, 100)])
           if s["precision"] is not None and s["precision"] >= a.target]
    if hit:
        det.threshold = hit[0]["threshold"]
        print(f"\nthreshold set to {det.threshold} (lowest with precision >= {a.target} out-of-sample)")
    else:
        print(f"\nno threshold reaches precision {a.target}; threshold left at {det.threshold}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    det.save(a.out)
    print(f"model written to {a.out}")


if __name__ == "__main__":
    main()
