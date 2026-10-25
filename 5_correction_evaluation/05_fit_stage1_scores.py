"""Step 15: which Stage 1 signals separate errors from clean tokens, and does a
learned score beat the hand-set one?

Input: the <page>.verdicts.jsonl files from 01_evaluate_full_corrector.py
(--verdict-dir). Each row needs its features (logged by the corrector).

Output:
  1. AUC of each signal alone against the error label (0.5 = no information)
  2. AUC and precision of the current hand-set score (suspicion)
  3. the same for a logistic regression trained on all pages except one,
     applied to that page (leave-one-page-out, so no page scores itself)

Usage: python3 05_fit_stage1_scores.py VERDICT_DIR [--target 0.85]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stage1_fit import auc, feature_names, leave_one_page_out  # noqa: E402


# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 5_correction_evaluation/05_fit_stage1_scores.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "verdict_dir": os.path.join(os.environ.get("TESSERACT_WORKSPACE", "/media/des/Tess/Tesseract2"),
            "synthetic_eval", "verdicts"),  # *.verdicts.jsonl from 01_evaluate_full_corrector.py
    "target": 0.85,
    "l2": 1.0,
}

def precision_table(scores, rows, thresholds):
    out = []
    for t in thresholds:
        flagged = [r for r, s in zip(rows, scores) if s >= t]
        err = sum(1 for r in flagged if r["status"] == "error")
        cln = sum(1 for r in flagged if r["status"] == "clean")
        judged = err + cln
        out.append((t, len(flagged), (err / judged) if judged else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdict_dir", nargs="?", default=CONFIG["verdict_dir"])
    ap.add_argument("--target", type=float, default=CONFIG["target"])
    ap.add_argument("--l2", type=float, default=CONFIG["l2"])
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.verdict_dir, "*.verdicts.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows = [r for r in rows if r["status"] in ("clean", "error")]
    if not rows or "features" not in rows[0]:
        raise SystemExit("No labelled rows with features. Re-run the evaluator with --verdict-dir.")
    labels = [1 if r["status"] == "error" else 0 for r in rows]
    print(f"Judged tokens: {len(rows)}  errors: {sum(labels)}  "
          f"base error rate: {sum(labels) / len(rows):.3f}")

    names = feature_names(rows)
    print("\n1. AUC per signal (0.5 = no information)")
    for k in names:
        vals = [r["features"][k] for r in rows]
        a = auc(vals, labels)
        print(f"   {k:<18} {a:.3f}")

    hand = [r["suspicion"] for r in rows]
    print(f"\n2. Hand-set score (suspicion): AUC {auc(hand, labels):.3f}")

    learned = leave_one_page_out(rows, names, l2=args.l2)
    print(f"3. Learned score, leave-one-page-out: AUC {auc(learned, labels):.3f}")

    thresholds = [round(x / 20, 2) for x in range(0, 20)]
    print("\nthreshold  hand: flagged  precision |  learned: flagged  precision")
    hand_t = precision_table(hand, rows, thresholds)
    learn_t = precision_table(learned, rows, thresholds)
    for (t, nh, ph), (_, nl, pl) in zip(hand_t, learn_t):
        fh = "  -  " if ph is None else f"{ph:.3f}"
        fl = "  -  " if pl is None else f"{pl:.3f}"
        print(f"{t:>9.2f}  {nh:>12}  {fh:>9} |  {nl:>13}  {fl:>9}")

    best = [t for t, n, p in learn_t if p is not None and p >= args.target]
    if best:
        print(f"\nLearned: lowest threshold reaching precision {args.target}: {best[0]}")
    else:
        print(f"\nLearned: no threshold reaches precision {args.target}.")


if __name__ == "__main__":
    main()
