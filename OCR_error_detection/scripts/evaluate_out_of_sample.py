#!/usr/bin/env python3
"""evaluate_out_of_sample.py — honest evaluation of the shipped detector.

Why this exists
---------------
The README's headline (recall 94.2 %, false flag 0.01 %) comes from a 67-page
validation slice of the 1,267 pages the model was fit on. It is a *selection*
number, not a held-out one. This script produces the numbers to quote instead,
using the data and code that ship in the repository:

  * the model is loaded exactly as production loads it (``DetectionEngine``,
    same lexicon filter, same calibration, same gates);
  * the evaluation rows are the 100 development pages (``page_0001`` …
    ``page_0100``), which ``Model/train_LightGBM.py`` excludes from the fit
    (``DEV_PAGES = 100``);
  * every development page is additionally scored by how much of its text also
    appears on a training page (word 5-grams, punctuation/digits stripped), so
    the report can separate "unseen document" from "document the model has
    effectively seen".

It also reports what the single recall number hides: calibration (ECE and a
reliability table), the recall split between out-of-lexicon and real-word
(in-lexicon) errors, and where the false flags sit.

Nothing here retrains anything or writes to the repository; it reads the
corpus, scores it, and prints/writes a JSON report.

Usage
-----
    python OCR_error_detection/scripts/evaluate_out_of_sample.py
    python OCR_error_detection/scripts/evaluate_out_of_sample.py \
        --model Model/detector_lgbm_conffree.pkl --report /tmp/oos.json
    python OCR_error_detection/scripts/evaluate_out_of_sample.py --limit 20000

The companion cross-validated number (text-grouped out-of-fold, no page
leakage at all) is produced by the trainer itself:

    python OCR_error_detection/Model/train_LightGBM.py --no-char-lm \
        --group-by text --out /tmp/m.pkl --report /tmp/r.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # OCR_error_detection/
_ROOT = os.path.dirname(_PKG)                    # repo root
for _p in (_ROOT, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from OCR_error_detection.detector import DetectionEngine  # noqa: E402

CORPUS = os.path.join(_PKG, "Model", "harmonized_corpus_jiwer.tsv")
SHIPPED_MODEL = os.path.join(_PKG, "Model", "detector_lgbm_conffree.pkl")

PUNCT_RE = re.compile(r"[^\w\s]|\d")

# Tiers used to separate "unseen document" from "effectively seen document".
# A page whose 5-grams are >= 90 % present on training pages is a near-duplicate
# in everything but name; 0 % is genuinely new text.
TIERS = (
    ("disjoint", lambda share: share == 0.0),
    ("partial", lambda share: 0.0 < share < 0.90),
    ("near_duplicate", lambda share: share >= 0.90),
)


def clean(text: str) -> str:
    """Strip punctuation and digits, collapse whitespace (same rule as the trainer)."""
    return " ".join(w for w in (PUNCT_RE.sub("", t).strip() for t in text.split()) if w)


def grams(text: str, n: int = 5) -> set:
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_corpus(path: str, dev_pages: int, eval_all: bool = False):
    if not os.path.exists(path):
        raise SystemExit(f"corpus not found: {path}")
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"] = df["ocr_word"].astype(str)
    # Page names are normally page_0001..; anything else is tolerated and simply
    # cannot take part in the page-number split (used by --eval-all, where the
    # whole file is the evaluation set).
    page_no = df["page"].astype(str).str.replace("page_", "", regex=False)
    df["page_no"] = pd.to_numeric(page_no, errors="coerce")
    if eval_all:
        return df, df.iloc[0:0]
    dev = df[df.page_no <= dev_pages].copy()
    prod = df[df.page_no > dev_pages]
    if dev.empty:
        raise SystemExit(f"no page in {path} has page number <= {dev_pages}; "
                         f"pass --eval-all to score every row")
    return dev, prod


def text_overlap(dev: pd.DataFrame, prod: pd.DataFrame) -> dict:
    """Share of each dev page's word 5-grams that also occur on a training page."""
    prod_grams = set()
    for words in prod.groupby("page")["gt_word"]:
        prod_grams |= grams(clean(" ".join(w for w in words[1].tolist() if isinstance(w, str))))
    share = {}
    for page, words in dev.groupby("page")["gt_word"]:
        g = grams(clean(" ".join(w for w in words.tolist() if isinstance(w, str))))
        share[page] = (len(g & prod_grams) / len(g)) if g else 0.0
    return share


def score(df: pd.DataFrame, engine: DetectionEngine) -> pd.DataFrame:
    """Score every row through the production path; keep the signals it used."""
    probs, in_lex, flips, labels = [], [], [], []
    t0 = time.time()
    for word, doc_freq in zip(df.ocr_word, df.doc_freq):
        verdict = engine.verdict(word, doc_freq=float(doc_freq))
        probs.append(verdict.suspicion)
        labels.append(verdict.label)
        in_lex.append(int(verdict.signals.in_lexicon))
        flips.append(len(verdict.signals.one_flip_words))
    print(f"  scored {len(df):,} rows in {time.time() - t0:.1f}s "
          f"({1000 * (time.time() - t0) / max(1, len(df)):.2f} ms/word)")
    out = df.copy()
    out["p"] = probs
    out["label"] = labels
    out["in_lexicon"] = in_lex
    out["one_flip"] = flips
    return out


def metrics(sub: pd.DataFrame, pass_below: float, flag_above: float) -> dict:
    y = sub.is_error.values
    p = sub.p.values
    flag = p >= flag_above
    gray = (p >= pass_below) & ~flag
    err, ok = y == 1, y == 0
    n_err, n_ok = int(err.sum()), int(ok.sum())

    def rate(mask, den):
        return float(mask[den].mean()) if den.any() else float("nan")

    from sklearn.metrics import average_precision_score, roc_auc_score
    recall = rate(flag | gray, err)
    false_flag = rate(flag, ok)
    return {
        "rows": int(len(sub)),
        "errors": n_err,
        "clean": n_ok,
        "error_rate": n_err / max(1, len(sub)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "recall": recall,
        "recall_flag_only": rate(flag, err),
        "false_flag": false_flag,
        "score": recall * (1.0 - false_flag),
        "gray_on_clean": rate(gray, ok),
    }


def calibration(sub: pd.DataFrame, n_bins: int = 10) -> dict:
    """Equal-count reliability bins + expected calibration error."""
    order = sub.p.values.argsort()
    chunks = [c for c in __import__("numpy").array_split(order, n_bins) if len(c)]
    bins, ece = [], 0.0
    for i, idx in enumerate(chunks):
        mean_p = float(sub.p.values[idx].mean())
        observed = float(sub.is_error.values[idx].mean())
        ece += abs(mean_p - observed) * len(idx) / len(sub)
        bins.append({"bin": i + 1, "n": int(len(idx)), "mean_p": round(mean_p, 6),
                     "observed": round(observed, 6),
                     "ratio": round(observed / mean_p, 3) if mean_p > 0 else None})
    return {"ece": round(float(ece), 5), "bins": bins}


def threshold_sweep(sub: pd.DataFrame) -> list:
    rows = []
    for flag_above in (0.50, 0.60, 0.65, 0.70, 0.80):
        for pass_below in (0.05, 0.10, 0.15, 0.30):
            m = metrics(sub, pass_below, flag_above)
            rows.append({"pass_below": pass_below, "flag_above": flag_above,
                         "recall": round(m["recall"], 5),
                         "flag_only": round(m["recall_flag_only"], 5),
                         "false_flag": round(m["false_flag"], 6)})
    return rows


def by_lexicon_state(sub: pd.DataFrame, pass_below: float, flag_above: float) -> dict:
    """OOV errors are catchable by construction; real-word errors are the hard class."""
    out = {}
    for state, name in ((0, "out_of_lexicon"), (1, "in_lexicon")):
        part = sub[sub.in_lexicon == state]
        if len(part) == 0:
            continue
        out[name] = metrics(part, pass_below, flag_above)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--model", default=SHIPPED_MODEL)
    ap.add_argument("--dev-pages", type=int, default=100,
                    help="page_0001..page_N are the held-out development pages")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a prefix only (smoke)")
    ap.add_argument("--eval-all", action="store_true",
                    help="score every row (a new corpus; no train/dev split)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def say(*a):
        if not args.quiet:
            print(*a, flush=True)

    say("=" * 74)
    say("Out-of-sample evaluation — shipped detector, development pages")
    say("=" * 74)

    dev, prod = load_corpus(args.corpus, args.dev_pages, eval_all=args.eval_all)
    if args.eval_all:
        say(f"evaluation rows {len(dev):,} ({dev.page.nunique()} pages) — every row is scored")
    else:
        say(f"development rows {len(dev):,} ({dev.page.nunique()} pages), "
            f"training rows {len(prod):,} ({prod.page.nunique()} pages)")
    say(f"real errors in development: {int(dev.is_error.sum()):,} "
        f"({100 * dev.is_error.mean():.2f} %)")

    if len(prod):
        say("\nText overlap of each development page with the training pages ...")
        share = text_overlap(dev, prod)
        overlap = Counter()
        for value in share.values():
            for name, test in TIERS:
                if test(value):
                    overlap[name] += 1
                    break
        say(f"  pages by tier: {dict(overlap)}")
    else:
        say("\nText overlap: skipped (--eval-all: this corpus has no training split; "
            "use scripts/make_eval_corpus.py --reference to compare it with the training text)")
        share = {}

    say(f"\nLoading detector: {args.model}")
    engine = DetectionEngine(lgbm_model_path=args.model, verbose=False)
    say(f"  lexicon {engine.lexicon_info['types']:,} types "
        f"(min freq {engine.lexicon_info['min_freq']}), "
        f"gates pass<{engine.pass_below} flag>={engine.flag_above}")
    if engine.lgbm_model is None:
        raise SystemExit("the detector has no LightGBM model — cannot evaluate")

    subset = dev if args.limit is None else dev.head(args.limit)
    say("\nScoring ...")
    scored = score(subset, engine)

    pass_below, flag_above = engine.pass_below, engine.flag_above
    overall = metrics(scored, pass_below, flag_above)
    say(f"\nAll development rows: AUC={overall['auc']:.4f} AP={overall['ap']:.3f} "
        f"recall={100 * overall['recall']:.2f} % "
        f"(flag only {100 * overall['recall_flag_only']:.2f} %) "
        f"false-flag={100 * overall['false_flag']:.3f} % "
        f"gray-on-clean={100 * overall['gray_on_clean']:.2f} %")

    tiers = {}
    for name, test in TIERS:
        pages = [p for p, v in share.items() if test(v)]
        part = scored[scored.page.isin(pages)]
        if len(part) == 0:
            continue
        m = metrics(part, pass_below, flag_above)
        tiers[name] = m
        say(f"  {name:15s} pages={len(pages):>3} rows={m['rows']:>6,} "
            f"AUC={m['auc']:.4f} recall={100 * m['recall']:6.2f} % "
            f"FF={100 * m['false_flag']:.3f} %")

    cal = calibration(scored)
    say(f"\nCalibration on unseen pages: ECE = {cal['ece']:.4f}")
    for b in cal["bins"]:
        say(f"  bin {b['bin']:>2}  n={b['n']:>5,}  mean P={b['mean_p']:.4f}  "
            f"observed={b['observed']:.4f}")

    lex = by_lexicon_state(scored, pass_below, flag_above)
    say("\nRecall by lexicon state (the single recall number hides this):")
    for name, m in lex.items():
        say(f"  {name:15s} rows={m['rows']:>6,} errors={m['errors']:>5,} "
            f"recall={100 * m['recall']:6.2f} % flag={100 * m['recall_flag_only']:5.2f} % "
            f"FF={100 * m['false_flag']:.3f} %")

    missed = scored[(scored.is_error == 1) & (scored.label == "pass")]
    say(f"\nMissed errors (PASS on a wrong word): {len(missed):,} "
        f"of {int(scored.is_error.sum()):,}")
    if len(missed):
        say(f"  in-lexicon share among misses: {100 * missed.in_lexicon.mean():.1f} % "
            f"(among all errors: {100 * scored[scored.is_error == 1].in_lexicon.mean():.1f} %)")
        say(f"  examples: {[(r.ocr_word, r.gt_word) for r in missed.head(8).itertuples()]}")

    sweep = threshold_sweep(scored)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.path.abspath(args.model),
        "corpus": os.path.abspath(args.corpus),
        "dev_pages": args.dev_pages,
        "gates": {"pass_below": pass_below, "flag_above": flag_above},
        "lexicon": engine.lexicon_info,
        "page_overlap": {p: round(v, 4) for p, v in sorted(share.items())},
        "overall": overall,
        "by_text_tier": tiers,
        "calibration": cal,
        "by_lexicon_state": lex,
        "missed_errors": {
            "count": int(len(missed)),
            "in_lexicon_share": float(missed.in_lexicon.mean()) if len(missed) else None,
            "examples": [[r.ocr_word, None if pd.isna(r.gt_word) else r.gt_word]
                         for r in missed.head(20).itertuples()],
        },
        "threshold_sweep": sweep,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        say(f"\nReport: {args.report}")
    return report


if __name__ == "__main__":
    main()
