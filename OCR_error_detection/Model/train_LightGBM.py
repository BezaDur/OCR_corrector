#!/usr/bin/env python3
"""
train_LightGBM.py
------------------
Trains the LightGBM OCR error detector exactly as the README describes the
FINAL model: conf-free, 14 features, main_i7 lexicon, frequency-filtered,
grouped cross-validation, isotonic calibration, gates 0.10 / 0.65.

Recipe (defaults, all overridable):
  * lexicon   : Model/main_i7_frq10.txt, freq >= 500  -> 54,057 canonical types
                (the README's "FINAL authoritative" dictionary; --min-freq)
  * corpus    : Model/harmonized_corpus_jiwer.tsv, 946,057 rows / 1,367 pages
  * features  : OCR_error_detection/features.py — the same module detector.py
                serves with, so training and inference cannot drift apart
  * dev pages : page_0001..page_0100 are held out of the production fit
                (the model is trained on the remaining 1,267 pages, as the
                README records: "1267 pages, 14 feats conf-free")
  * sweep     : --sweep-freqs reproduces the README's dictionary comparison
                table on a 1200 / 67 page split
  * grouping  : cross-validation is grouped by *text* (5-gram overlap links,
                docs/lgbm_step2_gate.md), not by page; page groups overlap
  * calibration: isotonic on out-of-fold scores; scale_pos_weight is left at 1
                so the calibrated score is a real P(error)
  * gates     : pass 0.10, flag 0.65 — stored in the payload; --tune-thresholds
                picks them from the out-of-fold sweep against --ff-target

Metric definitions (they match the README's table):
  recall     = share of real errors that are not PASSed (flag + gray)
  false_flag = share of clean words that reach FLAG
  score      = recall * (1 - false_flag)

Outputs:
  * Model/detector_lgbm_conffree.pkl — model + features + schema version +
    gates + calibrator + lexicon metadata (the detector reads the metadata and
    rebuilds the identical lexicon, so train/serve parity is by construction)
  * Model/train_report.json — every number printed here, for the record

Usage:
  python OCR_error_detection/Model/train_LightGBM.py                  # FINAL recipe
  python OCR_error_detection/Model/train_LightGBM.py --no-char-lm     # skip KenLM
  python OCR_error_detection/Model/train_LightGBM.py --sweep-freqs 10,100,500,1000
  python OCR_error_detection/Model/train_LightGBM.py --quick          # smoke run
"""

# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "repo_paths.py")):
    _root = _os.path.dirname(_root)
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import repo_paths  # noqa: E402  (imported after the path bootstrap)
# ----------------------------------------------------------------------------

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # OCR_error_detection/
_ROOT = os.path.dirname(_PARENT)          # repo root
for _p in (_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureContext, extract_frame,
)

HARMONIZED_TSV = os.path.join(_HERE, "harmonized_corpus_jiwer.tsv")
LEXICON_PATH = os.path.join(_HERE, "main_i7_frq10.txt")
CONFUSION_PATH = os.path.join(_HERE, "confusion_matrix.json")
CHAR_LM_PATH = repo_paths.char_kenlm()
MODEL_OUTPUT = os.path.join(_HERE, "detector_lgbm_conffree.pkl")
REPORT_OUTPUT = os.path.join(_HERE, "train_report.json")

# The README's FINAL dictionary: main_i7, freq >= 500 -> 54,057 clean types.
DEFAULT_MIN_FREQ = 500
DEFAULT_PASS_THRESH = 0.10
DEFAULT_FLAG_THRESH = 0.65
DEV_PAGES = 100          # page_0001..page_0100: development only
VAL_PAGES = 67           # 1200 train / 67 val split used for the sweep table

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.75,
    "min_child_samples": 30,
    # No scale_pos_weight: it inflates the positive class and breaks the claim
    # that the score is a calibrated probability. Isotonic calibration below
    # does that job (docs/lgbm_detector_review.md §4.6, recommendation 8).
    "scale_pos_weight": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_master_dict(filepath):
    """Raw surface form -> max frequency, as stored in the dictionary file."""
    print(f"Loading main dictionary: {filepath}...")
    freq_dict = {}
    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            w = parts[0]
            try:
                freq = int(parts[-1])
            except ValueError:
                freq = 1
            freq_dict[w] = max(freq_dict.get(w, 0), freq)
    print(f"Loaded {len(freq_dict):,} raw entries.")
    return freq_dict


def normalize_vocab(raw_dict, norm):
    """Canonical key -> max frequency. Same rule as DetectionEngine._load_vocabulary."""
    vocab = {}
    for w, f in raw_dict.items():
        k = norm(w)
        vocab[k] = max(vocab.get(k, 0), f)
    print(f"Normalized to {len(vocab):,} canonical keys.")
    return vocab


def filter_vocab(vocab, min_freq=None, max_types=None):
    """The README's frequency filter — the model's most important knob."""
    if min_freq and min_freq > 1:
        vocab = {w: f for w, f in vocab.items() if f >= min_freq}
    if max_types and len(vocab) > max_types:
        import heapq
        keep = heapq.nlargest(max_types, vocab, key=vocab.__getitem__)
        vocab = {w: vocab[w] for w in keep}
    return vocab


def build_context(lexicon_path, min_freq, max_types, char_lm_path=None, use_char_lm=True):
    """FeatureContext for both training and inference — the shared definition.

    With no KenLM file the char_kenlm_score column is NaN, which is what the
    README's model does ("KenLM OFF (gain 0)").
    """
    norm = AmharicNormalizer.normalize_word_strict
    raw = load_master_dict(lexicon_path)
    vocab = filter_vocab(normalize_vocab(raw, norm), min_freq, max_types)
    print(f"Lexicon after freq>={min_freq} filter: {len(vocab):,} types")

    char_lm = None
    if use_char_lm and char_lm_path and os.path.exists(char_lm_path):
        import kenlm
        print(f"Loading KenLM: {char_lm_path}...")
        char_lm = kenlm.Model(char_lm_path)
    elif use_char_lm:
        print("KenLM not found: training without it (char_kenlm_score = NaN).")

    print(f"Loading confusion matrix: {CONFUSION_PATH}...")
    confusion = ConfusionMatrix(CONFUSION_PATH)
    print(f"  reverse substitution entries: {len(confusion.reverse_sub_map):,}")
    if not confusion.reverse_sub_map:
        raise RuntimeError("confusion matrix produced no reverse substitutions; check the JSON keys")

    return FeatureContext(vocab=vocab, norm=norm,
                          reverse_sub_map=confusion.reverse_sub_map,
                          char_lm=char_lm)


def load_corpus(path):
    if not os.path.exists(path):
        raise SystemExit(f"corpus not found: {path}")
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"] = df["ocr_word"].astype(str)
    df["page_no"] = df["page"].str.replace("page_", "", regex=False).astype(int)
    print(f"Loaded {len(df):,} tokens across {df.page.nunique():,} pages.")
    print(f"Real errors: {df['is_error'].sum():,} ({100 * df['is_error'].mean():.2f}%)")
    return df


# --------------------------------------------------------------------------- #
# Grouping and folds
# --------------------------------------------------------------------------- #

def strict_text_groups(df, max_page_occurrences=5, min_shared=20, ngram=5):
    """Link pages that share 5-grams, then take connected components.

    Pages in this corpus overlap heavily (82.5% of 5-grams occur on more than
    one page), so page-level folds leak. Strict grouping follows
    docs/lgbm_step2_gate.md: a 5-gram that appears on more than
    `max_page_occurrences` pages is too common to link anything, and two pages
    are linked only when they share at least `min_shared` qualifying 5-grams.
    """
    per_page = defaultdict(list)
    gt_by_page = {p: g.tolist() for p, g in df.groupby("page")["gt_word"]}
    for page, words in gt_by_page.items():
        words = [w for w in words if isinstance(w, str) and w]
        grams = [tuple(words[i:i + ngram]) for i in range(max(0, len(words) - ngram + 1))]
        per_page[page] = set(grams)

    gram_pages = defaultdict(set)
    for page, grams in per_page.items():
        for g in grams:
            gram_pages[g].add(page)
    qualifying = {g: pages for g, pages in gram_pages.items()
                  if len(pages) <= max_page_occurrences}
    print(f"  5-grams: {len(gram_pages):,} distinct; qualifying (<= {max_page_occurrences} pages): "
          f"{len(qualifying):,}")

    pair_shared = Counter()
    for g, pages in qualifying.items():
        if len(pages) < 2:
            continue
        plist = sorted(pages)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                pair_shared[(plist[i], plist[j])] += 1

    parent = {p: p for p in per_page}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), shared in pair_shared.items():
        if shared >= min_shared:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    group_of = {p: find(p) for p in per_page}
    groups = defaultdict(list)
    for p, g in group_of.items():
        groups[g].append(p)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"  text groups: {len(groups):,} (largest {sizes[:5]})")
    return group_of


def grouped_folds(group_of, df, n_splits, seed=42):
    """Assign whole groups to folds, greedily balancing row counts."""
    rows_by_group = df.groupby(df["page"].map(group_of)).size()
    order = list(rows_by_group.sort_values(ascending=False).index)
    rng = random.Random(seed)
    rng.shuffle(order)                      # break ties between equal sizes
    order.sort(key=lambda g: -rows_by_group[g])
    folds = [[] for _ in range(n_splits)]
    load = [0] * n_splits
    for g in order:
        k = load.index(min(load))
        folds[k].append(g)
        load[k] += rows_by_group[g]
    page_group = pd.Series(df["page"].map(group_of).values, index=df["page"].values)
    page_group = page_group[~page_group.index.duplicated()]
    page_to_fold = {}
    for k, gs in enumerate(folds):
        pages = page_group.index[page_group.isin(gs)]
        for p in pages:
            page_to_fold[p] = k
    print(f"  folds ({n_splits}) rows per fold: {load}")
    return page_to_fold


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def metrics(y, prob, pass_below, flag_above):
    """README-shaped metrics. recall counts flag+gray (an error not passed)."""
    flag = prob >= flag_above
    gray = (prob >= pass_below) & ~flag
    err, ok = y == 1, y == 0
    recall = float((flag | gray)[err].mean()) if err.any() else float("nan")
    ff = float(flag[ok].mean()) if ok.any() else float("nan")
    return {
        "recall": recall,
        "recall_flag_only": float(flag[err].mean()) if err.any() else float("nan"),
        "false_flag": ff,
        "score": recall * (1.0 - ff) if recall == recall and ff == ff else float("nan"),
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "gray_on_clean": float(gray[ok].mean()) if ok.any() else float("nan"),
    }


def threshold_sweep(y, prob):
    """Pass gate / flag gate trade-off, as in docs/lgbm_detector_review.md §3."""
    rows = []
    for flag_above in (0.5, 0.6, 0.65, 0.7, 0.8):
        for pass_below in (0.05, 0.10, 0.15, 0.30):
            m = metrics(y, prob, pass_below, flag_above)
            rows.append({"pass_below": pass_below, "flag_above": flag_above, **m})
    return rows


def fmt_pct(x):
    return f"{100 * x:.2f}%" if x == x else "n/a"


def print_table(rows, columns, headers):
    widths = [max(len(h), max(len(str(r[c])) for r in rows)) for h, c in zip(headers, columns)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(str(r[c]).ljust(w) for c, w in zip(columns, widths)))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def fit_grouped_cv(X, y, fold_of_row, n_splits, rounds, early_stopping, inner_val_frac=0.1, seed=42):
    """Out-of-fold predictions with an inner early-stopping split.

    Early stopping never uses the evaluation fold (that would make the OOF
    metrics optimistic, docs/lgbm_detector_review.md §4.8): each fold trains on
    90% of its own training rows and stops on the other 10%.
    """
    oof = np.zeros(len(y))
    models = []
    rng = np.random.RandomState(seed)
    for k in range(n_splits):
        va_idx = np.where(fold_of_row == k)[0]
        tr_pool = np.where(fold_of_row != k)[0]
        if len(tr_pool) < 2:
            raise SystemExit(f"fold {k + 1} has {len(tr_pool)} training rows; "
                             f"use more data or fewer folds (got --folds {n_splits})")
        tr_pool = rng.permutation(tr_pool)
        n_inner = max(1, int(len(tr_pool) * inner_val_frac))
        inner_idx, tr_idx = tr_pool[:n_inner], tr_pool[n_inner:]
        trn = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=list(FEATURE_NAMES))
        inner = lgb.Dataset(X[inner_idx], label=y[inner_idx], reference=trn)
        clf = lgb.train(LGB_PARAMS, trn, num_boost_round=rounds,
                        valid_sets=[inner],
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[va_idx] = clf.predict(X[va_idx], num_iteration=clf.best_iteration)
        models.append(clf)
        print(f"  fold {k + 1}/{n_splits}: best_iteration={clf.best_iteration} "
              f"rows train={len(tr_idx):,} inner={len(inner_idx):,} eval={len(va_idx):,}")
    return oof, models


def train_production(X, y, rounds, seed=42):
    """Fit on every row given (the folded-out dev pages are already excluded)."""
    clf = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES)),
                    num_boost_round=rounds)
    return clf


def feature_importance_table(model):
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    rows = [{"feature": f, "gain": int(g), "splits": int(s)}
            for f, g, s in zip(FEATURE_NAMES, gain, split)]
    return sorted(rows, key=lambda r: -r["gain"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the LightGBM OCR error detector.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", default=LEXICON_PATH,
                    help="dictionary file (word + frequency)")
    ap.add_argument("--corpus", default=HARMONIZED_TSV,
                    help="harmonized corpus TSV")
    ap.add_argument("--min-freq", type=int, default=DEFAULT_MIN_FREQ,
                    help="drop lexicon entries below this frequency (README FINAL: 500)")
    ap.add_argument("--max-types", type=int, default=None,
                    help="keep at most this many types (default: no cap)")
    ap.add_argument("--char-lm", default=CHAR_LM_PATH, help="character KenLM model")
    ap.add_argument("--no-char-lm", action="store_true",
                    help="train without the KenLM feature (the column becomes NaN)")
    ap.add_argument("--out", default=MODEL_OUTPUT, help="model output path")
    ap.add_argument("--report", default=None, help="JSON report path (default: <out>.report.json)")
    ap.add_argument("--pass-thresh", type=float, default=DEFAULT_PASS_THRESH)
    ap.add_argument("--flag-thresh", type=float, default=DEFAULT_FLAG_THRESH)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--group-by", choices=("text", "page"), default="text",
                    help="cross-validation grouping (docs/lgbm_step2_gate.md: text)")
    ap.add_argument("--dev-pages", type=int, default=DEV_PAGES,
                    help="page_0001..page_N are held out of the production fit")
    ap.add_argument("--sweep-freqs", default=None,
                    help="e.g. 10,100,500,1000 — reproduce the README dictionary table "
                         "on a 1200/67 page split instead of training once")
    ap.add_argument("--val-pages", type=int, default=VAL_PAGES,
                    help="development pages used by --sweep-freqs")
    ap.add_argument("--tune-thresholds", action="store_true",
                    help="pick the gates from the out-of-fold sweep instead of the README's")
    ap.add_argument("--ff-target", type=float, default=0.0001,
                    help="false-flag target for --tune-thresholds (README FINAL: 0.01%%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 60 pages, 2 folds, 50 rounds, capped lexicon")
    ap.add_argument("--quick-pages", type=int, default=60,
                    help="pages kept by --quick (development pages stay held out)")
    return ap.parse_args(argv)


def run_sweep(args, ctx_by_freq, df, freqs):
    """The README's dictionary comparison: same split, one row per freq filter."""
    train_pages = sorted(p for p in df.page.unique()
                         if int(p.split("_")[1]) > args.dev_pages)
    rng = random.Random(args.seed)
    rng.shuffle(train_pages)
    val_pages = set(train_pages[:args.val_pages])
    # The dev pages are not part of the 1,267 training/sweep pages at all
    # (README: "1267 pages (1200 train / 67 val)"), so they stay out of the fit.
    tr = df[~df.page.isin(val_pages) & (df.page_no > args.dev_pages)]
    va = df[df.page.isin(val_pages)]
    print(f"\nSweep split: {len(tr.page.unique())} train pages / {len(va.page.unique())} val pages")

    rows = []
    for freq in freqs:
        print(f"\n--- min_freq={freq} ---")
        ctx = ctx_by_freq(freq)
        X_tr = np.array(extract_frame(zip(tr.ocr_word, tr.doc_freq), ctx), dtype=float)
        X_va = np.array(extract_frame(zip(va.ocr_word, va.doc_freq), ctx), dtype=float)
        clf = train_production(X_tr, tr.is_error.values, args.rounds, seed=args.seed)
        prob = clf.predict(X_va)
        m = metrics(va.is_error.values, prob, args.pass_thresh, args.flag_thresh)
        rows.append({"min_freq": freq, "types": len(ctx.vocab),
                     "auc": f"{m['auc']:.4f}", "recall": fmt_pct(m["recall"]),
                     "false_flag": f"{100 * m['false_flag']:.2f}%",
                     "score": f"{m['score']:.3f}"})
        print(f"  types={len(ctx.vocab):,}  AUC={m['auc']:.4f}  "
              f"recall={fmt_pct(m['recall'])}  FF={100 * m['false_flag']:.2f}%  "
              f"score={m['score']:.3f}")
    print("\nREADME table layout:")
    print_table(rows, ["min_freq", "types", "auc", "recall", "false_flag", "score"],
                ["freq>=", "Types", "Val AUC", "Recall", "False Flag", "Score"])
    return rows


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    print("=" * 72)
    print(f"Training LightGBM detector (feature schema v{FEATURE_SCHEMA_VERSION}, conf-free)")
    print("=" * 72)

    df = load_corpus(args.corpus)

    if args.quick:
        args.folds = 2
        args.rounds = 50
        args.max_types = args.max_types or 20000
        keep = dev_pages_kept = args.dev_pages + args.quick_pages
        df = df[df.page_no <= keep].copy()
        print(f"[quick] {args.quick_pages} production pages + {args.dev_pages} dev pages, "
              f"folds=2 rounds=50 max_types={args.max_types}")

    use_char_lm = not args.no_char_lm
    if use_char_lm and not os.path.exists(args.char_lm):
        use_char_lm = False

    def ctx_for(freq):
        return build_context(args.lexicon, freq, args.max_types,
                             args.char_lm, use_char_lm=use_char_lm)

    if args.sweep_freqs:
        freqs = [int(f) for f in str(args.sweep_freqs).split(",") if f.strip()]
        rows = run_sweep(args, ctx_for, df, freqs)
        report_path = args.report or os.path.join(
            os.path.dirname(args.out) or ".", "train_sweep_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"sweep": rows, "args": vars(args)}, fh, indent=2)
        print(f"\nSweep report: {report_path}")
        return

    ctx = ctx_for(args.min_freq)
    ids = ctx.vocab
    print(f"Lexicon metadata: freq>={args.min_freq} cap={args.max_types} "
          f"types={len(ctx.vocab):,}")

    dev = df[df.page_no <= args.dev_pages]
    prod = df[df.page_no > args.dev_pages]
    print(f"Development pages (held out of the fit): {dev.page.nunique()} "
          f"({len(dev):,} rows)")
    print(f"Production pages: {prod.page.nunique()} ({len(prod):,} rows)")

    # ---- Features (the same extract_frame the detector's serving path uses) ----
    t0 = time.time()
    X_prod = np.array(extract_frame(zip(prod.ocr_word, prod.doc_freq), ctx), dtype=float)
    y_prod = prod.is_error.values
    print(f"Feature extraction: {X_prod.shape[0]:,} rows x {X_prod.shape[1]} cols "
          f"in {time.time() - t0:.1f}s")

    # ---- Grouped cross-validation (out-of-fold metrics + calibration) ----
    if args.group_by == "text":
        group_of = strict_text_groups(prod)
    else:
        group_of = {p: p for p in prod.page.unique()}
        print(f"  page groups: {len(group_of):,} (page-level; overlapping text leaks)")
    page_to_fold = grouped_folds(group_of, prod, args.folds, seed=args.seed)
    fold_of_row = prod.page.map(page_to_fold).values

    print("\nGrouped cross-validation (grouped by %s)..." % args.group_by)
    oof, models = fit_grouped_cv(X_prod, y_prod, fold_of_row, args.folds,
                                 args.rounds, args.early_stopping, seed=args.seed)
    oof_metrics = metrics(y_prod, oof, args.pass_thresh, args.flag_thresh)
    print(f"\nOut-of-fold: AUC={oof_metrics['auc']:.4f} AP={oof_metrics['ap']:.4f} "
          f"recall={fmt_pct(oof_metrics['recall'])} "
          f"FF={100 * oof_metrics['false_flag']:.4f}% "
          f"score={oof_metrics['score']:.3f}")

    rounds = args.rounds
    if models and models[0].best_iteration:
        rounds = int(np.median([m.best_iteration or args.rounds for m in models]) * 1.1)
        print(f"Rounds for the production fit (median best_iteration x 1.1): {rounds}")

    # ---- Isotonic calibration on out-of-fold scores ----
    calibrator = None
    if len(set(y_prod)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y_prod)
        cal_oof = calibrator.predict(oof)
        cal_metrics = metrics(y_prod, cal_oof, args.pass_thresh, args.flag_thresh)
        print(f"Calibrated out-of-fold: AUC={cal_metrics['auc']:.4f} "
              f"recall={fmt_pct(cal_metrics['recall'])} "
              f"FF={100 * cal_metrics['false_flag']:.4f}%")
    else:
        cal_metrics = oof_metrics

    pass_thresh, flag_thresh = args.pass_thresh, args.flag_thresh
    if args.tune_thresholds:
        best = None
        for row in threshold_sweep(y_prod, calibrator.predict(oof) if calibrator is not None else oof):
            if row["false_flag"] <= args.ff_target:
                if best is None or row["recall"] > best["recall"]:
                    best = row
        if best:
            pass_thresh = best["pass_below"]
            flag_thresh = best["flag_above"]
            print(f"Tuned gates for FF<={100 * args.ff_target:.2f}%: "
                  f"pass={pass_thresh} flag={flag_thresh} "
                  f"(recall={fmt_pct(best['recall'])})")
        else:
            print(f"No gate met the FF target {100 * args.ff_target:.2f}%; "
                  f"keeping pass={pass_thresh} flag={flag_thresh}")

    # ---- Production model (trained on the 1,267 non-dev pages) ----
    print(f"\nTraining production model on {prod.page.nunique():,} pages "
          f"({len(prod):,} rows, {rounds} rounds)...")
    final_model = train_production(X_prod, y_prod, rounds, seed=args.seed)

    # ---- Dev-page check (the 100 pages the model has never seen) ----
    print("\n" + "=" * 72)
    print("Development pages (never trained on)")
    print("=" * 72)
    X_dev = np.array(extract_frame(zip(dev.ocr_word, dev.doc_freq), ctx), dtype=float)
    raw_dev = final_model.predict(X_dev)
    prob_dev = calibrator.predict(raw_dev) if calibrator is not None else raw_dev
    dev_metrics = metrics(dev.is_error.values, prob_dev, pass_thresh, flag_thresh)
    print(f"  AUC={dev_metrics['auc']:.4f}  recall={fmt_pct(dev_metrics['recall'])}  "
          f"FF={100 * dev_metrics['false_flag']:.4f}%  "
          f"score={dev_metrics['score']:.3f}  "
          f"(gray on clean words: {100 * dev_metrics['gray_on_clean']:.2f}%)")

    importance = feature_importance_table(final_model)
    print("\n[FEATURE IMPORTANCE (GAIN, production model)]")
    for row in importance:
        print(f"  {row['feature']:<20} gain={row['gain']:>10,} splits={row['splits']:>6,}")

    sweep = threshold_sweep(dev.is_error.values, prob_dev)
    print("\n[THRESHOLD SWEEP on the development pages]")
    print_table([{**r, "pass_below": f"{r['pass_below']:.2f}",
                  "flag_above": f"{r['flag_above']:.2f}",
                  "recall": fmt_pct(r["recall"]),
                  "false_flag": f"{100 * r['false_flag']:.2f}%"} for r in sweep],
                ["pass_below", "flag_above", "recall", "false_flag"],
                ["pass gate", "flag gate", "errors not passed", "clean flagged"])

    # ---- Save ----
    note = (f"conf-free main_i7 freq>={args.min_freq}: {len(ctx.vocab):,} types, "
            f"{prod.page.nunique():,} pages, {len(FEATURE_NAMES)} feats, "
            f"dev recall {100 * dev_metrics['recall']:.1f}% "
            f"FF {100 * dev_metrics['false_flag']:.2f}%")
    payload = {
        "model": final_model,
        "features": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "pass_thresh": pass_thresh,
        "flag_thresh": flag_thresh,
        "calibrator": calibrator,
        "note": note,
        # Lexicon metadata: the detector rebuilds exactly this vocabulary, so
        # training and serving cannot disagree about in_lexicon / log_corpus_freq.
        "lexicon_path": os.path.abspath(args.lexicon),
        "lexicon_min_freq": args.min_freq,
        "lexicon_max_types": args.max_types,
        "lexicon_types": len(ctx.vocab),
        "char_lm_used": bool(use_char_lm and ctx.char_lm is not None),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(payload, args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nModel saved: {args.out} ({size_mb:.1f} MB)")
    print(f"  {note}")

    report = {
        "args": vars(args),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "lexicon": {"path": os.path.abspath(args.lexicon),
                    "min_freq": args.min_freq, "max_types": args.max_types,
                    "types": len(ctx.vocab)},
        "groups": {"by": args.group_by,
                   "count": len(set(page_to_fold.values())) if args.group_by == "text" else prod.page.nunique()},
        "rows": {"production": int(len(prod)), "development": int(len(dev))},
        "pages": {"production": int(prod.page.nunique()), "development": int(dev.page.nunique())},
        "rounds": rounds,
        "gates": {"pass_below": pass_thresh, "flag_above": flag_thresh},
        "oof_metrics": oof_metrics,
        "calibrated_oof_metrics": cal_metrics,
        "dev_metrics": dev_metrics,
        "threshold_sweep": sweep,
        "feature_importance": importance,
        "note": note,
        "seconds": round(time.time() - t_start, 1),
    }
    report_path = args.report or os.path.splitext(args.out)[0] + ".report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"Report saved: {report_path}")
    print(f"Done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
