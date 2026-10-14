#!/usr/bin/env python3
"""
train_LightGBM.py
------------------
Trains the LightGBM OCR error detector on the harmonized corpus.

Features come from OCR_error_detection/features.py, the same module the
detector uses at inference, so training and serving cannot drift apart.

Inputs (paths below):
  * main dictionary (word list + frequencies), normalized to canonical keys
  * character KenLM model (optional with --no-char-lm; then the feature is NaN)
  * optical confusion matrix (key "sub")
  * harmonized corpus TSV (page, ocr_word, is_error, doc_freq, ...)

Output: Model/detector_lgbm.pkl with feature_schema_version = 2.

Not changed here yet (Stage 2 of docs/lgbm_detector_review.md):
  * tesseract_conf is not a feature (the corpus value is a placeholder)
  * scale_pos_weight, fixed thresholds, and fixed 400 rounds
"""

import argparse
import os
import sys
import time

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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureContext, extract_frame,
)

HARMONIZED_TSV = "harmonized_corpus_jiwer.tsv"
if not os.path.exists(HARMONIZED_TSV):
    HARMONIZED_TSV = os.path.join(_ROOT, "harmonized_corpus_jiwer.tsv")

MASTER_DICT_PATH = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt"
CHAR_LM_PATH = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm"

CONFUSION_PATH = os.path.join(_ROOT, "Model", "confusion_matrix.json")

MODEL_OUTPUT = os.path.join(_HERE, "detector_lgbm.pkl")


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


def build_context(master_path, char_lm_path, use_char_lm):
    norm = AmharicNormalizer.normalize_word_strict
    vocab = normalize_vocab(load_master_dict(master_path), norm)

    char_lm = None
    if use_char_lm:
        import kenlm
        print(f"Loading KenLM: {char_lm_path}...")
        char_lm = kenlm.Model(char_lm_path)
    else:
        print("Character KenLM disabled (--no-char-lm): char_kenlm_score will be NaN.")

    print(f"Loading confusion matrix: {CONFUSION_PATH}...")
    confusion = ConfusionMatrix(CONFUSION_PATH)
    print(f"  reverse substitution entries: {len(confusion.reverse_sub_map):,}")
    if not confusion.reverse_sub_map:
        raise RuntimeError("confusion matrix produced no reverse substitutions; check the JSON keys")

    return FeatureContext(vocab=vocab, norm=norm,
                          reverse_sub_map=confusion.reverse_sub_map,
                          char_lm=char_lm)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-char-lm", action="store_true",
                    help="train without the KenLM feature (feature is NaN)")
    ap.add_argument("--master-dict", default=MASTER_DICT_PATH)
    ap.add_argument("--char-lm", default=CHAR_LM_PATH)
    ap.add_argument("--out", default=MODEL_OUTPUT)
    args = ap.parse_args()

    print("=" * 65)
    print("Training LightGBM detector (feature schema v%d)" % FEATURE_SCHEMA_VERSION)
    print("=" * 65)

    if not os.path.exists(HARMONIZED_TSV):
        print(f"Error: {HARMONIZED_TSV} not found.")
        return

    ctx = build_context(args.master_dict, args.char_lm, use_char_lm=not args.no_char_lm)

    print(f"Loading {HARMONIZED_TSV}...")
    df = pd.read_csv(HARMONIZED_TSV, sep="\t")
    df = df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"] = df["ocr_word"].astype(str)
    print(f"Loaded {len(df):,} tokens across {df['page'].nunique():,} pages.")
    print(f"Real errors: {df['is_error'].sum():,} ({100 * df['is_error'].mean():.2f}%)")

    t0 = time.time()
    X = pd.DataFrame(
        extract_frame(zip(df["ocr_word"], df["doc_freq"].astype(int)), ctx),
        columns=FEATURE_NAMES,
    )
    print(f"Feature extraction complete in {time.time() - t0:.1f}s.")
    y = df["is_error"].values
    groups = df["page"].values

    # ---- Cross-validation (grouped by page). Unchanged settings from before. ----
    print("\nTraining LightGBM with 5-fold cross-validation (grouped by page)...")
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.75,
        "min_child_samples": 30,
        "scale_pos_weight": 2.0,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }
    models = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        trn = lgb.Dataset(X.iloc[tr], label=y[tr])
        val = lgb.Dataset(X.iloc[va], label=y[va], reference=trn)
        clf = lgb.train(params, trn, num_boost_round=500,
                        valid_sets=[trn, val],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = clf.predict(X.iloc[va], num_iteration=clf.best_iteration)
        models.append(clf)
        print(f"  fold {fold}: best_iteration={clf.best_iteration}")

    print("\n" + "=" * 65)
    print("Out-of-fold results (grouped by page)")
    print(f"ROC-AUC: {roc_auc_score(y, oof):.4f} | PR-AUC: {average_precision_score(y, oof):.4f}")
    print("=" * 65)

    importance = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    print("\n[FEATURE IMPORTANCE (GAIN, mean over folds)]")
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        print(f"  {name:<20}: {imp:.1f}")

    # ---- Production model on the full corpus (Stage 2 will hold out the eval pages). ----
    print("\nTraining production model on the full corpus...")
    final_model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=400)

    payload = {
        "model": final_model,
        "features": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "pass_thresh": 0.15,
        "flag_thresh": 0.60,
        "char_lm_used": not args.no_char_lm,
        "master_dict_path": args.master_dict,
        "confusion_matrix_path": CONFUSION_PATH,
    }
    joblib.dump(payload, args.out)
    print(f"\nProduction model saved to: {args.out}")


if __name__ == "__main__":
    main()
