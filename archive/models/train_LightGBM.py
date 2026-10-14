#!/usr/bin/env python3
"""
train_LightGBM.py
------------------
Trains Generalized LightGBM OCR Detector using permanent master dictionary (main_i7.txt),
graded corpus log-frequencies, KenLM character model, and document-level voting.
"""

import os
import re
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

HARMONIZED_TSV = "harmonized_corpus_jiwer.tsv"
if not os.path.exists(HARMONIZED_TSV):
    HARMONIZED_TSV = os.path.join(_PARENT, "harmonized_corpus_jiwer.tsv")

# Permanent Master Dictionary
MASTER_DICT_PATH = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt"

CHAR_LM_PATH = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm"
CONFUSION_PATH = os.path.join(_HERE, "confusion_matrix.json")
if not os.path.exists(CONFUSION_PATH):
    CONFUSION_PATH = os.path.join(_PARENT, "confusion_matrix.json")

MODEL_OUTPUT = os.path.join(_HERE, "detector_lgbm.pkl")
RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")

def load_master_dict(filepath):
    print(f"Loading permanent master dictionary: {filepath}...")
    freq_dict = {}
    with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            w = parts[0]
            try:
                freq = int(parts[-1])
            except ValueError:
                freq = 1
            freq_dict[w] = max(freq_dict.get(w, 0), freq)
    print(f"Loaded master dictionary with {len(freq_dict):,} words.")
    return freq_dict

def load_confusion_map(path):
    if not os.path.exists(path):
        print(f"[WARN] Confusion matrix not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rev = {}
    for p in data.get("substitutions", []):
        t, o = p.get("target"), p.get("ocr")
        if t and o: rev.setdefault(o, set()).add(t)
    return rev

def main():
    print("=" * 65)
    print("Training Generalized LightGBM with Master Dictionary & Frequencies")
    print("=" * 65)

    if not os.path.exists(HARMONIZED_TSV):
        print(f"Error: {HARMONIZED_TSV} not found.")
        return

    # 1. Load Permanent Master Dictionary
    master_dict = load_master_dict(MASTER_DICT_PATH)

    # 2. Load KenLM Character Model
    import kenlm
    print(f"Loading KenLM: {CHAR_LM_PATH}...")
    char_model = kenlm.Model(CHAR_LM_PATH)

    # 3. Load Optical Confusion Matrix
    print(f"Loading Confusion Matrix: {CONFUSION_PATH}...")
    reverse_map = load_confusion_map(CONFUSION_PATH)

    # 4. Load Harmonized Dataset
    print(f"Loading {HARMONIZED_TSV}...")
    df = pd.read_csv(HARMONIZED_TSV, sep="\t")
    df = df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"] = df["ocr_word"].astype(str)

    print(f"Loaded {len(df):,} tokens across {df['page'].nunique():,} pages.")
    print(f"Real Errors: {df['is_error'].sum():,} ({100*df['is_error'].mean():.2f}%)")

    # 5. Extract Feature Vectors
    print("\nExtracting feature vectors with continuous log frequencies...")
    t0 = time.time()

    unique_words = df["ocr_word"].unique()
    print(f"Evaluating {len(unique_words):,} unique words with KenLM & One-Flip...")

    kenlm_cache = {}
    for w in unique_words:
        if len(w) >= 2:
            spaced = " ".join(w)
            raw = char_model.score(spaced, bos=True, eos=True)
            kenlm_cache[w] = raw / (len(w) + 1.0)
        else:
            kenlm_cache[w] = -2.5

    one_flip_cache = {}
    for w in unique_words:
        cnt = 0
        chars = list(w)
        for i, ch in enumerate(chars):
            if ch in reverse_map:
                for cand in reverse_map[ch]:
                    cand_word = "".join(chars[:i] + [cand] + chars[i+1:])
                    if cand_word in master_dict:
                        cnt += 1
        one_flip_cache[w] = cnt

    # Vectorized / Mapped Feature Engineering
    # Continuous Log Corpus Frequency (0.0 if OOV)
    df["corpus_freq"] = df["ocr_word"].map(lambda w: master_dict.get(w, 0))
    df["log_corpus_freq"] = df["corpus_freq"].map(lambda f: math.log10(f + 1) if f > 0 else 0.0)
    df["in_lexicon"] = (df["corpus_freq"] > 0).astype(int)

    df["is_doc_singleton"] = (df["doc_freq"] == 1).astype(int)
    df["token_len"] = df["ocr_word"].str.len()

    df["one_flip_count"] = df["ocr_word"].map(one_flip_cache)
    df["has_one_flip"] = (df["one_flip_count"] > 0).astype(int)
    df["char_kenlm_score"] = df["ocr_word"].map(kenlm_cache)
    df["rare_glyph_ratio"] = df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS) / float(len(w)))

    print(f"Feature extraction complete in {time.time() - t0:.1f}s.")

    feature_cols = [
        "log_corpus_freq", "in_lexicon",
        "doc_freq", "is_doc_singleton",
        "token_len", "has_one_flip", "one_flip_count",
        "char_kenlm_score", "rare_glyph_ratio"
    ]
    X = df[feature_cols]
    y = df["is_error"].values
    groups = df["page"].values

    # 6. Train LightGBM with GroupKFold
    print("\nTraining LightGBM with 5-Fold Cross-Validation (Grouped by Page)...")
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.75,  # Force trees to learn KenLM and DocFreq, not just lexicon!
        'min_child_samples': 30,
        'scale_pos_weight': 2.0,   # Prioritize error recall
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }

    models = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        clf = lgb.train(
            params, trn_data, num_boost_round=500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_preds[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        models.append(clf)

    # 7. Evaluation
    auc = roc_auc_score(y, oof_preds)
    ap = average_precision_score(y, oof_preds)
    print(f"\n" + "=" * 65)
    print(f"Generalization Results across Unseen Pages")
    print(f"ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    print("=" * 65)

    PASS_THRESH = 0.15
    FLAG_THRESH = 0.60
    preds_label = np.where(oof_preds >= FLAG_THRESH, "FLAG",
                  np.where(oof_preds >= PASS_THRESH, "GRAY", "PASS"))

    err_mask = (y == 1)
    cor_mask = (y == 0)

    print(f"\n[PERFORMANCE ON {err_mask.sum():,} REAL ERRORS]")
    print(f"  ├── FLAG (Definite):        {(preds_label[err_mask] == 'FLAG').sum():,} ({100*(preds_label[err_mask] == 'FLAG').mean():.1f}%)")
    print(f"  ├── GRAY (Needs Word LM):  {(preds_label[err_mask] == 'GRAY').sum():,} ({100*(preds_label[err_mask] == 'GRAY').mean():.1f}%)")
    print(f"  └── MISSED (Passed):       {(preds_label[err_mask] == 'PASS').sum():,} ({100*(preds_label[err_mask] == 'PASS').mean():.1f}%)")

    print(f"\n[PERFORMANCE ON {cor_mask.sum():,} CLEAN WORDS]")
    print(f"  ├── PASS (Clean Pass):     {(preds_label[cor_mask] == 'PASS').sum():,} ({100*(preds_label[cor_mask] == 'PASS').mean():.1f}%)")
    print(f"  ├── GRAY (Scrutinized):    {(preds_label[cor_mask] == 'GRAY').sum():,} ({100*(preds_label[cor_mask] == 'GRAY').mean():.1f}%)")
    print(f"  └── FALSE ALARM (Flagged): {(preds_label[cor_mask] == 'FLAG').sum():,} ({100*(preds_label[cor_mask] == 'FLAG').mean():.2f}%)")

    importance = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    print("\n[BALANCED FEATURE IMPORTANCE (GAIN)]")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<22}: {imp:.1f}")

    # 8. Train Production Model and Save
    print(f"\nTraining production model on full dataset...")
    final_trn = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, final_trn, num_boost_round=400)

    payload = {
        "model": final_model,
        "features": feature_cols,
        "pass_thresh": PASS_THRESH,
        "flag_thresh": FLAG_THRESH,
        "master_dict_path": MASTER_DICT_PATH,
    }
    joblib.dump(payload, MODEL_OUTPUT)
    print(f"\nProduction model successfully saved to: {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()