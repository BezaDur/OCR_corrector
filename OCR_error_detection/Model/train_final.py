#!/usr/bin/env python3
"""
train_conffree.py — Conf-free final model (no ocr_conf feature)

Per user hypothesis: Tesseract confidence is unreliable (19% errors have conf>85, 7.5% correct have conf<70)
So we train without ocr_conf, relying only on linguistic features.

Uses 100 eval pages with real conf for evaluation, but training uses only linguistic features.
Splits 70/30 as in test_conf_free.py but now trains final model on 70 pages and evaluates on 30,
plus also trains on all 100 for production.

Features WITHOUT ocr_conf:
- log_corpus_freq, in_lexicon, doc_freq, is_doc_singleton, token_len, len_ratio,
  has_one_flip, one_flip_count, char_kenlm_score (NaN), rare_glyph_ratio,
  n_neighbors, mixed_script, has_digit, is_high_freq

14 features total, no ocr_conf.
"""

import os, re, math, json, unicodedata, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).parent
EVAL_DIR = _HERE.parent / "Evaluation"
LOCAL_LEXICON = _HERE / "local_lexicon.txt"
CONFUSION_PATH = _HERE / "confusion_matrix.json"
MODEL_OUTPUT = _HERE / "detector_lgbm_conffree.pkl"

RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")

def canonical_form(word, norm_fn):
    if not word:
        return word
    try:
        nfc = unicodedata.normalize('NFC', word)
    except:
        nfc = word
    try:
        return norm_fn(nfc)
    except:
        return nfc

def load_lexicon(path, max_types=15000):
    print(f"Loading lexicon {path} max {max_types}...")
    freq_dict = {}
    try:
        from OCR_error_detection.normalizer import AmharicNormalizer
        norm = AmharicNormalizer.normalize_word_strict
    except:
        norm = lambda x: x
    tmp = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            w = parts[0]
            try:
                freq = int(parts[-1]) if parts[-1].isdigit() else 1
            except:
                freq = 1
            nw = canonical_form(w, norm)
            if not nw:
                continue
            tmp[nw] = max(tmp.get(nw, 0), freq)
    import heapq
    keep = heapq.nlargest(max_types, tmp, key=tmp.__getitem__)
    freq_dict = {k: tmp[k] for k in keep}
    print(f"  Loaded {len(freq_dict):,} types")
    return freq_dict, norm

def load_confusion_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rev = defaultdict(set)
    if "sub" in data:
        for true_ch, obs_map in data["sub"].items():
            if isinstance(obs_map, dict):
                for obs_ch, cost in obs_map.items():
                    try:
                        c = float(cost)
                    except:
                        c = 4.5
                    if c <= 4.0 and true_ch != obs_ch:
                        rev[obs_ch].add(true_ch)
    result = {k: list(v) for k, v in rev.items()}
    print(f"  Confusion map {len(result)} chars")
    return result

def load_eval_data():
    analysis_path = EVAL_DIR / "analysis.tsv"
    rows = []
    with open(analysis_path, encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        idx_page = header.index('page')
        idx_ocr = header.index('ocr_word')
        idx_gt = header.index('gt_word')
        idx_err = header.index('is_error')
        idx_conf = header.index('tesseract_conf')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(idx_page, idx_ocr, idx_gt, idx_err, idx_conf):
                continue
            try:
                page = parts[idx_page]
                ocr_word = parts[idx_ocr]
                gt_word = parts[idx_gt]
                is_error_raw = int(parts[idx_err])
                is_error = 1 if is_error_raw!=0 else 0
                conf = float(parts[idx_conf])
                if conf == 75.0:
                    continue
                rows.append((page, ocr_word, gt_word, is_error, conf))
            except:
                continue
    df = pd.DataFrame(rows, columns=['page', 'ocr_word', 'gt_word', 'is_error', 'tesseract_conf'])
    print(f"Loaded eval data {len(df)} rows, {df['page'].nunique()} pages, errors {df['is_error'].sum()}")
    return df

def main():
    print("="*70)
    print("Training Conf-Free Model (no ocr_conf)")
    print("="*70)
    
    lexicon, norm_fn = load_lexicon(LOCAL_LEXICON, max_types=15000)
    reverse_map = load_confusion_map(CONFUSION_PATH)
    df = load_eval_data()
    
    # one_flip cache
    unique_words = df['ocr_word'].unique()
    print(f"Building one_flip cache for {len(unique_words)} words...")
    one_flip_cache = {}
    CONFUSION_CLASSES = (
        ("\u12a0", "\u12d0", "\u12d3"),
        ("\u1200", "\u1210", "\u1280"),
        ("\u1230", "\u1220"),
        ("\u1338", "\u1340"),
        ("\u1338", "\u1330"),
        ("\u1290", "\u1298"),
        ("\u1260", "\u1268"),
        ("\u12a8", "\u12a0"),
    )
    def confusion_alts(ch):
        out = []
        for cls in CONFUSION_CLASSES:
            if ch in cls:
                out.extend(c for c in cls if c != ch)
        return out
    
    for w in unique_words:
        cnt = 0
        w_norm = canonical_form(w, norm_fn)
        for i, ch in enumerate(w):
            alts = confusion_alts(ch) + reverse_map.get(ch, [])
            for alt in alts[:12]:
                cand = w[:i] + alt + w[i+1:]
                if cand == w:
                    continue
                cand_norm = canonical_form(cand, norm_fn)
                if cand_norm == w_norm:
                    continue
                if cand_norm in lexicon:
                    cnt += 1
                    break
            if cnt >= 8:
                break
        one_flip_cache[w] = cnt
    
    def build_features(dataframe):
        def get_freq(w):
            return lexicon.get(canonical_form(w, norm_fn), 0)
        df = dataframe.copy()
        df["corpus_freq"] = df["ocr_word"].map(get_freq)
        df["log_corpus_freq"] = df["corpus_freq"].map(lambda f: math.log10(f+1) if f>0 else 0.0)
        df["in_lexicon"] = (df["corpus_freq"]>0).astype(int)
        df["doc_freq"] = df.groupby(['page', 'ocr_word'])['ocr_word'].transform('count')
        df["is_doc_singleton"] = (df["doc_freq"]==1).astype(int)
        df["token_len"] = df["ocr_word"].str.len()
        df["len_ratio"] = df["token_len"]/4.14
        df["one_flip_count"] = df["ocr_word"].map(one_flip_cache)
        df["has_one_flip"] = (df["one_flip_count"]>0).astype(int)
        df["char_kenlm_score"] = float('nan')
        df["rare_glyph_ratio"] = df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS)/float(len(w)) if len(w)>0 else 0)
        df["n_neighbors"] = 0
        df["mixed_script"] = df["ocr_word"].apply(lambda w: 1 if any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in w) and any('\u1200' <= ch <= '\u137F' for ch in w) else 0)
        df["has_digit"] = df["ocr_word"].apply(lambda w: 1 if any(ch.isdigit() for ch in w) else 0)
        df["is_high_freq"] = (df["log_corpus_freq"]>3.0).astype(int)
        return df
    
    pages = sorted(df['page'].unique())
    random.seed(42)
    random.shuffle(pages)
    train_pages = set(pages[:70])
    test_pages = set(pages[70:])
    
    train_df = df[df['page'].isin(train_pages)].copy()
    test_df = df[df['page'].isin(test_pages)].copy()
    
    feature_cols = [
        "log_corpus_freq", "in_lexicon",
        "doc_freq", "is_doc_singleton",
        "token_len", "len_ratio",
        "has_one_flip", "one_flip_count",
        "char_kenlm_score", "rare_glyph_ratio",
        "n_neighbors", "mixed_script", "has_digit", "is_high_freq"
    ]
    
    train_feat = build_features(train_df)
    test_feat = build_features(test_df)
    
    X_train = train_feat[feature_cols]
    y_train = train_feat["is_error"].values
    X_test = test_feat[feature_cols]
    y_test = test_feat["is_error"].values
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    
    trn_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, trn_data, num_boost_round=300)
    pred = model.predict(X_test)
    
    # Evaluate
    print("\n--- Conf-Free Model Evaluation ---")
    for pass_t, flag_t in [(0.10, 0.65), (0.10, 0.70), (0.20, 0.65)]:
        labels = np.where(pred>=flag_t, "FLAG", np.where(pred>=pass_t, "GRAY", "PASS"))
        recall = (labels[y_test==1]!="PASS").mean()*100
        false_flag = (labels[y_test==0]=="FLAG").mean()*100
        pass_rate = (labels[y_test==0]=="PASS").mean()*100
        print(f"  thresh {pass_t}/{flag_t}: recall {recall:.1f}% pass {pass_rate:.1f}% false_flag {false_flag:.2f}%")
    
    # High-conf errors
    high_conf_mask = test_df['tesseract_conf']>85
    high_conf_errors = (y_test==1) & high_conf_mask.values
    if high_conf_errors.sum()>0:
        labels = np.where(pred>=0.65, "FLAG", np.where(pred>=0.10, "GRAY", "PASS"))
        caught = (labels[high_conf_errors]!="PASS").mean()*100
        print(f"  High-conf errors (conf>85) caught: {caught:.1f}% ({high_conf_errors.sum()} samples)")
    
    low_conf_correct_mask = (test_df['tesseract_conf']<70) & (y_test==0)
    if low_conf_correct_mask.sum()>0:
        labels = np.where(pred>=0.65, "FLAG", np.where(pred>=0.10, "GRAY", "PASS"))
        false_on_low = (labels[low_conf_correct_mask.values]=="FLAG").mean()*100
        print(f"  Low-conf correct (conf<70) false-flagged: {false_on_low:.1f}% ({low_conf_correct_mask.sum()} samples)")
    
    # Feature importance
    imp = sorted(zip(feature_cols, model.feature_importance(importance_type='gain')), key=lambda x: -x[1])
    print("\nFeature gains (conf-free):")
    for f,i in imp:
        print(f"  {f}: {i:.0f}")
    
    # Save final model trained on ALL 100 pages
    print("\n--- Training final conf-free on all 100 pages ---")
    full_feat = build_features(df)
    X_full = full_feat[feature_cols]
    y_full = full_feat["is_error"].values
    full_data = lgb.Dataset(X_full, label=y_full)
    final_model = lgb.train(params, full_data, num_boost_round=300)
    
    payload = {
        "model": final_model,
        "features": feature_cols,
        "pass_thresh": 0.10,
        "flag_thresh": 0.65,
        "calibrator": None,
        "note": "conf-free: no ocr_conf, 14 linguistic features only, 100 pages real eval"
    }
    joblib.dump(payload, MODEL_OUTPUT)
    print(f"Saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()
