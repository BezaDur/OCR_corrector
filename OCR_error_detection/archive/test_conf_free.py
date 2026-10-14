#!/usr/bin/env python3
"""
Test conf-free vs conf-included on small sample (100 eval pages with real conf)

Goal: Does removing ocr_conf improve accuracy, given that Tesseract conf is unreliable
(19% of errors have conf>85, 7.5% of correct have conf<70)?

We use only 100 eval pages where we have real conf from Evaluation/OCRed/*.conf.tsv
Split 70/30 train/test, train two models:
- Model WITH conf (15 features)
- Model WITHOUT conf (14 features, ocr_conf dropped)

Compare recall, false_flag, and specifically high-conf errors.
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
OCR_DIR = EVAL_DIR / "OCRed"
GT_DIR = EVAL_DIR / "GroundTruth"

LOCAL_LEXICON = _HERE / "local_lexicon.txt"
CONFUSION_PATH = _HERE / "confusion_matrix.json"

RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")
PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')

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
    # keep top
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
    """
    Load 100 eval pages with real conf and GT alignment.
    We will build dataframe from analysis.tsv which already has alignment + real conf
    """
    analysis_path = EVAL_DIR / "analysis.tsv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"{analysis_path} not found")
    
    # analysis.tsv header: page, ocr_word, gt_word, is_error, detector_verdict, detector_score, tesseract_conf, in_lexicon, one_flip, reason
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
                # is_error can be 0=correct, 1=error, 2/3 other? Treat >0 as error
                is_error = 1 if is_error_raw!=0 else 0
                conf = float(parts[idx_conf])
                if conf == 75.0:
                    continue # skip dummy
                rows.append((page, ocr_word, gt_word, is_error, conf))
            except:
                continue
    
    df = pd.DataFrame(rows, columns=['page', 'ocr_word', 'gt_word', 'is_error', 'tesseract_conf'])
    print(f"Loaded eval data {len(df)} rows from {analysis_path}, {df['page'].nunique()} pages")
    print(f"Errors: {df['is_error'].sum()}, Correct: {(df['is_error']==0).sum()}")
    return df

def main():
    print("="*70)
    print("Conf-free vs Conf-included small sample test")
    print("="*70)
    
    lexicon, norm_fn = load_lexicon(LOCAL_LEXICON, max_types=15000)
    reverse_map = load_confusion_map(CONFUSION_PATH)
    
    df = load_eval_data()
    
    # Feature: one_flip cache
    unique_words = df['ocr_word'].unique()
    print(f"Building one_flip cache for {len(unique_words)} unique words...")
    one_flip_cache = {}
    # Simple confusion alts
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
    
    def build_features(dataframe, include_conf=True):
        def get_freq(w):
            return lexicon.get(canonical_form(w, norm_fn), 0)
        df = dataframe.copy()
        df["corpus_freq"] = df["ocr_word"].map(get_freq)
        df["log_corpus_freq"] = df["corpus_freq"].map(lambda f: math.log10(f+1) if f>0 else 0.0)
        df["in_lexicon"] = (df["corpus_freq"]>0).astype(int)
        # doc_freq: we don't have it in analysis.tsv, simulate as 1 for singleton, else random
        # For this test, we will compute doc_freq from page groups
        # Count per page per word
        df["doc_freq"] = df.groupby(['page', 'ocr_word'])['ocr_word'].transform('count')
        df["is_doc_singleton"] = (df["doc_freq"]==1).astype(int)
        df["token_len"] = df["ocr_word"].str.len()
        df["len_ratio"] = df["token_len"]/4.14
        df["one_flip_count"] = df["ocr_word"].map(one_flip_cache)
        df["has_one_flip"] = (df["one_flip_count"]>0).astype(int)
        df["char_kenlm_score"] = float('nan')
        df["rare_glyph_ratio"] = df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS)/float(len(w)) if len(w)>0 else 0)
        df["ocr_conf"] = df["tesseract_conf"]/100.0 if include_conf else float('nan')
        # Actually if include_conf=False, we set NaN and later drop column
        df["n_neighbors"] = 0
        df["mixed_script"] = df["ocr_word"].apply(lambda w: 1 if any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in w) and any('\u1200' <= ch <= '\u137F' for ch in w) else 0)
        df["has_digit"] = df["ocr_word"].apply(lambda w: 1 if any(ch.isdigit() for ch in w) else 0)
        df["is_high_freq"] = (df["log_corpus_freq"]>3.0).astype(int)
        return df
    
    # Split pages 70/30
    pages = sorted(df['page'].unique())
    random.seed(42)
    random.shuffle(pages)
    train_pages = set(pages[:70])
    test_pages = set(pages[70:])
    
    train_df = df[df['page'].isin(train_pages)].copy()
    test_df = df[df['page'].isin(test_pages)].copy()
    
    print(f"Train pages {len(train_pages)} rows {len(train_df)}, Test pages {len(test_pages)} rows {len(test_df)}")
    
    feature_cols_with = [
        "log_corpus_freq", "in_lexicon",
        "doc_freq", "is_doc_singleton",
        "token_len", "len_ratio",
        "has_one_flip", "one_flip_count",
        "char_kenlm_score", "rare_glyph_ratio",
        "ocr_conf", "n_neighbors",
        "mixed_script", "has_digit", "is_high_freq"
    ]
    feature_cols_without = [c for c in feature_cols_with if c != "ocr_conf"]
    
    train_feat_with = build_features(train_df, include_conf=True)
    test_feat_with = build_features(test_df, include_conf=True)
    train_feat_without = build_features(train_df, include_conf=False)
    test_feat_without = build_features(test_df, include_conf=False)
    
    X_train_with = train_feat_with[feature_cols_with]
    y_train = train_feat_with["is_error"].values
    X_test_with = test_feat_with[feature_cols_with]
    y_test = test_feat_with["is_error"].values
    
    X_train_without = train_feat_without[feature_cols_without]
    X_test_without = test_feat_without[feature_cols_without]
    
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
    
    print("\n--- Training WITH conf ---")
    trn_data = lgb.Dataset(X_train_with, label=y_train)
    model_with = lgb.train(params, trn_data, num_boost_round=200)
    pred_with = model_with.predict(X_test_with)
    # Fix sklearn 1.5+ API
    try:
        auc_with = roc_auc_score(y_test, pred_with)
    except TypeError:
        auc_with = roc_auc_score(y_test, pred_with, multi_class='ovr')
    except ValueError:
        # fallback manual AUC
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, pred_with)
        auc_with = np.trapz(tpr, fpr)
    print(f"WITH conf AUC {auc_with:.4f}")
    # Feature importance
    imp_with = sorted(zip(feature_cols_with, model_with.feature_importance(importance_type='gain')), key=lambda x: -x[1])
    print("Feature gains WITH:")
    for f,i in imp_with:
        print(f"  {f}: {i:.0f}")
    
    print("\n--- Training WITHOUT conf (conf-free) ---")
    trn_data2 = lgb.Dataset(X_train_without, label=y_train)
    model_without = lgb.train(params, trn_data2, num_boost_round=200)
    pred_without = model_without.predict(X_test_without)
    try:
        auc_without = roc_auc_score(y_test, pred_without)
    except TypeError:
        auc_without = roc_auc_score(y_test, pred_without, multi_class='ovr')
    except ValueError:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, pred_without)
        auc_without = np.trapz(tpr, fpr)
    print(f"WITHOUT conf AUC {auc_without:.4f}")
    imp_without = sorted(zip(feature_cols_without, model_without.feature_importance(importance_type='gain')), key=lambda x: -x[1])
    print("Feature gains WITHOUT:")
    for f,i in imp_without:
        print(f"  {f}: {i:.0f}")
    
    # Evaluate with thresholds 0.10 / 0.65
    def evaluate(preds, y_true, test_df_local, name):
        print(f"\n=== Evaluation {name} ===")
        for pass_t, flag_t in [(0.10, 0.65), (0.20, 0.65), (0.10, 0.70)]:
            labels = np.where(preds>=flag_t, "FLAG", np.where(preds>=pass_t, "GRAY", "PASS"))
            recall = (labels[y_true==1]!="PASS").mean()*100
            false_flag = (labels[y_true==0]=="FLAG").mean()*100
            pass_rate = (labels[y_true==0]=="PASS").mean()*100
            print(f"  thresh {pass_t}/{flag_t}: recall {recall:.1f}% pass {pass_rate:.1f}% false_flag {false_flag:.2f}%")
        
        # High-conf errors: errors with conf>85
        high_conf_mask = test_df_local['tesseract_conf']>85
        high_conf_errors = (y_true==1) & high_conf_mask.values
        if high_conf_errors.sum()>0:
            # How many high-conf errors caught?
            for pass_t, flag_t in [(0.10, 0.65)]:
                labels = np.where(preds>=flag_t, "FLAG", np.where(preds>=pass_t, "GRAY", "PASS"))
                caught = (labels[high_conf_errors]!="PASS").mean()*100
                print(f"  High-conf errors (conf>85) caught: {caught:.1f}% ({high_conf_errors.sum()} samples)")
        
        low_conf_correct_mask = (test_df_local['tesseract_conf']<70) & (y_true==0)
        if low_conf_correct_mask.sum()>0:
            for pass_t, flag_t in [(0.10, 0.65)]:
                labels = np.where(preds>=flag_t, "FLAG", np.where(preds>=pass_t, "GRAY", "PASS"))
                false_on_low = (labels[low_conf_correct_mask.values]=="FLAG").mean()*100
                print(f"  Low-conf correct (conf<70) false-flagged: {false_on_low:.1f}% ({low_conf_correct_mask.sum()} samples)")
    
    evaluate(pred_with, y_test, test_df, "WITH conf")
    evaluate(pred_without, y_test, test_df, "WITHOUT conf (conf-free)")
    
    # Save comparison
    print("\n--- Summary ---")
    print(f"WITH conf AUC: {auc_with:.4f}")
    print(f"WITHOUT conf AUC: {auc_without:.4f}")
    print(f"Difference (without - with): {auc_without - auc_with:+.4f}")
    if auc_without > auc_with:
        print("=> Conf-free is BETTER (supports your hypothesis)")
    else:
        print("=> Conf-included is better on overall AUC, but check high-conf errors")
    
    # Also check correlation of conf with error
    print(f"\nConf stats in test set:")
    print(f"  Correct mean conf: {test_df[test_df['is_error']==0]['tesseract_conf'].mean():.1f}")
    print(f"  Error mean conf: {test_df[test_df['is_error']==1]['tesseract_conf'].mean():.1f}")

if __name__ == "__main__":
    main()
