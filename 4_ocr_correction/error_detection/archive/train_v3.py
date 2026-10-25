#!/usr/bin/env python3
"""
train_v3.py — Clean model training per roadmap Stage 1 & 2

- Strict evaluation isolation: 1367 pages -> train 1200, val 67, test 100 (held out)
- Real confidences: for eval pages we have real conf.tsv, for training pages we use NaN (not dummy 75)
- Fixed sub key, canonical NFC+Amharic normalization, NaN for missing KenLM/doc_freq/ocr_conf
- No scale_pos_weight distortion, no manual floors
- Isotonic calibration on val set
- Saves detector_lgbm_v3.pkl with calibrator
"""

import os
import re
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import unicodedata

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PARENT)

LOCAL_LEXICON = os.path.join(_HERE, "local_lexicon.txt")
CONFUSION_PATH = os.path.join(_HERE, "confusion_matrix.json")
if not os.path.exists(CONFUSION_PATH):
    CONFUSION_PATH = os.path.join(_PARENT, "confusion_matrix.json")

HARMONIZED_TSV = os.path.join(_HERE, "harmonized_corpus_jiwer.tsv")
MODEL_OUTPUT = os.path.join(_HERE, "detector_lgbm_v3.pkl")

EVAL_GT_DIR = os.path.join(_PARENT, "Evaluation", "GroundTruth")
EVAL_OCR_DIR = os.path.join(_PARENT, "Evaluation", "OCRed")

RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")
PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')

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

def _confusion_alts(ch: str):
    out=[]
    for cls in CONFUSION_CLASSES:
        if ch in cls:
            out.extend(c for c in cls if c!=ch)
    return out

def _canonical_form(word: str, norm_fn):
    if not word:
        return word
    try:
        nfc = unicodedata.normalize('NFC', word)
    except Exception:
        nfc = word
    try:
        return norm_fn(nfc)
    except Exception:
        return nfc

def load_lexicon(path: str, max_types: int = 150000):
    print(f"Loading lexicon: {path} (max {max_types})...")
    freq_dict={}
    is_raw=False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as probe:
            first=probe.readline()
            if len(first.strip().split())>5:
                is_raw=True
    except Exception:
        pass

    if is_raw:
        from collections import Counter
        counter=Counter()
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for tok in line.strip().split():
                    if tok:
                        counter[tok]+=1
        most_common=counter.most_common(max_types)
        try:
            from error_detection.normalizer import AmharicNormalizer
            norm = AmharicNormalizer.normalize_word_strict
        except Exception:
            norm=lambda x:x
        for w,freq in most_common:
            nw=_canonical_form(w, norm)
            if not nw:
                continue
            freq_dict[nw]=max(freq_dict.get(nw,0), freq)
    else:
        tmp={}
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                parts=line.strip().split()
                if not parts:
                    continue
                w=parts[0]
                try:
                    freq=int(parts[-1]) if parts[-1].isdigit() else 1
                except ValueError:
                    freq=1
                try:
                    from error_detection.normalizer import AmharicNormalizer
                    norm=AmharicNormalizer.normalize_word_strict
                    nw=_canonical_form(w, norm)
                except Exception:
                    nw=w
                if not nw:
                    continue
                tmp[nw]=max(tmp.get(nw,0), freq)
        if len(tmp)>max_types:
            import heapq
            keep=heapq.nlargest(max_types, tmp, key=tmp.__getitem__)
            freq_dict={k:tmp[k] for k in keep}
        else:
            freq_dict=tmp
    print(f"  Loaded {len(freq_dict):,} types")
    return freq_dict

def load_confusion_map(path: str):
    if not os.path.exists(path):
        print(f"[WARN] Confusion not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data=json.load(f)
    rev=defaultdict(set)
    if "sub" in data:
        for true_ch, obs_map in data["sub"].items():
            if isinstance(obs_map, dict):
                for obs_ch,cost in obs_map.items():
                    try:
                        c=float(cost)
                    except:
                        c=4.5
                    if c<=4.0 and true_ch!=obs_ch:
                        rev[obs_ch].add(true_ch)
            elif isinstance(obs_map, list):
                for obs_ch in obs_map:
                    if true_ch!=obs_ch:
                        rev[obs_ch].add(true_ch)
    if "substitutions" in data:
        for p in data["substitutions"]:
            if not isinstance(p, dict):
                continue
            true_ch=p.get("target") or p.get("true") or p.get("gt")
            obs_ch=p.get("ocr") or p.get("observed") or p.get("pred")
            cost=p.get("cost",2.2)
            if not true_ch or not obs_ch:
                continue
            try:
                c=float(cost)
            except:
                c=2.2
            if c<=4.0 and true_ch!=obs_ch:
                rev[obs_ch].add(true_ch)
    result={k:list(v) for k,v in rev.items()}
    print(f"  Confusion map: {len(result)} observed chars")
    return result

def clean_token(t):
    t=PUNCT_RE.sub('', t).strip()
    t=ETHIOPIC_PUNCT_RE.sub('', t).strip()
    return t

def main():
    print("="*70)
    print("Training Clean v3 Model — No Floors, No Leakage, Isotonic Calib")
    print("="*70)

    # 1. Load lexicon
    lexicon_path=LOCAL_LEXICON
    if not os.path.exists(lexicon_path):
        print(f"[ERROR] Lexicon not found at {lexicon_path}")
        return
    lexicon=load_lexicon(lexicon_path, max_types=150000)

    # 2. Confusion
    reverse_map=load_confusion_map(CONFUSION_PATH)

    # 3. Load harmonized corpus (1367 pages)
    if not os.path.exists(HARMONIZED_TSV):
        print(f"[ERROR] Harmonized TSV not found at {HARMONIZED_TSV}")
        return
    print(f"Loading {HARMONIZED_TSV}...")
    df=pd.read_csv(HARMONIZED_TSV, sep="\t")
    df=df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"]=df["ocr_word"].astype(str)
    print(f"  Total rows: {len(df):,}, pages: {df['page'].nunique():,}")

    # 4. Strict isolation: eval pages are page_0001..page_0100
    eval_page_names=set([f"page_{i:04d}" for i in range(1,101)])
    train_val_df=df[~df['page'].isin(eval_page_names)].copy()
    test_df=df[df['page'].isin(eval_page_names)].copy()
    print(f"  Train+Val (non-eval): {len(train_val_df):,} rows, {train_val_df['page'].nunique():,} pages")
    print(f"  Test (held-out eval 100): {len(test_df):,} rows, {test_df['page'].nunique():,} pages")

    # Split train_val into train 1200 pages and val 67 pages
    unique_pages=train_val_df['page'].unique()
    # Shuffle with seed for reproducibility
    np.random.seed(42)
    shuffled=np.random.permutation(unique_pages)
    train_pages=set(shuffled[:1200])
    val_pages=set(shuffled[1200:1267])  # 67 pages
    print(f"  Train pages: {len(train_pages)}, Val pages: {len(val_pages)}")

    train_df=train_val_df[train_val_df['page'].isin(train_pages)].copy()
    val_df=train_val_df[train_val_df['page'].isin(val_pages)].copy()

    print(f"  Train rows: {len(train_df):,}, Val rows: {len(val_df):,}")

    # 5. Feature engineering with canonical boundary and NaN for missing
    try:
        from error_detection.normalizer import AmharicNormalizer
        normalizer=AmharicNormalizer()
        norm_fn=lambda w: _canonical_form(w, normalizer.normalize_word_strict)
    except Exception:
        norm_fn=lambda w: unicodedata.normalize('NFC', w) if w else w

    # Precompute caches for unique words in train+val
    unique_words=pd.concat([train_df['ocr_word'], val_df['ocr_word']]).unique()
    print(f"  Unique words in train+val: {len(unique_words):,}")

    # One-flip cache
    one_flip_cache={}
    for w in unique_words:
        cnt=0
        w_norm=norm_fn(w)
        for i,ch in enumerate(w):
            alts=_confusion_alts(ch)+reverse_map.get(ch,[])
            for alt in alts[:12]:
                cand=w[:i]+alt+w[i+1:]
                if cand==w:
                    continue
                cand_norm=norm_fn(cand)
                if cand_norm==w_norm:
                    continue
                if cand_norm in lexicon:
                    cnt+=1
                    break
            if cnt>=8:
                break
        one_flip_cache[w]=cnt

    # For training, ocr_conf is dummy 75.0 -> use NaN per roadmap (dead confidence)
    # doc_freq is available but make it optional NaN handling
    # char_kenlm is NaN (no KenLM)

    def build_features(dataframe):
        # Use canonical for freq lookup
        def get_freq(w):
            return lexicon.get(norm_fn(w), 0)

        df=dataframe.copy()
        df["corpus_freq"]=df["ocr_word"].map(get_freq)
        df["log_corpus_freq"]=df["corpus_freq"].map(lambda f: math.log10(f+1) if f>0 else 0.0)
        df["in_lexicon"]=(df["corpus_freq"]>0).astype(int)
        df["is_doc_singleton"]=(df["doc_freq"]==1).astype(int)
        df["token_len"]=df["ocr_word"].str.len()
        df["len_ratio"]=df["token_len"]/4.14
        df["one_flip_count"]=df["ocr_word"].map(one_flip_cache)
        df["has_one_flip"]=(df["one_flip_count"]>0).astype(int)
        # NaN for missing per roadmap
        df["char_kenlm_score"]=float('nan')
        df["rare_glyph_ratio"]=df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS)/float(len(w)) if len(w)>0 else 0)
        # Real conf is dummy -> NaN
        df["ocr_conf"]=float('nan')
        df["n_neighbors"]=0  # skip for speed, but could compute
        df["mixed_script"]=df["ocr_word"].apply(lambda w: 1 if any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in w) and any('\u1200' <= ch <= '\u137F' for ch in w) else 0)
        df["has_digit"]=df["ocr_word"].apply(lambda w: 1 if any(ch.isdigit() for ch in w) else 0)
        df["is_high_freq"]=(df["log_corpus_freq"]>3.0).astype(int)
        return df

    train_feat=build_features(train_df)
    val_feat=build_features(val_df)

    feature_cols=[
        "log_corpus_freq", "in_lexicon",
        "doc_freq", "is_doc_singleton",
        "token_len", "len_ratio",
        "has_one_flip", "one_flip_count",
        "char_kenlm_score", "rare_glyph_ratio",
        "ocr_conf", "n_neighbors",
        "mixed_script", "has_digit", "is_high_freq"
    ]

    X_train=train_feat[feature_cols]
    y_train=train_feat["is_error"].values
    X_val=val_feat[feature_cols]
    y_val=val_feat["is_error"].values

    print(f"\nFeature cols: {feature_cols}")

    # 6. Train LightGBM WITHOUT scale_pos_weight distortion
    print("\nTraining LightGBM (no scale_pos_weight, no floors)...")
    params={
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }

    trn_data=lgb.Dataset(X_train, label=y_train)
    val_data=lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model=lgb.train(
        params, trn_data, num_boost_round=800,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    print(f"  Best iter: {model.best_iteration}")

    # OOF on val
    val_pred=model.predict(X_val, num_iteration=model.best_iteration)
    auc=roc_auc_score(y_val, val_pred)
    ap=average_precision_score(y_val, val_pred)
    print(f"  Val ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")

    # 7. Isotonic calibration on val set
    print("\nCalibrating with Isotonic Regression on val set...")
    ir=IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_pred, y_val)
    val_cal=ir.predict(val_pred)
    auc_cal=roc_auc_score(y_val, val_cal)
    print(f"  Calibrated Val ROC-AUC: {auc_cal:.4f}")

    # 8. Tune thresholds on val set for cost-sensitive decision
    # We want high recall and low false flag
    # Try thresholds
    best_f1=0
    best_thresh=(0.20, 0.65)
    print("\nThreshold tuning on val set:")
    for pass_t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for flag_t in [0.50, 0.55, 0.60, 0.65, 0.70]:
            if pass_t>=flag_t:
                continue
            labels=np.where(val_cal>=flag_t, "FLAG", np.where(val_cal>=pass_t, "GRAY", "PASS"))
            err_mask=(y_val==1)
            cor_mask=(y_val==0)
            if err_mask.sum()==0 or cor_mask.sum()==0:
                continue
            recall=(labels[err_mask]!="PASS").mean()
            false_flag=(labels[cor_mask]=="FLAG").mean()
            # Cost: we want recall >=0.95 and false_flag <0.02, optimize F1-like
            # Use score = recall - 2*false_flag
            score=recall - 2*false_flag
            if score>best_f1:
                best_f1=score
                best_thresh=(pass_t, flag_t)
            print(f"  pass={pass_t:.2f} flag={flag_t:.2f} -> recall {recall*100:.1f}% false_flag {false_flag*100:.2f}% score {score:.3f}")

    print(f"\nBest thresholds: pass={best_thresh[0]:.2f} flag={best_thresh[1]:.2f} score={best_f1:.3f}")

    # Use best thresholds
    PASS_THRESH, FLAG_THRESH=best_thresh

    # 9. Train final model on train+val (1267 pages) for production
    print("\nTraining final production model on train+val (1267 pages)...")
    full_df=pd.concat([train_df, val_df])
    full_feat=build_features(full_df)
    X_full=full_feat[feature_cols]
    y_full=full_feat["is_error"].values

    final_trn=lgb.Dataset(X_full, label=y_full)
    final_model=lgb.train(params, final_trn, num_boost_round=model.best_iteration)

    # Recalibrate isotonic on full? Use val calibrator for now, or fit on val again
    # For final, we keep calibrator trained on val

    payload={
        "model": final_model,
        "features": feature_cols,
        "pass_thresh": PASS_THRESH,
        "flag_thresh": FLAG_THRESH,
        "calibrator": ir,
        "lexicon_path": LOCAL_LEXICON,
        "confusion_path": CONFUSION_PATH,
        "note": "v3 clean: no leakage, no scale_pos_weight, isotonic calib, NaN for missing"
    }
    joblib.dump(payload, MODEL_OUTPUT)
    print(f"\nClean v3 model saved to: {MODEL_OUTPUT}")

    # 10. Quick test on held-out test set (100 eval pages) with real conf? But test_df has dummy conf
    # For test we need real conf from Evaluation/OCRed, but we have only dummy in harmonized
    # We'll evaluate later with evaluate_v2 which uses real conf files

if __name__=="__main__":
    main()
