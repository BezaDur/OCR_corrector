#!/usr/bin/env python3
"""
train_v4.py — Final model combining best of v2 and v3
- Uses all 1367 pages (no leakage concern for final production, but we will still evaluate on 100 eval held-out with real conf for verification)
- Real conf for eval pages (100), NaN for others (1267)
- No scale_pos_weight, no manual floors
- Isotonic calibration
- Saves as detector_lgbm_v4.pkl and also as v3 (overwrite) for default
"""

import os, re, json, math, unicodedata, joblib, numpy as np, pandas as pd
from collections import Counter, defaultdict

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

_HERE=os.path.dirname(os.path.abspath(__file__))
LOCAL_LEXICON=os.path.join(_HERE, "local_lexicon.txt")
CONFUSION_PATH=os.path.join(_HERE, "confusion_matrix.json")
HARMONIZED_TSV=os.path.join(_HERE, "harmonized_corpus_jiwer.tsv")
MODEL_OUTPUT=os.path.join(_HERE, "detector_lgbm_v4.pkl")
MODEL_V3_OUTPUT=os.path.join(_HERE, "detector_lgbm_v3.pkl")

EVAL_GT_DIR=os.path.join(os.path.dirname(_HERE), "Evaluation", "GroundTruth")
EVAL_OCR_DIR=os.path.join(os.path.dirname(_HERE), "Evaluation", "OCRed")

RARE_GLYPHS=set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")
PUNCT_RE=re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE=re.compile(r'[፡።፣፤፥፦፧፨]')
CONFUSION_CLASSES=(
    ("\u12a0", "\u12d0", "\u12d3"),
    ("\u1200", "\u1210", "\u1280"),
    ("\u1230", "\u1220"),
    ("\u1338", "\u1340"),
    ("\u1338", "\u1330"),
    ("\u1290", "\u1298"),
    ("\u1260", "\u1268"),
    ("\u12a8", "\u12a0"),
)
def _confusion_alts(ch):
    out=[]
    for cls in CONFUSION_CLASSES:
        if ch in cls:
            out.extend(c for c in cls if c!=ch)
    return out
def _canonical_form(word, norm_fn):
    if not word: return word
    try:
        nfc=unicodedata.normalize('NFC', word)
    except:
        nfc=word
    try:
        return norm_fn(nfc)
    except:
        return nfc

def load_lexicon(path, max_types=150000):
    print(f"Loading lexicon {path} max {max_types}...")
    freq_dict={}
    is_raw=False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as probe:
            first=probe.readline()
            if len(first.strip().split())>5:
                is_raw=True
    except: pass
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
            from OCR_error_detection.normalizer import AmharicNormalizer
            norm=AmharicNormalizer.normalize_word_strict
        except:
            norm=lambda x:x
        for w,freq in most_common:
            nw=_canonical_form(w, norm)
            if not nw: continue
            freq_dict[nw]=max(freq_dict.get(nw,0), freq)
    else:
        tmp={}
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                parts=line.strip().split()
                if not parts: continue
                w=parts[0]
                try:
                    freq=int(parts[-1]) if parts[-1].isdigit() else 1
                except: freq=1
                try:
                    from OCR_error_detection.normalizer import AmharicNormalizer
                    norm=AmharicNormalizer.normalize_word_strict
                    nw=_canonical_form(w, norm)
                except:
                    nw=w
                if not nw: continue
                tmp[nw]=max(tmp.get(nw,0), freq)
        if len(tmp)>max_types:
            import heapq
            keep=heapq.nlargest(max_types, tmp, key=tmp.__getitem__)
            freq_dict={k:tmp[k] for k in keep}
        else:
            freq_dict=tmp
    print(f"  Loaded {len(freq_dict):,} types")
    return freq_dict

def load_confusion_map(path):
    if not os.path.exists(path):
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
                    except: c=4.5
                    if c<=4.0 and true_ch!=obs_ch:
                        rev[obs_ch].add(true_ch)
    result={k:list(v) for k,v in rev.items()}
    print(f"  Confusion map {len(result)} chars")
    return result

def clean_token(t):
    t=PUNCT_RE.sub('', t).strip()
    t=ETHIOPIC_PUNCT_RE.sub('', t).strip()
    return t

def load_real_confs():
    # Load real confs for eval pages from Evaluation/OCRed
    real_confs={}
    if not os.path.exists(EVAL_OCR_DIR):
        return real_confs
    for fname in os.listdir(EVAL_OCR_DIR):
        if not fname.endswith(".conf.tsv"):
            continue
        base=fname.replace(".conf.tsv","")
        path=os.path.join(EVAL_OCR_DIR, fname)
        # Need mapping from ocr_word index to conf, but we need per row in harmonized?
        # Harmonized has page, ocr_word, doc_freq but no index. We can approximate by page+word
        # For simplicity, we will build a dict page -> list of (word, conf) in order
        # Then later we can map by page and doc position? This is approximate
        # Instead, we will directly use the conf files during feature building for eval pages
        # For harmonized, we will replace dummy conf with real conf by page average or by lookup
        # For now, load page -> word -> conf list
        try:
            with open(path, encoding='utf-8') as f:
                confs=[]
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("word"):
                        continue
                    parts=line.split("\t")
                    if len(parts)>=2:
                        try:
                            w=parts[0]
                            c=float(parts[1])
                            confs.append((w,c))
                        except: pass
                real_confs[base]=confs
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    print(f"Loaded real confs for {len(real_confs)} pages")
    return real_confs

def main():
    print("="*70)
    print("Training Final v4 — All 1367 pages, real conf for eval, no floors, isotonic")
    print("="*70)

    lexicon=load_lexicon(LOCAL_LEXICON, max_types=150000)
    reverse_map=load_confusion_map(CONFUSION_PATH)
    real_confs=load_real_confs()

    print(f"Loading {HARMONIZED_TSV}...")
    df=pd.read_csv(HARMONIZED_TSV, sep="\t")
    df=df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"]=df["ocr_word"].astype(str)
    print(f"  Total rows {len(df):,}, pages {df['page'].nunique():,}")

    # For real conf: for eval pages, we have real conf list, but harmonized rows are not in same order as conf.tsv?
    # The conf.tsv order matches ocr_output order, which should match harmonized order per page? Let's assume same order
    # We will replace tesseract_conf for eval pages with real confs from real_confs dict by position

    # Build mapping page -> list of real confs in order
    # For each eval page, get list of confs
    eval_pages=set([f"page_{i:04d}" for i in range(1,101)])
    # For training, we will create a new column real_conf
    df["real_conf"]=float('nan')
    for page in eval_pages:
        if page in real_confs:
            conf_list=real_confs[page]
            # conf_list is list of (word, conf) in order of ocr_output
            # Get rows for this page in df, in order
            mask=df['page']==page
            page_rows=df[mask]
            # If lengths match, assign
            if len(page_rows)==len(conf_list):
                # Assign confs
                conf_vals=[c for w,c in conf_list]
                df.loc[mask, "real_conf"]=conf_vals
            else:
                # Approximate by word matching: for each row, find first matching word conf
                # Build word->conf dict (first occurrence)
                word_to_conf={}
                for w,c in conf_list:
                    if w not in word_to_conf:
                        word_to_conf[w]=c
                # Map
                df.loc[mask, "real_conf"]=df.loc[mask, "ocr_word"].map(word_to_conf).fillna(75.0)
        else:
            # No real conf
            pass

    # For non-eval pages, real_conf stays NaN per roadmap
    # For eval pages where we couldn't map, fallback to 75? But we have mapping

    # Feature engineering
    try:
        from OCR_error_detection.normalizer import AmharicNormalizer
        normalizer=AmharicNormalizer()
        norm_fn=lambda w: _canonical_form(w, normalizer.normalize_word_strict)
    except Exception:
        norm_fn=lambda w: unicodedata.normalize('NFC', w) if w else w

    unique_words=df['ocr_word'].unique()
    print(f"Unique words {len(unique_words):,}")
    one_flip_cache={}
    for w in unique_words:
        cnt=0
        w_norm=norm_fn(w)
        for i,ch in enumerate(w):
            alts=_confusion_alts(ch)+reverse_map.get(ch,[])
            for alt in alts[:12]:
                cand=w[:i]+alt+w[i+1:]
                if cand==w: continue
                cand_norm=norm_fn(cand)
                if cand_norm==w_norm: continue
                if cand_norm in lexicon:
                    cnt+=1
                    break
            if cnt>=8:
                break
        one_flip_cache[w]=cnt

    def build_features(dataframe):
        def get_freq(w):
            return lexicon.get(norm_fn(w),0)
        df=dataframe.copy()
        df["corpus_freq"]=df["ocr_word"].map(get_freq)
        df["log_corpus_freq"]=df["corpus_freq"].map(lambda f: math.log10(f+1) if f>0 else 0.0)
        df["in_lexicon"]=(df["corpus_freq"]>0).astype(int)
        df["is_doc_singleton"]=(df["doc_freq"]==1).astype(int)
        df["token_len"]=df["ocr_word"].str.len()
        df["len_ratio"]=df["token_len"]/4.14
        df["one_flip_count"]=df["ocr_word"].map(one_flip_cache)
        df["has_one_flip"]=(df["one_flip_count"]>0).astype(int)
        df["char_kenlm_score"]=float('nan')
        df["rare_glyph_ratio"]=df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS)/float(len(w)) if len(w)>0 else 0)
        # Real conf where available, else NaN
        df["ocr_conf"]=df["real_conf"].apply(lambda x: x/100.0 if not pd.isna(x) else float('nan'))
        df["n_neighbors"]=0
        df["mixed_script"]=df["ocr_word"].apply(lambda w: 1 if any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in w) and any('\u1200' <= ch <= '\u137F' for ch in w) else 0)
        df["has_digit"]=df["ocr_word"].apply(lambda w: 1 if any(ch.isdigit() for ch in w) else 0)
        df["is_high_freq"]=(df["log_corpus_freq"]>3.0).astype(int)
        return df

    # Split: hold out 100 eval pages as test, train on 1267
    train_val_df=df[~df['page'].isin(eval_pages)].copy()
    test_df=df[df['page'].isin(eval_pages)].copy()

    # Further split train_val into train 1200 and val 67 for calibration
    unique_pages=train_val_df['page'].unique()
    np.random.seed(42)
    shuffled=np.random.permutation(unique_pages)
    train_pages=set(shuffled[:1200])
    val_pages=set(shuffled[1200:])

    train_df=train_val_df[train_val_df['page'].isin(train_pages)].copy()
    val_df=train_val_df[train_val_df['page'].isin(val_pages)].copy()

    print(f"Train rows {len(train_df):,}, Val rows {len(val_df):,}, Test rows {len(test_df):,}")

    train_feat=build_features(train_df)
    val_feat=build_features(val_df)
    test_feat=build_features(test_df)

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

    model=lgb.train(params, trn_data, num_boost_round=800, valid_sets=[trn_data, val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"Best iter {model.best_iteration}")
    val_pred=model.predict(X_val, num_iteration=model.best_iteration)
    print(f"Val ROC-AUC {roc_auc_score(y_val, val_pred):.4f} PR-AUC {average_precision_score(y_val, val_pred):.4f}")

    # Isotonic calibration
    ir=IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_pred, y_val)
    val_cal=ir.predict(val_pred)
    print(f"Calibrated Val ROC-AUC {roc_auc_score(y_val, val_cal):.4f}")

    # Tune thresholds
    best=(0.10, 0.65)
    best_score=-1
    for pass_t in [0.10, 0.15, 0.20]:
        for flag_t in [0.50, 0.55, 0.60, 0.65, 0.70]:
            if pass_t>=flag_t: continue
            labels=np.where(val_cal>=flag_t, "FLAG", np.where(val_cal>=pass_t, "GRAY", "PASS"))
            recall=(labels[y_val==1]!="PASS").mean()
            false_flag=(labels[y_val==0]=="FLAG").mean()
            score=recall - 2*false_flag
            if score>best_score:
                best_score=score
                best=(pass_t, flag_t)
    print(f"Best thresh {best} score {best_score}")

    PASS_THRESH, FLAG_THRESH=best

    # Final model on train+val (1267 pages) - no test
    full_df=pd.concat([train_df, val_df])
    full_feat=build_features(full_df)
    X_full=full_feat[feature_cols]
    y_full=full_feat["is_error"].values
    final_trn=lgb.Dataset(X_full, label=y_full)
    final_model=lgb.train(params, final_trn, num_boost_round=model.best_iteration)

    payload={
        "model": final_model,
        "features": feature_cols,
        "pass_thresh": PASS_THRESH,
        "flag_thresh": FLAG_THRESH,
        "calibrator": ir,
        "note": "v4 final: 1267 train pages NaN conf, isotonic, no floors, no scale_pos_weight"
    }
    joblib.dump(payload, MODEL_OUTPUT)
    print(f"Saved to {MODEL_OUTPUT}")

    # Also overwrite v3 as default authoritative
    joblib.dump(payload, MODEL_V3_OUTPUT)
    print(f"Also saved to {MODEL_V3_OUTPUT} as default v3")

    # Quick test on held-out test set with real conf
    X_test=test_feat[feature_cols]
    y_test=test_feat["is_error"].values
    raw_test=final_model.predict(X_test)
    cal_test=ir.predict(raw_test)
    for pass_t, flag_t in [(0.10,0.65), (0.20,0.65), best]:
        labels=np.where(cal_test>=flag_t, "FLAG", np.where(cal_test>=pass_t, "GRAY", "PASS"))
        recall=(labels[y_test==1]!="PASS").mean()*100
        false_flag=(labels[y_test==0]=="FLAG").mean()*100
        pass_rate=(labels[y_test==0]=="PASS").mean()*100
        print(f"Test (100 eval, real conf where mapped) pass={pass_t} flag={flag_t} -> recall {recall:.1f}% pass {pass_rate:.1f}% false_flag {false_flag:.2f}%")

if __name__=="__main__":
    main()
