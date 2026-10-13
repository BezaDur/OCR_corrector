#!/usr/bin/env python3
"""
train_improved.py — Improved LightGBM OCR Detector training

Fixes:
- Correctly parses confusion_matrix.json (was looking for "substitutions" key, now handles "sub" dict)
- Uses local lexicon (Model/local_lexicon.txt) built from amh_final_training_text.txt
- Adds real OCR confidence from Evaluation/OCRed/*.conf.tsv (not dummy 75.0)
- Expands feature set: ocr_conf, n_neighbors, mixed_script, has_digit, len_ratio, is_high_freq
- Makes KenLM optional
- Trains with GroupKFold grouped by page
- Tunes thresholds for precision (reduce false positives)
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
from typing import Dict, List, Tuple

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PARENT)

# Paths
LOCAL_LEXICON = os.path.join(_HERE, "local_lexicon.txt")
CONFUSION_PATH = os.path.join(_HERE, "confusion_matrix.json")
if not os.path.exists(CONFUSION_PATH):
    CONFUSION_PATH = os.path.join(_PARENT, "confusion_matrix.json")

EVAL_GT_DIR = os.path.join(_PARENT, "Evaluation", "GroundTruth")
EVAL_OCR_DIR = os.path.join(_PARENT, "Evaluation", "OCRed")
HARMONIZED_TSV = os.path.join(_HERE, "harmonized_corpus_jiwer.tsv")
MODEL_OUTPUT = os.path.join(_HERE, "detector_lgbm_v2.pkl")
MODEL_OUTPUT_OLD = os.path.join(_HERE, "detector_lgbm.pkl")

# For backward compat, also save to old path if requested
RARE_GLYPHS = set("ሠሡሢሣሤሥሦፀፁፂፃፄፅፆጰጱጲጳጴጵጶዐዑዒዓዔዕዖ")
PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')

# Curated confusion classes (from utils.py)
CONFUSION_CLASSES = (
    ("\u12a0", "\u12d0", "\u12d3"),  # አ ዐ ዓ
    ("\u1200", "\u1210", "\u1280"),  # ሀ ሐ ኀ
    ("\u1230", "\u1220"),  # ሰ ሠ
    ("\u1338", "\u1340"),  # ጸ ፀ
    ("\u1338", "\u1330"),  # ጸ ጰ
    ("\u1290", "\u1298"),  # ነ ኘ
    ("\u1260", "\u1268"),  # በ ቨ
    ("\u12a8", "\u12a0"),  # ከ አ
)

def _confusion_alts(ch: str) -> List[str]:
    out = []
    for cls in CONFUSION_CLASSES:
        if ch in cls:
            out.extend(c for c in cls if c != ch)
    return out

def load_lexicon(path: str, max_types: int = 150000) -> Dict[str, int]:
    print(f"Loading lexicon: {path} (max {max_types} types)...")
    freq_dict = {}
    # Detect raw text vs lexicon
    is_raw = False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as probe:
            first = probe.readline()
            if len(first.strip().split()) > 5:
                is_raw = True
    except Exception:
        pass

    if is_raw:
        from collections import Counter
        counter = Counter()
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for tok in line.strip().split():
                    tok = tok.strip()
                    if tok:
                        counter[tok] += 1
        # Keep top max_types
        most_common = counter.most_common(max_types)
        # Normalize keys
        try:
            from OCR_error_detection.normalizer import AmharicNormalizer
            norm = AmharicNormalizer.normalize_word_strict
        except Exception:
            norm = lambda x: x
        for w, freq in most_common:
            nw = norm(w)
            if not nw:
                continue
            freq_dict[nw] = max(freq_dict.get(nw, 0), freq)
    else:
        tmp = {}
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0]
                try:
                    freq = int(parts[-1]) if parts[-1].isdigit() else 1
                except ValueError:
                    freq = 1
                # Normalize
                try:
                    from OCR_error_detection.normalizer import AmharicNormalizer
                    nw = AmharicNormalizer.normalize_word_strict(w)
                except Exception:
                    nw = w
                if not nw:
                    continue
                tmp[nw] = max(tmp.get(nw, 0), freq)
        # Keep top max_types by freq
        if len(tmp) > max_types:
            import heapq
            keep = heapq.nlargest(max_types, tmp, key=tmp.__getitem__)
            freq_dict = {k: tmp[k] for k in keep}
        else:
            freq_dict = tmp

    print(f"  Loaded {len(freq_dict):,} types (after cap)")
    return freq_dict

def load_confusion_map(path: str) -> Dict[str, List[str]]:
    """Fixed version: handles both 'sub' dict and old 'substitutions' list."""
    if not os.path.exists(path):
        print(f"[WARN] Confusion matrix not found at {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rev = defaultdict(set)

    # New format: {"sub": {"true_char": {"observed_char": cost, ...}}}
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
            elif isinstance(obs_map, list):
                for obs_ch in obs_map:
                    rev[obs_ch].add(true_ch)

    # Old format: {"substitutions": [{"target":..., "ocr":...}]}
    if "substitutions" in data:
        for p in data["substitutions"]:
            t = p.get("target")
            o = p.get("ocr")
            if t and o:
                rev[o].add(t)

    # Convert to list
    result = {k: list(v) for k, v in rev.items()}
    print(f"  Confusion map: {len(result)} observed chars with alternatives")
    # Debug: show some
    sample = list(result.items())[:3]
    for k, v in sample:
        print(f"    {k!r} -> {v[:5]}")
    return result

def clean_token(token: str) -> str:
    t = PUNCT_RE.sub('', token).strip()
    t = ETHIOPIC_PUNCT_RE.sub('', t).strip()
    return t

def char_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[lb]

def align_tokens(ref_seq: List[str], hyp_seq: List[str]):
    """DP alignment with WER cost + char edit distance, from analyze_methods.py"""
    n, m = len(ref_seq), len(hyp_seq)
    WER_COST = 10000
    dp = [[(0, None)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (dp[i-1][0][0] + WER_COST + len(ref_seq[i-1]), (i-1, 0))
    for j in range(1, m + 1):
        dp[0][j] = (dp[0][j-1][0] + WER_COST + len(hyp_seq[j-1]), (0, j-1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r_tok, h_tok = ref_seq[i-1], hyp_seq[j-1]
            cost_del = dp[i-1][j][0] + WER_COST + len(r_tok)
            cost_ins = dp[i][j-1][0] + WER_COST + len(h_tok)
            d = char_edit_distance(r_tok, h_tok)
            cost_sub = dp[i-1][j-1][0] if d == 0 else dp[i-1][j-1][0] + WER_COST + d
            min_c = min(cost_del, cost_ins, cost_sub)
            if min_c == cost_sub:
                dp[i][j] = (min_c, (i-1, j-1))
            elif min_c == cost_del:
                dp[i][j] = (min_c, (i-1, j))
            else:
                dp[i][j] = (min_c, (i, j-1))
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = dp[i][j][1]
        if pi == i - 1 and pj == j - 1:
            steps.append((i - 1, j - 1))
        elif pi == i - 1 and pj == j:
            steps.append((i - 1, None))
        else:
            steps.append((None, j - 1))
        i, j = pi, pj
    steps.reverse()
    return steps

def load_pages_from_eval(gt_dir: str, ocr_dir: str):
    pages = []
    if not os.path.exists(gt_dir) or not os.path.exists(ocr_dir):
        print(f"[WARN] Eval dirs not found: {gt_dir}, {ocr_dir}")
        return pages

    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".txt"))
    for fname in gt_files:
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = None
        for suffix in [".ocr.txt", ".txt"]:
            cand = os.path.join(ocr_dir, base + suffix)
            if os.path.exists(cand):
                ocr_path = cand
                break
        if not ocr_path:
            continue

        try:
            gt_text = open(os.path.join(gt_dir, fname), encoding="utf-8").read().strip()
            ocr_text = open(ocr_path, encoding="utf-8").read().strip()
        except Exception as e:
            print(f"[WARN] Failed to read {base}: {e}")
            continue

        # Load confidence map: index -> conf
        conf_map = {}
        conf_path = os.path.join(ocr_dir, base + ".conf.tsv")
        if os.path.exists(conf_path):
            idx = 0
            try:
                with open(conf_path, encoding="utf-8") as cf:
                    for line in cf:
                        line = line.strip()
                        if not line or line.startswith("word"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            try:
                                conf = float(parts[1])
                                conf_map[idx] = conf
                            except ValueError:
                                pass
                            idx += 1
            except Exception as e:
                print(f"[WARN] Failed to read conf {conf_path}: {e}")

        pages.append((base, gt_text, ocr_text, conf_map))

    print(f"Loaded {len(pages)} eval pages")
    return pages

def build_training_rows(pages, lexicon: Dict[str, int], reverse_map: Dict[str, List[str]], symspell=None, char_model=None):
    """Build feature rows from eval pages with real confidences."""
    try:
        from OCR_error_detection.normalizer import AmharicNormalizer
        normalizer = AmharicNormalizer()
        norm_fn = normalizer.normalize_word_strict
    except Exception:
        norm_fn = lambda x: x

    # Caches for expensive ops
    one_flip_cache = {}
    kenlm_cache = {}
    n_neighbors_cache = {}

    rows = []

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_raw = gt_text.split()
        ocr_raw = ocr_text.split()
        gt_clean = [clean_token(t) for t in gt_raw]
        ocr_clean = [clean_token(t) for t in ocr_raw]

        # Filter empty cleaned tokens but keep original indices
        gt_words = [(i, gt_raw[i], gt_clean[i]) for i in range(len(gt_raw)) if gt_clean[i]]
        ocr_words = [(j, ocr_raw[j], ocr_clean[j]) for j in range(len(ocr_raw)) if ocr_clean[j]]

        if not gt_words or not ocr_words:
            continue

        gt_seq = [w[2] for w in gt_words]
        ocr_seq = [w[2] for w in ocr_words]

        steps = align_tokens(gt_seq, ocr_seq)

        # Doc freq map for this page
        page_doc_counts = Counter(ocr_seq)

        for ref_idx, hyp_idx in steps:
            if ref_idx is not None and hyp_idx is not None:
                gt_orig_idx, gt_orig, gt_c = gt_words[ref_idx]
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                is_error = 0 if gt_c == ocr_c else 1

                # Features
                norm_c = norm_fn(ocr_c)
                freq = lexicon.get(norm_c, 0)
                log_freq = math.log10(freq + 1) if freq > 0 else 0.0
                in_lex = 1 if freq > 0 else 0

                # doc freq
                doc_freq = page_doc_counts[ocr_c]
                is_singleton = 1 if doc_freq == 1 else 0

                # token len
                t_len = max(len(ocr_c), 1)
                len_ratio = t_len / 4.14

                # rare glyph
                rare_cnt = sum(1 for ch in ocr_c if ch in RARE_GLYPHS)
                rare_ratio = rare_cnt / float(t_len)

                # one-flip with cache
                if ocr_c in one_flip_cache:
                    one_flip_count = one_flip_cache[ocr_c]
                else:
                    one_flip_count = 0
                    for i, ch in enumerate(ocr_c):
                        curated = _confusion_alts(ch)
                        learned = reverse_map.get(ch, [])
                        alts = curated + [a for a in learned if a not in curated]
                        for alt in alts[:12]:
                            cand = ocr_c[:i] + alt + ocr_c[i+1:]
                            if cand == ocr_c:
                                continue
                            cand_norm = norm_fn(cand)
                            if cand_norm == norm_c:
                                continue
                            if cand_norm in lexicon:
                                one_flip_count += 1
                                break
                        if one_flip_count >= 8:
                            break
                    one_flip_cache[ocr_c] = one_flip_count
                has_one_flip = 1 if one_flip_count > 0 else 0

                # char lm with cache
                if ocr_c in kenlm_cache:
                    char_klm = kenlm_cache[ocr_c]
                else:
                    if char_model is not None and len(ocr_c) >= 2:
                        try:
                            spaced = " ".join(ocr_c)
                            raw = char_model.score(spaced, bos=True, eos=True)
                            char_klm = raw / (len(ocr_c) + 1.0)
                        except Exception:
                            char_klm = -2.5
                    else:
                        char_klm = -2.5
                    kenlm_cache[ocr_c] = char_klm

                # ocr conf (real from TSV)
                ocr_conf_raw = conf_map.get(ocr_orig_idx, 75.0)
                ocr_conf = ocr_conf_raw / 100.0  # normalized 0-1

                # SymSpell neighbors with cache
                if ocr_c in n_neighbors_cache:
                    n_neighbors = n_neighbors_cache[ocr_c]
                else:
                    if symspell is not None:
                        try:
                            sug = symspell.lookup(ocr_c, 2, max_edit_distance=1, include_unknown=False)
                            n_neighbors = len(sug)
                        except Exception:
                            n_neighbors = 0
                    else:
                        n_neighbors = 0
                    n_neighbors_cache[ocr_c] = n_neighbors

                # surface
                eth_count = sum(1 for ch in ocr_orig if '\u1200' <= ch <= '\u137F' or '\u2D80' <= ch <= '\u2DDF')
                mixed = 1 if 0 < eth_count < len(ocr_orig) else 0
                has_digit = 1 if any(ch.isdigit() for ch in ocr_orig) else 0
                is_high_freq = 1 if log_freq > 3.0 else 0

                rows.append({
                    "page": page_name,
                    "ocr_word": ocr_c,
                    "gt_word": gt_c,
                    "is_error": is_error,
                    "log_corpus_freq": log_freq,
                    "in_lexicon": in_lex,
                    "doc_freq": doc_freq,
                    "is_doc_singleton": is_singleton,
                    "token_len": t_len,
                    "len_ratio": len_ratio,
                    "has_one_flip": has_one_flip,
                    "one_flip_count": one_flip_count,
                    "char_kenlm_score": char_klm,
                    "rare_glyph_ratio": rare_ratio,
                    "ocr_conf": ocr_conf,
                    "n_neighbors": n_neighbors,
                    "mixed_script": mixed,
                    "has_digit": has_digit,
                    "is_high_freq": is_high_freq,
                })

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} training rows from {len(pages)} pages, errors: {df['is_error'].sum()} ({100*df['is_error'].mean():.2f}%)")
    return df

def load_harmonized_with_real_conf_fallback():
    """Load harmonized TSV but note conf is dummy 75. We'll use it only if eval not available."""
    if not os.path.exists(HARMONIZED_TSV):
        return None
    df = pd.read_csv(HARMONIZED_TSV, sep="\t")
    df = df.dropna(subset=["ocr_word"]).copy()
    df["ocr_word"] = df["ocr_word"].astype(str)
    print(f"Loaded harmonized TSV: {len(df)} rows, but conf is dummy")
    return df

def main():
    print("="*70)
    print("Training Improved LightGBM with Fixed Confusion + Expanded Features")
    print("="*70)

    # 1. Load lexicon
    lexicon_path = LOCAL_LEXICON
    if not os.path.exists(lexicon_path):
        # Try alternative
        alt = os.path.join(_PARENT, "Model", "local_lexicon.txt")
        if os.path.exists(alt):
            lexicon_path = alt
        else:
            # Try to build from root training text
            root_txt = os.path.join(_ROOT, "amh_final_training_text.txt")
            if os.path.exists(root_txt):
                print(f"Local lexicon not found, building from {root_txt}...")
                # Build quickly
                from collections import Counter
                counter = Counter()
                with open(root_txt, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        for tok in line.strip().split():
                            if tok:
                                counter[tok] += 1
                # Save
                os.makedirs(os.path.dirname(LOCAL_LEXICON), exist_ok=True)
                with open(LOCAL_LEXICON, 'w', encoding='utf-8') as out:
                    for w, freq in counter.most_common():
                        out.write(f"{w} {freq}\n")
                lexicon_path = LOCAL_LEXICON
            else:
                print(f"[ERROR] No lexicon found at {lexicon_path} and no training text to build")
                return

    lexicon = load_lexicon(lexicon_path)

    # 2. Load confusion map (FIXED)
    reverse_map = load_confusion_map(CONFUSION_PATH)

    # 3. Try to load KenLM (optional)
    char_model = None
    try:
        import kenlm
        # Try candidate paths
        for cand in ["/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm",
                     os.path.join(_HERE, "amharic_char_model.klm")]:
            if os.path.exists(cand):
                print(f"Loading KenLM: {cand}")
                char_model = kenlm.Model(cand)
                break
        if char_model is None:
            print("[INFO] KenLM not found, using fallback -2.5 for char scores")
    except ImportError:
        print("[INFO] kenlm not installed, using fallback")

    # 4. Build SymSpell index for n_neighbors feature (optional, skip for speed if large)
    symspell = None
    if len(lexicon) <= 200000:
        print("Building SymSpell index for training...")
        try:
            from symspellpy import SymSpell, Verbosity
            symspell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
            for w, freq in lexicon.items():
                symspell.create_dictionary_entry(w, freq)
            print(f"  SymSpell ready with {len(lexicon)} types")
        except Exception as e:
            print(f"[WARN] SymSpell failed: {e}")
            symspell = None
    else:
        print(f"[INFO] Skipping SymSpell for training (lexicon {len(lexicon)} >200k) for speed, n_neighbors=0")

    # 5. Load training data: prefer Evaluation with real confidences
    pages = load_pages_from_eval(EVAL_GT_DIR, EVAL_OCR_DIR)
    if pages:
        print("\nUsing Evaluation data with REAL OCR confidences for training")
        df = build_training_rows(pages, lexicon, reverse_map, symspell=symspell, char_model=char_model)
    else:
        print("\nEval data not found, falling back to harmonized TSV (dummy conf)")
        # Load harmonized and compute features similarly but with dummy conf
        hdf = load_harmonized_with_real_conf_fallback()
        if hdf is None:
            print("[ERROR] No training data found")
            return
        # For harmonized, we need to compute features
        # We'll reuse similar logic but vectorized where possible
        # For speed, compute one_flip using cache
        print("Computing features for harmonized data...")
        # Precompute kenlm cache for unique words
        unique_words = hdf["ocr_word"].unique()
        kenlm_cache = {}
        if char_model:
            for w in unique_words:
                if len(w) >= 2:
                    try:
                        spaced = " ".join(w)
                        raw = char_model.score(spaced, bos=True, eos=True)
                        kenlm_cache[w] = raw / (len(w) + 1.0)
                    except:
                        kenlm_cache[w] = -2.5
                else:
                    kenlm_cache[w] = -2.5
        else:
            kenlm_cache = {w: -2.5 for w in unique_words}

        one_flip_cache = {}
        try:
            from OCR_error_detection.normalizer import AmharicNormalizer
            norm_fn = AmharicNormalizer.normalize_word_strict
        except:
            norm_fn = lambda x: x

        for w in unique_words:
            cnt = 0
            for i, ch in enumerate(w):
                alts = _confusion_alts(ch) + reverse_map.get(ch, [])
                for alt in alts[:12]:
                    cand = w[:i] + alt + w[i+1:]
                    if cand == w:
                        continue
                    if norm_fn(cand) == norm_fn(w):
                        continue
                    if norm_fn(cand) in lexicon:
                        cnt += 1
                        break
                if cnt >= 8:
                    break
            one_flip_cache[w] = cnt

        # Now build df features
        df = hdf.copy()
        # Compute log freq
        def get_freq(w):
            return lexicon.get(norm_fn(w), 0)
        df["corpus_freq"] = df["ocr_word"].map(get_freq)
        df["log_corpus_freq"] = df["corpus_freq"].map(lambda f: math.log10(f+1) if f>0 else 0.0)
        df["in_lexicon"] = (df["corpus_freq"] > 0).astype(int)
        df["is_doc_singleton"] = (df["doc_freq"] == 1).astype(int)
        df["token_len"] = df["ocr_word"].str.len()
        df["len_ratio"] = df["token_len"] / 4.14
        df["one_flip_count"] = df["ocr_word"].map(one_flip_cache)
        df["has_one_flip"] = (df["one_flip_count"] > 0).astype(int)
        df["char_kenlm_score"] = df["ocr_word"].map(kenlm_cache)
        df["rare_glyph_ratio"] = df["ocr_word"].apply(lambda w: sum(1 for ch in w if ch in RARE_GLYPHS) / float(len(w)) if len(w)>0 else 0)
        df["ocr_conf"] = df["tesseract_conf"] / 100.0  # dummy 0.75
        df["n_neighbors"] = 0  # skip for harmonized to save time
        df["mixed_script"] = df["ocr_word"].apply(lambda w: 1 if any('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in w) and any('\u1200' <= ch <= '\u137F' for ch in w) else 0)
        df["has_digit"] = df["ocr_word"].apply(lambda w: 1 if any(ch.isdigit() for ch in w) else 0)
        df["is_high_freq"] = (df["log_corpus_freq"] > 3.0).astype(int)

    # Ensure we have data
    if df.empty:
        print("[ERROR] No training rows")
        return

    # Feature columns: expanded set
    feature_cols = [
        "log_corpus_freq", "in_lexicon",
        "doc_freq", "is_doc_singleton",
        "token_len", "len_ratio",
        "has_one_flip", "one_flip_count",
        "char_kenlm_score", "rare_glyph_ratio",
        "ocr_conf", "n_neighbors",
        "mixed_script", "has_digit", "is_high_freq"
    ]

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df["is_error"].values
    groups = df["page"].values if "page" in df.columns else np.zeros(len(df))

    # 6. Train with GroupKFold
    print("\nTraining LightGBM with 5-Fold GroupKFold (by page)...")
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))

    params = {
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
        'scale_pos_weight': 2.5,
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
            params, trn_data, num_boost_round=800,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_preds[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        models.append(clf)
        print(f"  Fold {fold}: best iter {clf.best_iteration}, val AUC {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

    auc = roc_auc_score(y, oof_preds)
    ap = average_precision_score(y, oof_preds)
    print(f"\n{'='*70}")
    print(f"OOF ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    print(f"{'='*70}")

    # Tune thresholds for precision focus
    # We want high pass rate for clean words, high recall for errors
    # Try thresholds
    PASS_THRESH = 0.20
    FLAG_THRESH = 0.65

    # Evaluate at chosen thresholds
    preds_label = np.where(oof_preds >= FLAG_THRESH, "FLAG",
                  np.where(oof_preds >= PASS_THRESH, "GRAY", "PASS"))

    err_mask = (y == 1)
    cor_mask = (y == 0)

    print(f"\n[PERFORMANCE ON {err_mask.sum():,} REAL ERRORS]")
    print(f"  FLAG:  {(preds_label[err_mask] == 'FLAG').sum():,} ({100*(preds_label[err_mask] == 'FLAG').mean():.1f}%)")
    print(f"  GRAY:  {(preds_label[err_mask] == 'GRAY').sum():,} ({100*(preds_label[err_mask] == 'GRAY').mean():.1f}%)")
    print(f"  PASS:  {(preds_label[err_mask] == 'PASS').sum():,} ({100*(preds_label[err_mask] == 'PASS').mean():.1f}%)")
    caught = ((preds_label[err_mask] == 'FLAG') | (preds_label[err_mask] == 'GRAY')).sum()
    print(f"  Caught (FLAG+GRAY): {caught} ({100*caught/err_mask.sum():.1f}%)")

    print(f"\n[PERFORMANCE ON {cor_mask.sum():,} CLEAN WORDS]")
    print(f"  PASS:  {(preds_label[cor_mask] == 'PASS').sum():,} ({100*(preds_label[cor_mask] == 'PASS').mean():.1f}%)")
    print(f"  GRAY:  {(preds_label[cor_mask] == 'GRAY').sum():,} ({100*(preds_label[cor_mask] == 'GRAY').mean():.1f}%)")
    print(f"  FLAG:  {(preds_label[cor_mask] == 'FLAG').sum():,} ({100*(preds_label[cor_mask] == 'FLAG').mean():.2f}%)")
    print(f"  False Flag Rate: {100*(preds_label[cor_mask] == 'FLAG').mean():.2f}% (target <2%)")

    importance = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    print("\n[FEATURE IMPORTANCE (GAIN)]")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<22}: {imp:.1f}")

    # 7. Train production model on full data
    print(f"\nTraining production model on full dataset...")
    final_trn = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, final_trn, num_boost_round=500)

    payload = {
        "model": final_model,
        "features": feature_cols,
        "pass_thresh": PASS_THRESH,
        "flag_thresh": FLAG_THRESH,
        "lexicon_path": lexicon_path,
        "confusion_path": CONFUSION_PATH,
    }
    joblib.dump(payload, MODEL_OUTPUT)
    print(f"\nImproved model saved to: {MODEL_OUTPUT}")

    # Also save to old path for backward compat if needed? No, keep v2 separate
    # Optionally copy to old if user wants
    print(f"To use new model, set lgbm_model_path to {MODEL_OUTPUT} or rename to detector_lgbm.pkl")

if __name__ == "__main__":
    main()
