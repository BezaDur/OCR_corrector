#!/usr/bin/env python3
"""
evaluate_v2.py — Evaluate improved detector v2 on Evaluation set with real confidences
"""

import os
import re
import sys
import csv
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from error_detection.detector import DetectionEngine

GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONFUSION = os.path.join(_HERE, "Model", "confusion_matrix.json")
if not os.path.exists(CONFUSION):
    CONFUSION = os.path.join(_HERE, "confusion_matrix.json")

LOCAL_LEXICON = os.path.join(_HERE, "Model", "local_lexicon.txt")
MODEL_V2 = os.path.join(_HERE, "Model", "detector_lgbm_v2.pkl")
MODEL_V1 = os.path.join(_HERE, "Model", "detector_lgbm.pkl")

PUNCT_RE = re.compile(r'[^\w\s]|\d')
ETHIOPIC_PUNCT_RE = re.compile(r'[፡።፣፤፥፦፧፨]')

def clean_token(t):
    t = PUNCT_RE.sub('', t).strip()
    t = ETHIOPIC_PUNCT_RE.sub('', t).strip()
    return t

def char_edit_distance(a,b):
    if a==b: return 0
    la,lb=len(a),len(b)
    prev=list(range(lb+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[lb]

def align_tokens(ref_seq, hyp_seq):
    n,m=len(ref_seq),len(hyp_seq)
    WER_COST=10000
    dp=[[(0,None)]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        dp[i][0]=(dp[i-1][0][0]+WER_COST+len(ref_seq[i-1]), (i-1,0))
    for j in range(1,m+1):
        dp[0][j]=(dp[0][j-1][0]+WER_COST+len(hyp_seq[j-1]), (0,j-1))
    for i in range(1,n+1):
        for j in range(1,m+1):
            r_tok,h_tok=ref_seq[i-1],hyp_seq[j-1]
            cost_del=dp[i-1][j][0]+WER_COST+len(r_tok)
            cost_ins=dp[i][j-1][0]+WER_COST+len(h_tok)
            d=char_edit_distance(r_tok,h_tok)
            cost_sub=dp[i-1][j-1][0] if d==0 else dp[i-1][j-1][0]+WER_COST+d
            min_c=min(cost_del,cost_ins,cost_sub)
            if min_c==cost_sub:
                dp[i][j]=(min_c,(i-1,j-1))
            elif min_c==cost_del:
                dp[i][j]=(min_c,(i-1,j))
            else:
                dp[i][j]=(min_c,(i,j-1))
    steps=[]
    i,j=n,m
    while i>0 or j>0:
        pi,pj=dp[i][j][1]
        if pi==i-1 and pj==j-1:
            steps.append((i-1,j-1))
        elif pi==i-1 and pj==j:
            steps.append((i-1,None))
        else:
            steps.append((None,j-1))
        i,j=pi,pj
    steps.reverse()
    return steps

def load_pages():
    pages=[]
    gt_files=sorted(f for f in os.listdir(GT_DIR) if f.endswith(".txt"))
    for fname in gt_files:
        base=fname.replace(".gt.txt","").replace(".txt","")
        ocr_path=None
        for suffix in [".ocr.txt",".txt"]:
            cand=os.path.join(OCR_DIR, base+suffix)
            if os.path.exists(cand):
                ocr_path=cand
                break
        if not ocr_path:
            continue
        gt_text=open(os.path.join(GT_DIR,fname),encoding='utf-8').read().strip()
        ocr_text=open(ocr_path,encoding='utf-8').read().strip()
        conf_map={}
        conf_path=os.path.join(OCR_DIR, base+".conf.tsv")
        if os.path.exists(conf_path):
            idx=0
            with open(conf_path,encoding='utf-8') as cf:
                for line in cf:
                    line=line.strip()
                    if not line or line.startswith("word"): continue
                    parts=line.split("\t")
                    if len(parts)>=2:
                        try:
                            conf=float(parts[1])
                            conf_map[idx]=conf
                        except: pass
                        idx+=1
        pages.append((base, gt_text, ocr_text, conf_map))
    return pages

def evaluate(detector, pages):
    stats={
        "pages":0, "gt_words":0, "ocr_words":0,
        "total_aligned":0, "substitutions":0,
        "gt_chars":0, "char_errors":0,
        "err_flag":0, "err_gray":0, "err_pass":0,
        "cor_pass":0, "cor_gray":0, "cor_flag":0,
    }
    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_raw=gt_text.split()
        ocr_raw=ocr_text.split()
        gt_clean=[clean_token(t) for t in gt_raw]
        ocr_clean=[clean_token(t) for t in ocr_raw]
        gt_words=[(i,gt_raw[i],gt_clean[i]) for i in range(len(gt_raw)) if gt_clean[i]]
        ocr_words=[(j,ocr_raw[j],ocr_clean[j]) for j in range(len(ocr_raw)) if ocr_clean[j]]
        if not gt_words or not ocr_words:
            continue
        stats["pages"]+=1
        stats["gt_words"]+=len(gt_words)
        stats["ocr_words"]+=len(ocr_words)
        gt_seq=[w[2] for w in gt_words]
        ocr_seq=[w[2] for w in ocr_words]
        steps=align_tokens(gt_seq, ocr_seq)
        page_doc_counts=Counter(ocr_seq)
        for ref_idx, hyp_idx in steps:
            if ref_idx is not None and hyp_idx is not None:
                gt_orig_idx, gt_orig, gt_c = gt_words[ref_idx]
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                is_match=(gt_c==ocr_c)
                if not is_match:
                    stats["char_errors"]+=char_edit_distance(gt_c, ocr_c)
                    stats["substitutions"]+=1
                stats["gt_chars"]+=len(gt_c)
                ocr_conf=conf_map.get(ocr_orig_idx)
                doc_freq=page_doc_counts[ocr_c]
                verdict=detector.verdict(ocr_orig, [], ocr_conf=ocr_conf, doc_freq=doc_freq)
                if not is_match:
                    if verdict.label=="flag": stats["err_flag"]+=1
                    elif verdict.label=="gray": stats["err_gray"]+=1
                    else: stats["err_pass"]+=1
                else:
                    if verdict.label=="pass": stats["cor_pass"]+=1
                    elif verdict.label=="gray": stats["cor_gray"]+=1
                    else: stats["cor_flag"]+=1
                stats["total_aligned"]+=1
    return stats

def main():
    pages=load_pages()
    print(f"Loaded {len(pages)} pages")

    # Evaluate V2
    print("\n=== Evaluating V2 Model ===")
    det_v2=DetectionEngine(
        dict_vocab_path=LOCAL_LEXICON,
        confusion_matrix_path=CONFUSION,
        lgbm_model_path=MODEL_V2,
        max_lexicon_types=150000,
        verbose=False
    )
    stats_v2=evaluate(det_v2, pages)
    print_stats(stats_v2, "V2")

    # Evaluate V1 if exists
    if os.path.exists(MODEL_V1):
        print("\n=== Evaluating V1 Model (old) ===")
        det_v1=DetectionEngine(
            dict_vocab_path=LOCAL_LEXICON,
            confusion_matrix_path=CONFUSION,
            lgbm_model_path=MODEL_V1,
            max_lexicon_types=150000,
            verbose=False
        )
        stats_v1=evaluate(det_v1, pages)
        print_stats(stats_v1, "V1")

        # Comparison
        print("\n=== Comparison ===")
        print(f"{'Metric':<30} {'V1':<15} {'V2':<15} {'Delta':<10}")
        def fmt(a,b):
            return f"{b-a:+.1f}%"
        v1_pass_rate = 100*stats_v1["cor_pass"]/max(1, stats_v1["cor_pass"]+stats_v1["cor_gray"]+stats_v1["cor_flag"])
        v2_pass_rate = 100*stats_v2["cor_pass"]/max(1, stats_v2["cor_pass"]+stats_v2["cor_gray"]+stats_v2["cor_flag"])
        v1_flag_rate = 100*stats_v1["cor_flag"]/max(1, stats_v1["cor_pass"]+stats_v1["cor_gray"]+stats_v1["cor_flag"])
        v2_flag_rate = 100*stats_v2["cor_flag"]/max(1, stats_v2["cor_pass"]+stats_v2["cor_gray"]+stats_v2["cor_flag"])
        v1_recall = 100*(stats_v1["err_flag"]+stats_v1["err_gray"])/max(1, stats_v1["substitutions"])
        v2_recall = 100*(stats_v2["err_flag"]+stats_v2["err_gray"])/max(1, stats_v2["substitutions"])
        print(f"{'Clean Pass Rate':<30} {v1_pass_rate:<15.1f} {v2_pass_rate:<15.1f} {v2_pass_rate-v1_pass_rate:+.1f}")
        print(f"{'False Flag Rate':<30} {v1_flag_rate:<15.2f} {v2_flag_rate:<15.2f} {v2_flag_rate-v1_flag_rate:+.2f}")
        print(f"{'Error Recall (flag+gray)':<30} {v1_recall:<15.1f} {v2_recall:<15.1f} {v2_recall-v1_recall:+.1f}")

def print_stats(stats, name):
    total_err=stats["substitutions"]
    total_clean=stats["total_aligned"]-total_err
    wer=100*total_err/max(stats["gt_words"],1)
    cer=100*stats["char_errors"]/max(stats["gt_chars"],1)
    print(f"\n{name} Results:")
    print(f"  Pages: {stats['pages']}, GT words: {stats['gt_words']}, OCR words: {stats['ocr_words']}, Aligned: {stats['total_aligned']}")
    print(f"  WER: {wer:.2f}%, CER: {cer:.2f}%, Word Acc: {100-wer:.2f}%")
    print(f"  Errors ({total_err}): FLAG {stats['err_flag']} ({100*stats['err_flag']/max(total_err,1):.1f}%), GRAY {stats['err_gray']} ({100*stats['err_gray']/max(total_err,1):.1f}%), PASS {stats['err_pass']} ({100*stats['err_pass']/max(total_err,1):.1f}%)")
    caught=stats['err_flag']+stats['err_gray']
    print(f"    Caught: {caught} ({100*caught/max(total_err,1):.1f}%)")
    print(f"  Clean ({total_clean}): PASS {stats['cor_pass']} ({100*stats['cor_pass']/max(total_clean,1):.1f}%), GRAY {stats['cor_gray']} ({100*stats['cor_gray']/max(total_clean,1):.1f}%), FLAG {stats['cor_flag']} ({100*stats['cor_flag']/max(total_clean,1):.2f}%)")

if __name__=="__main__":
    main()
