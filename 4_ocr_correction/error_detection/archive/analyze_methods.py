#!/usr/bin/env python3
"""
analyze_methods.py
------------------
Evaluates LightGBM OCR Error Detection on the 100 test pages.
Outputs: Evaluation/analysis.tsv
"""
import csv
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

try:
    from detector import DetectionEngine
except ImportError:
    from error_detection.detector import DetectionEngine

GT_DIR = os.path.join(_HERE, "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
CONF_DIR = OCR_DIR
CONFUSION = os.path.join(_HERE, "confusion_matrix.json")
OUTPUT_TSV = os.path.join(_HERE, "Evaluation", "analysis.tsv")
CHAR_LM_PATH = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/character_model/amharic_char_model.klm"
MASTER_DICT_PATH = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Data/dics_i7/main_i7.txt"

PUNCT_RE = re.compile(r'[^\w\s]|\d')

def clean_token(token): return PUNCT_RE.sub('', token).strip()
def char_edit_distance(a, b, _cache={}):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    key = (a, b)
    if key in _cache: return _cache[key]
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    _cache[key] = prev[lb]
    return prev[lb]

def align_tokens(ref_seq, hyp_seq):
    n, m = len(ref_seq), len(hyp_seq)
    WER_COST = 10000
    dp = [[(0, None)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = (dp[i-1][0][0] + WER_COST + len(ref_seq[i-1]), (i-1, 0))
    for j in range(1, m + 1): dp[0][j] = (dp[0][j-1][0] + WER_COST + len(hyp_seq[j-1]), (0, j-1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r_tok, h_tok = ref_seq[i-1], hyp_seq[j-1]
            cost_del = dp[i-1][j][0] + WER_COST + len(r_tok)
            cost_ins = dp[i][j-1][0] + WER_COST + len(h_tok)
            d = char_edit_distance(r_tok, h_tok)
            cost_sub = dp[i-1][j-1][0] if d == 0 else dp[i-1][j-1][0] + WER_COST + d
            min_c = min(cost_del, cost_ins, cost_sub)
            if min_c == cost_sub: dp[i][j] = (min_c, (i-1, j-1))
            elif min_c == cost_del: dp[i][j] = (min_c, (i-1, j))
            else: dp[i][j] = (min_c, (i, j-1))
    steps = []
    i, j = n, m
    n_ins = n_dels = 0
    while i > 0 or j > 0:
        pi, pj = dp[i][j][1]
        if pi == i - 1 and pj == j - 1: steps.append((i - 1, j - 1))
        elif pi == i - 1 and pj == j:
            steps.append((i - 1, None))
            n_dels += 1
        else:
            steps.append((None, j - 1))
            n_ins += 1
        i, j = pi, pj
    steps.reverse()
    return steps, n_ins, n_dels

def load_pages():
    pages = []
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".txt"): continue
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = next((os.path.join(OCR_DIR, base + s) for s in [".ocr.txt", ".txt"] if os.path.exists(os.path.join(OCR_DIR, base + s))), None)
        if not ocr_path: continue
        gt_text = open(os.path.join(GT_DIR, fname), encoding="utf-8").read().strip()
        ocr_text = open(ocr_path, encoding="utf-8").read().strip()
        conf_map = None
        conf_path = os.path.join(CONF_DIR, base + ".conf.tsv")
        if os.path.exists(conf_path):
            conf_map = {}
            idx = 0
            with open(conf_path, encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 2 and not line.startswith("word"):
                        try: conf_map[idx] = float(p[1])
                        except ValueError: pass
                        idx += 1
        pages.append((base, gt_text, ocr_text, conf_map))
    return pages

def main():
    pages = load_pages()
    if not pages: 
        print("No pages found!")
        return

    # Initialize ONLY ONCE using the permanent 4.9M master dictionary
    detector = DetectionEngine(
        dict_vocab_path=MASTER_DICT_PATH,
        confusion_matrix_path=CONFUSION,
        char_lm_path=CHAR_LM_PATH,
        verbose=True,
    )

    rows = []
    stats = {
        "pages": 0, "gt_words": 0, "ocr_words": 0,
        "total_aligned": 0, "substitutions": 0, "insertions": 0, "deletions": 0,
        "gt_chars": 0, "char_errors": 0,
        "correct_confs": [], "error_confs": [],
        "err_flag": 0, "err_gray": 0, "err_pass": 0,
        "cor_pass": 0, "cor_gray": 0, "cor_flag": 0,
    }

    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_raw, ocr_raw = gt_text.split(), ocr_text.split()
        gt_clean = [clean_token(t) for t in gt_raw]
        ocr_clean = [clean_token(t) for t in ocr_raw]

        gt_words = [(i, gt_raw[i], gt_clean[i]) for i in range(len(gt_raw)) if gt_clean[i]]
        ocr_words = [(j, ocr_raw[j], ocr_clean[j]) for j in range(len(ocr_raw)) if ocr_clean[j]]
        if not gt_words or not ocr_words: continue

        stats["pages"] += 1
        stats["gt_words"] += len(gt_words)
        stats["ocr_words"] += len(ocr_words)

        gt_seq = [w[2] for w in gt_words]
        ocr_seq = [w[2] for w in ocr_words]
        steps, n_ins, n_dels = align_tokens(gt_seq, ocr_seq)
        stats["insertions"] += n_ins
        stats["deletions"] += n_dels

        # Document-Level Word Frequency Map for this page
        page_doc_counts = Counter(ocr_seq)

        for ref_idx, hyp_idx in steps:
            if ref_idx is not None and hyp_idx is not None:
                gt_orig_idx, gt_orig, gt_c = gt_words[ref_idx]
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                is_match = (gt_c == ocr_c)

                if not is_match:
                    stats["char_errors"] += char_edit_distance(gt_c, ocr_c)
                    stats["substitutions"] += 1
                stats["gt_chars"] += len(gt_c)

                ocr_conf = conf_map.get(ocr_orig_idx) if conf_map else None
                ctx = [ocr_words[j][1] for j in range(max(0, hyp_idx - 2), hyp_idx)]
                doc_freq = page_doc_counts[ocr_c]

                # Run detector with LightGBM & document voting
                verdict = detector.verdict(ocr_orig, ctx, ocr_conf=ocr_conf, doc_freq=doc_freq)
                in_lex = detector._norm(ocr_c) in detector.vocab

                if not is_match:
                    is_error_val = 1
                    if ocr_conf is not None: stats["error_confs"].append(ocr_conf)
                    if verdict.label == "flag": stats["err_flag"] += 1
                    elif verdict.label == "gray": stats["err_gray"] += 1
                    else: stats["err_pass"] += 1
                else:
                    is_error_val = 0
                    if ocr_conf is not None: stats["correct_confs"].append(ocr_conf)
                    if verdict.label == "pass": stats["cor_pass"] += 1
                    elif verdict.label == "gray": stats["cor_gray"] += 1
                    else: stats["cor_flag"] += 1

                stats["total_aligned"] += 1
                rows.append({
                    "page": page_name, "ocr_word": ocr_c, "gt_word": gt_c,
                    "is_error": is_error_val, "detector_verdict": verdict.label,
                    "detector_score": f"{verdict.suspicion:.3f}",
                    "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                    "in_lexicon": 1 if in_lex else 0,
                    "one_flip": len(verdict.signals.one_flip_words),
                    "reason": verdict.reason,
                })

            elif ref_idx is not None:
                stats["gt_chars"] += len(gt_words[ref_idx][2])
                rows.append({
                    "page": page_name, "ocr_word": "", "gt_word": gt_words[ref_idx][2],
                    "is_error": 3, "detector_verdict": "", "detector_score": "",
                    "tesseract_conf": "", "in_lexicon": "", "one_flip": "", "reason": "DELETION",
                })
            elif hyp_idx is not None:
                ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
                ocr_conf = conf_map.get(ocr_orig_idx) if conf_map else None
                rows.append({
                    "page": page_name, "ocr_word": ocr_c, "gt_word": "",
                    "is_error": 2, "detector_verdict": "", "detector_score": "",
                    "tesseract_conf": f"{ocr_conf:.0f}" if ocr_conf is not None else "",
                    "in_lexicon": "", "one_flip": "", "reason": "INSERTION",
                })

    fieldnames = ["page", "ocr_word", "gt_word", "is_error", "detector_verdict", "detector_score", "tesseract_conf", "in_lexicon", "one_flip", "reason"]
    with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    total_errors = stats["substitutions"] + stats["insertions"] + stats["deletions"]
    wer = 100.0 * total_errors / max(stats["gt_words"], 1)
    cer = 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1)

    print("\n" + "=" * 65)
    print("Tesseract OCR Accuracy (PSM 3, amh_fast)")
    print("=" * 65)
    print(f"Pages:                    {stats['pages']}")
    print(f"GT words:                 {stats['gt_words']:,}")
    print(f"OCR words:                {stats['ocr_words']:,}")
    print(f"Aligned pairs:            {stats['total_aligned']:,}")
    print(f"Substitutions:            {stats['substitutions']:,}")
    print(f"WER:                      {wer:.2f}% | CER: {cer:.2f}%")
    print(f"Word accuracy:            {100 - wer:.2f}%")

    n_err = stats["substitutions"]
    caught = stats["err_flag"] + stats["err_gray"]

    print("\n" + "=" * 65)
    print(f"Production LightGBM Performance on Error Words ({n_err:,})")
    print("=" * 65)
    print(f"Caught (flag + gray):     {caught:,} ({100*caught/max(n_err,1):.1f}%)")
    print(f"  ├── FLAG (Definite):    {stats['err_flag']:,} ({100*stats['err_flag']/max(n_err,1):.1f}%)")
    print(f"  └── GRAY (Needs Word LM): {stats['err_gray']:,} ({100*stats['err_gray']/max(n_err,1):.1f}%)")
    print(f"Missed (passed):          {stats['err_pass']:,} ({100*stats['err_pass']/max(n_err,1):.1f}%)")

    n_cor = stats["total_aligned"] - stats["substitutions"]
    print("\n" + "=" * 65)
    print(f"Clean Words Verification (Correct: {n_cor:,})")
    print("=" * 65)
    print(f"Correctly Passed:         {stats['cor_pass']:,} ({100*stats['cor_pass']/max(n_cor,1):.1f}%)")
    print(f"Sent to Gray:             {stats['cor_gray']:,} ({100*stats['cor_gray']/max(n_cor,1):.1f}%)")
    print(f"False Flagged (alarm):    {stats['cor_flag']:,} ({100*stats['cor_flag']/max(n_cor,1):.2f}%)")

if __name__ == "__main__":
    main()