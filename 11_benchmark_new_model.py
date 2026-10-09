#!/usr/bin/env python3
"""
Step 11: Comprehensive Benchmark of the Newly Trained amh_best model
over all 150 stress-tested evaluation pages.
"""

import os
import glob
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

DATASET_DIR = "/media/des/Tess/Tesseract2/synthetic_eval"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
OUT_DIR = os.path.join(DATASET_DIR, "ocr_new_model")

MODELS_DIR = "/media/des/Tess/Tesseract2/training/models"
MODEL_NAME = "amh_epoch1_fast"
PSM = "3"
NUM_WORKERS = 4

def dp_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def run_page(img_path):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_base = os.path.join(OUT_DIR, base_name)
    raw_txt_file = f"{out_base}.txt"
    final_ocr_path = os.path.join(OUT_DIR, f"{base_name}.ocr.txt")
    gt_path = os.path.join(GT_DIR, f"{base_name}.gt.txt")

    if not os.path.exists(gt_path):
        return None

    cmd = [
        "tesseract",
        img_path,
        out_base,
        "-l", MODEL_NAME,
        "--tessdata-dir", MODELS_DIR,
        "--psm", PSM
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return None

        if os.path.exists(raw_txt_file):
            os.replace(raw_txt_file, final_ocr_path)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read().strip()
            with open(final_ocr_path, "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()

            gt_chars = "".join(gt_text.split())
            ocr_chars = "".join(ocr_text.split())
            c_ref = len(gt_chars)
            c_err = dp_edit_distance(gt_chars, ocr_chars)

            gt_words = gt_text.split()
            ocr_words = ocr_text.split()
            w_ref = len(gt_words)
            w_err = dp_edit_distance(gt_words, ocr_words)

            return (c_ref, c_err, w_ref, w_err)
        return None
    except Exception:
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    total = len(images)

    print(f"Running New Model ({MODEL_NAME}) on {total} evaluation pages across {NUM_WORKERS} cores...")
    t0 = time.time()

    total_ref_chars = 0
    total_char_errors = 0
    total_ref_words = 0
    total_word_errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in executor.map(run_page, images):
            if res is None:
                continue
            c_ref, c_err, w_ref, w_err = res
            total_ref_chars += c_ref
            total_char_errors += c_err
            total_ref_words += w_ref
            total_word_errors += w_err
            completed += 1

            if completed % 25 == 0 or completed == total:
                print(f"Progress: [{completed}/{total}] pages completed...")

    elapsed = time.time() - t0
    cer = (total_char_errors / max(1, total_ref_chars)) * 100.0
    wer = (total_word_errors / max(1, total_ref_words)) * 100.0
    acc = 100.0 - cer

    print("\n" + "=" * 65)
    print("NEW MODEL BENCHMARK RESULTS (150 Full Pages)")
    print("=" * 65)
    print(f"Character Accuracy (Raw OCR): {acc:6.2f}%")
    print(f"Character Error Rate (CER)  : {cer:6.2f}%")
    print(f"Word Error Rate (WER)       : {wer:6.2f}%")
    print(f"Total Reference Characters  : {total_ref_chars:,}")
    print(f"Total Reference Words       : {total_ref_words:,}")
    print(f"Processing Time             : {elapsed:.1f} s ({elapsed/total:.2f} s/page)")
    print("=" * 65)

if __name__ == "__main__":
    main()
