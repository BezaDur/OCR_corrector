#!/usr/bin/env python3
"""
Step 6: End-to-end evaluation benchmark.
"""
import os
import sys
import glob
import time

from amharic_ocr_corrector import AmharicOCRCorrector

DATASET_DIR = "/media/des/Tess/Tesseract2/synthetic_eval"
GT_DIR = os.path.join(DATASET_DIR, "ground_truth")
RAW_OCR_DIR = os.path.join(DATASET_DIR, "ocr_new_model")
CORRECTED_OUT_DIR = os.path.join(DATASET_DIR, "ocr_corrected")
KENLM_BIN = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary"
CONFUSION_JSON = "/media/des/Tess/Tesseract2/confusion_matrix.json"
DICT_FILE = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt"

# Old machine: start sequential. Each worker loads KenLM + 5M vocab (~1GB+).
USE_MP = False
NUM_WORKERS = 2


def dp_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


_corrector = None


def build_corrector():
    return AmharicOCRCorrector(
        kenlm_path=KENLM_BIN,
        confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE,
        beam_width=5,
        max_optical_cost=4.8,
        lambda_optical=0.9,
        lambda_lm=2.0,
        lambda_morph=1.5,
        max_candidates=25,
    )


def init_worker():
    global _corrector
    _corrector = AmharicOCRCorrector(
        kenlm_path=KENLM_BIN,
        confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE,
        beam_width=4,
        max_optical_cost=4.0,
        lambda_optical=1.2,
        lambda_lm=1.8,
        lambda_morph=1.5
    )


def process_single_page(gt_path):
    global _corrector
    base_name = os.path.basename(gt_path).replace(".gt.txt", "")
    raw_ocr_path = os.path.join(RAW_OCR_DIR, f"{base_name}.ocr.txt")
    corrected_path = os.path.join(CORRECTED_OUT_DIR, f"{base_name}.corr.txt")
    if not os.path.exists(raw_ocr_path):
        return None

    t0 = time.time()
    print(f"  -> {base_name}", flush=True)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()
    with open(raw_ocr_path, "r", encoding="utf-8") as f:
        raw_ocr_text = f.read().strip()

    corrected_text = _corrector.correct_text_block(raw_ocr_text)

    os.makedirs(CORRECTED_OUT_DIR, exist_ok=True)
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_text + "\n")

    gt_chars = "".join(gt_text.split())
    raw_chars = "".join(raw_ocr_text.split())
    corr_chars = "".join(corrected_text.split())
    c_ref = len(gt_chars)
    raw_c_err = dp_edit_distance(gt_chars, raw_chars)
    corr_c_err = dp_edit_distance(gt_chars, corr_chars)

    gt_words = gt_text.split()
    raw_words = raw_ocr_text.split()
    corr_words = corrected_text.split()
    w_ref = len(gt_words)
    raw_w_err = dp_edit_distance(gt_words, raw_words)
    corr_w_err = dp_edit_distance(gt_words, corr_words)

    dt = time.time() - t0
    print(f"     done {base_name} in {dt:.2f}s", flush=True)
    return (c_ref, raw_c_err, corr_c_err, w_ref, raw_w_err, corr_w_err)


def main():
    os.makedirs(CORRECTED_OUT_DIR, exist_ok=True)
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    total_pages = len(gt_files)
    if total_pages == 0:
        print("Error: No Ground Truth files found.", flush=True)
        return

    # Smoke-test one sentence before touching the whole eval set.
    # Comment this out once you have seen a corrected line printed.
    print("Smoke-testing one dummy sentence...", flush=True)
    probe = build_corrector() if not USE_MP else None

    t_start = time.time()
    totals = [0, 0, 0, 0, 0, 0]
    completed = 0

    def accumulate(res):
        nonlocal completed
        if res is None:
            return
        for i, v in enumerate(res):
            totals[i] += v
        completed += 1
        print(f"Progress: [{completed}/{total_pages}]", flush=True)

    if USE_MP:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Multi-core: {total_pages} pages, {NUM_WORKERS} workers", flush=True)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as ex:
            for res in ex.map(process_single_page, gt_files, chunksize=1):
                accumulate(res)
    else:
        print(f"Sequential CPU benchmark on {total_pages} pages...", flush=True)
        global _corrector
        _corrector = probe if probe is not None else build_corrector()
        if probe is not None:
            print("Smoke test:", _corrector.correct_sentence("እንደምን አደርክ ።"), flush=True)
        for gt_path in gt_files:
            accumulate(process_single_page(gt_path))

    elapsed_time = time.time() - t_start
    total_ref_chars, total_raw_char_errors, total_corrected_char_errors, \
        total_ref_words, total_raw_word_errors, total_corrected_word_errors = totals

    raw_cer = (total_raw_char_errors / max(1, total_ref_chars)) * 100.0
    corr_cer = (total_corrected_char_errors / max(1, total_ref_chars)) * 100.0
    raw_wer = (total_raw_word_errors / max(1, total_ref_words)) * 100.0
    corr_wer = (total_corrected_word_errors / max(1, total_ref_words)) * 100.0
    char_error_reduction = (
        (total_raw_char_errors - total_corrected_char_errors)
        / max(1, total_raw_char_errors)
    ) * 100.0

    print("\n" + "=" * 70)
    print("FINAL OCR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Reference Characters: {total_ref_chars:,}")
    print(f"Total Reference Words:      {total_ref_words:,}")
    print(f"Total Pages Processed:      {total_pages}")
    print(f"Total Processing Time:      {elapsed_time:.1f} s "
          f"({elapsed_time / max(1, total_pages):.2f} s/page on CPU)")
    print("-" * 70)
    print(f"{'METRIC':<25} | {'RAW TESSERACT':<18} | {'WITH POST-CORRECTION':<18}")
    print("-" * 70)
    print(f"{'Character Error Rate (CER)':<25} | {raw_cer:6.2f}%            | {corr_cer:6.2f}%")
    print(f"{'Character Accuracy':<25} | {100.0 - raw_cer:6.2f}%            | {100.0 - corr_cer:6.2f}%")
    print(f"{'Word Error Rate (WER)':<25} | {raw_wer:6.2f}%            | {corr_wer:6.2f}%")
    print("-" * 70)
    print(f"Relative Error Reduction:   {char_error_reduction:.2f}% of raw OCR errors corrected!")
    print("=" * 70)


if __name__ == "__main__":
    main()