#!/usr/bin/env python3
"""
Step 6: End-to-end evaluation benchmark.
"""
import os
import glob
import time
import argparse
import json
from amharic_canonical_normalizer import normalize_for_metric

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
    """Exact Levenshtein using O(min(m,n)) memory; accepts text or word lists."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, value in enumerate(s1, 1):
        current = [i]
        for j, other in enumerate(s2, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (value != other)))
        previous = current
    return previous[-1]


_corrector = None
CORRECTOR_OPTIONS = dict(beam_width=5, max_optical_cost=4.8, lambda_optical=0.9,
                         lambda_lm=2.0, lambda_morph=1.5, max_candidates=25)
CANONICAL_METRICS = False


def build_corrector():
    return AmharicOCRCorrector(
        kenlm_path=KENLM_BIN, confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE, **CORRECTOR_OPTIONS,
    )


def init_worker(config=None):
    global _corrector, KENLM_BIN, CONFUSION_JSON, DICT_FILE
    global RAW_OCR_DIR, CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS
    if config is not None:
        (KENLM_BIN, CONFUSION_JSON, DICT_FILE, RAW_OCR_DIR, CORRECTED_OUT_DIR,
         CORRECTOR_OPTIONS, CANONICAL_METRICS) = config
    _corrector = build_corrector()


def process_single_page(gt_path):
    global _corrector
    base_name = os.path.basename(gt_path).replace(".gt.txt", "")
    raw_ocr_path = os.path.join(RAW_OCR_DIR, f"{base_name}.ocr.txt")
    corrected_path = os.path.join(CORRECTED_OUT_DIR, f"{base_name}.corr.txt")
    if not os.path.exists(raw_ocr_path):
        raise FileNotFoundError(raw_ocr_path)

    t0 = time.time()
    print(f"  -> {base_name}", flush=True)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()
    with open(raw_ocr_path, "r", encoding="utf-8") as f:
        raw_ocr_text = f.read().strip()

    correction_start = time.perf_counter()
    corrected_text = _corrector.correct_text_block(raw_ocr_text)
    correction_seconds = time.perf_counter() - correction_start
    token_count = len(raw_ocr_text.split())

    os.makedirs(CORRECTED_OUT_DIR, exist_ok=True)
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_text + "\n")

    if CANONICAL_METRICS:
        gt_text = normalize_for_metric(gt_text)
        raw_ocr_text = normalize_for_metric(raw_ocr_text)
        corrected_text = normalize_for_metric(corrected_text)

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
    return (c_ref, raw_c_err, corr_c_err, w_ref, raw_w_err, corr_w_err,
            correction_seconds, token_count)


def main():
    os.makedirs(CORRECTED_OUT_DIR, exist_ok=True)
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    total_pages = len(gt_files)
    if total_pages == 0:
        raise FileNotFoundError(f"No Ground Truth files found in {GT_DIR}")

    missing = [path for path in gt_files if not os.path.exists(os.path.join(
        RAW_OCR_DIR, os.path.basename(path).replace(".gt.txt", ".ocr.txt")))]
    if missing:
        raise FileNotFoundError(f"Missing raw OCR for {len(missing)} of {total_pages} pages; first: {missing[0]}")
    print("Correction settings:", json.dumps(CORRECTOR_OPTIONS, sort_keys=True), flush=True)
    print("Metric mode:", "canonical" if CANONICAL_METRICS else "strict", flush=True)
    print("Inputs:", KENLM_BIN, CONFUSION_JSON, DICT_FILE, RAW_OCR_DIR, flush=True)
    init_start = time.perf_counter()

    # Smoke-test one sentence before touching the whole eval set.
    # Comment this out once you have seen a corrected line printed.
    print("Smoke-testing one dummy sentence...", flush=True)
    probe = build_corrector() if not USE_MP else None

    t_start = time.time()
    totals = [0, 0, 0, 0, 0, 0, 0.0, 0]
    if probe is not None:
        print(f"Model initialization: {time.perf_counter() - init_start:.2f}s", flush=True)
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
        worker_config = (KENLM_BIN, CONFUSION_JSON, DICT_FILE, RAW_OCR_DIR,
                         CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker,
                                 initargs=(worker_config,)) as ex:
            for res in ex.map(process_single_page, gt_files, chunksize=1):
                accumulate(res)
    else:
        print(f"Sequential CPU benchmark on {total_pages} pages...", flush=True)
        global _corrector
        _corrector = probe if probe is not None else build_corrector()
        if probe is not None:
            print("Smoke test:", _corrector.correct_sentence("እንደምን አደርክ ።"), flush=True)
            _corrector.clear_caches()  # Start page evaluation without smoke-test cache entries.
        for gt_path in gt_files:
            accumulate(process_single_page(gt_path))

    elapsed_time = time.time() - t_start
    total_ref_chars, total_raw_char_errors, total_corrected_char_errors, \
        total_ref_words, total_raw_word_errors, total_corrected_word_errors, \
        correction_seconds, token_count = totals

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
    print(f"Total Pages Processed:      {completed}/{total_pages}")
    print(f"Correction-only worker time: {correction_seconds:.2f}s "
          f"({token_count / max(correction_seconds, 1e-9):.1f} input tokens/s)")
    if not USE_MP:
        print("Candidate cache:", _corrector._generate_candidates_cached.cache_info())
        print("Morphology cache:", _corrector._validate_morphology_cached.cache_info())
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


def configure_from_cli():
    global GT_DIR, RAW_OCR_DIR, CORRECTED_OUT_DIR, KENLM_BIN, CONFUSION_JSON, DICT_FILE
    global USE_MP, NUM_WORKERS, CANONICAL_METRICS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", default=GT_DIR)
    parser.add_argument("--raw-dir", default=RAW_OCR_DIR)
    parser.add_argument("--output-dir", default=CORRECTED_OUT_DIR)
    parser.add_argument("--kenlm", default=KENLM_BIN)
    parser.add_argument("--confusion", default=CONFUSION_JSON)
    parser.add_argument("--dictionary", default=DICT_FILE)
    parser.add_argument("--backend", choices=("trie", "symspell"), default="trie")
    parser.add_argument("--trie-search", choices=("adaptive", "full"), default="adaptive")
    parser.add_argument("--symspell-distance", type=int, default=1)
    parser.add_argument("--max-optical-cost", type=float, default=4.8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--canonical-metrics", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    GT_DIR, RAW_OCR_DIR, CORRECTED_OUT_DIR = args.gt_dir, args.raw_dir, args.output_dir
    KENLM_BIN, CONFUSION_JSON, DICT_FILE = args.kenlm, args.confusion, args.dictionary
    NUM_WORKERS, USE_MP = args.workers, args.workers > 1
    CANONICAL_METRICS = args.canonical_metrics
    CORRECTOR_OPTIONS.update(candidate_backend=args.backend, trie_search=args.trie_search,
                             symspell_max_edit_distance=args.symspell_distance,
                             max_optical_cost=args.max_optical_cost, score_eos=not args.no_eos)


if __name__ == "__main__":
    configure_from_cli()
    main()