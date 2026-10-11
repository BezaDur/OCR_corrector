#!/usr/bin/env python3
"""
Step 6: End-to-end evaluation benchmark.
"""
import os
import glob
import time
import argparse
import json
from pathlib import Path
from ocr_metrics import COUNT_FIELDS, edit_distance, score_texts, percentages
from amharic_canonical_normalizer import normalize_for_metric

from amharic_ocr_corrector import AmharicOCRCorrector

MORPHEMES_JSON = None
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


dp_edit_distance = edit_distance  # Backward-compatible import for existing callers.


_corrector = None
# NOTE: these must match AmharicOCRCorrector's constructor at HEAD. The older
# `candidate_backend` / `trie_search` / `symspell_max_edit_distance` options
# belonged to a previous revision that is no longer in the tree; passing them
# raises TypeError. See docs/architecture_review.md section 2.
CORRECTOR_OPTIONS = dict(beam_width=5, max_optical_cost=4.8, lambda_optical=0.9,
                         lambda_lm=2.0, lambda_morph=1.5, max_candidates=25)
MORPHEMES_JSON = None
CANONICAL_METRICS = False
JSON_REPORT = None
EXPECTED_PAGES = None
MAX_DAMAGE_RATE = None


def build_corrector():
    return AmharicOCRCorrector(
        kenlm_path=KENLM_BIN, confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE, morphemes_json_path=MORPHEMES_JSON,
        **CORRECTOR_OPTIONS,
    )


def init_worker(config=None):
    global _corrector, KENLM_BIN, CONFUSION_JSON, DICT_FILE, MORPHEMES_JSON
    global RAW_OCR_DIR, CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS
    if config is not None:
        (KENLM_BIN, CONFUSION_JSON, DICT_FILE, MORPHEMES_JSON, RAW_OCR_DIR,
         CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS) = config
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

    strict = score_texts(gt_text, raw_ocr_text, corrected_text)
    texts = (gt_text, raw_ocr_text, corrected_text)
    normalized = tuple(normalize_for_metric(text) for text in texts)
    canonical = strict.copy() if normalized == texts else score_texts(*normalized)
    dt = time.time() - t0
    print(f"     done {base_name} in {dt:.2f}s (correction {correction_seconds:.3f}s)", flush=True)
    return dict(page=base_name, strict=strict, canonical=canonical,
                correction_seconds=correction_seconds, input_tokens=token_count)


def main():
    if Path(CORRECTED_OUT_DIR).resolve() in (Path(RAW_OCR_DIR).resolve(), Path(GT_DIR).resolve()):
        raise ValueError('Use a separate corrected-output directory, not raw OCR or ground truth')
    os.makedirs(CORRECTED_OUT_DIR, exist_ok=True)
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.gt.txt")))
    total_pages = len(gt_files)
    if total_pages == 0:
        raise FileNotFoundError(f"No Ground Truth files found in {GT_DIR}")

    if EXPECTED_PAGES is not None and total_pages != EXPECTED_PAGES:
        raise ValueError(f"Expected {EXPECTED_PAGES} pages, found {total_pages}")
    missing = [path for path in gt_files if not os.path.exists(os.path.join(
        RAW_OCR_DIR, os.path.basename(path).replace(".gt.txt", ".ocr.txt")))]
    if missing:
        raise FileNotFoundError(f"Missing raw OCR for {len(missing)} of {total_pages} pages; first: {missing[0]}")
    print("Correction settings:", json.dumps(CORRECTOR_OPTIONS, sort_keys=True), flush=True)
    print("Reporting strict AND canonical metrics; headline mode:",
          "canonical" if CANONICAL_METRICS else "strict", flush=True)
    print("Inputs:", KENLM_BIN, CONFUSION_JSON, DICT_FILE, RAW_OCR_DIR, flush=True)
    init_start = time.perf_counter()

    probe = build_corrector() if not USE_MP else None
    initialization_seconds = time.perf_counter() - init_start if probe is not None else None
    t_start = time.perf_counter()
    totals = {mode: dict.fromkeys(COUNT_FIELDS, 0) for mode in ('strict', 'canonical')}
    pages = []
    if probe is not None:
        print(f"Model initialization: {initialization_seconds:.2f}s", flush=True)

    def accumulate(res):
        if res is None:
            raise RuntimeError('Unexpected missing page result')
        for mode in totals:
            for key in COUNT_FIELDS:
                totals[mode][key] += res[mode][key]
        pages.append(res)
        print(f"Progress: [{len(pages)}/{total_pages}]", flush=True)

    if USE_MP:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Multi-core: {total_pages} pages, {NUM_WORKERS} workers", flush=True)
        worker_config = (KENLM_BIN, CONFUSION_JSON, DICT_FILE, MORPHEMES_JSON,
                         RAW_OCR_DIR, CORRECTED_OUT_DIR, CORRECTOR_OPTIONS,
                         CANONICAL_METRICS)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker,
                                 initargs=(worker_config,)) as ex:
            for res in ex.map(process_single_page, gt_files, chunksize=1):
                accumulate(res)
    else:
        print(f"Sequential CPU benchmark on {total_pages} pages...", flush=True)
        global _corrector
        _corrector = probe if probe is not None else build_corrector()
        for gt_path in gt_files:
            accumulate(process_single_page(gt_path))

    elapsed_time = time.perf_counter() - t_start
    report = summarize(totals, pages, elapsed_time, initialization_seconds)
    if JSON_REPORT:
        path = Path(JSON_REPORT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if MAX_DAMAGE_RATE is not None:
        mode = 'canonical' if CANONICAL_METRICS else 'strict'
        upper = report['metrics'][mode]['rates']['clean_token_damage_rate_upper']
        if upper is None or upper > MAX_DAMAGE_RATE:
            raise RuntimeError(f"{mode} clean-token damage gate failed: {upper}; limit {MAX_DAMAGE_RATE}%")
    return report


def summarize(totals, pages, elapsed_time, initialization_seconds):
    correction_seconds = sum(page['correction_seconds'] for page in pages)
    token_count = sum(page['input_tokens'] for page in pages)
    report = dict(
        metrics={mode: dict(counts=counts, rates=percentages(counts)) for mode, counts in totals.items()},
        pages=pages, completed_pages=len(pages), expected_pages=EXPECTED_PAGES,
        correction_worker_seconds=correction_seconds, evaluation_wall_seconds=elapsed_time,
        initialization_seconds=initialization_seconds,
        correction_tokens_per_worker_second=token_count / correction_seconds if correction_seconds else None,
        morphology=(_corrector.morphology_stats()
                    if (not USE_MP and _corrector is not None) else None),
        options=CORRECTOR_OPTIONS.copy(), workers=NUM_WORKERS if USE_MP else 1,
        damage_gate_mode='canonical' if CANONICAL_METRICS else 'strict',
        max_clean_damage_rate=MAX_DAMAGE_RATE,
        inputs=dict(kenlm=KENLM_BIN, dictionary=DICT_FILE, confusion=CONFUSION_JSON,
                    ground_truth=GT_DIR, raw_ocr=RAW_OCR_DIR),
        damage_definition='Gold tokens matching in every co-optimal raw alignment; min/max losses '
                          'across co-optimal corrected alignments. Not causal provenance.',
    )
    print('\nFINAL OCR BENCHMARK RESULTS')
    print(f"Pages: {len(pages)}; correction-only worker time: {correction_seconds:.3f}s; "
          f"evaluation wall time (including metrics/I/O): {elapsed_time:.3f}s")
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}%'
    for mode, result in report['metrics'].items():
        counts, rates = result['counts'], result['rates']
        print(f"[{mode.upper()}] CER raw/corrected: {fmt(rates['raw_cer'])} / {fmt(rates['corrected_cer'])}")
        print(f"Character accuracy raw/corrected: {fmt(rates['raw_character_accuracy'])} / "
              f"{fmt(rates['corrected_character_accuracy'])}")
        print(f"WER raw/corrected: {fmt(rates['raw_wer'])} / {fmt(rates['corrected_wer'])}")
        print(f"Character error reduction: {fmt(rates['character_error_reduction'])}")
        print(f"Inserted words raw/corrected (selected optimal alignment): "
              f"{counts['raw_insertions']} / {counts['corrected_insertions']}")
        print(f"Clean-token damage (alignment-derived): {fmt(rates['clean_token_damage_rate_lower'])} "
              f"to {fmt(rates['clean_token_damage_rate_upper'])}; "
              f"damaged {counts['damaged_clean_tokens']}..{counts['max_damaged_clean_tokens']} / "
              f"{counts['clean_reference_tokens']} clean reference tokens; "
              f"ambiguous raw positions excluded: {counts['ambiguous_raw_reference_tokens']}")
    print('Correction tokens/worker-second:', report['correction_tokens_per_worker_second'])
    if not USE_MP and _corrector is not None:
        print('Candidate cache:', _corrector._generate_candidates_cached.cache_info())
        print('Morphology cache:', _corrector._validate_morphology_cached.cache_info())
        print('Morphology:', json.dumps(_corrector.morphology_stats(), sort_keys=True))
    return report


def configure_from_cli():
    global GT_DIR, RAW_OCR_DIR, CORRECTED_OUT_DIR, KENLM_BIN, CONFUSION_JSON, DICT_FILE
    global USE_MP, NUM_WORKERS, CANONICAL_METRICS, JSON_REPORT, EXPECTED_PAGES, MAX_DAMAGE_RATE
    global MORPHEMES_JSON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", default=GT_DIR)
    parser.add_argument("--raw-dir", default=RAW_OCR_DIR)
    parser.add_argument("--output-dir", default=CORRECTED_OUT_DIR)
    parser.add_argument("--kenlm", default=KENLM_BIN)
    parser.add_argument("--confusion", default=CONFUSION_JSON)
    parser.add_argument("--dictionary", default=DICT_FILE)
    parser.add_argument(
        "--backend", choices=("symspell",), default="symspell",
        help="Kept for compatibility. The trie backend from an earlier revision "
             "is not present at HEAD; only 'symspell' is available.",
    )
    parser.add_argument("--morphemes-json", default=None,
                        help="Path to morphemes.json for the morpheme-peel guard")
    parser.add_argument("--max-optical-cost", type=float, default=4.8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--canonical-metrics", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--json-report")
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--max-clean-damage-rate", type=float,
                        help="Optional upper damage-rate gate in percent; headline metric mode applies")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--lambda-optical", type=float, default=0.9)
    parser.add_argument("--lambda-lm", type=float, default=2.0)
    parser.add_argument("--lambda-morph", type=float, default=1.5)
    parser.add_argument("--require-morphemes-json", action="store_true",
                        help="Fail instead of continuing with the peel guard disabled")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.expected_pages is not None and args.expected_pages < 1:
        parser.error("--expected-pages must be positive")
    if args.max_clean_damage_rate is not None and not 0 <= args.max_clean_damage_rate <= 100:
        parser.error("--max-clean-damage-rate must be between 0 and 100")
    JSON_REPORT, EXPECTED_PAGES = args.json_report, args.expected_pages
    MORPHEMES_JSON = args.morphemes_json
    MAX_DAMAGE_RATE = args.max_clean_damage_rate
    GT_DIR, RAW_OCR_DIR, CORRECTED_OUT_DIR = args.gt_dir, args.raw_dir, args.output_dir
    KENLM_BIN, CONFUSION_JSON, DICT_FILE = args.kenlm, args.confusion, args.dictionary
    NUM_WORKERS, USE_MP = args.workers, args.workers > 1
    CANONICAL_METRICS = args.canonical_metrics
    CORRECTOR_OPTIONS.update(beam_width=args.beam_width,
                             lambda_optical=args.lambda_optical,
                             lambda_lm=args.lambda_lm, lambda_morph=args.lambda_morph,
                             max_optical_cost=args.max_optical_cost,
                             score_eos=not args.no_eos,
                             require_morphemes_json=args.require_morphemes_json)


if __name__ == "__main__":
    configure_from_cli()
    main()