#!/usr/bin/env python3
"""
Step 6: End-to-end evaluation benchmark.
"""
# Make the repo-root shared modules importable from this subfolder.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import glob
import time
import argparse
import json
from pathlib import Path
from ocr_metrics import COUNT_FIELDS, edit_distance, score_texts, percentages
from amharic_canonical_normalizer import normalize_for_metric

from amharic_ocr_corrector import AmharicOCRCorrector
from tsv_confidence import align_confidences

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
TSV_DIR = None  # folder of Tesseract .tsv files; enables per-word confidences
JSON_REPORT = None
EXPECTED_PAGES = None
MAX_DAMAGE_RATE = None
LIMIT_PAGES = None


def build_corrector():
    return AmharicOCRCorrector(
        kenlm_path=KENLM_BIN, confusion_matrix_path=CONFUSION_JSON,
        dict_vocab_path=DICT_FILE, morphemes_json_path=MORPHEMES_JSON,
        **CORRECTOR_OPTIONS,
    )


def init_worker(config=None):
    global _corrector, KENLM_BIN, CONFUSION_JSON, DICT_FILE, MORPHEMES_JSON
    global RAW_OCR_DIR, CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS, TSV_DIR
    if config is not None:
        (KENLM_BIN, CONFUSION_JSON, DICT_FILE, MORPHEMES_JSON, RAW_OCR_DIR,
         CORRECTED_OUT_DIR, CORRECTOR_OPTIONS, CANONICAL_METRICS, TSV_DIR) = config
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

    line_confs, conf_stats = None, None
    if TSV_DIR:
        tsv_path = os.path.join(TSV_DIR, f"{base_name}.tsv")
        if os.path.exists(tsv_path):
            line_confs, conf_stats = align_confidences(tsv_path, raw_ocr_text.splitlines())
            print(f"     confidences {base_name}: {conf_stats}", flush=True)
        else:
            print(f"     no TSV for {base_name}; running without confidences", flush=True)
    dropped_before = getattr(_corrector, "_conf_dropped", 0)
    correction_start = time.perf_counter()
    if line_confs is None:
        corrected_text = _corrector.correct_text_block(raw_ocr_text)
    else:
        corrected_text = _corrector.correct_text_block(raw_ocr_text, line_confidences=line_confs)
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
    if conf_stats is not None:
        print(f"     confidences dropped in correction: "
              f"{getattr(_corrector, '_conf_dropped', 0) - dropped_before}", flush=True)
    return dict(page=base_name, strict=strict, canonical=canonical,
                correction_seconds=correction_seconds, input_tokens=token_count,
                confidence_alignment=conf_stats)


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
    if LIMIT_PAGES is not None:
        # Smoke runs: take the first N pages in sorted order. Checked against the
        # full directory above, so --expected-pages still guards the dataset.
        gt_files = gt_files[:LIMIT_PAGES]
        total_pages = len(gt_files)
        print(f"Limiting run to the first {total_pages} page(s) (--limit-pages)", flush=True)
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
                         CANONICAL_METRICS, TSV_DIR)
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
        limit_pages=LIMIT_PAGES,
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
    global MORPHEMES_JSON, LIMIT_PAGES, TSV_DIR
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
    parser.add_argument("--char-lm", default=None,
                        help="Path to the character-level KenLM model (Stage 1 signal)")
    parser.add_argument("--morphemes-json", default=None,
                        help="Path to morphemes.json for the morpheme-peel guard")
    parser.add_argument("--max-optical-cost", type=float, default=4.8)
    # The anchor barrier is the ONLY "do no harm" mechanism in the system, and
    # it was not tunable from the CLI. It is the direct knob on clean-token
    # damage: raising it protects more verified tokens, lowering it lets the
    # LM rewrite them. Essential for the Stage 4 gate experiment.
    parser.add_argument("--anchor-margin", type=float, default=3.5,
                        help="Score bonus for keeping a verified token "
                             "(default 3.5). Raise to reduce clean-token damage.")
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--canonical-metrics", action="store_true")
    parser.add_argument("--tsv-dir", default=None,
                        help="Folder of Tesseract <page>.tsv files; adds per-word confidences to Stage 1")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--json-report")
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--limit-pages", type=int, default=None,
                        help="Run only the first N pages (sorted by name) for a quick smoke test")
    parser.add_argument("--max-clean-damage-rate", type=float,
                        help="Optional upper damage-rate gate in percent; headline metric mode applies")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--lambda-optical", type=float, default=0.9)
    parser.add_argument("--lambda-lm", type=float, default=2.0)
    parser.add_argument("--lambda-morph", type=float, default=1.5)
    parser.add_argument("--require-morphemes-json", action="store_true",
                        help="Fail instead of continuing with the peel guard disabled")
    # Morphology is NOT a passive feature: with a working analyzer,
    # is_known_word() accepts any *decomposable* OOV token, and verified tokens
    # gain the anchor barrier's protection and skip the pairwise-reversal
    # search. Measured on real corpus data that is 56% of OOV tokens -- so this
    # flag is the single most important ablation in the system.
    # --- restored / retired options ----------------------------------------
    # These four were present at HEAD but pointed at a corrector revision that
    # no longer exists (candidate_backend, trie_search, trusted_*_frequency,
    # symspell_max_edit_distance). At HEAD they were accepted by argparse and
    # then passed to a constructor that rejected them, so every run with them
    # died with a TypeError. One has been restored for real; the rest now fail
    # with an explanation instead of "unrecognized arguments".
    parser.add_argument("--symspell-distance", type=int, default=1,
                        dest="symspell_distance",
                        help="SymSpell max edit distance (default 1; "
                             "2+ will OOM on a large lexicon)")
    # Retired: accepted so the error is actionable rather than "unrecognized".
    parser.add_argument("--trie-search", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--trusted-long-word-frequency", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--trusted-short-word-frequency", type=int,
                        help=argparse.SUPPRESS)

    # Lexicon size caps. SymSpell costs ~1.03 KB per dictionary type
    # (measured), so an uncapped multi-million-type lexicon will OOM.
    # Measured on real corpus data: the top 300k types cover 92.4% of running
    # tokens, the top 500k cover 96.9%, and 65% of types occur exactly once --
    # those singletons are mostly OCR noise, so capping tends to help
    # precision as well as memory.
    parser.add_argument("--max-lexicon-types", type=int, default=None,
                        help="Keep only the N most frequent dictionary types")
    parser.add_argument("--dict-min-freq", type=int, default=None,
                        help="Drop dictionary types seen fewer than N times")
    parser.add_argument(
        "--no-morphology", dest="use_morphology", action="store_false", default=None,
        help="Disable the morphological analyzer (A/B against the default)")
    parser.add_argument(
        "--require-morphology", dest="use_morphology", action="store_true",
        help="Fail if the morphological analyzer cannot be imported")
    args = parser.parse_args()

    retired = [
        ("--trie-search", args.trie_search,
         "the trie candidate backend was removed; SymSpell + the weighted "
         "optical reversal generator replaced it"),
        ("--trusted-long-word-frequency", args.trusted_long_word_frequency,
         "trusted_*_frequency thresholds were removed; the anchor barrier "
         "plus the peel guard replaced them"),
        ("--trusted-short-word-frequency", args.trusted_short_word_frequency,
         "trusted_*_frequency thresholds were removed; the anchor barrier "
         "plus the peel guard replaced them"),
    ]
    used = [name for name, val, _ in retired if val]
    if used:
        parser.error(
            "removed option(s): " + ", ".join(used) + ". " +
            "; ".join(reason for name, val, reason in retired if val) +
            ". Drop them from the command line.")

    if args.symspell_distance < 1:
        parser.error("--symspell-distance must be >= 1")
    if args.symspell_distance > 1:
        print(f"[WARN] --symspell-distance {args.symspell_distance}: the "
              f"SymSpell index grows steeply with edit distance (~1.03 KB/type "
              f"at distance 1). This will likely OOM on a large lexicon; "
              f"consider --max-lexicon-types alongside it.", flush=True)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.expected_pages is not None and args.expected_pages < 1:
        parser.error("--expected-pages must be positive")
    if args.limit_pages is not None and args.limit_pages < 1:
        parser.error("--limit-pages must be positive")
    if args.max_clean_damage_rate is not None and not 0 <= args.max_clean_damage_rate <= 100:
        parser.error("--max-clean-damage-rate must be between 0 and 100")
    JSON_REPORT, EXPECTED_PAGES = args.json_report, args.expected_pages
    LIMIT_PAGES = args.limit_pages
    MORPHEMES_JSON = args.morphemes_json
    MAX_DAMAGE_RATE = args.max_clean_damage_rate
    GT_DIR, RAW_OCR_DIR, CORRECTED_OUT_DIR = args.gt_dir, args.raw_dir, args.output_dir
    KENLM_BIN, CONFUSION_JSON, DICT_FILE = args.kenlm, args.confusion, args.dictionary
    NUM_WORKERS, USE_MP = args.workers, args.workers > 1
    CANONICAL_METRICS = args.canonical_metrics
    TSV_DIR = args.tsv_dir
    CORRECTOR_OPTIONS.update(beam_width=args.beam_width,
                             lambda_optical=args.lambda_optical,
                             lambda_lm=args.lambda_lm, lambda_morph=args.lambda_morph,
                             max_optical_cost=args.max_optical_cost,
                             score_eos=not args.no_eos,
                             require_morphemes_json=args.require_morphemes_json,
                             use_morphology=args.use_morphology,
                             max_lexicon_types=args.max_lexicon_types,
                             dict_min_freq=args.dict_min_freq,
                             symspell_max_edit_distance=args.symspell_distance,
                             anchor_margin=args.anchor_margin,
                             char_lm_path=args.char_lm,
                             max_candidates=args.max_candidates)


if __name__ == "__main__":
    configure_from_cli()
    main()