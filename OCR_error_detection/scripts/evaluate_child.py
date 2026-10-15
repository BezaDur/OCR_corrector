#!/usr/bin/env python3
"""evaluate_child.py — evaluate the child detector on the Evaluation/ set.

Runs the shipped DetectionEngine (the one the parent corrector uses) over
every page in OCR_error_detection/Evaluation/ (GroundTruth vs OCRed) and
reports:

  * WER / CER — how bad the raw OCR is
  * error recall — share of OCR errors the detector flags or grays
  * false flag rate — share of clean words wrongly flagged
  * PASS / GRAY / FLAG breakdown for errors and clean words

Usage:
    python3 OCR_error_detection/scripts/evaluate_child.py [--max-pages N] [--verbose]
    python3 OCR_error_detection/scripts/evaluate_child.py --min-recall 90 --max-false-flag 1

    # a specific model, served with the lexicon it was trained on
    python3 OCR_error_detection/scripts/evaluate_child.py \
        --model /tmp/loc_f10.pkl \
        --lexicon OCR_error_detection/Model/local_lexicon.txt --min-freq 10

Compare several configurations on these same held-out pages with
scripts/compare_lexicon_configs.py.

Exit code 0 when the evaluation completes and meets the quality gates
(default: recall >= 80%, false flag <= 2%); 1 otherwise.
"""
import argparse
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from OCR_error_detection import ConfidencePolicy, DetectionEngine
from OCR_error_detection.alignment import align_tokens

GT_DIR = os.path.join(os.path.dirname(_HERE), "Evaluation", "GroundTruth")
OCR_DIR = os.path.join(os.path.dirname(_HERE), "Evaluation", "OCRed")

PUNCT_RE = re.compile(r"[^\w\s]|\d")
ETHIOPIC_PUNCT_RE = re.compile(r"[፡።፣፤፥፦፧፨]")


def clean_token(tok):
    tok = PUNCT_RE.sub("", tok or "").strip()
    return ETHIOPIC_PUNCT_RE.sub("", tok).strip()


def char_edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def load_pages(max_pages=0):
    pages = []
    gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".txt"))
    for fname in gt_files:
        base = fname.replace(".gt.txt", "").replace(".txt", "")
        ocr_path = None
        for suffix in (".ocr.txt", ".txt"):
            candidate = os.path.join(OCR_DIR, base + suffix)
            if os.path.exists(candidate):
                ocr_path = candidate
                break
        if not ocr_path:
            continue
        with open(os.path.join(GT_DIR, fname), encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(ocr_path, encoding="utf-8") as f:
            ocr_text = f.read().strip()
        conf_map = {}
        conf_path = os.path.join(OCR_DIR, base + ".conf.tsv")
        if os.path.exists(conf_path):
            idx = 0
            with open(conf_path, encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line or line.startswith("word"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            conf_map[idx] = float(parts[1])
                        except ValueError:
                            pass
                        idx += 1
        pages.append((base, gt_text, ocr_text, conf_map))
        if max_pages and len(pages) >= max_pages:
            break
    return pages


def evaluate(detector, pages, verbose=False):
    stats = Counter()
    for page_name, gt_text, ocr_text, conf_map in pages:
        gt_raw = gt_text.split()
        ocr_raw = ocr_text.split()
        gt_words = [(i, t, clean_token(t)) for i, t in enumerate(gt_raw)]
        ocr_words = [(j, t, clean_token(t)) for j, t in enumerate(ocr_raw)]
        gt_words = [w for w in gt_words if w[2]]
        ocr_words = [w for w in ocr_words if w[2]]
        if not gt_words or not ocr_words:
            continue
        stats["pages"] += 1
        stats["gt_words"] += len(gt_words)
        stats["ocr_words"] += len(ocr_words)

        gt_seq = [w[2] for w in gt_words]
        ocr_seq = [w[2] for w in ocr_words]
        steps, _dist, _ins, _del, _matches = align_tokens(gt_seq, ocr_seq)
        page_doc_counts = Counter(ocr_seq)

        for ref_idx, hyp_idx in steps:
            if ref_idx is None or hyp_idx is None:
                continue
            gt_c = gt_words[ref_idx][2]
            ocr_orig_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
            is_error = gt_c != ocr_c
            stats["gt_chars"] += len(gt_c)
            if is_error:
                stats["errors"] += 1
                stats["char_errors"] += char_edit_distance(gt_c, ocr_c)
            else:
                stats["clean"] += 1
            stats["aligned"] += 1

            ocr_conf = conf_map.get(ocr_orig_idx)
            doc_freq = page_doc_counts[ocr_c]
            verdict = detector.verdict(ocr_orig, [], ocr_conf=ocr_conf, doc_freq=doc_freq)

            key = "err" if is_error else "cor"
            if verdict.label == "flag":
                stats[key + "_flag"] += 1
            elif verdict.label == "gray":
                stats[key + "_gray"] += 1
            else:
                stats[key + "_pass"] += 1

        if verbose:
            print(f"  {page_name}: {len(ocr_words)} OCR words aligned", flush=True)
    return stats


def print_stats(stats):
    errors = stats["errors"]
    clean = stats["clean"]
    wer = 100.0 * errors / max(stats["gt_words"], 1)
    cer = 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1)

    caught = stats["err_flag"] + stats["err_gray"]
    recall = 100.0 * caught / max(errors, 1)
    recall_flag = 100.0 * stats["err_flag"] / max(errors, 1)
    false_flag = 100.0 * stats["cor_flag"] / max(clean, 1)
    clean_pass = 100.0 * stats["cor_pass"] / max(clean, 1)

    print()
    print("=== OCR_error_detection — Evaluation/ set ===")
    print(f"Pages: {stats['pages']}, GT words: {stats['gt_words']:,}, "
          f"OCR words: {stats['ocr_words']:,}, Aligned: {stats['aligned']:,}")
    print(f"WER: {wer:.2f}%  (substitutions {errors:,}), CER: {cer:.2f}%")
    print()
    print(f"ERROR words ({errors:,}):")
    print(f"  FLAG {stats['err_flag']:,} ({100.0 * stats['err_flag'] / max(errors, 1):.1f}%), "
          f"GRAY {stats['err_gray']:,} ({100.0 * stats['err_gray'] / max(errors, 1):.1f}%), "
          f"PASS {stats['err_pass']:,} ({100.0 * stats['err_pass'] / max(errors, 1):.1f}%)")
    print(f"  Recall (flag+gray): {recall:.1f}%   (flag only: {recall_flag:.1f}%)")
    print()
    print(f"CLEAN words ({clean:,}):")
    print(f"  PASS {stats['cor_pass']:,} ({clean_pass:.1f}%), "
          f"GRAY {stats['cor_gray']:,}, FLAG {stats['cor_flag']:,}")
    print(f"  False flag rate: {false_flag:.2f}%")
    print()
    print(f"Score (recall − false_flag): {recall - false_flag:.1f}")
    return recall, false_flag


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-pages", type=int, default=0,
                    help="evaluate only the first N pages (default: all)")
    ap.add_argument("--verbose", action="store_true", help="per-page progress")
    ap.add_argument("--model", default=None,
                    help="LightGBM payload to serve (default: auto-resolve, "
                         "which picks Model/detector_lgbm_conffree.pkl)")
    ap.add_argument("--lexicon", default=None,
                    help="lexicon for --model. A model trained on local_lexicon "
                         "must be served with it, or in_lexicon / "
                         "log_corpus_freq come from the wrong vocabulary")
    ap.add_argument("--confidence-policy", action="store_true",
                    help="apply the post-model confidence policy layer "
                         "(ConfidencePolicy.tuned(): promote gray->pass at "
                         "conf>=0.82, widen gray->flag at conf<0.65 & s>=0.35). "
                         "Off by default so this script keeps reproducing the "
                         "baseline the other numbers in this repo were measured at.")
    ap.add_argument("--policy-demote-pass-conf", type=float, default=None,
                    help="also enable policy rule C (pass->gray below this "
                         "confidence). Requires --confidence-policy. Disabled by "
                         "default: measured at 5.8%% marginal precision, it sends "
                         "~847 extra clean words to the corrector per 100 pages "
                         "and lowers attention precision.")
    ap.add_argument("--min-freq", type=int, default=None,
                    help="lexicon frequency filter to apply to --lexicon")
    ap.add_argument("--min-recall", type=float, default=80.0,
                    help="exit 1 if error recall (flag+gray) is below this %%")
    ap.add_argument("--max-false-flag", type=float, default=2.0,
                    help="exit 1 if the false-flag rate on clean words exceeds this %%")
    args = ap.parse_args(argv)

    pages = load_pages(args.max_pages)
    if not pages:
        print(f"No evaluation pages found under {GT_DIR}", file=sys.stderr)
        return 1
    print(f"Loaded {len(pages)} pages from Evaluation/")

    print("Loading DetectionEngine (lexicon + LightGBM + confusion)...", flush=True)
    engine_kwargs = {"verbose": args.verbose}
    if args.model:
        engine_kwargs["lgbm_model_path"] = args.model
    if args.lexicon:
        engine_kwargs["dict_vocab_path"] = args.lexicon
        if args.min_freq:
            engine_kwargs["dict_min_freq"] = args.min_freq
    elif args.min_freq:
        print("--min-freq needs --lexicon; ignored", file=sys.stderr)
    if args.policy_demote_pass_conf is not None and not args.confidence_policy:
        print("--policy-demote-pass-conf needs --confidence-policy; ignored",
              file=sys.stderr)
    if args.confidence_policy:
        pol = ConfidencePolicy.tuned()
        if args.policy_demote_pass_conf is not None:
            pol.demote_pass_conf = args.policy_demote_pass_conf
        engine_kwargs["confidence_policy"] = pol
        print(f"  confidence policy: promote gray->pass at conf>={pol.promote_pass_conf}, "
              f"widen gray->flag at conf<{pol.flag_below_conf} & s>={pol.flag_min_suspicion}"
              + (f", demote pass->gray at conf<{pol.demote_pass_conf}"
                 if pol.demote_pass_conf is not None else "  (rule C off)"))
    detector = DetectionEngine(**engine_kwargs)
    if args.model or args.lexicon:
        print(f"  model : {detector.model_path}")
        print(f"  lexicon: {detector.lexicon_info['path']} "
              f"(min freq {detector.lexicon_info['min_freq']}, "
              f"{detector.lexicon_info['types']:,} types)")

    stats = evaluate(detector, pages, verbose=args.verbose)
    if not stats["aligned"]:
        print("No aligned words — nothing evaluated", file=sys.stderr)
        return 1
    recall, false_flag = print_stats(stats)

    if recall < args.min_recall or false_flag > args.max_false_flag:
        print(f"\nFAIL: recall {recall:.1f}% (min {args.min_recall}%) / "
              f"false flag {false_flag:.2f}% (max {args.max_false_flag}%)",
              file=sys.stderr)
        return 1
    print(f"\nOK: recall {recall:.1f}% >= {args.min_recall}%, "
          f"false flag {false_flag:.2f}% <= {args.max_false_flag}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
