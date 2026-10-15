#!/usr/bin/env python3
"""compare_lexicon_configs.py — held-out comparison of detector configurations.

Why this exists
---------------
The dictionary table in README.md / OCR_error_detection/README.md is measured on
67 validation pages drawn from the *same distribution* as the 1,267 training
pages (`train_LightGBM.py --sweep-freqs`). Those numbers are in-distribution.

This script measures the same configurations on the 100 `Evaluation/` pages,
which are held out of training, using the exact protocol of
`scripts/evaluate_child.py` (GroundTruth vs OCRed token alignment, then one
`DetectionEngine.verdict()` per aligned OCR token). Every model is served
through the same code path with its own matching lexicon, so the comparison is
apples-to-apples — including the shipped `Model/detector_lgbm_conffree.pkl`,
which is a pre-schema-2 pickle and is served with legacy imputation.

Usage
-----
    # the shipped model plus every candidate found under --model-dir
    python OCR_error_detection/scripts/compare_lexicon_configs.py

    # explicit configs: name|model.pkl|lexicon.txt|min_freq   (lexicon/- = auto)
    python OCR_error_detection/scripts/compare_lexicon_configs.py \
        --config 'shipped|-|-|-' \
        --config 'm7_f1000|/tmp/det_cmp/m7_f1000.pkl|-|1000'

    # fewer pages while iterating
    python OCR_error_detection/scripts/compare_lexicon_configs.py --max-pages 10

Exit code is always 0 unless a config fails to load; this is a measurement tool,
not a quality gate (use evaluate_child.py for gates).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHILD = os.path.dirname(_HERE)                 # OCR_error_detection/
_ROOT = os.path.dirname(_CHILD)                 # repo root
for _p in (_ROOT, _CHILD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_DIR = os.path.join(_CHILD, "Model")
SHIPPED = os.path.join(MODEL_DIR, "detector_lgbm_conffree.pkl")
MAIN_I7 = os.path.join(MODEL_DIR, "main_i7_frq10.txt")
LOCAL_LEXICON = os.path.join(MODEL_DIR, "local_lexicon.txt")


def _load_eval_module():
    """Import scripts/evaluate_child.py (the folder is not a package)."""
    path = os.path.join(_HERE, "evaluate_child.py")
    spec = importlib.util.spec_from_file_location("evaluate_child", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# name -> (model path or None for shipped, lexicon path or None, min_freq or None)
KNOWN_CANDIDATES = [
    ("shipped conffree (main_i7 f>=500)", SHIPPED, None, None),
    ("retrained main_i7 f>=500", os.path.join(MODEL_DIR, "detector_lgbm_final.pkl"), MAIN_I7, 500),
    ("main_i7 f>=1000", None, MAIN_I7, 1000),
    ("main_i7 f>=100", None, MAIN_I7, 100),
    ("main_i7 f>=10", None, MAIN_I7, 10),
    ("local_lexicon f>=10", None, LOCAL_LEXICON, 10),
    ("local_lexicon f>=3", None, LOCAL_LEXICON, 3),
    ("local_lexicon f>=1", None, LOCAL_LEXICON, 1),
]


def parse_config(spec):
    """'name|model|lexicon|min_freq' -> dict. '-' or '' means auto/None."""
    parts = [p.strip() for p in spec.split("|")]
    while len(parts) < 4:
        parts.append("")
    name, model, lexicon, min_freq = parts[:4]

    def _none(v):
        return None if v in ("", "-", "None", None) else v

    model = _none(model)
    lexicon = _none(lexicon)
    if model and not os.path.isabs(model):
        model = os.path.join(_ROOT, model)
    if lexicon and not os.path.isabs(lexicon):
        lexicon = os.path.join(_ROOT, lexicon)
    return {
        "name": name or os.path.basename(model or "auto"),
        "model": model,
        "lexicon": lexicon,
        "min_freq": int(min_freq) if _none(min_freq) is None else int(min_freq),
    }


def default_configs(model_dir):
    """The shipped model, then any candidate pickles found in model_dir."""
    configs = [{"name": "shipped conffree", "model": SHIPPED,
                "lexicon": None, "min_freq": None}]
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".pkl") or fname == os.path.basename(SHIPPED):
                continue
            path = os.path.join(model_dir, fname)
            cfg = {"name": os.path.splitext(fname)[0], "model": path,
                   "lexicon": None, "min_freq": None}
            # A model trained from local_lexicon must be served with it, or
            # in_lexicon / log_corpus_freq come from the wrong vocabulary.
            if "loc" in fname:
                cfg["lexicon"] = LOCAL_LEXICON
            elif "m7" in fname or "main_i7" in fname:
                cfg["lexicon"] = MAIN_I7
            # Recover the training filter from the payload when it records one.
            try:
                import joblib
                payload = joblib.load(path)
                if isinstance(payload, dict):
                    if payload.get("lexicon_min_freq"):
                        cfg["min_freq"] = int(payload["lexicon_min_freq"])
                    lp = payload.get("lexicon_path")
                    if lp and os.path.exists(lp):
                        cfg["lexicon"] = lp
            except Exception:
                pass
            configs.append(cfg)
    return configs


def build_engine(cfg, verbose=False):
    from OCR_error_detection import DetectionEngine

    kwargs = {"verbose": verbose}
    if cfg.get("model"):
        kwargs["lgbm_model_path"] = cfg["model"]
    if cfg.get("lexicon"):
        kwargs["dict_vocab_path"] = cfg["lexicon"]
        if cfg.get("min_freq"):
            kwargs["dict_min_freq"] = cfg["min_freq"]
    return DetectionEngine(**kwargs)


def summarize(stats):
    """Same arithmetic as evaluate_child.print_stats, returned as a dict."""
    errors = stats["errors"]
    clean = stats["clean"]
    caught = stats["err_flag"] + stats["err_gray"]
    recall = 100.0 * caught / max(errors, 1)
    recall_flag = 100.0 * stats["err_flag"] / max(errors, 1)
    ff = 100.0 * stats["cor_flag"] / max(clean, 1)
    return {
        "errors": errors,
        "clean": clean,
        "recall": recall,
        "recall_flag_only": recall_flag,
        "false_flag": ff,
        "err_pass_pct": 100.0 * stats["err_pass"] / max(errors, 1),
        "clean_pass_pct": 100.0 * stats["cor_pass"] / max(clean, 1),
        "gray_on_clean_pct": 100.0 * stats["cor_gray"] / max(clean, 1),
        "score": (recall / 100.0) * (1.0 - ff / 100.0),
        "wer": 100.0 * errors / max(stats["gt_words"], 1),
        "cer": 100.0 * stats["char_errors"] / max(stats["gt_chars"], 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", default=[],
                    help="'name|model.pkl|lexicon.txt|min_freq'; repeatable, "
                         "'-' means auto-resolve")
    ap.add_argument("--model-dir", default="/tmp/det_cmp",
                    help="with no --config: directory of candidate pickles "
                         "(default /tmp/det_cmp)")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--json-report", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    ev = _load_eval_module()
    pages = ev.load_pages(args.max_pages)
    if not pages:
        print("no evaluation pages found", file=sys.stderr)
        return 1
    print(f"Loaded {len(pages)} held-out pages from Evaluation/\n")

    if args.config:
        configs = [parse_config(c) for c in args.config]
    else:
        configs = default_configs(args.model_dir)

    rows = []
    for cfg in configs:
        if cfg.get("model") and not os.path.exists(cfg["model"]):
            print(f"[skip] {cfg['name']}: no such model {cfg['model']}")
            continue
        label = cfg["name"]
        print(f"--- {label} ---", flush=True)
        t0 = time.time()
        try:
            det = build_engine(cfg, verbose=args.verbose)
        except Exception as exc:
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            rows.append({"config": label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        init_s = time.time() - t0
        t1 = time.time()
        stats = ev.evaluate(det, pages, verbose=False)
        eval_s = time.time() - t1
        m = summarize(stats)
        m.update({
            "config": label,
            "lexicon_types": det.lexicon_info.get("types"),
            "lexicon_min_freq": det.lexicon_info.get("min_freq"),
            "lexicon": os.path.basename(det.lexicon_info.get("path") or ""),
            "model": os.path.basename(det.model_path or "(rule fallback)"),
            "gates": f"{det.pass_below}/{det.flag_above}",
            "init_seconds": round(init_s, 2),
            "ms_per_word": round(eval_s * 1000.0 / max(stats["aligned"], 1), 3),
        })
        rows.append(m)
        print(f"  lexicon={m['lexicon']} f>={m['lexicon_min_freq']} "
              f"({m['lexicon_types']:,} types)  gates={m['gates']}", flush=True)
        print(f"  recall={m['recall']:.2f}%  flag-only={m['recall_flag_only']:.2f}%  "
              f"FF={m['false_flag']:.3f}%  score={m['score']:.4f}", flush=True)
        print(f"  errors PASSed={m['err_pass_pct']:.2f}%  clean PASS={m['clean_pass_pct']:.2f}%  "
              f"clean GRAY={m['gray_on_clean_pct']:.2f}%", flush=True)
        print(f"  init={m['init_seconds']}s  {m['ms_per_word']} ms/word\n", flush=True)

    ok = [r for r in rows if "recall" in r]
    if ok:
        print("=" * 100)
        print("HELD-OUT COMPARISON — 100 Evaluation/ pages, evaluate_child.py protocol")
        print("=" * 100)
        hdr = ("config", "lexicon", "types", "recall", "flag-only", "FF",
               "err PASS", "clean PASS", "score", "ms/word")
        print(f"{hdr[0]:<34}{hdr[1]:<20}{hdr[2]:>9}{hdr[3]:>9}{hdr[4]:>10}"
              f"{hdr[5]:>8}{hdr[6]:>9}{hdr[7]:>11}{hdr[8]:>8}{hdr[9]:>9}")
        for r in sorted(ok, key=lambda x: -x["score"]):
            print(f"{r['config']:<34}{r['lexicon'] + ' f>=' + str(r['lexicon_min_freq']):<20}"
                  f"{r['lexicon_types']:>9,}{r['recall']:>8.2f}%{r['recall_flag_only']:>9.2f}%"
                  f"{r['false_flag']:>7.3f}%{r['err_pass_pct']:>8.2f}%"
                  f"{r['clean_pass_pct']:>10.2f}%{r['score']:>8.4f}{r['ms_per_word']:>9}")
        best = max(ok, key=lambda x: x["score"])
        print(f"\nBest by score on held-out pages: {best['config']} "
              f"(score {best['score']:.4f}, recall {best['recall']:.2f}%, "
              f"FF {best['false_flag']:.3f}%)")

    if args.json_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_report)), exist_ok=True)
        with open(args.json_report, "w", encoding="utf-8") as fh:
            json.dump({"pages": len(pages), "protocol": "evaluate_child.py",
                       "results": rows}, fh, indent=2)
        print(f"Report: {args.json_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
