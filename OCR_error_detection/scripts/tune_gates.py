#!/usr/bin/env python3
"""tune_gates.py — choose the PASS / FLAG gates on held-out data, defensibly.

Why this exists
---------------
The shipped gates (pass 0.10 / flag 0.65) come from README.md, but the flag gate
is unreachable in practice: flag-only recall on the held-out Evaluation/ pages is
2.6%, so ~90% of real errors sit in GRAY and the three-way verdict is effectively
two-way. The trainer's own `threshold_sweep` cannot pick a replacement because its
ranking metric `score = recall(flag+gray) * (1 - FF(flag))` is gameable — `recall`
counts GRAY as caught while `FF` counts only FLAG, so lowering the pass gate raises
the score at zero measured cost.

This script does three things the trainer does not:

1. **Exact 2-D sweep.** The four rates are separable: err_pass and clean_pass
   depend only on the pass gate, err_flag and clean_flag only on the flag gate. So
   the whole grid is computed from two 1-D passes — no sampling, no approximation.

2. **An honest objective.** Instead of one gameable score it reports the full
   operating point (what is kept, what is reviewed, what is auto-corrected) and a
   **frontier**: for each false-flag budget, the best achievable error capture.
   You pick a risk appetite; the script tells you what it buys.

3. **Page-level cross-validation for the selection itself.** Choosing gates on the
   same 100 pages you then report is selection bias. `--folds N` selects gates on
   N-1 folds of pages and measures them on the held-out fold, then aggregates —
   so the reported numbers come from gates that never saw those pages.

Metrics (all as percentages)
    err_pass    real errors verdicted PASS   -> missed entirely      (lower better)
    err_gray    real errors verdicted GRAY   -> reviewed, not fixed
    err_flag    real errors verdicted FLAG   -> auto-corrected       (higher better)
    clean_pass  clean words verdicted PASS   -> correctly untouched  (higher better)
    clean_gray  clean words verdicted GRAY   -> review burden
    clean_flag  clean words verdicted FLAG   -> damaged by correction (lower better)
    gray_share  GRAY as a share of ALL tokens -> operational cost of the middle band

Usage
-----
    # the shipped model, exact sweep + 5-fold page CV
    python OCR_error_detection/scripts/tune_gates.py

    # a specific model served with the lexicon it was trained on
    python OCR_error_detection/scripts/tune_gates.py \
        --model /tmp/det_cmp/loc_f10.pkl \
        --lexicon OCR_error_detection/Model/local_lexicon.txt --min-freq 10

    # frontier against several false-flag budgets
    python OCR_error_detection/scripts/tune_gates.py --ff-budget 0.05,0.1,0.25,0.5,1.0

    # cache the collected probabilities so repeated sweeps don't re-run the detector
    python OCR_error_detection/scripts/tune_gates.py --dump-probs /tmp/probs.npz
    python OCR_error_detection/scripts/tune_gates.py --load-probs /tmp/probs.npz
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHILD = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CHILD)
for _p in (_ROOT, _CHILD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

MODEL_DIR = os.path.join(_CHILD, "Model")
SHIPPED = os.path.join(MODEL_DIR, "detector_lgbm_conffree.pkl")
MAIN_I7 = os.path.join(MODEL_DIR, "main_i7_frq10.txt")
LOCAL_LEXICON = os.path.join(MODEL_DIR, "local_lexicon.txt")

CURRENT_PASS = 0.10
CURRENT_FLAG = 0.65


def _load_eval_module():
    path = os.path.join(_HERE, "evaluate_child.py")
    spec = importlib.util.spec_from_file_location("evaluate_child_for_gates", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(model=None, lexicon=None, min_freq=None, max_pages=0, verbose=False):
    """Run the detector over the held-out pages; return (y, prob, page) arrays.

    y = 1 for a real OCR error, 0 for a clean word. prob is the calibrated
    P(error) the gates are applied to, so any gate pair can be scored afterwards
    without re-running the model.
    """
    from OCR_error_detection import DetectionEngine

    ev = _load_eval_module()
    pages = ev.load_pages(max_pages)
    if not pages:
        raise SystemExit(f"no evaluation pages under {ev.GT_DIR}")

    kwargs = {"verbose": verbose}
    if model:
        kwargs["lgbm_model_path"] = model
    if lexicon:
        kwargs["dict_vocab_path"] = lexicon
        if min_freq:
            kwargs["dict_min_freq"] = min_freq
    det = DetectionEngine(**kwargs)

    ys, probs, pids = [], [], []
    t0 = time.time()
    for pi, (name, gt_text, ocr_text, conf_map) in enumerate(pages):
        gt_words = [(i, t, ev.clean_token(t)) for i, t in enumerate(gt_text.split())]
        ocr_words = [(j, t, ev.clean_token(t)) for j, t in enumerate(ocr_text.split())]
        gt_words = [w for w in gt_words if w[2]]
        ocr_words = [w for w in ocr_words if w[2]]
        if not gt_words or not ocr_words:
            continue
        steps = ev.align_tokens([w[2] for w in gt_words], [w[2] for w in ocr_words])[0]
        counts = {}
        for _i, _t, c in ocr_words:
            counts[c] = counts.get(c, 0) + 1
        for ref_idx, hyp_idx in steps:
            if ref_idx is None or hyp_idx is None:
                continue
            gt_c = gt_words[ref_idx][2]
            ocr_idx, ocr_orig, ocr_c = ocr_words[hyp_idx]
            v = det.verdict(ocr_orig, [], ocr_conf=conf_map.get(ocr_idx),
                            doc_freq=counts[ocr_c])
            ys.append(1 if gt_c != ocr_c else 0)
            probs.append(float(v.suspicion))
            pids.append(pi)
        if verbose:
            print(f"  {name}: {len(ocr_words)} tokens", flush=True)

    info = {
        "pages": len(set(pids)),
        "tokens": len(ys),
        "errors": int(sum(ys)),
        "model": os.path.basename(det.model_path or "(rule fallback)"),
        "model_path": det.model_path,
        "lexicon": os.path.basename(det.lexicon_info.get("path") or ""),
        "lexicon_min_freq": det.lexicon_info.get("min_freq"),
        "lexicon_types": det.lexicon_info.get("types"),
        "seconds": round(time.time() - t0, 1),
    }
    return np.array(ys, dtype=np.int8), np.array(probs, dtype=np.float64), \
        np.array(pids, dtype=np.int32), info


# --------------------------------------------------------------------------- #
# Exact separable sweep
# --------------------------------------------------------------------------- #

def rate_below(probs, gate):
    """Share of `probs` strictly below `gate` (verdict PASS)."""
    return float((probs < gate).mean()) if len(probs) else float("nan")


def rate_at_or_above(probs, gate):
    """Share of `probs` at or above `gate` (verdict FLAG)."""
    return float((probs >= gate).mean()) if len(probs) else float("nan")


def operating_point(y, prob, pass_below, flag_above):
    """The full verdict mix at one gate pair."""
    err, ok = y == 1, y == 0
    pe, po = prob[err], prob[ok]
    ep, ef = rate_below(pe, pass_below), rate_at_or_above(pe, flag_above)
    cp, cf = rate_below(po, pass_below), rate_at_or_above(po, flag_above)
    eg = max(0.0, 1.0 - ep - ef)
    cg = max(0.0, 1.0 - cp - cf)
    n = len(prob)
    gray_all = (prob >= pass_below) & (prob < flag_above)
    return {
        "pass_below": round(pass_below, 4),
        "flag_above": round(flag_above, 4),
        "err_pass": 100 * ep,
        "err_gray": 100 * eg,
        "err_flag": 100 * ef,
        "clean_pass": 100 * cp,
        "clean_gray": 100 * cg,
        "clean_flag": 100 * cf,
        "err_capture": 100 * (1 - ep),
        "gray_share": 100 * float(gray_all.mean()) if n else float("nan"),
    }


def pass_curve(y, prob, pass_grid):
    """Decision 1: PASS vs "needs attention" — a precision/recall curve.

    attention = clean words that are NOT passed. In the parent corrector both
    GRAY and FLAG go on to candidate generation, so attention is real work and
    real damage risk, not a freebie. `attention_precision` is the share of the
    tokens the corrector would work on that really are errors; that ratio is
    what drives the net-harm result in the committed stage-5 benchmark.

    Neither direction is free, so this curve is reported rather than optimized
    against a single number: raising the gate spares clean words and misses more
    errors, lowering it does the reverse.
    """
    err, ok = y == 1, y == 0
    pe, po = prob[err], prob[ok]
    ne, no = float(err.sum()), float(ok.sum())
    rows = []
    for pb in pass_grid:
        ep = rate_below(pe, pb)
        cp = rate_below(po, pb)
        n_att_err = (1 - ep) * ne
        n_att_ok = (1 - cp) * no
        rows.append({
            "pass_below": pb,
            "err_pass": 100 * ep,
            "err_capture": 100 * (1 - ep),
            "clean_pass": 100 * cp,
            "attention": 100 * (1 - cp),
            "attention_precision": 100 * n_att_err / (n_att_err + n_att_ok)
                                   if (n_att_err + n_att_ok) else float("nan"),
            "n_err_missed": int(round(ep * ne)),
            "n_clean_attention": int(round((1 - cp) * no)),
        })
    return rows


def flag_curve(y, prob, flag_grid):
    """Decision 2: among tokens at or above the gate, FLAG (auto-correct) vs not.

    `precision` is the decisive number: of the words FLAG sends to correction,
    what share were actually errors. Below ~50% the detector damages at least as
    many clean words as it fixes, and the committed stage-5 benchmark
    (5_correction_evaluation/benchmarks/benchmark_full_symspell.json) already
    shows correction is net-harmful at the current mix — so precision, not
    volume, is what a flag gate should be chosen on.
    """
    err, ok = y == 1, y == 0
    pe, po = prob[err], prob[ok]
    ne, no = int(err.sum()), int(ok.sum())
    rows = []
    for fa in flag_grid:
        ef = rate_at_or_above(pe, fa)
        cf = rate_at_or_above(po, fa)
        n_ef, n_cf = ef * ne, cf * no
        rows.append({
            "flag_above": fa,
            "err_flag": 100 * ef,
            "clean_flag": 100 * cf,
            "n_err_flagged": int(round(n_ef)),
            "n_clean_flagged": int(round(n_cf)),
            "precision": 100 * n_ef / (n_ef + n_cf) if (n_ef + n_cf) else float("nan"),
        })
    return rows


def pick_gates(y, prob, pass_grid, flag_grid, miss_budget, ff_budget,
               min_precision=None):
    """Choose one gate pair under two explicit budgets, one per decision.

    Pass gate: the HIGHEST gate whose err_pass stays within `miss_budget` (the
    share of real errors you accept missing outright). Highest, because among
    gates that meet the recall floor it spares the most clean words from
    candidate generation — and the committed stage-5 benchmark shows correction
    damages clean tokens, so sparing them is worth something.

    Flag gate: the LOWEST gate whose clean_flag stays within `ff_budget` and, if
    given, whose FLAG precision is at least `min_precision`. Lowest, because that
    auto-corrects the most true errors within the damage budget.
    """
    pc = pass_curve(y, prob, pass_grid)
    fc = flag_curve(y, prob, flag_grid)

    feasible_p = [r for r in pc if r["err_pass"] <= miss_budget + 1e-9]
    pb = max((r["pass_below"] for r in feasible_p), default=None)

    feasible_f = [r for r in fc
                  if r["clean_flag"] <= ff_budget + 1e-9
                  and (min_precision is None
                       or (r["precision"] == r["precision"]
                           and r["precision"] >= min_precision))]
    fa = min((r["flag_above"] for r in feasible_f), default=None)
    if pb is not None and fa is not None and pb > fa:
        pb = fa  # never let the PASS gate sit above the FLAG gate
    if pb is None or fa is None:
        return None
    return operating_point(y, prob, pb, fa)


def page_cv(y, prob, pids, pass_grid, flag_grid, miss_budget, ff_budget,
            min_precision=None, folds=5, seed=42):
    """Select gates on N-1 page folds, measure them on the held-out fold.

    Out-of-fold rates are pooled from raw counts (not averaged fold-to-fold), so
    a fold with few errors does not carry the same weight as one with many.
    Also reports how often each gate pair was chosen: a choice that is stable
    across folds is the evidence that it is not fitted to particular pages.
    """
    uniq = sorted(set(pids.tolist()))
    rng = np.random.default_rng(seed)
    order = np.array(uniq)
    rng.shuffle(order)
    assign = {p: i % folds for i, p in enumerate(order)}
    fold_of = np.array([assign[p] for p in pids])

    picks = []
    tot = {k: 0.0 for k in ("err", "err_pass", "err_flag", "clean", "clean_pass",
                            "clean_flag", "tokens", "gray_all")}
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        chosen = pick_gates(y[tr], prob[tr], pass_grid, flag_grid,
                            miss_budget, ff_budget, min_precision)
        if chosen is None:
            continue
        pb, fa = chosen["pass_below"], chosen["flag_above"]
        picks.append((pb, fa))
        y_te, p_te = y[te], prob[te]
        err, ok = y_te == 1, y_te == 0
        tot["err"] += int(err.sum())
        tot["clean"] += int(ok.sum())
        tot["tokens"] += int(len(y_te))
        tot["err_pass"] += float((p_te[err] < pb).sum())
        tot["err_flag"] += float((p_te[err] >= fa).sum())
        tot["clean_pass"] += float((p_te[ok] < pb).sum())
        tot["clean_flag"] += float((p_te[ok] >= fa).sum())
        tot["gray_all"] += float(((p_te >= pb) & (p_te < fa)).sum())

    if not picks:
        return {"folds": folds, "folds_used": 0, "out_of_fold": {}, "gate_picks": []}

    pc = lambda a, b: 100.0 * a / b if b else float("nan")  # noqa: E731
    n_ef, n_cf = tot["err_flag"], tot["clean_flag"]
    agg = {
        "err_pass": pc(tot["err_pass"], tot["err"]),
        "err_flag": pc(tot["err_flag"], tot["err"]),
        "err_capture": pc(tot["err"] - tot["err_pass"], tot["err"]),
        "clean_pass": pc(tot["clean_pass"], tot["clean"]),
        "clean_flag": pc(tot["clean_flag"], tot["clean"]),
        "flag_precision": pc(n_ef, n_ef + n_cf),
        "gray_share": pc(tot["gray_all"], tot["tokens"]),
    }
    from collections import Counter
    pick_counts = Counter(picks)
    return {
        "miss_budget": miss_budget,
        "ff_budget": ff_budget,
        "min_precision": min_precision,
        "folds": folds,
        "folds_used": len(picks),
        "out_of_fold": agg,
        "gate_picks": [{"pass_below": p, "flag_above": f, "folds": c}
                       for (p, f), c in pick_counts.most_common()],
    }


def prob_distribution(prob, y):
    """Isotonic calibration is steppy: show which gate values are even distinct."""
    err, ok = y == 1, y == 0
    qs = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    return {
        "all_quantiles": {f"p{q}": round(float(np.percentile(prob, q)), 4) for q in qs},
        "error_quantiles": {f"p{q}": round(float(np.percentile(prob[err], q)), 4) for q in qs},
        "clean_quantiles": {f"p{q}": round(float(np.percentile(prob[ok], q)), 4) for q in qs},
        "distinct_values": int(len(np.unique(np.round(prob, 6)))),
        "share_ge_current_flag": round(float((prob >= CURRENT_FLAG).mean()) * 100, 3),
        "share_ge_0.50": round(float((prob >= 0.50).mean()) * 100, 3),
        "share_ge_0.30": round(float((prob >= 0.30).mean()) * 100, 3),
    }


def fmt_row(m):
    return (f"  pass<{m['pass_below']:.2f} flag>={m['flag_above']:.2f} | "
            f"err: PASS {m['err_pass']:6.2f}%  GRAY {m['err_gray']:6.2f}%  FLAG {m['err_flag']:6.2f}% | "
            f"clean: PASS {m['clean_pass']:6.2f}%  GRAY {m['clean_gray']:6.2f}%  FLAG {m['clean_flag']:6.3f}% | "
            f"capture {m['err_capture']:6.2f}%  gray_share {m['gray_share']:5.2f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None, help="LightGBM payload (default: shipped)")
    ap.add_argument("--lexicon", default=None, help="lexicon the model was trained on")
    ap.add_argument("--min-freq", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--pass-grid", default="0.00-0.60:0.01",
                    help="lo-hi:step for the pass gate")
    ap.add_argument("--flag-grid", default="0.20-0.99:0.01",
                    help="lo-hi:step for the flag gate")
    ap.add_argument("--ff-budget", default="0.05,0.10,0.25,0.50,1.00",
                    help="comma-separated false-flag budgets (%%) for the frontier")
    ap.add_argument("--miss-budget", default="5,7.5,10,18",
                    help="comma-separated ceilings (%%) on real errors verdicted "
                         "PASS, i.e. missed outright")
    ap.add_argument("--min-precision", type=float, default=60.0,
                    help="minimum FLAG precision (%%) a flag gate must keep")
    ap.add_argument("--folds", type=int, default=5, help="page folds for CV (0 = skip)")
    ap.add_argument("--dump-probs", default=None, help="save collected probabilities (.npz)")
    ap.add_argument("--load-probs", default=None, help="reuse collected probabilities (.npz)")
    ap.add_argument("--json-report", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    def grid(spec):
        span, step = spec.split(":")
        lo, hi = (float(x) for x in span.split("-"))
        step = float(step)
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 4) for i in range(n)]

    pass_grid = grid(args.pass_grid)
    flag_grid = grid(args.flag_grid)
    budgets = [float(x) for x in str(args.ff_budget).split(",") if x.strip()]
    misses = [float(x) for x in str(args.miss_budget).split(",") if x.strip()]

    if args.load_probs and os.path.exists(args.load_probs):
        z = np.load(args.load_probs, allow_pickle=True)
        y, prob, pids = z["y"], z["prob"], z["pids"]
        info = json.loads(str(z["info"])) if "info" in z else {}
        print(f"Loaded {len(y):,} token probabilities from {args.load_probs}")
    else:
        print("Collecting calibrated probabilities on the held-out Evaluation/ pages...",
              flush=True)
        y, prob, pids, info = collect(args.model, args.lexicon, args.min_freq,
                                      args.max_pages, args.verbose)
        if args.dump_probs:
            os.makedirs(os.path.dirname(os.path.abspath(args.dump_probs)), exist_ok=True)
            np.savez(args.dump_probs, y=y, prob=prob, pids=pids,
                     info=json.dumps(info))
            print(f"Saved probabilities to {args.dump_probs}")

    print(f"\nmodel   : {info.get('model')}")
    if info.get("lexicon_types"):
        print(f"lexicon : {info.get('lexicon')} f>={info.get('lexicon_min_freq')} "
              f"({info.get('lexicon_types'):,} types)")
    print(f"pages   : {info.get('pages')}   tokens: {info.get('tokens'):,}   "
          f"errors: {info.get('errors'):,} ({100 * y.mean():.2f}%)")

    dist = prob_distribution(prob, y)
    print("\n=== calibrated probability distribution ===")
    print(f"  distinct values (isotonic is steppy): {dist['distinct_values']:,}")
    print(f"  all   quantiles: {dist['all_quantiles']}")
    print(f"  error quantiles: {dist['error_quantiles']}")
    print(f"  clean quantiles: {dist['clean_quantiles']}")
    print(f"  tokens with P >= current flag gate {CURRENT_FLAG}: {dist['share_ge_current_flag']}%")
    print(f"  tokens with P >= 0.50: {dist['share_ge_0.50']}%   P >= 0.30: {dist['share_ge_0.30']}%")

    print("\n=== current gates vs nearby alternatives (exact, same pages) ===")
    shown = [(0.10, 0.65), (0.05, 0.65), (0.10, 0.50), (0.05, 0.50),
             (0.10, 0.40), (0.05, 0.40), (0.10, 0.30), (0.05, 0.30),
             (0.02, 0.30), (0.02, 0.25)]
    for pb, fa in shown:
        tag = "  <- CURRENT" if (pb, fa) == (CURRENT_PASS, CURRENT_FLAG) else ""
        print(fmt_row(operating_point(y, prob, pb, fa)) + tag)

    pc_rows = pass_curve(y, prob, pass_grid)
    fc_rows = flag_curve(y, prob, flag_grid)

    print("\n=== DECISION 1: the PASS gate (leave it alone vs needs attention) ===")
    print("    'attention' = clean words NOT passed; in the parent corrector both")
    print("    GRAY and FLAG go on to candidate generation, so this is real cost")
    print(f"  {'pass gate':>10}{'err PASS':>10}{'missed n':>10}{'capture':>9}"
          f"{'attention':>11}{'attn n':>8}{'attn precision':>16}")
    for r in pc_rows:
        if r["pass_below"] * 100 % 5 > 1e-9:
            continue
        tag = "  <- CURRENT" if abs(r["pass_below"] - CURRENT_PASS) < 1e-9 else ""
        print(f"  {r['pass_below']:>10.2f}{r['err_pass']:>9.2f}%{r['n_err_missed']:>10}"
              f"{r['err_capture']:>8.2f}%{r['attention']:>10.2f}%"
              f"{r['n_clean_attention']:>8}{r['attention_precision']:>15.1f}%{tag}")

    print("\n=== DECISION 2: the FLAG gate (auto-correct vs review) ===")
    print("    precision = of the words FLAG sends to correction, the share that")
    print("    really were errors. Below 50% it damages as many as it fixes.")
    print(f"  {'flag gate':>10}{'err FLAG':>10}{'clean FLAG':>12}{'n err':>8}"
          f"{'n clean':>9}{'precision':>11}")
    for r in fc_rows:
        if r["flag_above"] * 100 % 5 > 1e-9:
            continue
        tag = "  <- CURRENT" if abs(r["flag_above"] - CURRENT_FLAG) < 1e-9 else ""
        prec = r["precision"]
        ps = f"{prec:>10.1f}%" if prec == prec else "        n/a"
        print(f"  {r['flag_above']:>10.2f}{r['err_flag']:>9.2f}%{r['clean_flag']:>11.3f}%"
              f"{r['n_err_flagged']:>8}{r['n_clean_flagged']:>9}{ps}{tag}")

    print("\n=== recommended gate pairs under explicit budgets ===")
    print(f"    (errors missed <= miss budget, clean FLAG <= FF budget, "
          f"FLAG precision >= {args.min_precision:.0f}%)")
    recs = []
    for mb in misses:
        for fb in budgets:
            m = pick_gates(y, prob, pass_grid, flag_grid, mb, fb, args.min_precision)
            if m is None:
                print(f"  miss<={mb:5.2f}%  FF<={fb:5.2f}%   infeasible on this grid")
                continue
            m["miss_budget"] = mb
            m["ff_budget"] = fb
            pcr = next((r for r in pc_rows if abs(r["pass_below"] - m["pass_below"]) < 1e-9), None)
            fcr = next((r for r in fc_rows if abs(r["flag_above"] - m["flag_above"]) < 1e-9), None)
            m["attention_precision"] = pcr["attention_precision"] if pcr else float("nan")
            m["flag_precision"] = fcr["precision"] if fcr else float("nan")
            m["n_err_flagged"] = fcr["n_err_flagged"] if fcr else None
            m["n_clean_flagged"] = fcr["n_clean_flagged"] if fcr else None
            recs.append(m)
            print(f"  miss<={mb:5.2f}%  FF<={fb:5.2f}%  ->  "
                  f"pass<{m['pass_below']:.2f} flag>={m['flag_above']:.2f} | "
                  f"capture {m['err_capture']:6.2f}%  attn prec {m['attention_precision']:5.1f}%  "
                  f"err FLAG {m['err_flag']:5.2f}%  clean FLAG {m['clean_flag']:6.3f}%  "
                  f"FLAG prec {m['flag_precision']:5.1f}%")

    cv = None
    if args.folds and recs:
        base = recs[0]
        print(f"\n=== page-level {args.folds}-fold CV of the selection itself ===")
        print(f"    budgets: miss<={base['miss_budget']:.2f}%, "
              f"FF<={base['ff_budget']:.2f}%, precision>={args.min_precision:.0f}%")
        print("    gates are chosen on other pages, then measured on pages they never saw")
        cv = page_cv(y, prob, pids, pass_grid, flag_grid,
                     base["miss_budget"], base["ff_budget"], args.min_precision,
                     folds=args.folds)
        a = cv["out_of_fold"]
        print(f"  folds used: {cv['folds_used']}/{args.folds}")
        print(f"  out-of-fold: err capture {a['err_capture']:.2f}%  "
              f"err PASS {a['err_pass']:.2f}%  err FLAG {a['err_flag']:.2f}%")
        print(f"               clean PASS {a['clean_pass']:.2f}%  "
              f"clean FLAG {a['clean_flag']:.3f}%  FLAG precision {a['flag_precision']:.1f}%  "
              f"gray share {a['gray_share']:.2f}%")
        print("  gate choices across folds:")
        for p in cv["gate_picks"]:
            print(f"    pass<{p['pass_below']:.2f} flag>={p['flag_above']:.2f}  "
                  f"chosen in {p['folds']} fold(s)")

    if args.json_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_report)), exist_ok=True)
        with open(args.json_report, "w", encoding="utf-8") as fh:
            json.dump({"info": info, "distribution": dist,
                       "pass_curve": pc_rows, "flag_curve": fc_rows,
                       "recommendations": recs, "cv": cv,
                       "budgets": {"miss": misses, "ff": budgets,
                                   "min_precision": args.min_precision},
                       "current": operating_point(y, prob, CURRENT_PASS, CURRENT_FLAG),
                       "grids": {"pass": pass_grid, "flag": flag_grid}}, fh, indent=2)
        print(f"\nReport: {args.json_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
