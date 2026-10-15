#!/usr/bin/env python3
"""probe_late_fusion.py — measure a proposed post-model confidence policy layer.

Collects, for every aligned token on the held-out Evaluation/ pages: the base
verdict, the calibrated probability, the signals the proposed rules read
(in_lexicon, one_flip_words, ocr_conf), and the truth label. Then replays the
rules offline so each one's reach, precision and net effect can be measured
without re-running the detector.

Also tests the two confidence scales, because the codebase stores both:
`verdict(ocr_conf=...)` receives Tesseract's 0-100 value, and rescales it onto
`sig.ocr_conf` in 0-1. A rule written against one and fed the other silently
does nothing (or everything).
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CHILD = os.path.dirname(_HERE)
for _p in (_ROOT, _CHILD):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_ev():
    path = os.path.join(_HERE, "evaluate_child.py")
    spec = importlib.util.spec_from_file_location("ev_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(model=None, lexicon=None, min_freq=None, max_pages=0):
    from error_detection import DetectionEngine

    ev = load_ev()
    pages = ev.load_pages(max_pages)
    kw = {"verbose": False}
    if model:
        kw["lgbm_model_path"] = model
    if lexicon:
        kw["dict_vocab_path"] = lexicon
        if min_freq:
            kw["dict_min_freq"] = min_freq
    det = DetectionEngine(**kw)

    rows = []
    for pi, (name, gt_text, ocr_text, conf_map) in enumerate(pages):
        gt = [w for w in (ev.clean_token(t) for t in gt_text.split()) if w]
        ocr_raw = ocr_text.split()
        ow = []
        for j, t in enumerate(ocr_raw):
            c = ev.clean_token(t)
            if c:
                ow.append((j, t, c))
        if not gt or not ow:
            continue
        steps = ev.align_tokens(gt, [w[2] for w in ow])[0]
        counts = {}
        for _j, _t, c in ow:
            counts[c] = counts.get(c, 0) + 1
        for ref, hyp in steps:
            if ref is None or hyp is None:
                continue
            oi, orig, oc = ow[hyp]
            raw_conf = conf_map.get(oi)
            v = det.verdict(orig, [], ocr_conf=raw_conf, doc_freq=counts[oc])
            sig = v.signals
            rows.append({
                "page": pi,
                "is_error": 1 if gt[ref] != oc else 0,
                "label": v.label,
                "s": float(v.suspicion),
                "in_lexicon": bool(getattr(sig, "in_lexicon", False)),
                "n_one_flip": len(getattr(sig, "one_flip_words", []) or []),
                "conf01": None if getattr(sig, "ocr_conf", None) is None else float(sig.ocr_conf),
                "conf100": None if raw_conf is None else float(raw_conf),
                "token": orig,
            })
    return rows, det


def confusion(rows, label_of):
    """Verdict mix + the metrics evaluate_child reports, for a labelling fn."""
    cnt = {"err": {"pass": 0, "gray": 0, "flag": 0},
           "clean": {"pass": 0, "gray": 0, "flag": 0}}
    for r in rows:
        cnt["err" if r["is_error"] else "clean"][label_of(r)] += 1
    ne = sum(cnt["err"].values())
    nc = sum(cnt["clean"].values())
    caught = cnt["err"]["flag"] + cnt["err"]["gray"]
    recall = 100 * caught / max(ne, 1)
    flag_only = 100 * cnt["err"]["flag"] / max(ne, 1)
    ff = 100 * cnt["clean"]["flag"] / max(nc, 1)
    return {
        "err_flag": cnt["err"]["flag"], "err_gray": cnt["err"]["gray"],
        "err_pass": cnt["err"]["pass"], "clean_pass": cnt["clean"]["pass"],
        "clean_gray": cnt["clean"]["gray"], "clean_flag": cnt["clean"]["flag"],
        "n_err": ne, "n_clean": nc,
        "recall": recall, "flag_only": flag_only, "false_flag": ff,
        "score": recall - ff,
        "flag_precision": 100 * cnt["err"]["flag"] / max(cnt["err"]["flag"] + cnt["clean"]["flag"], 1),
        # tokens the parent corrector would actually work on (gray + flag)
        "err_attention": ne - cnt["err"]["pass"],
        "clean_attention": nc - cnt["clean"]["pass"],
    }


def attention_precision(m):
    tot = m["err_attention"] + m["clean_attention"]
    return 100 * m["err_attention"] / tot if tot else float("nan")


def show(tag, m):
    print(f"{tag:<44} err FLAG {m['err_flag']:>5} GRAY {m['err_gray']:>5} PASS {m['err_pass']:>4} | "
          f"clean PASS {m['clean_pass']:>6} GRAY {m['clean_gray']:>5} FLAG {m['clean_flag']:>4} | "
          f"recall {m['recall']:6.2f}% flag-only {m['flag_only']:5.2f}% FF {m['false_flag']:5.3f}% "
          f"score {m['score']:6.2f} FLAGprec {m['flag_precision']:5.1f}% attnprec {attention_precision(m):5.2f}%")


def rule_stats(rows, name, predicate, frm, to):
    """How many tokens a rule can fire on, and what it moves."""
    fired = [r for r in rows if r["label"] == frm and predicate(r)]
    te = sum(1 for r in fired if r["is_error"])
    tc = len(fired) - te
    return {"rule": name, "from": frm, "to": to, "fires": len(fired),
            "errors": te, "clean": tc,
            "precision": 100 * te / len(fired) if fired else float("nan"),
            "max_possible": len(fired)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--min-freq", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--json-report", default=None)
    args = ap.parse_args(argv)

    if args.load and os.path.exists(args.load):
        rows = json.load(open(args.load, encoding="utf-8"))
        print(f"Loaded {len(rows):,} token rows from {args.load}\n")
    else:
        print("Collecting per-token signals over the held-out pages...", flush=True)
        rows, _det = collect(args.model, args.lexicon, args.min_freq, args.max_pages)
        if args.dump:
            json.dump(rows, open(args.dump, "w", encoding="utf-8"))
            print(f"Dumped {len(rows):,} rows to {args.dump}")
        print()

    base = confusion(rows, lambda r: r["label"])
    show("BASELINE (gates 0.10 / 0.65)", base)

    # ---- confidence availability, on both scales ----
    have = [r for r in rows if r["conf01"] is not None]
    print(f"\n=== confidence availability ===")
    print(f"  tokens with a confidence value: {len(have):,} of {len(rows):,} "
          f"({100 * len(have) / len(rows):.1f}%)")
    if have:
        import statistics
        c01 = sorted(r["conf01"] for r in have)
        print(f"  sig.ocr_conf (0-1 scale): median {statistics.median(c01):.3f}  "
              f"min {c01[0]:.3f}  max {c01[-1]:.3f}")
        for t in (0.45, 0.55, 0.82):
            below = sum(1 for v in c01 if v < t)
            print(f"    conf01 <  {t}: {below:>7,} tokens ({100 * below / len(c01):6.3f}%)"
                  f"   >= {t}: {len(c01) - below:>7,}")
        for t in (45, 55, 82):
            below = sum(1 for v in c01 if v * 100 < t)
            print(f"    conf   <  {t}: {below:>7,} tokens ({100 * below / len(c01):6.3f}%)")

    # ---- how many tokens can each proposed rule possibly touch? ----
    print("\n=== reach of each proposed rule, as written (0-1 scale) ===")
    rules = [
        rule_stats(rows, "A: gray->pass  in_lex & no-flip & conf>=0.82",
                   lambda r: r["in_lexicon"] and r["n_one_flip"] == 0
                   and r["conf01"] is not None and r["conf01"] >= 0.82,
                   "gray", "pass"),
        rule_stats(rows, "B: gray->flag  conf<0.55 & s>=0.35",
                   lambda r: r["conf01"] is not None and r["conf01"] < 0.55
                   and r["s"] >= 0.35, "gray", "flag"),
        rule_stats(rows, "C: pass->gray  conf<0.45",
                   lambda r: r["conf01"] is not None and r["conf01"] < 0.45,
                   "pass", "gray"),
    ]
    for r in rules:
        print(f"  {r['rule']:<48} fires on {r['fires']:>6,} tokens  "
              f"(errors {r['errors']}, clean {r['clean']}, precision "
              f"{r['precision'] if r['precision'] == r['precision'] else float('nan'):.1f}%)")

    print("\n=== same rules if thresholds were meant for the 0-100 scale ===")
    rules100 = [
        rule_stats(rows, "A': gray->pass  in_lex & no-flip & conf>=82",
                   lambda r: r["in_lexicon"] and r["n_one_flip"] == 0
                   and r["conf01"] is not None and r["conf01"] * 100 >= 82,
                   "gray", "pass"),
        rule_stats(rows, "B': gray->flag  conf<55 & s>=0.35",
                   lambda r: r["conf01"] is not None and r["conf01"] * 100 < 55
                   and r["s"] >= 0.35, "gray", "flag"),
        rule_stats(rows, "C': pass->gray  conf<45",
                   lambda r: r["conf01"] is not None and r["conf01"] * 100 < 45,
                   "pass", "gray"),
    ]
    for r in rules100:
        print(f"  {r['rule']:<48} fires on {r['fires']:>6,} tokens  "
              f"(errors {r['errors']}, clean {r['clean']}, precision "
              f"{r['precision'] if r['precision'] == r['precision'] else float('nan'):.1f}%)")

    # ---- replay both interpretations end to end ----
    def apply_policy(scale):
        def label_of(r):
            lab, s = r["label"], r["s"]
            c = r["conf01"]
            if c is None:
                return lab
            c = c * 100 if scale == 100 else c
            hiA, loB, pB, loC = (82, 55, 0.35, 45) if scale == 100 else (0.82, 0.55, 0.35, 0.45)
            if lab == "gray" and r["in_lexicon"] and r["n_one_flip"] == 0 and c >= hiA:
                return "pass"
            if lab == "gray" and c < loB and s >= pB:
                return "flag"
            if lab == "pass" and c < loC:
                return "gray"
            return lab
        return label_of

    print("\n=== end-to-end replay ===")
    m01 = confusion(rows, apply_policy(1))
    show("policy as written (conf on 0-1)", m01)
    m100 = confusion(rows, apply_policy(100))
    show("policy with 0-100 thresholds", m100)

    print("\n=== claimed 'after' from the proposal, for comparison ===")
    print(f"{'claimed':<44} err FLAG   462 GRAY  3549 PASS  256 | "
          f"clean PASS  48248 GRAY  9781 FLAG   89 | "
          f"recall  94.00% flag-only 10.80% FF 0.150% score  93.80")

    # ---- which deltas are arithmetically reachable? ----
    print("\n=== can the claimed deltas be produced at all? ===")
    n_low45 = sum(1 for r in rows if r["conf01"] is not None and r["conf01"] < 0.45)
    n_low55 = sum(1 for r in rows if r["conf01"] is not None and r["conf01"] < 0.55)
    print(f"  claimed clean PASS change      : {m01['clean_pass'] - base['clean_pass']:+d} "
          f"(proposal claims {48248 - base['clean_pass']:+d})")
    print(f"  claimed clean GRAY change      : {m01['clean_gray'] - base['clean_gray']:+d} "
          f"(proposal claims {9781 - base['clean_gray']:+d})")
    print(f"  claimed err FLAG change        : {m01['err_flag'] - base['err_flag']:+d} "
          f"(proposal claims {462 - base['err_flag']:+d})")
    print(f"  tokens with conf01 < 0.45      : {n_low45}  <- upper bound on Rule C's reach")
    print(f"  tokens with conf01 < 0.55      : {n_low55}  <- upper bound on Rule B's reach")
    print(f"  Rule C would need to fire on >= {48248 - base['clean_pass'] + 1:,} clean words "
          f"to produce the claimed PASS drop.")
    print(f"  Rule B would need to fire on >= {462 - base['err_flag']:,} errors "
          f"to produce the claimed FLAG rise.")

    if args.json_report:
        out = {"baseline": base, "policy_0_1": m01, "policy_0_100": m100,
               "rules": rules + rules100,
               "confidence_population": {"lt_045": n_low45, "lt_055": n_low55,
                                         "tokens": len(rows), "with_conf": len(have)}}
        json.dump(out, open(args.json_report, "w", encoding="utf-8"), indent=2)
        print(f"\nReport: {args.json_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
