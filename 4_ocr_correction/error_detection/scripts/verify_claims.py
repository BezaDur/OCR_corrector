#!/usr/bin/env python3
"""verify_claims.py — re-derive the detector's public numbers and try to break them.

Every performance claim in this repository should be checkable without trusting
the person who wrote it down.  This script checks the claims in
``docs/detector_out_of_sample.md`` against the data and the shipped model, in
four groups:

  A. is the data what the documents say it is?
  B. do the headline numbers come out again?
  C. can the numbers be an artifact (label leakage, one lucky page, a rule that
     is as good as the model)?
  D. are the evaluation pages independent samples at all?

Group D is the uncomfortable one.  The corpus is not 1,367 independent pages: it
is 55 documents cut into overlapping windows, so ~9 rows in 10 repeat text that
another row already contains.  The script measures that rather than asserting it.

Exit code is 1 if any check fails, so it can gate a merge.

Usage
-----
    python 4_ocr_correction/error_detection/scripts/verify_claims.py            # ~3 minutes
    python 4_ocr_correction/error_detection/scripts/verify_claims.py --fast     # ~1 minute
    python 4_ocr_correction/error_detection/scripts/verify_claims.py --json /tmp/claims.json

Nothing here writes into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                    # 4_ocr_correction/error_detection/
_STAGE = os.path.dirname(_PKG)                   # 4_ocr_correction/
_ROOT = os.path.dirname(_STAGE)                  # repo root
for _p in (_HERE, _ROOT, _STAGE, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate_out_of_sample as eos  # noqa: E402

CLAIMS_JSON = os.path.join(_ROOT, "docs", "out_of_sample_shipped.json")
EVAL_DIR = os.path.join(_ROOT, "Evaluation")

RESULTS: list[dict] = []


def record(group: str, ident: str, title: str, ok: bool | None, detail: str):
    """ok=None means the check is informational (no pass/fail)."""
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append({"group": group, "id": ident, "title": title,
                    "status": status, "detail": detail})
    print(f"  [{status}] {ident:4s} {title}\n         {detail}")
    return ok


def close(actual, expected, tol):
    return abs(float(actual) - float(expected)) <= tol


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def bootstrap(values: np.ndarray, weights_groups, n: int = 200, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap: resample groups, return the 2.5/97.5 percentile of the mean."""
    groups = list(weights_groups.items())
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [rng.choice(groups) for _ in range(len(groups))]
        num = sum(values[m].sum() for _, m in pick)
        den = sum(m.sum() for _, m in pick)
        if den:
            means.append(num / den)
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]


# --------------------------------------------------------------------------- #
# A. data
# --------------------------------------------------------------------------- #

def check_data(df: pd.DataFrame, dev: pd.DataFrame, prod: pd.DataFrame, fast: bool):
    print("\nA. Is the data what the documents say it is?")
    kinds = df.match_type.value_counts().to_dict()
    record("A", "A1", "corpus shape",
           len(df) == 946_057 and df.page.nunique() == 1_367,
           f"{len(df):,} rows, {df.page.nunique():,} pages, "
           f"clean {kinds.get('MATCH', 0):,} / SUB {kinds.get('SUBSTITUTION', 0):,} / "
           f"INS {kinds.get('INSERTION', 0):,} (expected 946,057 / 1,367 / 884,976 / 59,992 / 1,089)")

    record("A", "A2", "evaluation slice",
           len(dev) == 62_479 and int(dev.is_error.sum()) == 4_361 and dev.page.nunique() == 100,
           f"{len(dev):,} rows, {int(dev.is_error.sum()):,} errors, "
           f"{dev.page.nunique()} pages (expected 62,479 / 4,361 / 100)")

    if fast:
        print("  [SKIP] A3 label rebuild (--fast)")
        return
    import re
    punct = re.compile(r"[^\w\s]|\d")

    def toks(t):
        return [c for c in (punct.sub("", w).strip() for w in t.split()) if c]

    from error_detection.alignment import align_tokens
    agree = total = label_agree = 0
    for fname in sorted(os.listdir(os.path.join(EVAL_DIR, "GroundTruth"))):
        stem = fname.replace(".gt.txt", "")
        gt = toks(open(os.path.join(EVAL_DIR, "GroundTruth", fname), encoding="utf-8").read())
        ocr = toks(open(os.path.join(EVAL_DIR, "OCRed", f"{stem}.ocr.txt"), encoding="utf-8").read())
        steps, *_ = align_tokens(gt, ocr)
        built = []
        for ref_i, hyp_i in steps:
            if hyp_i is None:
                continue
            w = ocr[hyp_i]
            built.append((w, None, 1, "INSERTION") if ref_i is None else
                         ((w, w, 0, "MATCH") if gt[ref_i] == w else (w, gt[ref_i], 1, "SUBSTITUTION")))
        ref = dev[dev.page == stem][["ocr_word", "gt_word", "is_error", "match_type"]]
        n = min(len(built), len(ref))
        total += n
        ref_rows = list(ref.itertuples(index=False))
        for i in range(n):
            b, r = built[i], ref_rows[i]
            if b[2] == r.is_error and b[3] == r.match_type:
                label_agree += 1
            if (b[0] == r.ocr_word and b[2] == r.is_error and b[3] == r.match_type
                    and ((b[1] is None and pd.isna(r.gt_word)) or b[1] == r.gt_word)):
                agree += 1
    record("A", "A3", "labels are reproducible from raw GT/OCR text",
           label_agree == total,
           f"is_error+match_type rebuilt from Evaluation/ agree on {label_agree:,}/{total:,}; "
           f"gt_word attribution agrees on {agree:,} (differences are merged-token ties)")


# --------------------------------------------------------------------------- #
# B. headline numbers
# --------------------------------------------------------------------------- #

def check_numbers(scored: pd.DataFrame, report: dict, overall: dict, tiers: dict, lex: dict):
    print("\nB. Do the headline numbers come out again?")
    exp = report["overall"]
    for key, tol, name in (
        ("recall", 5e-4, "recall (flag + gray)"),
        ("recall_flag_only", 5e-4, "flag-only recall"),
        ("false_flag", 5e-5, "false flag on clean words"),
        ("auc", 5e-4, "AUC"),
        ("ap", 2e-3, "average precision"),
    ):
        record("B", f"B-{key}", name, close(overall[key], exp[key], tol),
               f"measured {overall[key]:.5f} vs committed {exp[key]:.5f}")
    ece = eos.calibration(scored)["ece"]
    record("B", "B-ece", "calibration error", close(ece, report["calibration"]["ece"], 5e-4),
           f"measured ECE {ece:.4f} vs committed {report['calibration']['ece']:.4f}")

    for name, committed in report["by_text_tier"].items():
        got = tiers.get(name, {})
        record("B", f"B-tier-{name}", f"text tier '{name}'",
               got.get("rows") == committed["rows"] and close(got.get("recall", 0), committed["recall"], 5e-4),
               f"rows {got.get('rows')} vs {committed['rows']}, "
               f"recall {got.get('recall', float('nan')):.4f} vs {committed['recall']:.4f}")

    for name, committed in report["by_lexicon_state"].items():
        got = lex.get(name, {})
        record("B", f"B-lex-{name}", f"lexicon state '{name}'",
               got.get("rows") == committed["rows"] and close(got.get("recall", 0), committed["recall"], 5e-4),
               f"rows {got.get('rows')} vs {committed['rows']}, "
               f"recall {got.get('recall', float('nan')):.4f} vs {committed['recall']:.4f}")


# --------------------------------------------------------------------------- #
# C. could the number be an artifact?
# --------------------------------------------------------------------------- #

def check_artifacts(scored: pd.DataFrame, engine, pass_below: float, flag_above: float):
    print("\nC. Could the numbers be an artifact?")

    # C1 label permutation: if the metric path leaks the label, shuffling breaks nothing.
    rng = np.random.default_rng(0)
    y = scored.is_error.to_numpy().copy()
    aucs = []
    from sklearn.metrics import roc_auc_score
    for _ in range(5):
        rng.shuffle(y)
        aucs.append(roc_auc_score(y, scored.p.to_numpy()))
    record("C", "C1", "shuffling the labels destroys the score",
           all(0.47 <= a <= 0.53 for a in aucs),
           f"AUC after shuffling the error labels: {[round(a, 3) for a in aucs]} (expected ~0.50)")

    # C2 cluster bootstrap by page.
    yv = scored.is_error.to_numpy()
    err_flag = ((scored.label == "flag") | (scored.label == "gray")).to_numpy()[yv == 1]
    err_page = scored.page.to_numpy()[yv == 1]
    lo, hi = bootstrap(err_flag, {p: err_page == p for p in scored.page.unique()})
    record("C", "C2", "page-cluster bootstrap of recall", None,
           f"95 % interval {100 * lo:.2f}–{100 * hi:.2f} % (treats the 100 pages as independent; "
           f"section D shows why that is optimistic)")

    # C3 does a plain dictionary rule do as well?
    oov = scored.in_lexicon.to_numpy() == 0
    err = scored.is_error.to_numpy() == 1
    ok = ~err
    rule_recall = float(oov[err].mean())
    rule_ff = float(oov[ok].mean())
    model_recall = float(((scored.label == "flag") | (scored.label == "gray"))[err].mean())
    model_ff = float((scored.label == "flag")[ok].mean())
    record("C", "C3", "the model beats the obvious rule",
           model_recall > rule_recall and model_ff < rule_ff / 10,
           f"\"flag every word outside the lexicon\": recall {100 * rule_recall:.2f} %, "
           f"false flag {100 * rule_ff:.2f} % | model: recall {100 * model_recall:.2f} %, "
           f"false flag {100 * model_ff:.3f} %")

    # C4 determinism.
    sample = scored.head(500)
    a = np.array([engine.verdict(w, doc_freq=float(d)).suspicion
                  for w, d in zip(sample.ocr_word, sample.doc_freq)])
    b = np.array([engine.verdict(w, doc_freq=float(d)).suspicion
                  for w, d in zip(sample.ocr_word, sample.doc_freq)])
    record("C", "C4", "scoring is deterministic", bool(np.array_equal(a, b)),
           f"two passes over 500 rows identical: {np.array_equal(a, b)}")


# --------------------------------------------------------------------------- #
# D. independence audit
# --------------------------------------------------------------------------- #

def check_independence(df: pd.DataFrame, dev: pd.DataFrame, prod: pd.DataFrame, scored: pd.DataFrame,
                       fast: bool):
    print("\nD. Are the evaluation pages independent samples?")

    def page_words(sub):
        return {p: [c for c in (eos.PUNCT_RE.sub("", w).strip()
                                for w in g.tolist() if isinstance(w, str)) if c]
                for p, g in sub.groupby("page")["gt_word"]}

    def first_seen(pages_in_order, words, n=5):
        seen, novel, total = set(), 0, 0
        for p in pages_in_order:
            seq = words[p]
            total += len(seq)
            gs = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
            new = [g for g in gs if g not in seen]
            novel += len(new)
            seen.update(gs)
        return total, novel, len(seen)

    words = page_words(df)
    order = sorted(words, key=lambda p: int(p.split("_")[1]))
    total, novel, distinct = first_seen(order, words)
    dup = 1 - (novel + 4) / total
    record("D", "D1", "how much of the corpus is repeated text", None,
           f"{total:,} word rows but only {novel + 4:,} first-seen 5-grams "
           f"({100 * dup:.0f} % of rows repeat text already present); "
           f"{distinct:,} distinct 5-grams overall")

    dev_order = [p for p in order if int(p.split("_")[1]) <= 100]
    dtotal, dnovel, _ = first_seen(dev_order, words)
    record("D", "D2", "how much unique text the 100 dev pages contain", None,
           f"{dtotal:,} rows judge only ~{dnovel + 4:,} distinct source words "
           f"({100 * (dnovel + 4) / dtotal:.1f} % of rows) — about "
           f"{(dnovel + 4) / 600:.0f} page-equivalents of text")

    # D2b: are the OCR events duplicated too, or just the underlying text?
    seen_ctx, keep = set(), []
    for p in sorted(scored.page.unique(), key=lambda x: int(x.split("_")[1])):
        rows = scored[scored.page == p]
        seq = rows.ocr_word.astype(str).tolist()
        idx = rows.index.to_numpy()
        for i in range(len(seq)):
            ctx = tuple(seq[max(0, i - 2):i + 3])
            if ctx not in seen_ctx:
                seen_ctx.add(ctx)
                keep.append(idx[i])
    uniq = scored.loc[keep]
    y = uniq.is_error.to_numpy() == 1
    caught = (uniq.label != "pass").to_numpy()
    r = float(caught[y].mean()) if y.any() else float("nan")
    ci = 1.96 * float(np.sqrt(r * (1 - r) / max(1, int(y.sum()))))
    record("D", "D2b", "recall survives keeping only first-seen OCR contexts", None,
           f"{len(uniq):,} of {len(scored):,} rows are new 5-word OCR contexts; recall on them "
           f"{100 * r:.2f} % ± {100 * ci:.2f} pp (n={int(y.sum()):,} errors)")

    # D3 document structure
    def continues(a, b, k=10):
        seq_a, seq_b = words[a], words[b]
        kk = min(k, len(seq_b), len(seq_a))
        if kk <= 0:
            return False
        head = tuple(seq_b[:kk])
        return any(tuple(seq_a[i:i + kk]) == head for i in range(len(seq_a) - kk + 1))

    docs, starts = 1, []
    for a, b in zip(order, order[1:]):
        if not continues(a, b):
            docs += 1
            starts.append(b)
    dev_docs = 1 + sum(1 for s in starts if int(s.split("_")[1]) <= 100)
    record("D", "D3", "the corpus is documents, not pages", None,
           f"{docs} documents in {len(order):,} pages (page starts are usually found ~10-60 words "
           f"into the previous page); the 100 dev pages span {dev_docs} of them")

    straddles = continues("page_0100", "page_0101") if "page_0101" in words else None
    record("D", "D4", "does the train/dev split cut a document in half?", None,
           f"page_0101 continues page_0100: {straddles} "
           f"(True means the last dev pages overlap training pages)")

    if fast:
        print("  [SKIP] D5 grouping sensitivity (--fast)")
        return
    from Model.train_LightGBM import document_groups, strict_text_groups
    import contextlib
    import io
    counts = {}
    for cut in (5, 20, 50):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g = strict_text_groups(prod, max_page_occurrences=cut)
        counts[cut] = len(set(g.values()))
    record("D", "D5", "why the old text grouping was abandoned", None,
           f"shared-5-gram components at max_page_occurrences=5/20/50 -> "
           f"{counts[5]:,}/{counts[20]:,}/{counts[50]:,} groups. A sliding window repeats "
           f"every 5-gram on 10-40 pages, so the cutoff discards the linking evidence; "
           f"no setting recovers the 55 documents.")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doc_of = document_groups(prod)
    n_docs = len(set(doc_of.values()))
    record("D", "D6", "the documented grouping method finds documents", 45 <= n_docs <= 60,
           f"--group-by document finds {n_docs} groups in the 1,267 training pages "
           f"(documents say 51)")


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=eos.CORPUS)
    ap.add_argument("--model", default=eos.SHIPPED_MODEL)
    ap.add_argument("--claims", default=CLAIMS_JSON)
    ap.add_argument("--fast", action="store_true", help="skip the two slow checks")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.claims):
        raise SystemExit(f"committed report not found: {args.claims}")
    report = json.load(open(args.claims, encoding="utf-8"))

    print("=" * 78)
    print("Verifying the detector's claims against the data in this repository")
    print("=" * 78)
    t0 = time.time()

    dev, prod = eos.load_corpus(args.corpus, 100)
    df = pd.concat([dev, prod])
    check_data(df, dev, prod, args.fast)

    print(f"\nLoading {os.path.basename(args.model)} and scoring {len(dev):,} rows ...")
    engine = eos.DetectionEngine(lgbm_model_path=args.model, verbose=False)
    scored = eos.score(dev, engine)
    pass_below, flag_above = engine.pass_below, engine.flag_above
    overall = eos.metrics(scored, pass_below, flag_above)
    print(f"  recall {100 * overall['recall']:.2f} %, false flag {100 * overall['false_flag']:.3f} %, "
          f"AUC {overall['auc']:.4f}")

    share = eos.text_overlap(dev, prod)
    tiers = {}
    for name, test in eos.TIERS:
        pages = [p for p, v in share.items() if test(v)]
        part = scored[scored.page.isin(pages)]
        if len(part):
            tiers[name] = eos.metrics(part, pass_below, flag_above)
    lex = eos.by_lexicon_state(scored, pass_below, flag_above)

    check_numbers(scored, report, overall, tiers, lex)
    check_artifacts(scored, engine, pass_below, flag_above)
    check_independence(df, dev, prod, scored, args.fast)

    print("\n" + "=" * 78)
    counts = Counter(r["status"] for r in RESULTS)
    for group in ("A", "B", "C", "D"):
        rows = [r for r in RESULTS if r["group"] == group]
        if not rows:
            continue
        fails = [r["id"] for r in rows if r["status"] == "FAIL"]
        print(f"  {group}: {len(rows) - len(fails)}/{len(rows)} passed"
              + (f"   FAILED: {', '.join(fails)}" if fails else ""))
    print(f"  {time.time() - t0:.0f}s")
    print("=" * 78)
    if counts.get("FAIL"):
        print("\nOne or more claims did not reproduce. Do not quote them until they do.")
    else:
        print("\nEvery hard claim reproduced. Read group D before quoting the intervals.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "checks": RESULTS}, fh, indent=2)
        print(f"wrote {args.json}")
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
