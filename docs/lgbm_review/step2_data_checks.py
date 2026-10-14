"""Step 2 data checks (go / no-go gate) for the LightGBM detector.

Runs on the repository only (no dictionary or KenLM needed). Checks:
  C1  label consistency: is_error agrees with ocr_word != gt_word
  C2  confidence: is the tesseract_conf column informative in the training file?
  C3  doc_freq semantics: count within the page (must match inference)
  C4  page inventory: pages present vs expected
  C5  OCR coverage: harmonized rows vs the evaluation OCR output (both cleaned the same way)
  C6  evaluation independence: how much of the evaluation ground truth is in training
  C7  training-internal overlap and page groups for grouped cross-validation
  C8  punctuation and digits: absent from the harmonized training rows

Usage:  python3 docs/lgbm_review/step2_data_checks.py
"""
import collections
import csv
import difflib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OCR = os.path.join(ROOT, "OCR_error_detection")
TSV = os.path.join(OCR, "harmonized_corpus_jiwer.tsv")
EVAL_GT = os.path.join(OCR, "Evaluation", "GroundTruth")
EVAL_OCR = os.path.join(OCR, "Evaluation", "OCRed")
N = 5  # n-gram size for overlap checks
# Same cleaning as features.clean_surface / PUNCT_RE. The harmonized TSV was built with it
# (no punctuation or digits in any row), so the evaluation text must be cleaned the same way
# before any comparison. Comparing raw text with cleaned text understates overlap.
PUNCT_RE = re.compile(r"[^\w\s]|\d")


def clean_tokens(seq):
    return [w for w in (PUNCT_RE.sub("", t).strip() for t in seq) if w]


def read_tokens(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read().split()


def grams(seq, n=N):
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def union_find(keys):
    par = {k: k for k in keys}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    return par, find


def main():
    rows = list(csv.DictReader(open(TSV, encoding="utf-8", errors="ignore"), delimiter="\t"))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["page"]].append(r)
    print(f"rows {len(rows):,}  pages {len(by):,}")

    # C1 labels
    bad = sum((r["ocr_word"] != r["gt_word"]) != (r["is_error"] == "1") for r in rows)
    err = sum(r["is_error"] == "1" for r in rows)
    print(f"C1 label consistency: {bad} disagreements; error rate {100 * err / len(rows):.2f}%")

    # C2 confidence
    conf = collections.Counter(r["tesseract_conf"] for r in rows)
    print(f"C2 confidence: distinct values {len(conf)}; most common {conf.most_common(1)[0]}")

    # C3 doc_freq semantics on a sample page
    sample = by["page_0001"]
    within = collections.Counter(r["ocr_word"] for r in sample)
    match = sum(int(r["doc_freq"]) == within[r["ocr_word"]] for r in sample)
    print(f"C3 doc_freq equals count within page: {match}/{len(sample)} on page_0001")

    # C4 page inventory
    nums = {int(p.split("_")[1]) for p in by}
    missing = [i for i in range(1, max(nums) + 1) if i not in nums]
    print(f"C4 pages {len(nums)} of 1..{max(nums)}; missing {missing}")

    # C5 coverage vs evaluation OCR (same page names)
    ev_total = tr_total = aligned = 0
    for i in range(1, 101):
        p = f"page_{i:04d}"
        ev = clean_tokens(read_tokens(os.path.join(EVAL_OCR, p + ".ocr.txt")))
        tr = [r["ocr_word"] for r in by[p]]
        sm = difflib.SequenceMatcher(None, ev, tr, autojunk=False)
        aligned += sum(b.size for b in sm.get_matching_blocks())
        ev_total += len(ev)
        tr_total += len(tr)
    print(f"C5 eval OCR tokens {ev_total:,}; harmonized rows {tr_total:,}; "
          f"identical aligned tokens {aligned:,} ({100 * aligned / ev_total:.1f}% of eval OCR)")

    # C8 punctuation and digits: the harmonized training rows contain none
    tr_marked = sum(bool(PUNCT_RE.search(r["ocr_word"])) for r in rows)
    raw_tokens = raw_marked = 0
    for i in range(1, 101):
        for t in read_tokens(os.path.join(EVAL_OCR, f"page_{i:04d}.ocr.txt")):
            raw_tokens += 1
            raw_marked += bool(PUNCT_RE.search(t))
    print(f"C8 punctuation/digits: training rows {tr_marked}; eval OCR tokens with them "
          f"{raw_marked:,}/{raw_tokens:,} ({100 * raw_marked / raw_tokens:.1f}%)")

    # C6 evaluation ground truth inside training ground truth
    tr_gt = {p: [r["gt_word"] for r in rs] for p, rs in by.items()}
    tr_grams = {p: grams(s) for p, s in tr_gt.items()}
    all_grams = set().union(*tr_grams.values())
    hit = total = 0
    lines_hit = lines_total = 0
    same_page_share = []
    all_text = "\n".join(" ".join(s) for s in tr_gt.values())
    for fname in sorted(os.listdir(EVAL_GT)):
        p = fname.replace(".gt.txt", "")
        text = open(os.path.join(EVAL_GT, fname), encoding="utf-8", errors="ignore").read()
        g = grams(clean_tokens(text.split()))
        hit += len(g & all_grams)
        total += len(g)
        same_page_hit = len(g & tr_grams.get(p, set()))
        same_page_share.append(same_page_hit / len(g))
        for line in text.splitlines():
            cl = " ".join(clean_tokens(line.split()))
            if len(cl.split()) < 4:
                continue
            lines_total += 1
            if cl in all_text:
                lines_hit += 1
    print(f"C6 eval 5-grams found in training GT: {hit:,}/{total:,} ({100 * hit / total:.1f}%)")
    print(f"C6 eval lines (4+ words) verbatim in training GT: {lines_hit:,}/{lines_total:,}")
    print(f"C6 share of each eval page's 5-grams found on the SAME-numbered training page: "
          f"mean {sum(same_page_share) / len(same_page_share):.3f}, min {min(same_page_share):.3f}")

    # C7 training-internal overlap and groups
    idx = collections.defaultdict(set)
    for p, s in tr_gt.items():
        for g in grams(s):
            idx[g].add(p)
    shared = sum(len(ps) > 1 for ps in idx.values())
    print(f"C7 distinct 5-grams {len(idx):,}; on more than one page {100 * shared / len(idx):.1f}%")
    for cap in (50, 5):
        pair = collections.Counter()
        for ps in idx.values():
            if 1 < len(ps) <= cap:
                ps = sorted(ps)
                for a in range(len(ps)):
                    for b in range(a + 1, len(ps)):
                        pair[(ps[a], ps[b])] += 1
        par, find = union_find(tr_gt)
        for (a, b), c in pair.items():
            if c >= 20:
                par[find(a)] = find(b)
        comp = collections.Counter(find(p) for p in tr_gt)
        sizes = sorted(comp.values(), reverse=True)
        print(f"C7 groups (5-gram pages <= {cap}, link >= 20 shared): {len(comp)} groups; "
              f"largest {sizes[:5]}")


if __name__ == "__main__":
    sys.exit(main())
