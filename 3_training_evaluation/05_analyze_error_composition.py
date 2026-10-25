"""Measure the error composition that determines the whole design.

Two questions that every proposal in this thread silently assumes an answer to:

  Q1. NON-WORD vs REAL-WORD: what fraction of errors produce a token that is
      not in the lexicon? Non-word errors are trivially detectable; real-word
      errors are invisible to the lexicon and need context.

  Q2. CORRECTABILITY CEILING: what fraction of errors are within weighted edit
      distance 1 of the true word? That is the hard ceiling for SymSpell at
      max_dictionary_edit_distance=1. If it is low, no amount of detection
      tuning will help -- the generator cannot reach the answer.

Runs two ways:
  --pairs GT_DIR OCR_DIR   real aligned ground truth / OCR pages
  (default)                synthetic corruption driven by the REAL confusion
                           matrix, so it runs anywhere
"""
# Make the repo-root shared modules importable from this subfolder.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import collections
import json
import math
import os
import random
import sys

_REPO = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_REPO / "tools"))
import repo_paths  # noqa: E402
import stage_paths  # noqa: E402
stage_paths.register(str(_REPO))

CONFUSION = repo_paths.confusion_matrix()
CORPUS = os.path.join(str(_REPO), "1_data_preparation", "data",
                      "amh_final_training_text.txt")


# --------------------------------------------------------------------------- #
# synthetic corruption driven by the real confusion matrix
# --------------------------------------------------------------------------- #
# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 3_training_evaluation/05_analyze_error_composition.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "pairs_gt": None,  # set both pairs_* to analyse real GT/OCR dirs
    "pairs_ocr": None,
    "n_chars": 4_000_000,
    "target_cer": 34.4,
    "target_wer": 36.86,
    "seed": 7,
}

class Channel:
    """P(observed | true) built from confusion_matrix.json (cost = -ln P)."""

    def __init__(self, path):
        c = json.load(open(path, encoding="utf-8"))
        self.sub = {t: {o: math.exp(-cost) for o, cost in m.items()}
                    for t, m in c.get("sub", {}).items()}
        # P(true char deleted), P(spurious char inserted)
        self.dels = {t: math.exp(-cost) for t, cost in c.get("del", {}).items()}
        self.ins = {o: math.exp(-cost) for o, cost in c.get("ins", {}).items()}
        self.ins_chars = sorted(self.ins, key=self.ins.__getitem__, reverse=True)
        self.ins_w = [self.ins[c] for c in self.ins_chars]

    def error_mass(self, ch):
        s = sum(self.sub.get(ch, {}).values())
        return s + self.dels.get(ch, 0.0)

    def corrupt_char(self, ch, rng):
        """Return a replacement string for `ch` (may be '', 1 char, or 2)."""
        subs = self.sub.get(ch)
        opts, weights = [], []
        if subs:
            for o, p in subs.items():
                opts.append(("sub", o))
                weights.append(p)
        d = self.dels.get(ch, 0.0)
        if d > 0:
            opts.append(("del", ""))
            weights.append(d)
        if not opts:
            return ch
        kind, rep = rng.choices(opts, weights=weights)[0]
        out = "" if kind == "del" else rep
        # occasional spurious insertion next to this character
        if rng.random() < self._ins_rate:
            extra = rng.choices(self.ins_chars, weights=self.ins_w)[0]
            out = out + extra if rng.random() < 0.5 else extra + out
        return out

    def corrupt(self, text, rng, per_char_error_rate, ins_rate, burst=0.0):
        """Corrupt `text`.

        burst=0 -> errors spread uniformly over characters.
        burst=1 -> errors CLUSTER: each token draws its own error rate, so a
        few tokens are heavily damaged and most are untouched.

        This distinction is not cosmetic. It sets CER/WER, and CER/WER is what
        decides whether edit-distance correction can work at all:
          CER/WER ~0.33  -> ~1.4 of 4.1 chars wrong per bad word  (fixable)
          CER/WER ~0.93  -> ~3.9 of 4.1 chars wrong per bad word  (hopeless)
        Uniform spreading cannot reproduce a high CER/WER; only bursty
        damage can.
        """
        self._ins_rate = ins_rate
        out = []
        tok_rate = None
        for ch in text:
            if ch in (" ", "\n"):
                out.append(ch)
                tok_rate = None
                continue
            if tok_rate is None:
                if burst <= 0:
                    tok_rate = per_char_error_rate
                else:
                    # heavy-tailed per-token rate: most tokens near zero
                    u = rng.random()
                    if u < burst:
                        tok_rate = per_char_error_rate * rng.uniform(2.0, 6.0)
                    else:
                        tok_rate = per_char_error_rate * rng.uniform(0.0, 0.25)
                    tok_rate = min(tok_rate, 0.95)
            if rng.random() < tok_rate * (0.3 + 0.7 * self.error_mass(ch)):
                out.append(self.corrupt_char(ch, rng))
            else:
                out.append(ch)
        return "".join(out)


# --------------------------------------------------------------------------- #
def load_corpus(n_chars, seed=0):
    with open(CORPUS, encoding="utf-8", errors="ignore") as f:
        text = f.read(n_chars)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    random.Random(seed).shuffle(lines)
    return lines


def build_lexicon(lines):
    c = collections.Counter()
    for l in lines:
        c.update(l.split())
    return c


def edit_distance_leq1(a, b):
    """True if a and b are within Levenshtein distance 1 (cheap check)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                      # substitution
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1
    if la < lb:
        a, b = b, a                   # ensure a is longer
        la, lb = lb, la
    for i in range(la):               # single deletion from longer
        if a[:i] + a[i + 1:] == b:
            return True
    return False


def analyse(pairs, lexicon, label):
    """pairs: list of (true_token, ocr_token) -- already aligned 1:1."""
    total = len(pairs)
    wrong = [(t, o) for t, o in pairs if t != o]
    n_wrong = len(wrong)

    nonword = [p for p in wrong if p[1] not in lexicon]
    realword = [p for p in wrong if p[1] in lexicon]
    within1 = [p for p in wrong if edit_distance_leq1(p[0], p[1])]
    oov_within1 = [p for p in nonword if edit_distance_leq1(p[0], p[1])]
    rw_within1 = [p for p in realword if edit_distance_leq1(p[0], p[1])]

    print(f"\n{'='*66}\n{label}\n{'='*66}")
    print(f"aligned tokens            : {total:,}")
    print(f"tokens in error           : {n_wrong:,} ({100*n_wrong/max(total,1):.2f}%)")
    print()
    print("--- Q1: non-word vs real-word (drives where to invest) ---")
    print(f"  NON-WORD  (OCR token not in lexicon) : {len(nonword):7,} "
          f"({100*len(nonword)/max(n_wrong,1):5.1f}% of errors)")
    print(f"  REAL-WORD (OCR token IS in lexicon)  : {len(realword):7,} "
          f"({100*len(realword)/max(n_wrong,1):5.1f}% of errors)")
    print()
    print("--- Q2: correctability ceiling for SymSpell d=1 ---")
    print(f"  errors within edit distance 1 of truth: {len(within1):7,} "
          f"({100*len(within1)/max(n_wrong,1):5.1f}% of errors)")
    print(f"    of non-word errors                 : {len(oov_within1):7,} "
          f"({100*len(oov_within1)/max(len(nonword),1):5.1f}%)")
    print(f"    of real-word errors                : {len(rw_within1):7,} "
          f"({100*len(rw_within1)/max(len(realword),1):5.1f}%)")
    print()
    print(f"  => HARD CEILING at d=1: the corrector can reach at most "
          f"{100*len(within1)/max(n_wrong,1):.1f}% of errors")
    print(f"     even with a perfect detector and perfect ranking.")

    # baseline detector: flag everything OOV
    flagged = [(t, o) for t, o in pairs if o not in lexicon]
    tp = sum(1 for t, o in flagged if t != o)
    fp = len(flagged) - tp
    fn = n_wrong - tp
    prec = tp / max(len(flagged), 1)
    rec = tp / max(n_wrong, 1)
    print()
    print("--- baseline detector: flag every OOV token ---")
    print(f"  flagged {len(flagged):,} ({100*len(flagged)/max(total,1):.1f}% of tokens)")
    print(f"  precision {prec:.3f}   recall {rec:.3f}   "
          f"false positives {fp:,}   missed errors {fn:,}")
    return dict(total=total, wrong=n_wrong, nonword=len(nonword),
                realword=len(realword), within1=len(within1),
                prec=prec, rec=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-gt", default=CONFIG["pairs_gt"], help="dir of *.gt.txt")
    ap.add_argument("--pairs-ocr", default=CONFIG["pairs_ocr"], help="dir of *.ocr.txt")
    ap.add_argument("--n-chars", type=int, default=CONFIG["n_chars"])
    ap.add_argument("--target-cer", type=float, default=CONFIG["target_cer"],
                    help="calibrate synthetic corruption to this CER")
    ap.add_argument("--target-wer", type=float, default=CONFIG["target_wer"],
                    help="...and to this WER")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines = load_corpus(args.n_chars, args.seed)
    lexicon = build_lexicon(lines)
    print(f"lexicon: {len(lexicon):,} types from {len(lines):,} lines")

    if args.pairs_gt and args.pairs_ocr:
        pairs = load_real_pairs(args.pairs_gt, args.pairs_ocr)
        analyse(pairs, lexicon, "REAL DATA: " + args.pairs_gt)
        return

    # ---- synthetic, driven by the real confusion matrix -------------------
    ch = Channel(CONFUSION)
    # DISJOINT lexicon: the clean tokens must be able to be OOV, otherwise the
    # OOV detector scores a circular precision of 1.000.
    half = len(lines) // 2
    lexicon = build_lexicon(lines[half:])
    sample = lines[:2000]
    print(f"disjoint lexicon: {len(lexicon):,} types "
          f"(corrupting a different {len(sample):,} lines)")

    # calibrate burstiness + rate to match BOTH the measured CER and WER
    best = None
    for burst in (0.0, 0.2, 0.35, 0.5, 0.7):
        rate = 0.10
        for _ in range(30):
            pairs, cer, wer = corrupt_pairs(ch, sample, rng, rate, 0.03, burst)
            if abs(cer - args.target_cer) < 0.7:
                break
            rate *= (args.target_cer / max(cer, 0.01)) ** 0.7
            rate = max(0.005, min(0.9, rate))
        err = abs(cer - args.target_cer) + abs(wer - args.target_wer)
        print(f"  burst={burst:.2f} rate={rate:.3f} -> CER {cer:5.2f}% "
              f"WER {wer:5.2f}%  CER/WER {cer/max(wer,0.01):.3f}")
        if best is None or err < best[0]:
            best = (err, burst, rate, cer, wer, pairs)

    err, burst, rate, cer, wer, pairs = best
    print(f"\nBEST MATCH: burst={burst:.2f} rate={rate:.3f} "
          f"CER {cer:.2f}% (target {args.target_cer}) "
          f"WER {wer:.2f}% (target {args.target_wer}) "
          f"CER/WER {cer/max(wer,0.01):.3f}")
    print(f"chars wrong per erroneous word: "
          f"{cer/max(wer,0.01) * 4.14:.2f} of 4.14")
    analyse(pairs, lexicon,
            f"SYNTHETIC matched to the real eval set "
            f"(CER {cer:.2f}%, WER {wer:.2f}%)")


def corrupt_pairs(ch, lines, rng, rate, ins_rate, burst=0.0):
    """Corrupt lines and return 1:1 aligned (true, ocr) token pairs.

    Only lines whose token count is preserved are usable for 1:1 alignment;
    we corrupt until we have enough of them.
    """
    pairs = []
    tot_c = err_c = tot_w = err_w = 0
    for line in lines:
        toks = line.split()
        if not toks or len(toks) < 3:
            continue
        corrupt = ch.corrupt(line, rng, rate, ins_rate, burst)
        ctoks = corrupt.split()
        tot_c += len(line)
        err_c += sum(1 for a, b in zip(line, corrupt) if a != b)
        if len(ctoks) != len(toks):
            continue          # segmentation changed -> skip (aligned separately)
        pairs.extend(zip(toks, ctoks))
        tot_w += len(toks)
        err_w += sum(1 for a, b in zip(toks, ctoks) if a != b)
    cer = 100.0 * err_c / max(tot_c, 1)
    wer = 100.0 * err_w / max(tot_w, 1)
    return pairs, cer, wer


def load_real_pairs(gt_dir, ocr_dir):
    """Align GT and OCR at token level using the same DP as step 03."""
    import glob
    from ocr_metrics import edit_distance  # noqa  (stage_paths registers stage 3)
    pairs = []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.gt.txt"))):
        base = os.path.basename(gt_path).replace(".gt.txt", "")
        ocr_path = os.path.join(ocr_dir, f"{base}.ocr.txt")
        if not os.path.exists(ocr_path):
            continue
        g = open(gt_path, encoding="utf-8").read().split()
        o = open(ocr_path, encoding="utf-8").read().split()
        # token-level Levenshtein alignment via difflib on token ids
        sm = difflib.SequenceMatcher(a=g, b=o, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                pairs.extend(zip(g[i1:i2], o[j1:j2]))
            elif tag == "replace":
                n = min(i2 - i1, j2 - j1)
                pairs.extend(zip(g[i1:i1 + n], o[j1:j1 + n]))
    return pairs


if __name__ == "__main__":
    import difflib  # noqa: E402
    main()
