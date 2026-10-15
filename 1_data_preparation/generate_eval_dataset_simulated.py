#!/usr/bin/env python3
"""
Generate a 100-page synthetic evaluation dataset from real Amharic text
and the real confusion matrix.

Produces:
  evaluation_data/ground_truth/page_NNNN.gt.txt   (clean text)
  evaluation_data/ocr_output/page_NNNN.ocr.txt    (corrupted text)

Each page is a contiguous chunk of ~35 lines from amh.training_text.
Corruption uses the real confusion_matrix.json with calibrated burstiness
to match realistic OCR error profiles (~2-5% CER, ~4-8% WER).
"""
import json
import math
import os
import random

# ---- Config ----
CORPUS = "amh.training_text"
CONFUSION = "confusion_matrix.json"
OUTPUT_DIR = "evaluation_data"
NUM_PAGES = 100
LINES_PER_PAGE = 35
SEED = 42
TARGET_CER = 5.0   # character error rate %
TARGET_BURST = 0.3  # burstiness: 0=uniform, 1=clustered


class Channel:
    """P(observed | true) built from confusion_matrix.json."""

    def __init__(self, path):
        c = json.load(open(path, encoding="utf-8"))
        self.sub = {
            t: {o: math.exp(-cost) for o, cost in m.items()}
            for t, m in c.get("sub", {}).items()
        }
        self.dels = {t: math.exp(-cost) for t, cost in c.get("del", {}).items()}
        self.ins = {o: math.exp(-cost) for o, cost in c.get("ins", {}).items()}
        self.ins_chars = sorted(self.ins, key=self.ins.__getitem__, reverse=True)
        self.ins_w = [self.ins[c] for c in self.ins_chars]

    def error_mass(self, ch):
        s = sum(self.sub.get(ch, {}).values())
        return s + self.dels.get(ch, 0.0)

    def corrupt_char(self, ch, rng):
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
        # occasional spurious insertion
        if self.ins_chars and rng.random() < 0.03:
            extra = rng.choices(self.ins_chars, weights=self.ins_w)[0]
            out = out + extra if rng.random() < 0.5 else extra + out
        return out

    def corrupt_text(self, text, rng, per_char_error_rate, burst=0.0):
        """Corrupt text character by character."""
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
                    u = rng.random()
                    if u < burst:
                        tok_rate = per_char_error_rate * rng.uniform(2.0, 4.0)
                    else:
                        tok_rate = per_char_error_rate * rng.uniform(0.0, 0.3)
                    tok_rate = min(tok_rate, 0.6)
            if rng.random() < tok_rate:
                out.append(self.corrupt_char(ch, rng))
            else:
                out.append(ch)
        return "".join(out)


def load_corpus(path, n_lines):
    """Load n_lines of real Amharic text, skipping very short lines."""
    lines = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if len(stripped) > 10:  # skip headers/blanks
                lines.append(stripped)
            if len(lines) >= n_lines:
                break
    return lines


def main():
    rng = random.Random(SEED)
    channel = Channel(CONFUSION)

    # Load enough lines for NUM_PAGES * LINES_PER_PAGE
    needed = NUM_PAGES * LINES_PER_PAGE + 200  # extra buffer
    print(f"Loading {needed} lines from {CORPUS}...")
    all_lines = load_corpus(CORPUS, needed)
    print(f"  loaded {len(all_lines)} lines")

    if len(all_lines) < NUM_PAGES * LINES_PER_PAGE:
        print(f"  WARNING: only {len(all_lines)} lines available, "
              f"need {NUM_PAGES * LINES_PER_PAGE}")

    # Shuffle to avoid sequential bias
    rng.shuffle(all_lines)

    gt_dir = os.path.join(OUTPUT_DIR, "ground_truth")
    ocr_dir = os.path.join(OUTPUT_DIR, "ocr_output")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)

    # Calibrate the per-char error rate to hit target CER
    # Use a fixed calibration corpus segment
    calib_text = "\n".join(all_lines[:500])
    calib_rng = random.Random(999)
    rate = 0.04
    for _ in range(20):
        calib_rng.seed(999)
        corrupted = channel.corrupt_text(calib_text, calib_rng, rate, TARGET_BURST)
        n_chars = len(calib_text)
        n_errors = sum(1 for a, b in zip(calib_text, corrupted) if a != b)
        cer = 100.0 * n_errors / max(n_chars, 1)
        if abs(cer - TARGET_CER) < 0.3:
            break
        rate *= (TARGET_CER / max(cer, 0.01)) ** 0.7
        rate = max(0.001, min(0.5, rate))

    # Verify on a fresh sample
    calib_rng.seed(777)
    corrupted = channel.corrupt_text(calib_text, calib_rng, rate, TARGET_BURST)
    n_errors = sum(1 for a, b in zip(calib_text, corrupted) if a != b)
    actual_cer = 100.0 * n_errors / max(len(calib_text), 1)
    print(f"Calibrated: rate={rate:.4f}, target CER={TARGET_CER}%, actual={actual_cer:.2f}%")

    # Generate pages
    line_idx = 0
    total_gt_tokens = 0
    total_errors = 0

    for page_num in range(1, NUM_PAGES + 1):
        if line_idx + LINES_PER_PAGE > len(all_lines):
            # Wrap around
            line_idx = 0
            rng.shuffle(all_lines)

        page_lines = all_lines[line_idx:line_idx + LINES_PER_PAGE]
        line_idx += LINES_PER_PAGE

        gt_text = "\n".join(page_lines) + "\n"

        # Corrupt with fresh burstiness per page
        ocr_text = channel.corrupt_text(gt_text, rng, rate, TARGET_BURST)

        # Count errors for this page
        gt_chars = gt_text.replace(" ", "").replace("\n", "")
        ocr_chars = ocr_text.replace(" ", "").replace("\n", "")
        page_errors = sum(1 for a, b in zip(gt_chars, ocr_chars) if a != b)
        total_errors += page_errors
        total_gt_tokens += len(gt_text.split())

        base = f"page_{page_num:04d}"
        with open(os.path.join(gt_dir, f"{base}.gt.txt"), "w", encoding="utf-8") as f:
            f.write(gt_text)
        with open(os.path.join(ocr_dir, f"{base}.ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_text)

        if page_num % 25 == 0:
            print(f"  [{page_num}/{NUM_PAGES}] pages generated")

    # Summary stats
    cer = 100.0 * total_errors / max(
        len("".join(all_lines[:NUM_PAGES * LINES_PER_PAGE])
            .replace(" ", "").replace("\n", "")), 1)
    print(f"\nDone: {NUM_PAGES} pages")
    print(f"  GT dir:  {gt_dir}")
    print(f"  OCR dir: {ocr_dir}")
    print(f"  Total GT tokens: ~{total_gt_tokens:,}")
    print(f"  Measured CER: {cer:.2f}%")
    print(f"  Measured total char errors: {total_errors:,}")


if __name__ == "__main__":
    main()