#!/usr/bin/env python3
"""Spot-check the corrector on the five known page-1 failure patterns.

NOTE ON MEMORY: `main_i7.txt` is a multi-million-type lexicon and SymSpell
costs ~1.03 KB per dictionary type (measured), so loading it uncapped needs
~6 GB and will be OOM-killed on most machines. The caps below keep the run
inside a few hundred MB. See docs/segmenter_integration.md §7.

Measured trade-off on real corpus data: the top 300k types cover 92.4% of
running tokens and the top 500k cover 96.9%, while 65% of types occur exactly
once (and in an OCR-mined lexicon most singletons ARE OCR errors). So capping
usually helps precision as well as memory.

Override with --max-lexicon-types / --dict-min-freq. Pass 0 to disable a cap.
"""
# Make the repo-root shared modules importable from this subfolder.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import os
import sys
import time

from amharic_ocr_corrector import AmharicOCRCorrector

KENLM = "/mnt/mintdata/Users/Des/Mint/Models/KenLM/word_model/amharic_5gram.binary"
CONFUSION = "/media/des/Tess/Tesseract2/confusion_matrix.json"
LEXICON = "/mnt/mintdata/Users/Des/Mint/4_Dataset_wo/Projects/p16_morpheme_analysis/data/main_i7.txt"

# Defaults chosen to keep peak RSS in the hundreds of MB rather than GB.
DEFAULT_MAX_TYPES = int(os.environ.get("AMHARIC_MAX_LEXICON_TYPES", 500_000))
DEFAULT_MIN_FREQ = int(os.environ.get("AMHARIC_DICT_MIN_FREQ", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-lexicon-types", type=int, default=DEFAULT_MAX_TYPES,
                    help="Keep only the N most frequent types (0 = no cap)")
    ap.add_argument("--dict-min-freq", type=int, default=DEFAULT_MIN_FREQ,
                    help="Drop types seen fewer than N times (0 = no cap)")
    ap.add_argument("--no-morphology", action="store_true",
                    help="Disable the morphological analyzer (A/B)")
    args = ap.parse_args()

    corrector = AmharicOCRCorrector(
        kenlm_path=KENLM,
        confusion_matrix_path=CONFUSION,
        dict_vocab_path=LEXICON,
        max_lexicon_types=args.max_lexicon_types or None,
        dict_min_freq=args.dict_min_freq or None,
        use_morphology=False if args.no_morphology else None,
    )

    test_sentences = [
        # 1. Real-word slip in context (አና -> እና, አንደሚያደርጉ -> እንደሚያደርጉ)
        "የኢትዮጵያ፣ ግብጽ አና ሱዳን ባለሙያዎች በግድቡ የቴክኒክ አና የሕግ ጉዳይ ዙሪያ ውይይት አንደሚያደርጉ ይጠበቃል",
        # 2. Ghost loop artifact (ሶስትዮሽቨ -> ሶስትዮሽ)
        "ሶስተኛው ዙር የሕዳሴው ግድብ ሶስትዮሽቨ ውይይት በግብጽ ተጀመረ።",
        # 3. Optical root error (በሀዳሴው -> በህዳሴው)
        "ሶስተኛው ተመሳሳይ ውይይት ዛሬ በግብጽ ካይሮ በመካሄድ ላይ መሆኑን በሀዳሴው ግድብ የኢትዮጵያ",
        # 4. Multi-order corruption (ፕርጀከት -> ፕሮጀክት)
        "ትልቁ ፕርጀከት ነው፡፡",
        # 5. Clean token protection test (Verified words that must not be stripped or damaged)
        "ተከትሎም እርምጃን ወሰዱ፤ መወሰኗ ተረጋገጠ፤ በህገ-መንግስቱን መሰረት ዓ/ም ተጠናቀቀ።",
    ]

    print("=" * 70)
    print("TESTING FULL NORMALIZED PIPELINE (NO HARDCODING)")
    print("=" * 70)
    print(f"lexicon types kept : {len(corrector.vocab):,}")
    print(f"morphology         : "
          f"{'on' if corrector.segmenter is not None else 'off'}")
    print("=" * 70)

    for s in test_sentences:
        t0 = time.time()
        res = corrector.correct_sentence(s)
        dt = (time.time() - t0) * 1000.0
        print(f"RAW OCR   : {s}")
        print(f"CORRECTED : {res} ({dt:.2f} ms)\n")

    print("=" * 70)
    print("morphology_stats:", corrector.morphology_stats())


if __name__ == "__main__":
    sys.exit(main())
