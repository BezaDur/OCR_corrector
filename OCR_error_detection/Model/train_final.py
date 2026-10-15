#!/usr/bin/env python3
"""
train_final.py
---------------
The FINAL model recipe, in one command.

This is the model the README calls authoritative:

    main_i7, freq >= 500 -> 54,057 clean types, 14 conf-free features,
    1,267 pages (the 100 evaluation pages held out), isotonic calibration,
    pass gate 0.10 / flag gate 0.65.

It is a thin wrapper around train_LightGBM.py, which holds the implementation.
Run one of:

    python OCR_error_detection/Model/train_final.py            # train + write the model
    python OCR_error_detection/Model/train_final.py --no-save  # dry run, report only
    python OCR_error_detection/Model/train_final.py --sweep    # README dictionary table

Nothing here retrains the shipped Model/detector_lgbm_conffree.pkl unless you
run it with --save (the default writes to Model/detector_lgbm_final.pkl so an
existing artifact is never overwritten by accident).
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PARENT)
for _p in (_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_LightGBM as trainer  # noqa: E402  (same folder)

DEFAULT_OUT = os.path.join(_HERE, "detector_lgbm_final.pkl")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the README's FINAL detector.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="model output path (default: Model/detector_lgbm_final.pkl)")
    ap.add_argument("--no-save", action="store_true",
                    help="train and report, but write no model into the repository")
    ap.add_argument("--sweep", action="store_true",
                    help="reproduce the README's dictionary comparison table instead")
    ap.add_argument("--no-char-lm", action="store_true",
                    help="train without the character KenLM (the README's model has gain 0 for it)")
    ap.add_argument("--quick", action="store_true", help="smoke run")
    args, passthrough = ap.parse_known_args(argv)

    if args.no_save:
        out = "/tmp/detector_lgbm_final_dryrun.pkl"
        report = "/tmp/detector_lgbm_final_dryrun.report.json"
        print("[dry run] no model will be written into the repository")
    else:
        out = args.out
        report = os.path.splitext(out)[0] + ".report.json"

    print("FINAL recipe: main_i7 freq>=%d, 14 conf-free features, "
          "100 dev pages held out, isotonic calibration, gates %.2f/%.2f"
          % (trainer.DEFAULT_MIN_FREQ, trainer.DEFAULT_PASS_THRESH,
             trainer.DEFAULT_FLAG_THRESH))

    trainer_argv = [
        "--min-freq", str(trainer.DEFAULT_MIN_FREQ),
        "--out", out,
        "--report", report,
        "--pass-thresh", str(trainer.DEFAULT_PASS_THRESH),
        "--flag-thresh", str(trainer.DEFAULT_FLAG_THRESH),
    ]
    if args.no_char_lm:
        trainer_argv.append("--no-char-lm")
    if args.sweep:
        trainer_argv += ["--sweep-freqs", "10,100,500,1000"]
    if args.quick:
        trainer_argv.append("--quick")
    trainer_argv += passthrough
    trainer.main(trainer_argv)


if __name__ == "__main__":
    main()
