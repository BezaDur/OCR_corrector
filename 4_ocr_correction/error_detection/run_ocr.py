#!/usr/bin/env python3
"""
Run Tesseract OCR on all images in Evaluation/images/.

Produces two files per image in Evaluation/OCRed/:
  page_NNNN.ocr.txt   — the OCR text
  page_NNNN.conf.tsv  — per-word confidence scores

One-time setup: symlink the model into system tessdata so Tesseract
finds it AND its config files (tsv, hocr) in the same place:
    sudo ln -s ${TESSERACT_WORKSPACE:-/media/des/Tess/Tesseract2}/training/2_training/models/packaged/amh_fast.traineddata \
               /usr/share/tesseract-ocr/5/tessdata/amh_fast.traineddata

Usage:
    python run_ocr.py
"""
import csv
import glob
import os
import subprocess
import argparse
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # 4_ocr_correction/error_detection
_STAGE4 = os.path.dirname(_HERE)                      # 4_ocr_correction
_ROOT = os.path.dirname(_STAGE4)                      # repo root
# Training output lives outside the repo; TESSERACT_WORKSPACE points at it.
for _p in (_STAGE4, os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import repo_paths  # noqa: E402
IMAGES_DIR = os.path.join(_ROOT, "Evaluation", "images")
OUTPUT_DIR = os.path.join(_ROOT, "Evaluation", "OCRed")
LANG = "amh_fast"
PSM = "3"


# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 4_ocr_correction/error_detection/run_ocr.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "images": IMAGES_DIR,
    "out": OUTPUT_DIR,
}

def run_tesseract(image_path, output_base):
    """Run Tesseract on one image. Returns (text, word_confidences)."""

    # --- Text output ---
    cmd = ["tesseract", image_path, output_base,
           "-l", LANG, "--psm", PSM]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [ERROR] text failed: {r.stderr.strip()}")
        return "", []

    text = ""
    txt_path = output_base + ".txt"
    if os.path.exists(txt_path):
        text = open(txt_path, encoding="utf-8").read().strip()
        os.remove(txt_path)

    # --- TSV output for word confidences ---
    tsv_base = output_base + "_tsv"
    cmd_tsv = ["tesseract", image_path, tsv_base,
               "-l", LANG, "--psm", PSM, "tsv"]
    r2 = subprocess.run(cmd_tsv, capture_output=True, text=True)

    word_confs = []
    tsv_path = tsv_base + ".tsv"
    if os.path.exists(tsv_path):
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                word = row.get("text", "").strip()
                level = row.get("level", "0")
                if level == "5" and word:
                    try:
                        conf = float(row.get("conf", "-1"))
                    except ValueError:
                        conf = -1.0
                    word_confs.append((word, conf))
        os.remove(tsv_path)
    else:
        if text:
            print(f"    [WARN] TSV not created (word confs unavailable)")
            if r2.stderr.strip():
                print(f"    [WARN] {r2.stderr.strip()[:200]}")

    return text, word_confs


def main():
    global IMAGES_DIR, OUTPUT_DIR
    ap = argparse.ArgumentParser(description="OCR page images with the deployment settings.")
    ap.add_argument("--images", default=CONFIG["images"], help="folder of page images")
    ap.add_argument("--out", default=CONFIG["out"], help="folder for .ocr.txt and .conf.tsv")
    args = ap.parse_args()
    IMAGES_DIR, OUTPUT_DIR = args.images, args.out

    print(f"Python: {sys.executable}")
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
        print(f"Tesseract: {r.stdout.strip().split(chr(10))[0]}")
    except FileNotFoundError:
        print("ERROR: Tesseract not found.")
        sys.exit(1)

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    images = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(IMAGES_DIR, pat)))
    images.sort()

    if not images:
        print(f"No images in {IMAGES_DIR}")
        sys.exit(1)

    # Quick model check
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_base = os.path.join(OUTPUT_DIR, "_model_check")
    test_r = subprocess.run(["tesseract", images[0], tmp_base,
                             "-l", LANG, "--psm", PSM],
                            capture_output=True, text=True)
    for ext in [".txt", ".tsv"]:
        p = tmp_base + ext
        if os.path.exists(p):
            os.remove(p)
    if test_r.returncode != 0:
        print(f"\nERROR: Tesseract can't load model '{LANG}'.")
        print(f"Fix (one-time):")
        models_dir = repo_paths.workspace("training", "models")
        print(f"  sudo ln -s {models_dir}/{LANG}.traineddata \\")
        print(f"             /usr/share/tesseract-ocr/5/tessdata/{LANG}.traineddata")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"Model: {LANG}")
    print(f"PSM: {PSM}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for i, img_path in enumerate(images, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_base = os.path.join(OUTPUT_DIR, base)

        text, word_confs = run_tesseract(img_path, out_base)

        with open(out_base + ".ocr.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")

        with open(out_base + ".conf.tsv", "w", encoding="utf-8") as f:
            f.write("word\tconfidence\n")
            for word, conf in word_confs:
                f.write(f"{word}\t{conf}\n")

        n = len(word_confs)
        avg = (sum(c for _, c in word_confs) / n) if n else 0
        print(f"  [{i}/{len(images)}] {base}: {n} words, avg conf {avg:.1f}%")

    print(f"\nDone. OCR in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()