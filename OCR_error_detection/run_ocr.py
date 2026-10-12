#!/usr/bin/env python3
"""
Run Tesseract OCR on all images in Evaluation/images/.

Produces two files per image in Evaluation/OCRed/:
  page_NNNN.ocr.txt   — the OCR text
  page_NNNN.conf.tsv  — per-word confidence scores

Usage:
    python OCR_error_detection/run_ocr.py
"""
import csv
import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(_HERE, "Evaluation", "images")
OUTPUT_DIR = os.path.join(_HERE, "Evaluation", "OCRed")
LANG = "amh_epoch2_fast"
TESSDATA_DIR = "/media/des/Tess/Tesseract2/training/models"
PSM = "3"


def run_tesseract(image_path, output_base):
    """Run Tesseract on one image. Returns (text, word_confidences)."""
    # plain text
    r = subprocess.run(
        ["tesseract", image_path, output_base,
         "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [ERROR] {r.stderr.strip()}")
        return "", []

    text = ""
    txt_path = output_base + ".txt"
    if os.path.exists(txt_path):
        text = open(txt_path, encoding="utf-8").read().strip()
        os.remove(txt_path)

    # TSV with confidence
    r = subprocess.run(
        ["tesseract", image_path, output_base + "_tsv",
         "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM, "tsv"],
        capture_output=True, text=True,
    )
    word_confs = []
    tsv_path = output_base + "_tsv.tsv"
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

    return text, word_confs


def main():
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    images = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(IMAGES_DIR, pat)))
    images.sort()

    if not images:
        print(f"No images in {IMAGES_DIR}")
        sys.exit(1)

    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Tesseract not found. Install: sudo apt install tesseract-ocr tesseract-ocr-amh")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(images)} images. Output: {OUTPUT_DIR}\n")

    for i, img_path in enumerate(images, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_base = os.path.join(OUTPUT_DIR, base)

        text, word_confs = run_tesseract(img_path, out_base)

        # save OCR text
        with open(out_base + ".ocr.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")

        # save confidence
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