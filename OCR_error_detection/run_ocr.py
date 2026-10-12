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
LANG = "amh_fast"
TESSDATA_DIR = "/media/des/Tess/Tesseract2/training/models"
PSM = "3"


def run_tesseract(image_path, output_base):
    """Run Tesseract on one image using TSV mode.
    Returns (text, word_confidences)."""
    # Single Tesseract call in TSV mode
    cmd = ["tesseract", image_path, output_base,
           "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM, "tsv"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [ERROR] {r.stderr.strip()}")
        return "", []

    tsv_path = output_base + ".tsv"
    if not os.path.exists(tsv_path):
        print(f"    [ERROR] TSV not created: {tsv_path}")
        return "", []

    # Parse TSV: collect word-level text and confidences,
    # and reconstruct full text by line
    word_confs = []
    lines = []
    current_line_words = []

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        prev_block_num = None
        prev_par_num = None
        prev_line_num = None

        for row in reader:
            level = row.get("level", "0")
            text = row.get("text", "").strip()
            conf = row.get("conf", "-1")
            block_num = row.get("block_num", "0")
            par_num = row.get("par_num", "0")
            line_num = row.get("line_num", "0")

            # Level 5 = word
            if level == "5" and text:
                try:
                    conf_val = float(conf)
                except ValueError:
                    conf_val = -1.0
                word_confs.append((text, conf_val))
                current_line_words.append(text)

            # Level 4 = line — flush the line words
            if level == "4":
                if current_line_words:
                    lines.append(" ".join(current_line_words))
                    current_line_words = []

        # Flush last line
        if current_line_words:
            lines.append(" ".join(current_line_words))

    os.remove(tsv_path)
    text = "\n".join(lines)
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
        print("Tesseract not found.")
        sys.exit(1)

    # Check model exists
    model_path = os.path.join(TESSDATA_DIR, LANG + ".traineddata")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        if os.path.isdir(TESSDATA_DIR):
            available = [f for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata")]
            if available:
                print(f"Available in {TESSDATA_DIR}:")
                for f in sorted(available):
                    print(f"  {f}")
            else:
                print(f"No .traineddata files in {TESSDATA_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(images)} images")
    print(f"Model: {model_path}")
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