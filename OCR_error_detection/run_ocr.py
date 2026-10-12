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
    """Run Tesseract on one image. Returns (text, word_confidences)."""
    # --- Call 1: plain text output ---
    cmd = ["tesseract", image_path, output_base,
           "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [ERROR] text call failed (rc={r.returncode}): {r.stderr.strip()}")
        return "", []

    text = ""
    txt_path = output_base + ".txt"
    if os.path.exists(txt_path):
        text = open(txt_path, encoding="utf-8").read().strip()
        os.remove(txt_path)
    else:
        print(f"    [WARN] no .txt at {txt_path}")

    # --- Call 2: TSV output for word confidences ---
    tsv_base = output_base + "_tsv"
    cmd_tsv = ["tesseract", image_path, tsv_base,
               "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM, "tsv"]
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
        print(f"    [WARN] TSV not created: {tsv_path}")
        print(f"    [WARN] tsv rc={r2.returncode}")
        print(f"    [WARN] tsv stderr: {r2.stderr.strip()[:200]}")
        # Try hocr as fallback
        hocr_base = output_base + "_hocr"
        cmd_hocr = ["tesseract", image_path, hocr_base,
                    "-l", LANG, "--tessdata-dir", TESSDATA_DIR, "--psm", PSM, "hocr"]
        r3 = subprocess.run(cmd_hocr, capture_output=True, text=True)
        hocr_path = hocr_base + ".hocr"
        if os.path.exists(hocr_path):
            print(f"    [INFO] hocr output works — TSV not supported by this Tesseract")
            os.remove(hocr_path)
        else:
            print(f"    [INFO] hocr also failed (rc={r3.returncode}): {r3.stderr.strip()[:200]}")

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

    # Print Tesseract version
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
        print(r.stdout.strip().split("\n")[0])
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