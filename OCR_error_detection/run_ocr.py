#!/usr/bin/env python3
"""
Run Tesseract OCR on all images in Evaluation/images/.

Produces two files per image in Evaluation/OCRed/:
  page_NNNN.ocr.txt   — the OCR text (one word per line, matching GT format)
  page_NNNN.conf.tsv  — per-word confidence scores from Tesseract

Usage (run from repo root or from OCR_error_detection/):
    python OCR_error_detection/run_ocr.py

    # Override paths:
    python OCR_error_detection/run_ocr.py \
        --images OCR_error_detection/Evaluation/images \
        --output OCR_error_detection/Evaluation/OCRed \
        --lang amh --psm 6

Prerequisites:
    - Tesseract installed with Amharic language data
      Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-amh
      macOS:         brew install tesseract tesseract-lang
"""
import argparse
import csv
import glob
import os
import subprocess
import sys


def run_tesseract(image_path, output_base, lang="amh", psm="6"):
    """Run Tesseract on one image. Returns (text, word_confidences).

    word_confidences: list of (word, confidence) tuples.
    """
    # 1. Get plain text
    cmd_txt = [
        "tesseract", image_path, output_base,
        "-l", lang, "--psm", psm,
    ]
    r = subprocess.run(cmd_txt, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERROR] tesseract text failed: {r.stderr.strip()}")
        return "", []

    txt_path = output_base + ".txt"
    text = ""
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            text = f.read().strip()

    # 2. Get TSV output (has per-word confidence)
    cmd_tsv = [
        "tesseract", image_path, output_base + "_tsv",
        "-l", lang, "--psm", psm, "tsv",
    ]
    r = subprocess.run(cmd_tsv, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] tesseract tsv failed: {r.stderr.strip()}")
        return text, []

    tsv_path = output_base + "_tsv.tsv"
    word_confs = []
    if os.path.exists(tsv_path):
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                word = row.get("text", "").strip()
                conf_str = row.get("conf", "-1")
                level = row.get("level", "0")
                # level 5 = word level
                if level == "5" and word:
                    try:
                        conf = float(conf_str)
                    except ValueError:
                        conf = -1.0
                    word_confs.append((word, conf))
        os.remove(tsv_path)  # clean up temp TSV

    return text, word_confs


def main():
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on images with confidence output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images",
        default=os.path.join(os.path.dirname(__file__), "Evaluation", "images"),
        help="Directory of input images (default: Evaluation/images)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "Evaluation", "OCRed"),
        help="Output directory (default: Evaluation/OCRed)",
    )
    parser.add_argument("--lang", default="amh", help="Tesseract language (default: amh)")
    parser.add_argument("--psm", default="6", help="Page segmentation mode (default: 6)")
    args = parser.parse_args()

    # Collect images
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(os.path.join(args.images, pat)))
    image_files.sort()

    if not image_files:
        print(f"No images found in {args.images}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {args.images}")
    print(f"Output: {args.output}")
    print(f"Language: {args.lang}, PSM: {args.psm}")
    print()

    os.makedirs(args.output, exist_ok=True)

    # Check tesseract is available
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: tesseract not found.")
        print("Install: sudo apt install tesseract-ocr tesseract-ocr-amh")
        sys.exit(1)

    for i, img_path in enumerate(image_files, 1):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_base = os.path.join(args.output, basename)

        text, word_confs = run_tesseract(img_path, output_base, args.lang, args.psm)

        # Save OCR text (word per line to match ground truth format)
        ocr_txt_path = output_base + ".ocr.txt"
        with open(ocr_txt_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        # Save per-word confidence
        conf_path = output_base + ".conf.tsv"
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write("word\tconfidence\n")
            for word, conf in word_confs:
                f.write(f"{word}\t{conf}\n")

        n_words = len(word_confs)
        avg_conf = (sum(c for _, c in word_confs) / n_words) if n_words else 0
        print(f"  [{i}/{len(image_files)}] {basename}: "
              f"{n_words} words, avg confidence {avg_conf:.1f}%")

    print(f"\nDone. OCR output in {args.output}")


if __name__ == "__main__":
    main()