#!/usr/bin/env python3
"""
Run Tesseract OCR on all images in Evaluation/images/.

Produces two files per image in Evaluation/OCRed/:
  page_NNNN.ocr.txt   — the OCR text
  page_NNNN.conf.tsv  — per-word confidence scores

Usage:
    python OCR_error_detection/run_ocr.py
"""
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


def run_tesseract(image_path):
    """Run Tesseract on one image via pytesseract.
    Returns (text, word_confidences).
    """
    try:
        import pytesseract
    except ImportError:
        print("pytesseract not installed. Run: pip install pytesseract")
        sys.exit(1)

    from PIL import Image

    # Point pytesseract to the right tessdata
    old_val = os.environ.get("TESSDATA_PREFIX", "")
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

    try:
        img = Image.open(image_path)

        # Get full TSV data (level 5 = word)
        tsv = pytesseract.image_to_data(img, lang=LANG, config=f"--psm {PSM}",
                                         output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(img, lang=LANG, config=f"--psm {PSM}").strip()

        word_confs = []
        n = len(tsv["level"])
        for i in range(n):
            if str(tsv["level"][i]) == "5":
                word = str(tsv["text"][i]).strip()
                if word:
                    try:
                        conf = float(tsv["conf"][i])
                    except (ValueError, TypeError):
                        conf = -1.0
                    word_confs.append((word, conf))
    finally:
        os.environ["TESSDATA_PREFIX"] = old_val

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

    # Check pytesseract
    try:
        import pytesseract
        print(f"pytesseract {pytesseract.__version__} (get_tesseract_version={pytesseract.get_tesseract_version()})")
    except ImportError:
        print("ERROR: pytesseract not installed. Run: pip install pytesseract")
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

        text, word_confs = run_tesseract(img_path)

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