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


def find_system_tessdata():
    """Find the system tessdata directory by asking Tesseract."""
    try:
        r = subprocess.run(["tesseract", "--print-parameters"],
                           capture_output=True, text=True)
        # tessdata path is in `tessedit_char_whitelist` line? No.
        # Better: check common locations or env var
        pass
    except Exception:
        pass

    candidates = [
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
        "/usr/local/share/tessdata",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    # Use env var
    env = os.environ.get("TESSDATA_PREFIX", "")
    if env and os.path.isdir(env):
        return env

    return None


def ensure_model_in_tessdata():
    """Symlink the fine-tuned model into the system tessdata dir
    so Tesseract can find it without --tessdata-dir
    (which would break the tsv/hocr config file lookup)."""
    model_src = os.path.join(TESSDATA_DIR, LANG + ".traineddata")
    if not os.path.exists(model_src):
        print(f"ERROR: Model not found: {model_src}")
        if os.path.isdir(TESSDATA_DIR):
            available = [f for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata")]
            if available:
                print(f"Available in {TESSDATA_DIR}:")
                for f in sorted(available):
                    print(f"  {f}")
            else:
                print(f"No .traineddata files in {TESSDATA_DIR}")
        sys.exit(1)

    system_td = find_system_tessdata()
    if not system_td:
        print("ERROR: Cannot find system tessdata directory.")
        print("Set TESSDATA_PREFIX env variable to your tessdata path.")
        sys.exit(1)

    model_dest = os.path.join(system_td, LANG + ".traineddata")

    # Already there?
    if os.path.exists(model_dest):
        return

    # Symlink
    try:
        os.symlink(model_src, model_dest)
        print(f"Symlinked model: {model_dest} -> {model_src}")
    except PermissionError:
        print(f"ERROR: Cannot symlink to {system_td} (permission denied).")
        print(f"Run manually: sudo ln -s {model_src} {model_dest}")
        sys.exit(1)


def run_tesseract(image_path):
    """Run Tesseract on one image via pytesseract.
    Returns (text, word_confidences).
    """
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)

    # Don't use --tessdata-dir (it breaks tsv/hocr config lookup).
    # Model is accessible from the system tessdata dir via symlink.
    config = f"--psm {PSM}"

    tsv = pytesseract.image_to_data(img, lang=LANG, config=config,
                                     output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(img, lang=LANG, config=config).strip()

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

    return text, word_confs


def main():
    print(f"Python: {sys.executable}")
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True)
        print(f"Tesseract: {r.stdout.strip().split(chr(10))[0]}")
    except FileNotFoundError:
        print("ERROR: Tesseract not found.")
        sys.exit(1)

    try:
        import pytesseract
        print(f"pytesseract {pytesseract.__version__}")
    except ImportError:
        print(f"\nERROR: pytesseract not importable by {sys.executable}")
        print(f"Fix: {sys.executable} -m pip install pytesseract Pillow")
        sys.exit(1)

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    images = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(IMAGES_DIR, pat)))
    images.sort()

    if not images:
        print(f"No images in {IMAGES_DIR}")
        sys.exit(1)

    ensure_model_in_tessdata()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Found {len(images)} images")
    print(f"Model: {LANG}")
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