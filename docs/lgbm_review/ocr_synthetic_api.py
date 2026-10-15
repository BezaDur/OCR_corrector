"""OCR the synthetic page images with Tesseract's Python API (tesserocr).

Same output as 4_ocr_correction/error_detection/run_ocr.py: for each image
    <out>/page_NNNN.ocr.txt    OCR text
    <out>/page_NNNN.conf.tsv   word<TAB>confidence per word
Used where the `tesseract` command-line binary is not installed. Same engine, same
language model (amh_fast), same page segmentation mode (3). Check it on one page against
the command-line output before trusting it (see the note in the plan document).

Usage:
    python docs/lgbm_review/ocr_synthetic_api.py \
        --images synthetic_eval/images --out synthetic_eval/OCRed \
        --models models --lang amh_fast --psm 3
"""
import argparse
import glob
import os

import tesserocr
from tesserocr import PyTessBaseAPI, RIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True, help="folder that holds <lang>.traineddata")
    ap.add_argument("--lang", default="amh_fast")
    ap.add_argument("--psm", type=int, default=3)
    args = ap.parse_args()

    images = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        images.extend(glob.glob(os.path.join(args.images, pat)))
    images.sort()
    if not images:
        raise SystemExit(f"no images in {args.images}")
    os.makedirs(args.out, exist_ok=True)

    api = PyTessBaseAPI(path=args.models, lang=args.lang, psm=args.psm)
    print(f"tesseract {tesserocr.tesseract_version().splitlines()[0]}; model {args.lang}; psm {args.psm}")
    for i, img in enumerate(images, 1):
        base = os.path.splitext(os.path.basename(img))[0]
        api.SetImageFile(img)
        text = api.GetUTF8Text().strip()
        api.Recognize()
        words = []
        it = api.GetIterator()
        if it is not None:
            while True:
                w = it.GetUTF8Text(RIL.WORD)
                if w and w.strip():
                    words.append((w.strip(), it.Confidence(RIL.WORD)))
                if not it.Next(RIL.WORD):
                    break
        with open(os.path.join(args.out, base + ".ocr.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        with open(os.path.join(args.out, base + ".conf.tsv"), "w", encoding="utf-8") as f:
            f.write("word\tconfidence\n")
            for w, c in words:
                f.write(f"{w}\t{c:.6f}\n")
        print(f"  [{i}/{len(images)}] {base}: {len(words)} words")
    api.End()
    print(f"done: {len(images)} pages in {args.out}")


if __name__ == "__main__":
    main()
