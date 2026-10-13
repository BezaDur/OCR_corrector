"""Render synthetic evaluation pages from source text.

The rendered text is the ground truth (no transcription). Each page image gets a
degradation level (clean / light / medium) so results can be broken down later.

Input text file: UTF-8. Pages are separated by a line that contains only '---'.
Within a page, each line is a line of text; blank lines are kept as paragraph breaks.
Long lines are wrapped to the page width.

Output (same layout as OCR_error_detection/Evaluation/):
    <out>/images/page_NNNN.png
    <out>/GroundTruth/page_NNNN.gt.txt
    <out>/manifest.csv

Usage (on any machine with Pillow and an Ethiopic font):
    python docs/lgbm_review/make_synthetic_set.py \
        --text new_source.txt --font /path/a.ttf /path/b.ttf ... \
        --out fresh_eval --source "publisher / title" --collected 2026-10 \
        --size 36 --dpi 300 --width 2480 --height 3508

Then OCR the images with the deployment settings (run_ocr.py --images ... --out ...).
"""
import argparse
import csv
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import numpy as np

LEVELS = {
    # blur radius (px), gaussian noise sigma (0-255 scale), max skew (degrees), contrast factor
    "clean": dict(blur=0.0, noise=0.0, skew=0.0, contrast=1.0),
    "light": dict(blur=0.6, noise=6.0, skew=0.4, contrast=0.9),
    "medium": dict(blur=1.1, noise=14.0, skew=1.0, contrast=0.8),
}


def split_pages(path):
    pages, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "---":
                pages.append(cur)
                cur = []
            else:
                cur.append(line)
    if any(l.strip() for l in cur):
        pages.append(cur)
    return [p for p in pages if any(l.strip() for l in p)]


def wrap(font, text, max_width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if font.getlength(trial) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(page_lines, font, width, height, margin, line_gap):
    """Draw whole source lines only. Returns (image, rendered_lines, truncated).

    A source line is drawn only if all its wrapped lines fit on the page. The ground
    truth is the rendered lines, so it always matches the image."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    y = margin
    max_w = width - 2 * margin
    line_h = font.size + line_gap
    rendered, truncated = [], False
    for raw in page_lines:
        if not raw.strip():
            y += line_h // 2
            rendered.append("")
            continue
        wrapped = wrap(font, raw.strip(), max_w)
        needed = len(wrapped) * line_h
        if y + needed > height - margin:
            truncated = True
            break
        for ln in wrapped:
            draw.text((margin, y), ln, font=font, fill=0)
            y += line_h
        rendered.append(raw)
    return img, rendered, truncated


def degrade(img, level, rng):
    p = LEVELS[level]
    if p["skew"]:
        angle = rng.uniform(-p["skew"], p["skew"])
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    if p["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    if p["contrast"] != 1.0:
        arr = np.asarray(img, dtype=np.float32)
        arr = 255 - (255 - arr) * p["contrast"]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if p["noise"]:
        arr = np.asarray(img, dtype=np.float32)
        rs = np.random.default_rng(rng.randrange(2**32))
        arr = arr + rs.normal(0, p["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return ImageOps.autocontrast(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--font", required=True, nargs="+",
                    help="one or more TTF/OTF files with Ethiopic glyphs; pages rotate through them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True, help="where the text came from (manifest)")
    ap.add_argument("--collected", required=True, help="date collected, e.g. 2026-10")
    ap.add_argument("--size", type=int, default=36, help="font size in pixels")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=int, default=2480)
    ap.add_argument("--height", type=int, default=3508)
    ap.add_argument("--margin", type=int, default=160)
    ap.add_argument("--line-gap", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    fonts = [(path, ImageFont.truetype(path, args.size)) for path in args.font]
    pages = split_pages(args.text)
    if not pages:
        raise SystemExit("no pages found in the text file")

    img_dir = os.path.join(args.out, "images")
    gt_dir = os.path.join(args.out, "GroundTruth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    rng = random.Random(args.seed)
    levels = list(LEVELS)

    rows = []
    for i, lines in enumerate(pages, 1):
        name = f"page_{i:04d}"
        level = levels[(i - 1) % len(levels)]
        font_path, font = fonts[(i - 1) % len(fonts)]
        img, rendered, truncated = render(lines, font, args.width, args.height,
                                          args.margin, args.line_gap)
        img = degrade(img, level, rng)
        img.save(os.path.join(img_dir, name + ".png"), dpi=(args.dpi, args.dpi))
        with open(os.path.join(gt_dir, name + ".gt.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rendered).strip() + "\n")
        rows.append(dict(page=name, source=args.source, date_collected=args.collected,
                         font=os.path.basename(font_path), font_size=args.size,
                         degradation=level, seed=args.seed,
                         notes="truncated: lines not rendered were dropped from GT" if truncated else ""))

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} pages to {args.out}")


if __name__ == "__main__":
    main()
