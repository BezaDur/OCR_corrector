#!/usr/bin/env python3
"""
Generate 100 synthetic evaluation pages with images and ground truth.
Then run Tesseract OCR to produce paired OCR output.

Output:
  eval_data/images/page_NNNN.png          (degraded document images)
  eval_data/ground_truth/page_NNNN.gt.txt (clean text)
  eval_data/ocr_output/page_NNNN.ocr.txt  (Tesseract OCR output)
"""
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CORPUS = "amh.training_text"
FONTS_DIR = "eval_fonts"
OUTPUT_DIR = "eval_data"
NUM_PAGES = 100
LINES_PER_PAGE = 28
SEED = 42

PAGE_W, PAGE_H = 1240, 1754
MARGIN_L, MARGIN_R = 120, 120
MARGIN_T, MARGIN_B = 120, 120
LINE_SPACING = 14


def load_fonts(fonts_dir, size):
    fonts = []
    for f in sorted(os.listdir(fonts_dir)):
        if f.lower().endswith('.ttf'):
            path = os.path.join(fonts_dir, f)
            try:
                ImageFont.truetype(path, size)
                fonts.append(path)
                print(f"  font OK: {f}")
            except Exception as e:
                print(f"  SKIP {f}: {e}")
    return fonts


def wrap_text(paragraphs, font, max_width):
    wrapped = []
    dummy = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(dummy)
    for para in paragraphs:
        tokens = para.strip().split()
        if not tokens:
            wrapped.append("")
            continue
        line = []
        for tok in tokens:
            test = " ".join(line + [tok])
            bbox = d.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                line.append(tok)
            else:
                if line:
                    wrapped.append(" ".join(line))
                line = [tok]
        if line:
            wrapped.append(" ".join(line))
    return wrapped


def render_page(lines, font_path, font_size):
    bg = random.randint(246, 255)
    img = Image.new("RGB", (PAGE_W, PAGE_H), color=(bg, bg, bg))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    max_w = PAGE_W - MARGIN_L - MARGIN_R
    max_h = PAGE_H - MARGIN_T - MARGIN_B
    wrapped = wrap_text(lines, font, max_w)
    page_lines = []
    y = MARGIN_T
    for line in wrapped:
        if not line:
            y += LINE_SPACING * 2
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        lh = bbox[3] - bbox[1]
        if y + lh > MARGIN_T + max_h:
            break
        draw.text((MARGIN_L, y), line, font=font, fill=(15, 15, 15))
        page_lines.append(line)
        y += lh + LINE_SPACING
    return img, page_lines


def degrade(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # ink bleed
    k = np.ones((2, 2), np.uint8)
    if random.random() > 0.5:
        img = cv2.erode(img, k, iterations=1)
    img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.6))
    # spine shadow
    h, w = img.shape[:2]
    sw = random.randint(int(w * 0.06), int(w * 0.12))
    grad = np.linspace(random.uniform(0.6, 0.8), 1.0, sw)
    grad = np.tile(grad, (h, 1))
    if random.random() > 0.5:
        img[:, w - sw:] = (img[:, w - sw:] * grad[:, :, np.newaxis]).astype(np.uint8)
    else:
        img[:, :sw] = (img[:, :sw] * grad[:, :, np.newaxis]).astype(np.uint8)
    # noise
    noise = np.random.normal(0, random.uniform(1.0, 3.0), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def load_corpus(path, n_lines):
    lines = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if len(s) > 15:
                lines.append(s)
            if len(lines) >= n_lines:
                break
    return lines


def main():
    rng = random.Random(SEED)
    font_size = rng.randint(30, 38)
    fonts = load_fonts(FONTS_DIR, font_size)
    if not fonts:
        print("ERROR: no fonts"); return

    needed = NUM_PAGES * LINES_PER_PAGE + 200
    print(f"Loading {needed} lines from {CORPUS}...")
    all_lines = load_corpus(CORPUS, needed)
    print(f"  loaded {len(all_lines)} lines")
    rng.shuffle(all_lines)

    img_dir = os.path.join(OUTPUT_DIR, "images")
    gt_dir = os.path.join(OUTPUT_DIR, "ground_truth")
    ocr_dir = os.path.join(OUTPUT_DIR, "ocr_output")
    for d in (img_dir, gt_dir, ocr_dir):
        os.makedirs(d, exist_ok=True)

    line_idx = 0
    total = len(all_lines)
    count = 0

    for p in range(1, NUM_PAGES + 1):
        if line_idx + LINES_PER_PAGE > total:
            line_idx = 0
            rng.shuffle(all_lines)

        chunk = all_lines[line_idx:line_idx + LINES_PER_PAGE]
        line_idx += LINES_PER_PAGE

        font_path = rng.choice(fonts)
        clean_img, page_lines = render_page(chunk, font_path, font_size)
        if not page_lines:
            continue

        degraded = degrade(clean_img)
        base = f"page_{p:04d}"
        cv2.imwrite(os.path.join(img_dir, f"{base}.png"), degraded)
        with open(os.path.join(gt_dir, f"{base}.gt.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(page_lines).strip() + "\n")

        count += 1
        if p % 25 == 0:
            print(f"  [{p}/{NUM_PAGES}] done")

    print(f"\nRendered {count} pages to {OUTPUT_DIR}/")
    return count


if __name__ == "__main__":
    main()