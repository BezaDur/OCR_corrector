#!/usr/bin/env python3
"""
Step 1: Production document synthesis engine using OpenCV and Pillow.
Features:
  - Targeted Font Fallback (prevents .notdef hollow boxes forever).
  - Explicit Typographic Pairings for Brana, FreeSerif, and Geez-Manuscript.
  - Physically-modeled document degradation: spine curvature, ink bleed,
    non-uniform lighting gradients, and sensor noise.
"""

# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "repo_paths.py")):
    _root = _os.path.dirname(_root)
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import repo_paths  # noqa: E402  (imported after the path bootstrap)
# ----------------------------------------------------------------------------

import os
import sys
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# ----------------- Configuration -----------------
FONTS_DIR = repo_paths.fonts()
GT_SOURCE_FILE = repo_paths.workspace("ground_truth/merged.txt")
OUTPUT_DIR = repo_paths.workspace("synthetic_eval")

# Explicit typographic pairings for fonts with incomplete cmaps
FALLBACK_PAIRS = {
    "Brana-Regular.ttf": os.path.join(FONTS_DIR, "AbyssinicaSIL-R.ttf"),
    "FreeSerifBold.ttf": os.path.join(FONTS_DIR, "FreeSerif.ttf"),
    "FreeSerifItalic.ttf": os.path.join(FONTS_DIR, "FreeSerif.ttf"),
    "FreeSerifBoldItalic.ttf": os.path.join(FONTS_DIR, "FreeSerif.ttf"),
    "Geez-Manuscript-Zemen-COLR.ttf": os.path.join(FONTS_DIR, "AbyssinicaSIL-R.ttf"),
}

# Universal default fallback if an unexpected missing glyph occurs
DEFAULT_FALLBACK_PATH = os.path.join(FONTS_DIR, "NotoSansEthiopic-Regular.ttf")

# Whitelist of verified, real-world document fonts + manuscript font
VERIFIED_FONTS = [
    "AbyssinicaSIL-R.ttf",
    "nyala.ttf",
    "Brana-Regular.ttf",
    "NotoSansEthiopic-Regular.ttf",
    "NotoSansEthiopic-Bold.ttf",
    "NotoSansEthiopic-VariableFont_wdth,wght.ttf",
    "jiret.ttf",
    "jiretsl.ttf",
    "FreeSerif.ttf",
    "washrab.ttf",
    "washra-BoldSlant.ttf",
    "washra-BS.ttf",
    "hiwua.ttf",
    "wookianos.ttf",
    "yebse.ttf",
    "Geez-Manuscript-Zemen-COLR.ttf"
]

NUM_PAGES_TO_GENERATE = 150

PAGE_WIDTH = 1654
PAGE_HEIGHT = 2338
MARGIN_LEFT = 150
MARGIN_RIGHT = 150
MARGIN_TOP = 150
MARGIN_BOTTOM = 150
DEFAULT_FONT_SIZE = 34
LINE_SPACING = 16

FONT_CMAP_CACHE = {}
FONT_OBJ_CACHE = {}
# -------------------------------------------------


def get_font_codepoints(font_path):
    if font_path not in FONT_CMAP_CACHE:
        try:
            tt = TTFont(font_path)
            cps = set()
            for table in tt['cmap'].tables:
                if table.isUnicode():
                    cps.update(table.cmap.keys())
            FONT_CMAP_CACHE[font_path] = cps
        except Exception as e:
            print(f"[WARN] Could not parse cmap for {font_path}: {e}")
            FONT_CMAP_CACHE[font_path] = set()
    return FONT_CMAP_CACHE[font_path]


def get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in FONT_OBJ_CACHE:
        FONT_OBJ_CACHE[key] = ImageFont.truetype(font_path, size)
    return FONT_OBJ_CACHE[key]


def draw_text_with_targeted_fallback(draw, xy, text, primary_font_path, font_size, fill=(15, 15, 15)):
    """
    Renders text honoring the primary font, falling back cleanly to the
    designated companion font when a glyph is missing in the primary cmap.
    """
    primary_font_name = os.path.basename(primary_font_path)
    primary_cps = get_font_codepoints(primary_font_path)
    primary_font = get_cached_font(primary_font_path, font_size)

    fallback_path = FALLBACK_PAIRS.get(primary_font_name, DEFAULT_FALLBACK_PATH)
    fallback_font = get_cached_font(fallback_path, font_size)

    cur_x, cur_y = xy
    for char in text:
        cp = ord(char)
        use_font = primary_font if cp in primary_cps else fallback_font
        
        draw.text((cur_x, cur_y), char, font=use_font, fill=fill)
        
        bbox = draw.textbbox((cur_x, cur_y), char, font=use_font)
        char_w = bbox[2] - bbox[0]
        cur_x += char_w


def wrap_text_to_lines(paragraphs, font, max_width):
    wrapped_lines = []
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    for para in paragraphs:
        tokens = para.strip().split()
        if not tokens:
            continue

        current_line = []
        for token in tokens:
            test_line = " ".join(current_line + [token])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(token)
            else:
                if current_line:
                    wrapped_lines.append(" ".join(current_line))
                    current_line = [token]
                else:
                    wrapped_lines.append(token)
                    current_line = []

        if current_line:
            wrapped_lines.append(" ".join(current_line))

        # Blank line between paragraphs
        wrapped_lines.append("")

    return wrapped_lines


def render_clean_page(paragraphs_subset, font_path, font_size, max_width, max_height):
    bg = random.randint(248, 255)
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(bg, bg, bg))
    draw = ImageDraw.Draw(canvas)

    primary_font = get_cached_font(font_path, font_size)
    wrapped = wrap_text_to_lines(paragraphs_subset, primary_font, max_width)

    page_lines = []
    y = MARGIN_TOP

    for line in wrapped:
        if not line:
            y += LINE_SPACING * 2
            continue

        bbox = draw.textbbox((0, 0), line, font=primary_font)
        line_height = bbox[3] - bbox[1]

        if y + line_height > (MARGIN_TOP + max_height):
            break

        # Uses the targeted fallback renderer character-by-character
        draw_text_with_targeted_fallback(
            draw=draw,
            xy=(MARGIN_LEFT, y),
            text=line,
            primary_font_path=font_path,
            font_size=font_size,
            fill=(15, 15, 15)
        )

        page_lines.append(line)
        y += line_height + LINE_SPACING

    return canvas, page_lines


def apply_book_spine_shadow(img_bgr, side="left"):
    h, w = img_bgr.shape[:2]
    shadow_width = random.randint(int(w * 0.08), int(w * 0.16))
    gradient = np.linspace(random.uniform(0.55, 0.75), 1.0, shadow_width)
    gradient = np.tile(gradient, (h, 1))

    if side == "right":
        gradient = np.fliplr(gradient)
        img_bgr[:, w - shadow_width:] = (img_bgr[:, w - shadow_width:] * gradient[:, :, np.newaxis]).astype(np.uint8)
    else:
        img_bgr[:, :shadow_width] = (img_bgr[:, :shadow_width] * gradient[:, :, np.newaxis]).astype(np.uint8)

    return img_bgr


def apply_lighting_gradient(img_bgr):
    h, w = img_bgr.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    mask = 1.0 - (0.15 * (xx**2 + yy**2) / 2.0)
    return (img_bgr * mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


def apply_ink_bleed(img_bgr):
    kernel = np.ones((2, 2), np.uint8)
    if random.random() > 0.5:
        img_bgr = cv2.erode(img_bgr, kernel, iterations=1)
    return cv2.GaussianBlur(img_bgr, (3, 3), random.uniform(0.3, 0.6))


def apply_sensor_noise(img_bgr):
    noise = np.random.normal(0, random.uniform(1.5, 3.5), img_bgr.shape)
    return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_page(img_bgr):
    img = apply_ink_bleed(img_bgr)
    img = apply_book_spine_shadow(img, side=random.choice(["left", "right"]))
    img = apply_lighting_gradient(img)
    img = apply_sensor_noise(img)
    
    angle = random.uniform(-0.5, 0.5)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def load_available_fonts(fonts_path):
    fonts = []
    for f in VERIFIED_FONTS:
        p = os.path.join(fonts_path, f)
        if os.path.exists(p):
            fonts.append(p)
        else:
            print(f"[NOTE] Font not found (skipped): {f}")

    if not fonts:
        print("Error: No whitelisted fonts found!")
        sys.exit(1)

    print(f"Loaded {len(fonts)} verified document fonts.")
    return fonts


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Generate synthetic evaluation pages (fonts → degraded page images + ground truth).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts-dir", default=FONTS_DIR,
                    help="directory with the Ethiopic fonts")
    ap.add_argument("--gt-source", default=GT_SOURCE_FILE,
                    help="merged ground-truth text to draw paragraphs from")
    ap.add_argument("--out-dir", default=OUTPUT_DIR,
                    help="where to write the pages + ground truth")
    ap.add_argument("--pages", type=int, default=NUM_PAGES_TO_GENERATE,
                    help="how many pages to generate (default %(default)s)")
    ap.add_argument("-v", "--verbose", action="store_true", help="more output")
    return ap.parse_args(argv)


def main(argv=None):
    global FONTS_DIR, GT_SOURCE_FILE, OUTPUT_DIR, NUM_PAGES_TO_GENERATE
    args = _parse_args(argv)
    FONTS_DIR, GT_SOURCE_FILE = args.fonts_dir, args.gt_source
    OUTPUT_DIR, NUM_PAGES_TO_GENERATE = args.out_dir, args.pages

    images_out = os.path.join(OUTPUT_DIR, "images")
    gt_out = os.path.join(OUTPUT_DIR, "ground_truth")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    if not os.path.exists(GT_SOURCE_FILE):
        print(f"Error: Source ground truth not found at {GT_SOURCE_FILE}")
        sys.exit(1)

    font_paths = load_available_fonts(FONTS_DIR)

    print(f"Reading text from {GT_SOURCE_FILE}...")
    sample_paragraphs = []
    with open(GT_SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                sample_paragraphs.append(stripped)
            if len(sample_paragraphs) >= NUM_PAGES_TO_GENERATE * 40:
                break

    print(f"Buffered {len(sample_paragraphs)} paragraphs. Generating {NUM_PAGES_TO_GENERATE} pages...")

    usable_width = PAGE_WIDTH - (MARGIN_LEFT + MARGIN_RIGHT)
    usable_height = PAGE_HEIGHT - (MARGIN_TOP + MARGIN_BOTTOM)

    para_cursor = 0
    total_paras = len(sample_paragraphs)

    for page_idx in range(1, NUM_PAGES_TO_GENERATE + 1):
        if para_cursor >= total_paras:
            print("Reached end of source text buffer.")
            break

        chosen_font_path = random.choice(font_paths)
        chunk = sample_paragraphs[para_cursor : para_cursor + 35]

        clean_pil_img, page_lines = render_clean_page(
            paragraphs_subset=chunk,
            font_path=chosen_font_path,
            font_size=DEFAULT_FONT_SIZE,
            max_width=usable_width,
            max_height=usable_height
        )

        if not page_lines:
            break

        consumed_paras = max(1, len("\n".join(page_lines).split("\n\n")))
        para_cursor += consumed_paras

        cv_img = cv2.cvtColor(np.array(clean_pil_img), cv2.COLOR_RGB2BGR)
        degraded_img = degrade_page(cv_img)

        base_name = f"page_{page_idx:04d}"
        img_file_path = os.path.join(images_out, f"{base_name}.png")
        gt_file_path = os.path.join(gt_out, f"{base_name}.gt.txt")

        cv2.imwrite(img_file_path, degraded_img)

        with open(gt_file_path, "w", encoding="utf-8") as gf:
            gf.write("\n".join(page_lines).strip() + "\n")

        if page_idx % 25 == 0 or page_idx == NUM_PAGES_TO_GENERATE:
            print(f"Generated [{page_idx}/{NUM_PAGES_TO_GENERATE}] pages...")

    print("\nDataset generation finished successfully.")
    print(f"Images:       {images_out}")
    print(f"Ground Truth: {gt_out}")


if __name__ == "__main__":
    main()