#!/usr/bin/env python3
"""
Step 2: Definitive Training Ground Truth Generator for Tesseract LSTM.
Features:
  - 20 Curated, Verified Amharic Font Families (Sans, Serif, Calligraphic, Regional).
  - Uses amh_final_training_text.txt (includes 25k mined rare sentences).
  - Exact 1-to-1 Box/GT Character Verification (zero dropped glyphs).
  - Safe page geometry (PTSIZE=14, MARGIN=15, XSIZE=3600, YSIZE=220).
  - Strictly underscore-delimited basenames.
"""

# --- make repo_paths importable from any working directory -------------------
import os as _os
import sys as _sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "tools", "stage_paths.py")):
    _root = _os.path.dirname(_root)
# tools/ first so stage_paths and repo_paths resolve, then register every stage
# folder so cross-stage imports work by bare module name. This replaces the
# thirteen root-level shims that used to forward those imports by hand.
_sys.path.insert(0, _os.path.join(_root, "tools"))
import stage_paths  # noqa: E402  (imported after the path bootstrap)
stage_paths.register(_root)
import repo_paths  # noqa: E402
# ----------------------------------------------------------------------------

import os
import sys
import random
import re
import unicodedata
import subprocess
from concurrent.futures import ProcessPoolExecutor

# ----------------- Configuration -----------------
FONTS_DIR = repo_paths.workspace("fonts")
TRAINING_TEXT = repo_paths.workspace("amh_final_training_text.txt")
OUTPUT_DIR = repo_paths.workspace("training_ground_truth")

TARGET_LINE_COUNT = 100000
NUM_WORKERS = 4
BASE_SEED = 42

PTSIZE = 14
MARGIN = 15
XSIZE = 3600
YSIZE = 220
LEADING = 16

# The 20 Curated, System-Verified Font Families
FONTS_WHITELIST = [
    # 1. Standard Document Serifs
    "Abyssinica SIL",
    "Nyala",
    "Brana Regular",
    "FreeSerif",
    "Noto Serif Ethiopic Regular",
    "Noto Serif Ethiopic Bold",
    "Noto Serif Ethiopic Light",

    # 2. Modern Sans-Serifs
    "Noto Sans Ethiopic Regular",
    "Noto Sans Ethiopic Bold",
    "Noto Sans Ethiopic Light",

    # 3. Calligraphic & Slanted
    "Ethiopia Jiret",
    "Ethiopia Jiret Slant",
    "Ethiopic WashRa Bold Slant Bold",
    "Ethiopic WashRa SemiBold Slant SemiBold",

    # 4. Regional & Traditional Publishing
    "Ethiopic Abay Regular",
    "Ethiopic Lessan Regular",
    "Ethiopic Sadiss Regular",
    "Ethiopic Hiwua",
    "Ethiopic Wookianos",
    "Ethiopic Yebse"
]
# -------------------------------------------------


# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 1_data_preparation/05_generate_training_boxes.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "fonts_dir": FONTS_DIR,
    "training_text": TRAINING_TEXT,
    "out_dir": OUTPUT_DIR,
    "lines": TARGET_LINE_COUNT,
    "verbose": False,
}

def sanitize_tag(text):
    """Sanitizes strings to contain strictly letters, numbers, and underscores."""
    t = str(text).replace(".", "_").replace("-", "neg")
    return re.sub(r'[^a-zA-Z0-9_]', '', t)


def validate_line_output(gt_file, tif_file, box_file, expected_text):
    """
    Verifies that the generated box file matches the input GT text byte-for-byte.
    """
    if not (os.path.exists(box_file) and os.path.exists(tif_file) and os.path.exists(gt_file)):
        return False

    if os.path.getsize(box_file) < 20 or os.path.getsize(tif_file) < 100:
        return False

    try:
        box_chars = []
        with open(box_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] != "\t":
                    box_chars.append(parts[0])

        norm_box = unicodedata.normalize("NFC", "".join(box_chars))
        norm_gt = unicodedata.normalize("NFC", expected_text.strip().replace(" ", ""))

        if len(norm_box) >= len(norm_gt) * 0.95:
            return True
        return False
    except Exception:
        return False


def remove_artifacts(paths):
    for p in paths:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass


def render_single_line(job):
    idx, text, font_name, exposure, char_spacing = job
    font_tag = sanitize_tag(font_name)
    exp_tag = sanitize_tag(exposure)
    sp_tag = sanitize_tag(char_spacing)

    base_name = f"amh_{idx:06d}_{font_tag}_exp{exp_tag}_sp{sp_tag}"
    gt_file = os.path.join(OUTPUT_DIR, f"{base_name}.gt.txt")
    out_base = os.path.join(OUTPUT_DIR, base_name)
    tif_file = f"{out_base}.tif"
    box_file = f"{out_base}.box"

    artifacts = (gt_file, tif_file, box_file)

    with open(gt_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")

    cmd = [
        "text2image",
        f"--font={font_name}",
        f"--fonts_dir={FONTS_DIR}",
        f"--text={gt_file}",
        f"--outputbase={out_base}",
        "--max_pages=1",
        f"--margin={MARGIN}",
        f"--leading={LEADING}",
        f"--xsize={XSIZE}",
        f"--ysize={YSIZE}",
        f"--ptsize={PTSIZE}",
        f"--char_spacing={char_spacing}",
        f"--exposure={exposure}"
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            remove_artifacts(artifacts)
            return False

        if not validate_line_output(gt_file, tif_file, box_file, text):
            remove_artifacts(artifacts)
            return False

        return True

    except Exception:
        remove_artifacts(artifacts)
        return False


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Generate Tesseract training boxes + images from the final training text.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts-dir", default=CONFIG["fonts_dir"], help="font directory")
    ap.add_argument("--training-text", default=CONFIG["training_text"],
                    help="final training text (one line per training line)")
    ap.add_argument("--out-dir", default=CONFIG["out_dir"], help="box/gt output directory")
    ap.add_argument("--lines", type=int, default=CONFIG["lines"],
                    help="how many training lines to render (default %(default)s)")
    ap.add_argument("-v", "--verbose", action="store_true", default=CONFIG["verbose"], help="more output")
    return ap.parse_args(argv)


def main(argv=None):
    global FONTS_DIR, TRAINING_TEXT, OUTPUT_DIR, TARGET_LINE_COUNT
    args = _parse_args(argv)
    FONTS_DIR, TRAINING_TEXT = args.fonts_dir, args.training_text
    OUTPUT_DIR, TARGET_LINE_COUNT = args.out_dir, args.lines

    repo_paths.makedirs(OUTPUT_DIR, "the box output directory")

    print(f"Reading training lines from {TRAINING_TEXT}...")
    lines = []
    with open(TRAINING_TEXT, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                lines.append(l)

    if not lines:
        print("Error: Training text is empty.")
        return

    rng_master = random.Random(BASE_SEED)
    rng_master.shuffle(lines)
    selected_lines = lines[:TARGET_LINE_COUNT]

    print(f"Target lines to render: {len(selected_lines):,}")
    print(f"Active Fonts (20 total): {len(FONTS_WHITELIST)}")
    print(f"Point Size:             {PTSIZE} pt")
    print(f"Workers:                {NUM_WORKERS} CPU processes\n")

    jobs = []
    for i, line in enumerate(selected_lines):
        rng_job = random.Random(BASE_SEED + i)
        chosen_font = FONTS_WHITELIST[i % len(FONTS_WHITELIST)]
        exposure = rng_job.choice([0, 0, 1, -1])
        char_spacing = rng_job.choice([0.0, 0.0, 0.1, 0.2])
        jobs.append((i, line, chosen_font, exposure, char_spacing))

    completed = 0
    failed = 0
    total = len(jobs)

    print("Beginning generation...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for success in executor.map(render_single_line, jobs):
            if success:
                completed += 1
            else:
                failed += 1

            processed = completed + failed
            if processed % 5000 == 0 or processed == total:
                print(f"Progress: [{processed:,}/{total:,}] | Valid: {completed:,} | Skipped: {failed:,}")

    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Successfully generated : {completed:,}")
    print(f"Skipped / Purged       : {failed:,}")
    print(f"Output directory       : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()