#!/usr/bin/env python3
"""
Step 2: Batch OCR inference over synthetic evaluation pages using fine-tuned Tesseract.
Uses multi-processing on CPU for maximum throughput.
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
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- Configuration -----------------
DATASET_DIR = repo_paths.workspace("synthetic_eval")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
OCR_OUT_DIR = os.path.join(DATASET_DIR, "ocr_pred")

LANG = "amh"
TESSDATA_DIR = "/usr/share/tesseract-ocr/5/tessdata"
PSM = "3"

# Number of parallel CPU workers (adjust to match your core count, e.g., 4 or 8)
NUM_WORKERS = 4
# -------------------------------------------------


def run_tesseract_on_page(image_path):
    """
    Executes Tesseract CLI on a single image and writes the OCR result to disk.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_base = os.path.join(OCR_OUT_DIR, base_name)
    out_txt_file = f"{out_base}.txt"

    # Expected standardized target name: page_XXXX.ocr.txt
    final_ocr_path = os.path.join(OCR_OUT_DIR, f"{base_name}.ocr.txt")

    cmd = [
        "tesseract",
        image_path,
        out_base,
        "-l", LANG,
        "--tessdata-dir", TESSDATA_DIR,
        "--psm", PSM
    ]

    try:
        # Run Tesseract suppressing verbose stderr
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            return (base_name, False, result.stderr.strip())

        # Rename out_base.txt to page_XXXX.ocr.txt
        if os.path.exists(out_txt_file):
            os.replace(out_txt_file, final_ocr_path)
            return (base_name, True, None)
        else:
            return (base_name, False, "Output text file not created.")

    except Exception as e:
        return (base_name, False, str(e))


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run Tesseract over the evaluation images and collect per-word output.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=DATASET_DIR,
                    help="synthetic evaluation set (images/ + ground_truth/)")
    ap.add_argument("--images-dir", default=IMAGES_DIR, help="page images")
    ap.add_argument("--out-dir", default=OCR_OUT_DIR, help="OCR output directory")
    ap.add_argument("-v", "--verbose", action="store_true", help="more output")
    return ap.parse_args(argv)


def main(argv=None):
    global DATASET_DIR, IMAGES_DIR, OCR_OUT_DIR
    args = _parse_args(argv)
    DATASET_DIR, IMAGES_DIR = args.dataset_dir, args.images_dir
    OCR_OUT_DIR = args.out_dir

    repo_paths.makedirs(OCR_OUT_DIR, "the OCR output directory")

    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    total_images = len(image_files)

    if total_images == 0:
        print(f"Error: No .png images found in {IMAGES_DIR}")
        return

    print(f"Found {total_images} pages to process.")
    print(f"Running Tesseract (Lang: {LANG}, PSM: {PSM}) using {NUM_WORKERS} CPU processes...")

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_tesseract_on_page, img_path): img_path for img_path in image_files}

        for future in as_completed(futures):
            base_name, success, error_msg = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                print(f"[FAIL] {base_name}: {error_msg}")

            if (completed + failed) % 25 == 0 or (completed + failed) == total_images:
                print(f"Progress: [{completed + failed}/{total_images}] completed...")

    print("\nOCR Processing Complete.")
    print(f"Successfully processed: {completed}/{total_images}")
    if failed > 0:
        print(f"Failed pages: {failed}")
    print(f"Results stored in: {OCR_OUT_DIR}")


if __name__ == "__main__":
    main()