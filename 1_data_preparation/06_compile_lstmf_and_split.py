#!/usr/bin/env python3
"""
Step 8: High-Speed Multi-Core .lstmf Compiler & Dataset Splitter.
Features:
  - Spawns across 4 CPU cores to convert .box/.tif pairs into .lstmf tensors.
  - Generates 90/10 Train/Eval lists with deterministic shuffling.
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
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor

# ----------------- Configuration -----------------
GT_DIR = repo_paths.workspace("training_ground_truth")
LISTS_DIR = repo_paths.workspace("training/lists")
NUM_WORKERS = 4
TRAIN_RATIO = 0.90
BASE_SEED = 42
# -------------------------------------------------


# ============================================================================
# RUN SETTINGS — edit these values, then run this script with NO arguments:
#     python 1_data_preparation/06_compile_lstmf_and_split.py
# Every value is the default for the CLI flag of the same name (--kebab-case);
# passing the flag still overrides it, but you never need to. See HOW_TO_RUN.md.
# ============================================================================
CONFIG = {
    "gt_dir": GT_DIR,
    "lists_dir": LISTS_DIR,
    "verbose": False,
}

def compile_single_lstmf(tif_path):
    """
    Executes Tesseract to compile one .tif + .box pair into a .lstmf tensor.
    """
    base_path = os.path.splitext(tif_path)[0]
    box_path = f"{base_path}.box"
    lstmf_path = f"{base_path}.lstmf"

    if not os.path.exists(box_path):
        return None

    cmd = [
        "tesseract",
        tif_path,
        base_path,
        "--psm", "6",
        "--dpi", "300",
        "lstm.train"
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(lstmf_path) and os.path.getsize(lstmf_path) > 100:
            return lstmf_path
        return None
    except Exception:
        return None


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Compile training images into .lstmf files and write the train/eval lists.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default=CONFIG["gt_dir"],
                    help="directory with the generated .tif/.box/.gt.txt triples")
    ap.add_argument("--lists-dir", default=CONFIG["lists_dir"],
                    help="where to write list.train / list.eval")
    ap.add_argument("-v", "--verbose", action="store_true", default=CONFIG["verbose"], help="more output")
    return ap.parse_args(argv)


def main(argv=None):
    global GT_DIR, LISTS_DIR
    args = _parse_args(argv)
    GT_DIR, LISTS_DIR = args.gt_dir, args.lists_dir

    repo_paths.makedirs(LISTS_DIR, "the lstmf list directory")

    tif_files = sorted(glob.glob(os.path.join(GT_DIR, "*.tif")))
    total_files = len(tif_files)

    if total_files == 0:
        print(f"Error: No .tif files found in {GT_DIR}")
        return

    print(f"Found {total_files:,} line images.")
    print(f"Compiling into .lstmf tensors using {NUM_WORKERS} CPU processes...\n")

    valid_lstmf = []
    failed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in executor.map(compile_single_lstmf, tif_files):
            if res:
                valid_lstmf.append(res)
            else:
                failed += 1

            processed = len(valid_lstmf) + failed
            if processed % 10000 == 0 or processed == total_files:
                print(f"Compilation Progress: [{processed:,}/{total_files:,}] | Compiled: {len(valid_lstmf):,}")

    print(f"\nCompilation Complete! Valid .lstmf records: {len(valid_lstmf):,}")

    # Deterministic 90/10 Split
    rng = random.Random(BASE_SEED)
    rng.shuffle(valid_lstmf)

    split_idx = int(len(valid_lstmf) * TRAIN_RATIO)
    train_set = valid_lstmf[:split_idx]
    eval_set = valid_lstmf[split_idx:]

    train_list_path = os.path.join(LISTS_DIR, "list.train")
    eval_list_path = os.path.join(LISTS_DIR, "list.eval")

    with open(train_list_path, "w", encoding="utf-8") as f:
        for p in train_set:
            f.write(p + "\n")

    with open(eval_list_path, "w", encoding="utf-8") as f:
        for p in eval_set:
            f.write(p + "\n")

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total .lstmf samples : {len(valid_lstmf):,}")
    print(f"Training set (90%)   : {len(train_set):,} -> {train_list_path}")
    print(f"Evaluation set (10%) : {len(eval_set):,} -> {eval_list_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
