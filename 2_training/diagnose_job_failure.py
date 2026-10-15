#!/usr/bin/env python3
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

import random
import subprocess
import os

TRAINING_TEXT = repo_paths.require(repo_paths.workspace("amh.training_text"),
                                  "the Tesseract training text")
UNICHARSET_FILE = repo_paths.require(repo_paths.workspace("amh.unicharset"),
                                    "amh.unicharset")
FONTS_DIR = repo_paths.require(repo_paths.fonts(), "the training font directory")

FONTS_WHITELIST = [
    "Abyssinica SIL",
    "Nyala",
    "Brana",
    "Noto Sans Ethiopic",
    "Noto Sans Ethiopic, Bold",
    "Ethiopia Jiret",
    "Ethiopia Jiret Slant",
    "FreeSerif",
    "Ethiopic WashRa Bold, Bold",
    "Ethiopic WashRa SemiBold, SemiBold",
    "Ethiopic WashRa SemiBold Slant Semi-Bold",
    "Ethiopic WashRa Bold Slant Bold",
    "Ethiopic Hiwua",
    "Ethiopic Wookianos",
    "Ethiopic Yebse"
]

with open(TRAINING_TEXT, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

random.seed(42)
random.shuffle(lines)

print(f"Inspecting first 1000 actual jobs...")

failures = []
for i in range(1000):
    text = lines[i]
    chosen_font = FONTS_WHITELIST[i % len(FONTS_WHITELIST)]
    exposure = random.choice([0, 0, 1, -1])
    char_spacing = random.choice([0.0, 0.0, -0.1, 0.25])
    ptsize = random.choice([28, 32, 32, 36])

    gt_file = "/tmp/diag_gt.txt"
    with open(gt_file, "w", encoding="utf-8") as tf:
        tf.write(text + "\n")

    cmd = [
        "text2image",
        f"--font={chosen_font}",
        f"--text={gt_file}",
        "--outputbase=/tmp/diag_out",
        "--max_pages=1",
        "--strip_unrenderable_words",
        "--leading=32",
        "--xsize=3200",
        "--ysize=300",
        f"--ptsize={ptsize}",
        f"--char_spacing={char_spacing}",
        f"--exposure={exposure}",
        f"--fonts_dir={FONTS_DIR}",
        f"--unicharset_file={UNICHARSET_FILE}"
    ]

    res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        err = res.stderr.strip().replace("\n", " | ")
        failures.append((i, chosen_font, len(text), err))

print(f"\nResult: {len(failures)}/1000 jobs failed.")
print("-" * 75)
for idx, font, length, err in failures[:10]:
    print(f"Job {idx:2d} | Font: {font:<28} | Len: {length:2d} | Error: {err[:80]}")
