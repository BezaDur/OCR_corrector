# repo_paths.py — one place for the data roots every stage script needs.
#
# The pipeline reads and writes several trees that are far too large to keep in
# git (Tesseract training output, synthetic evaluation pages, KenLM models, the
# raw text corpora). Historically every script hard-coded the author's machine
# paths, so nothing could run anywhere else. Each root is now read from an
# environment variable with the original location as the default, so the
# original machine keeps working and any other machine only has to set the
# variables it actually has data for.
#
#   AMHARIC_DATA_ROOT      default /mnt/mintdata/Users/Des/Mint
#                          (KenLM models, dictionaries, raw datasets)
#   TESSERACT_WORKSPACE    default /media/des/Tess/Tesseract2
#                          (synthetic_eval, training/, amh.training_text, ...)
#   TESSERACT_TRAINING     default /media/des/Tess/Tesseract/TessTraining
#                          (fonts/, langdata/)
#   AMHARIC_GOOGLE_BEST    default /media/des/Tess/Tesseract/Google_best
#
# Point overrides (take precedence over the roots above):
#   AMHARIC_WORD_KENLM     amharic_5gram.binary
#   AMHARIC_CHAR_KENLM     amharic_char_model.klm
#   AMHARIC_MAIN_I7_DICT   main_i7.txt (word + frequency)
#   AMHARIC_RAW_DATASET    raw text corpus tree
#   AMHARIC_WORD_DATASET   04_OCR_tess_word_defrag
#
# Usage from a script in a stage folder:
#
#   import os, sys
#   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#   import repo_paths
#   FONTS_DIR = repo_paths.fonts()
#   OUTPUT_DIR = repo_paths.workspace("synthetic_eval")
#   repo_paths.require(FONTS_DIR, "the font directory")   # clear error if absent
#
# `python repo_paths.py` prints the resolved roots.

import os

DATA_ROOT = os.environ.get("AMHARIC_DATA_ROOT", "/mnt/mintdata/Users/Des/Mint")
TESS_WORKSPACE = os.environ.get("TESSERACT_WORKSPACE", "/media/des/Tess/Tesseract2")
TESS_TRAINING = os.environ.get("TESSERACT_TRAINING", "/media/des/Tess/Tesseract/TessTraining")
GOOGLE_BEST = os.environ.get("AMHARIC_GOOGLE_BEST", "/media/des/Tess/Tesseract/Google_best")

# Full names, for scripts that read better with them.
TESSERACT_WORKSPACE = TESS_WORKSPACE
TESSERACT_TRAINING = TESS_TRAINING

# Derived defaults; each is individually overridable.
WORD_KENLM = os.environ.get(
    "AMHARIC_WORD_KENLM",
    os.path.join(DATA_ROOT, "Models", "KenLM", "word_model", "amharic_5gram.binary"))
CHAR_KENLM = os.environ.get(
    "AMHARIC_CHAR_KENLM",
    os.path.join(DATA_ROOT, "Models", "KenLM", "character_model", "amharic_char_model.klm"))
MAIN_I7_DICT = os.environ.get(
    "AMHARIC_MAIN_I7_DICT",
    os.path.join(DATA_ROOT, "4_Dataset_wo", "Projects", "p16_morpheme_analysis",
                 "data", "main_i7.txt"))
RAW_DATASET = os.environ.get("AMHARIC_RAW_DATASET", os.path.join(DATA_ROOT, "1_Dataset_raw"))
WORD_DATASET = os.environ.get(
    "AMHARIC_WORD_DATASET",
    os.path.join(DATA_ROOT, "2_Dataset_text", "04_OCR_tess_word_defrag"))

# Repository copies of the small data files (these are in git).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_MODEL_DIR = os.path.join(REPO_ROOT, "OCR_error_detection", "Model")
VENDORED_SEGMENTER_DATA = os.path.join(REPO_ROOT, "vendor", "segmenter", "data")


def data(*parts):
    """AMHARIC_DATA_ROOT join (raw corpora, KenLM models, dictionaries)."""
    return os.path.join(DATA_ROOT, *parts)


def workspace(*parts):
    """TESSERACT_WORKSPACE join (synthetic_eval, ocr_pred, training, ...)."""
    return os.path.join(TESS_WORKSPACE, *parts)


def training(*parts):
    """TESSERACT_TRAINING join."""
    return os.path.join(TESS_TRAINING, *parts)


def fonts(*parts):
    """Font directory used by the page generator."""
    return training("Fonts", *parts)


def langdata(*parts):
    """langdata directory (amh.training_text, amh.unicharset)."""
    return training("langdata", *parts)


def word_kenlm():
    """Word KenLM binary used by the corrector's Viterbi stage."""
    return WORD_KENLM


def char_kenlm():
    """Character KenLM model (optional; the detector's column is NaN without it)."""
    return CHAR_KENLM


def main_i7_dict():
    """main_i7-style dictionary (word + frequency).

    The copy shipped in the repository wins when it is present: it is the file
    the detector's model was trained against (main_i7, freq >= 10), so scripts
    run offline and against the vocabulary the model actually saw.
    """
    local = os.path.join(REPO_MODEL_DIR, "main_i7_frq10.txt")
    if os.path.exists(local):
        return local
    return MAIN_I7_DICT


def confusion_matrix():
    """Optical confusion matrix: the in-repo copy wins (the detector ships one)."""
    local = os.path.join(REPO_MODEL_DIR, "confusion_matrix.json")
    if os.path.exists(local):
        return local
    return workspace("confusion_matrix.json")


def morphemes_json():
    """Morphemes file: vendored copy first, then the analysis project."""
    local = os.path.join(VENDORED_SEGMENTER_DATA, "morphemes.json")
    if os.path.exists(local):
        return local
    return data("4_Dataset_wo", "Projects", "p16_morpheme_analysis", "data", "morphemes.json")


def makedirs(path, what="output directory"):
    """mkdir -p, with a message naming the environment variable on failure.

    Output directories live outside the repo (Tesseract workspace), so a
    missing/unwritable root has to say which variable to set instead of raising
    a bare PermissionError from deep inside the script.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"cannot create {what}: {path}\n  {exc}\n"
            "Set TESSERACT_WORKSPACE (and friends) to a writable location — see "
            "repo_paths.py, or run `python repo_paths.py` to see what resolves "
            "where.")
    return path


def require(path, what):
    """Return `path`, or exit with a message naming the variable to set."""
    if path and os.path.exists(path):
        return path
    raise SystemExit(
        f"{what} not found: {path}\n"
        "Set the matching environment variable (see repo_paths.py) or pass the "
        "path on the command line. Roots in use:\n"
        f"  AMHARIC_DATA_ROOT   = {DATA_ROOT}\n"
        f"  TESSERACT_WORKSPACE = {TESS_WORKSPACE}\n"
        f"  TESSERACT_TRAINING  = {TESS_TRAINING}")


def describe():
    """Human-readable summary of the resolved roots."""
    return (
        f"data root          : {DATA_ROOT}\n"
        f"tesseract workspace: {TESS_WORKSPACE}\n"
        f"tesseract training : {TESS_TRAINING}\n"
        f"google_best        : {GOOGLE_BEST}\n"
        f"word KenLM         : {WORD_KENLM}\n"
        f"char KenLM         : {CHAR_KENLM}\n"
        f"main_i7 dictionary : {main_i7_dict()}\n"
        f"confusion matrix   : {confusion_matrix()}\n"
        f"morphemes json     : {morphemes_json()}\n"
        f"raw dataset        : {RAW_DATASET}\n"
        f"word dataset       : {WORD_DATASET}"
    )


if __name__ == "__main__":
    print(describe())
