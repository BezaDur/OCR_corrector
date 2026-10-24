# 1_data_preparation — build the training and evaluation data

**Parent project, step 1: everything starts from the text and the fonts.**

## What it does

| Script | What it produces |
|---|---|
| `01_generate_eval_pages.py` | synthetic evaluation pages (fonts, curvature, bleed, lighting, noise) |
| `generate_eval_pages.py`, `generate_eval_dataset.py` | older entry points for the same job |
| `07_generate_training_boxes.py` | Tesseract training box files from rendered lines |
| `08_compile_lstmf_and_split.py` | `.lstmf` compilation and the train/eval split |
| `assemble_balanced_training_data.py`, `build_final_training_data.py` | assemble the final training text |
| `audit_training_text.py`, `audit_rare_glyphs.py` | coverage and sanity checks on that text |
| `rare_character_miner.py`, `discover_character_inventory.py` | rare glyphs, character inventory |
| `check_pango_names.py`, `clean_font_directories.py`, `inspect_and_install_fonts.py` | font discovery/installation |
| `diagnose_font_accuracy.py`, `diagnose_font_cmaps.py` | does a font actually have the glyphs? |

## Data in the repo

| File | Size | Contents |
|---|---:|---|
| `data/amh_final_training_text.txt` | 58,248,932 B (55.6 MiB) | 359,776 lines, **4,420,102 tokens**, **638,125 unique** (22.7 M chars) |
| `data/mined_rare_sentences.txt` | same size and line count | a different 51 MB of content, not a copy of the file above; `rare_character_miner.py` and `assemble_balanced_training_data.py` read/write it through `repo_paths.workspace(...)`, i.e. the external workspace, not this copy |
| `data/amh.training_text`, `amh.unicharset`, `amh.lstm-unicharset`, `google_raw.lstm-unicharset` | | Tesseract inputs and the character sets |
| `fonts.conf` | | fontconfig file used by the renderer |

## What has to be present

Every script gets its paths from `repo_paths.py`:

```bash
export TESSERACT_TRAINING=/path/to/TessTraining     # Fonts/, langdata/
export TESSERACT_WORKSPACE=/path/to/Tesseract2      # synthetic_eval/, training output
export AMHARIC_DATA_ROOT=/path/to/Mint              # KenLM models, raw corpora
python repo_paths.py                                # verify before running anything
```

Page synthesis also needs `opencv-python`, `Pillow`, `fonttools`, the fonts
(`fonts/` has 20 Ethiopic TTFs) and a writable workspace. `augraphy` is not
imported by any script; the checked-in `augraphy_cache/` is a leftover from an
earlier experiment.

## How to run

From the repository root:

```bash
python 1_data_preparation/01_generate_eval_pages.py
python 1_data_preparation/07_generate_training_boxes.py
python 1_data_preparation/08_compile_lstmf_and_split.py
```

## Output

`amh.training_text`, `amh.lstm-unicharset`, `amh.unicharset` for step 2, and the
evaluation pages that step 3 scores. A copy of the 100-page evaluation set used
by the detector is checked in under `OCR_error_detection/Evaluation/`.
