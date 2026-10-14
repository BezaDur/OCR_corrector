# 1_data_preparation — Prepare data for OCR training

**Parent project, step 1: all starts from preparing data for main OCR training.**

## What it does
- Generate evaluation pages (01_generate_eval_pages.py)
- Generate training boxes (07_generate_training_boxes.py)
- Compile lstmf and split (08_compile_lstmf_and_split.py)
- Assemble balanced training data, audit rare glyphs, discover character inventory
- Final training text: data/amh_final_training_text.txt (4.4M tokens, 56MB, 359k lines)

## Data
- `data/amh_final_training_text.txt` — 4.4M tokens, raw training text (moved from root)
- Fonts, char inventory, etc.

## How to run
```bash
python 01_generate_eval_pages.py
python 07_generate_training_boxes.py
python 08_compile_lstmf_and_split.py
```

## Output
- amh.training_text, amh.lstm-unicharset, amh.unicharset for Tesseract training
- eval pages for 3_training_evaluation
