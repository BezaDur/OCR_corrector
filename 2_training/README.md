# 2_training — train and package the Tesseract LSTM

**Parent project, step 2: once the data from step 1 exists, train on it.**

## What it does

| Script | What it does |
|---|---|
| `01_train_epochs.sh` | Stefan-Weil style sequential epoch runner, 5 epochs, from `true_google_best/amh.lstm` |
| `02_run_epoch2.sh` | fail-safe single epoch-2 run aimed at the `ቯ`/`ጰ` confusions |
| `03_package_best_and_fast.sh` | takes the newest checkpoint and packages `amh_best` / `amh_fast` `.traineddata` |
| `diagnose_job_failure.py` | checks training text, unicharset and lists when a job dies |

All three shell scripts read `TESSERACT_WORKSPACE` (default
`/media/des/Tess/Tesseract2`) and expect:

```
$TESSERACT_WORKSPACE/training/lists/list.train        # from step 1 (06_compile_lstmf_and_split.py)
$TESSERACT_WORKSPACE/training/lists/list.eval
$TESSERACT_WORKSPACE/training/amh/amh.traineddata     # training config
$TESSERACT_WORKSPACE/true_google_best/amh.lstm        # starting checkpoint
```

## How to run

```bash
export TESSERACT_WORKSPACE=/path/to/Tesseract2
bash 2_training/01_train_epochs.sh
bash 2_training/03_package_best_and_fast.sh
```

Tesseract's training tools (`lstmtraining`, `combine_tessdata`) must be on
`PATH`; they are not pip packages.

## What is checked in

| File | Size | What it is |
|---|---:|---|
| `models/best/amh.traineddata` | 8.1 MB | best packaged model |
| `models/best/amh.lstm` | 3.3 MB | its LSTM checkpoint |
| `models/google_amh.lstm` | 428 KB | the Google baseline checkpoint |
| `../models/amh_best.traineddata` | 4.7 MB | earlier "best" package at the root |
| `../models/amh_fast.traineddata` | 644 KB | earlier "fast" package at the root |
| `config/amh.unicharset` | | character set |
| `config/confusion_matrix.json` | | **stale copy — read by no code path**; the authoritative matrix is `4_ocr_correction/error_detection/Model/confusion_matrix.json` |

The training inputs are not kept here: `amh.training_text` (24.9 MB),
`amh.unicharset` and `amh.lstm-unicharset` live in `1_data_preparation/data/`
next to the corpus, `amh_final_training_text.txt` (4.42 M tokens).
`2_training/data/` holds only a pointer note.

## Output

An `amh.traineddata` that step 3 evaluates and the corrector's Stage 1 model was
trained against OCR output from. Nothing in the repository runs Tesseract by
default; `3_training_evaluation/01_run_ocr_eval.py` does, and it needs the
`tesseract` binary plus a `tessdata` directory.
