# 2_training — Train OCR model

**Parent project, step 2: once we prepare data, we train on it.**

## What it does
- Run Tesseract LSTM training epochs (01_train_epochs.sh, 02_run_epoch2.sh)
- Package best and fast models (03_package_best_and_fast.sh)
- Diagnose job failures

## How to run
```bash
bash 01_train_epochs.sh
bash 03_package_best_and_fast.sh
```

## Output
- amh.traineddata (OCR engine)
- google_amh.lstm, etc.
