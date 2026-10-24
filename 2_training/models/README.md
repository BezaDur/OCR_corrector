# 2_training/models — packaged OCR models

| File | Size | What it is |
|---|---:|---|
| `best/amh.traineddata` | 8.1 MB | best packaged model (moved from `true_google_best/`) |
| `best/amh.lstm` | 3.3 MB | its LSTM checkpoint |
| `google_amh.lstm` | 428 KB | the Google baseline checkpoint |

Two more packaged variants live at the repository root in `models/`:
`amh_best.traineddata` (4.7 MB) and `amh_fast.traineddata` (644 KB). To use a
model, copy its `.traineddata` into a tessdata directory as `amh.traineddata`;
`3_training_evaluation/02_run_ocr_eval.py` expects `tesseract` and that
directory to exist.
