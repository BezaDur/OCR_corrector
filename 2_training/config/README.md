# 2_training/config — Tesseract config and character set

- `amh.unicharset` — the character set the model was trained with.
- `confusion_matrix.json` — **stale copy, read by no code path.** It differs from
  `4_ocr_correction/error_detection/Model/confusion_matrix.json`, which is the single
  authoritative matrix read by both the corrector and the detector. Kept only as
  a record of the matrix at training time; do not edit it expecting an effect.
