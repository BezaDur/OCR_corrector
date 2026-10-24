# 2_training/data — pointer

This folder is intentionally empty apart from this note. The Tesseract inputs
(`amh.training_text` 24.9 MB, `amh.unicharset`, `amh.lstm-unicharset`,
`google_raw.lstm-unicharset`) are in **`1_data_preparation/data/`**, next to the
corpus they were built from (`amh_final_training_text.txt`, 4,420,102 tokens).

The per-run lists (`list.train`, `list.eval`) are generated into
`$TESSERACT_WORKSPACE/training/lists/` by
`1_data_preparation/08_compile_lstmf_and_split.py`; they are not in git.
