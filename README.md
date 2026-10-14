# OCR Corrector — Amharic

This repo does two things:
1. **Fine-tune Tesseract OCR** for Amharic (Stages 1-2)
2. **Fix OCR mistakes** after OCR (Stages 4-5)

The focus now is **Stage 1: Error Detection** — decide which words are wrong.

## Quick start

```bash
# Install
pip install symspellpy lightgbm scikit-learn pandas

# Use detector (auto-finds conf-free model)
python -c "
from OCR_error_detection import DetectionEngine
det = DetectionEngine(dict_vocab_path='OCR_error_detection/Model/local_lexicon.txt', max_lexicon_types=5000)
print(det.verdict('ሰላም').label)   # pass
print(det.verdict('በሸታ').label)   # gray (needs fix, even with high conf)
"

# Run tests
python -m unittest OCR_error_detection.tests.test_detector -v
```

## Repo layout

```
1_data_preparation/        make training and eval data
2_training/                train OCR model
3_training_evaluation/     measure OCR model
4_ocr_correction/          build corrector
5_correction_evaluation/   measure corrector
OCR_error_detection/       **main: error detection (LightGBM conf-free, authoritative)**
  Model/                   models and dictionary (2 models, 15M lexicon, 49M corpus)
    detector_lgbm_conffree.pkl  NEW authoritative, 14 feats, no conf, 982K
    detector_lgbm_v3.pkl        previous clean baseline, 15 feats NaN conf, 4.8MB
    local_lexicon.txt           750k types
    confusion_matrix.json       47K
    train_conffree.py           final training script (conf-free)
  Evaluation/              100 pages images + GT + OCR + confidences + analysis.tsv (62k rows)
  archive/                 old deprecated scripts
  tests/                   40 tests
archive/                   deprecated at root level
  models/                  old models (v1, v2, v4)
docs/repo_layout.md        detailed stage map
docs/future_improvements.md future directions
```

## OCR_error_detection — plain guide

See `OCR_error_detection/README.md` for plain guide, results, and future directions.

**One-line:**
- Old: 9.85% false flags, 56% clean kept
- WITH conf (v2): 0.85% false flag, 87.9% recall, but misses 44% high-conf errors (conf>85), false-flags 11.7% low-conf correct
- **WITHOUT conf small sample (100 pages, 15k lexicon): 0.07% false flag (12x better), 90.8% recall, 90.8% high-conf errors caught vs 56%, 0.1% low-conf false-flag vs 11.7%, 0 ሕ/ኀ errors**
- **WITHOUT conf FULL FINAL (1267 pages, 150k lexicon, 4.8MB, isotonic): val recall 94.2% false_flag 0.04% at 0.10/0.70, pass 0.10 flag 0.70, 14 linguistic feats only**

**Why conf-free is better:** Tesseract conf is unreliable — 19% of real errors have conf>85 (confident but wrong), 7.5% correct have conf<70. If model trusts conf, it misses dangerous high-conf errors and false-flags low-conf correct. Conf-free relies only on linguistic features (dictionary, one-flip, rare glyphs, doc_freq) and gives same verdict regardless of conf — correct behavior.

**Authoritative FINAL model:** `OCR_error_detection/Model/detector_lgbm_conffree.pkl` (14 feats, no ocr_conf, 4.8MB FULL, 1267 pages, 150k lexicon, isotonic, 94.2% val recall, 0.04% false_flag) — prioritized in `CANDIDATE_MODEL_PATHS`

Previous clean baseline `detector_lgbm_v3.pkl` (15 feats, NaN conf, 4.8MB) kept as fallback.

## Deprecated — moved to archive

To avoid confusion, old scripts were moved:

- `suspicion.py` (old rule-based) → `archive/suspicion_old.py` + new shim `suspicion.py` (deprecated warning, delegates to DetectionEngine)
- `detection/` (old logistic) → `archive/deprecated_detectors/detection_old/`
- Old training/eval in `OCR_error_detection/` → `OCR_error_detection/archive/`
- Old models → `archive/models/`

New code should use `OCR_error_detection.detector.DetectionEngine`, not `suspicion.py`.

## Future

See `OCR_error_detection/README.md` → Future improvements section for 11 directions (better dict, word LM, morphology, more data, insertion/deletion flips, doc voting, end-to-end, active learning, etc.)
See `docs/future_improvements.md` for detailed roadmap.

## Conf-free experiment

Run `python OCR_error_detection/Model/train_conffree.py` to train conf-free model (70/30 split, 15k lexicon, ~1 min)
Run `python OCR_error_detection/archive/test_conf_free.py` to compare WITH vs WITHOUT conf (shows 90.8% vs 56.1% high-conf caught)
