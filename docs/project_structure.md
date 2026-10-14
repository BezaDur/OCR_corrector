# Project Structure — Sensible Organization

**You asked: Organizing it in a sensible manner, not just data but also scripts and other stuff is good practice. Also, directory structures, I want files relevant to detection should be inside the directory.**

## Current Layout (before reorg) — Problems

```
. (repo root, messy)
├── 1_data_preparation/ (OK, but data file at root)
├── 2_training/ (OK)
├── 3_training_evaluation/ (OK, but metrics at root)
├── 4_ocr_correction/ (only has 03_build_confusion_matrix.py, main corrector at root)
├── 5_correction_evaluation/ (OK)
├── detection/ (old logistic regression, deprecated, confusing with OCR_error_detection)
├── OCR_error_detection/ (new LightGBM, child, self-contained, OK)
│   ├── Model/ (detector_lgbm_conffree.pkl, main_i7_frq10.txt, etc.)
│   ├── Evaluation/ (100 pages)
│   └── ...
├── amharic_ocr_corrector.py (MAIN CORRECTOR, should be in 4_ocr_correction/)
├── normalizer.py (PARENT normalizer, bigger, should be in 4_ocr_correction/, duplicate with child)
├── suspicion.py (fallback, should be in 4_ocr_correction/)
├── amharic_canonical_normalizer.py (eval util, should be in 3_ or 4_)
├── ocr_metrics.py (eval metric, should be in 3_)
├── tsv_gt.py, tsv_confidence.py (eval, should be in 3_)
├── confusion_matrix.json (duplicate, root and Model/)
├── amh_final_training_text.txt (4.4M tokens, 56MB, training text, should be in 1_data_preparation/data/)
├── tests/ (mixed: old detection tests + parent tests)
├── docs/
│   ├── components_guide.md (OK)
│   └── ...
└── ...
```

**Problems:**
- Root files mixed: parent corrector, child detection, old detection, eval utils all at root
- Duplicates: confusion_matrix.json at root and Model/, normalizer.py at root and child
- detection/ vs OCR_error_detection/ confusing (old vs new)
- No clear import parent → child
- Data file amh_final_training_text.txt at root, not in data folder
- Evaluation data in multiple places: eval_data/, evaluation_data/, OCR_error_detection/Evaluation/

## Proposed Layout (after reorg) — Clean

```
.
├── README.md (overall project overview, pipeline 1→2→3→4→5 + child)
├── requirements.txt (pip install -r requirements.txt)
├── .gitignore (ignore training/, fonts/, eval_data/images/, etc.)
├── docs/
│   ├── components_guide.md (plain explanation: KenLM, SymSpell, LightGBM, dict, model, with diagrams)
│   ├── architecture.md (parent vs child, data flow, how parent imports child via adapter)
│   └── project_structure.md (this file)
│
├── 1_data_preparation/ (prepare data for OCR training)
│   ├── README.md (what it does, how to run)
│   ├── 01_generate_eval_pages.py (generate 100 eval pages)
│   ├── 07_generate_training_boxes.py
│   ├── 08_compile_lstmf_and_split.py
│   ├── assemble_balanced_training_data.py
│   ├── build_final_training_data.py
│   ├── data/
│   │   └── amh_final_training_text.txt (4.4M tokens, 56MB, moved from root)
│   └── ...
│
├── 2_training/ (train Tesseract LSTM)
│   ├── README.md
│   ├── 09_train_epochs.sh
│   ├── 10_package_best_and_fast.sh
│   └── ...
│
├── 3_training_evaluation/ (evaluate OCR, CER/WER)
│   ├── README.md
│   ├── 02_run_ocr_eval.py
│   ├── 06_diagnose_ocr_engine.py
│   ├── ocr_metrics.py (moved from root, canonical metrics)
│   ├── tsv_gt.py (moved from root, ground-truth alignment)
│   ├── tsv_confidence.py (moved from root)
│   ├── tsv_gt_stats.py (moved from root)
│   └── ...
│
├── 4_ocr_correction/ (parent corrector, main correction logic)
│   ├── README.md (explains corrector, how it calls child detection via adapter)
│   ├── amharic_ocr_corrector.py (main corrector, moved from root, 80K lines)
│   ├── normalizer.py (parent normalizer, bigger, punctuation/numerals/abbrev, moved from root)
│   ├── amharic_canonical_normalizer.py (moved from root, canonicalize for metrics)
│   ├── suspicion.py (fallback SuspicionScorer, moved from root, deprecated but kept)
│   ├── 03_build_confusion_matrix.py (builds confusion_matrix.json from OCR errors)
│   ├── confusion_matrix.json (single source lives in OCR_error_detection/Model/confusion_matrix.json; no symlinks)
│   ├── trace_candidate.py (debug candidate generation)
│   └── ...
│
├── 5_correction_evaluation/ (evaluate full corrector)
│   ├── README.md
│   ├── 04_evaluate_full_corrector.py
│   ├── 05_collect_error_cases.py
│   ├── 13_audit_correction_decisions.py
│   ├── stage1_labels.py (moved from root, labels for eval)
│   ├── stage1_fit.py (moved from root, fit for eval)
│   └── ...
│
├── OCR_error_detection/ (child project, self-contained error detection)
│   ├── README.md (child overview, final model main_i7 freq>=500 54k types recall 94.2% FF 0.01%)
│   ├── __init__.py (exports DetectionEngine, CorrectorDetectionAdapter, etc.)
│   ├── detector.py (main, LightGBM, default 50k types freq>=10 no SymSpell, 2.9s init 0.56ms/word)
│   ├── adapter.py (for parent to import child, reuses vocab, no double load)
│   ├── normalizer.py (child normalizer, smaller, lexicon keying only, ሐ→ሀ etc.)
│   ├── signals.py (surface, lexical, one-flip signals)
│   ├── confusion.py (ConfusionMatrix, reverse map)
│   ├── scoring.py (combine_signals, deprecated, LightGBM authoritative now)
│   ├── alignment.py (align OCR vs GT for labels)
│   ├── utils.py (ETHIOPIC, RARE_GLYPH_CLASSES, etc.)
│   ├── run_ocr.py (run OCR for eval)
│   ├── Model/
│   │   ├── detector_lgbm_conffree.pkl (FINAL, 3.5MB, main_i7 freq>=500 54k types, 1267 pages, 14 feats conf-free)
│   │   ├── main_i7_frq10.txt (737k types, 15MB, from bigger data, freq>=10, max freq 5.6M, default dict for all needs)
│   │   ├── local_lexicon.txt (750k types, 15MB, from 4.4M tokens, fallback)
│   │   ├── confusion_matrix.json (single source, 47K, learned from real errors)
│   │   ├── harmonized_corpus_jiwer.tsv (1367 pages, 946k words, 49MB, training data)
│   │   └── train_final.py (train final model)
│   ├── Evaluation/
│   │   ├── analysis.tsv (62k rows, page, ocr_word, gt_word, is_error, conf, etc.)
│   │   ├── GroundTruth/ (100 pages, page_*.gt.txt)
│   │   ├── OCRed/ (100 pages, page_*.ocr.txt + page_*.conf.tsv)
│   │   └── images/ (ignored by git, generated)
│   ├── tests/
│   │   ├── test_detector.py (child tests)
│   │   └── ...
│   ├── scripts/
│   │   ├── audit_decision.py (moved from root ocr_decision_audit.py)
│   │   └── ...
│   └── docs/ (child docs, if needed)
│
├── detection/ (old logistic regression, deprecated, will archive)
│   ├── README.md (explains deprecated, use OCR_error_detection instead)
│   ├── detector.py (old ErrorDetector, logistic)
│   ├── train.py
│   ├── evaluate.py
│   └── ...
│
├── archive/
│   ├── detection_old/ (old detection/ moved here)
│   ├── OCR_error_detection_archive/ (old versions of child)
│   └── ...
│
├── vendor/
│   └── segmenter/ (morphological suite, vendored)
│       ├── segmenter.py
│       ├── seg_normalizer.py
│       └── ...
│
├── eval_data/ (deprecated, use OCR_error_detection/Evaluation/)
├── evaluation_data/ (deprecated, use OCR_error_detection/Evaluation/)
├── tests/ (parent tests, or move to 5_correction_evaluation/tests/)
│   ├── test_ocr_corrector.py (parent corrector tests)
│   └── ...
│
└── config/
    └── ...
```

## What Goes Where — Rules

1. **Files relevant to detection inside OCR_error_detection/**:
   - detector.py, adapter.py, normalizer.py (child), signals.py, confusion.py, scoring.py, alignment.py, utils.py, run_ocr.py
   - Model/* (pkl, txt dicts, json confusion, tsv corpus)
   - Evaluation/* (GroundTruth, OCRed, analysis.tsv)
   - tests/* (child tests)
   - scripts/* (train_final.py, audit_decision.py)

2. **Parent corrector files inside 4_ocr_correction/**:
   - amharic_ocr_corrector.py (main corrector)
   - normalizer.py (parent, bigger)
   - suspicion.py (fallback)
   - amharic_canonical_normalizer.py
   - 03_build_confusion_matrix.py

3. **Evaluation metrics inside 3_training_evaluation/**:
   - ocr_metrics.py, tsv_gt.py, tsv_confidence.py, tsv_gt_stats.py

4. **Data preparation files inside 1_data_preparation/**:
   - amh_final_training_text.txt moved to 1_data_preparation/data/

5. **Old detection/ archived**:
   - detection/ → archive/detection_old/, README says deprecated, use OCR_error_detection

6. **No duplicates**:
   - confusion_matrix.json single source in OCR_error_detection/Model/
   - Root confusion_matrix.json removed (no symlinks; load the Model/ copy by path)
   - normalizer.py separate: parent (bigger) in 4_ocr_correction/, child (smaller) in OCR_error_detection/

7. **Parent imports child via adapter**:
   - 4_ocr_correction/amharic_ocr_corrector.py does `from OCR_error_detection.adapter import CorrectorDetectionAdapter`
   - Adapter reuses parent's vocab, no double load
   - Child doesn't import parent (no circular)

## How to Import Child to Parent — Code

```python
# In 4_ocr_correction/amharic_ocr_corrector.py
from OCR_error_detection import DetectionEngine
from OCR_error_detection.adapter import CorrectorDetectionAdapter

# Option 1: Direct use (self-contained)
det = DetectionEngine(
    dict_vocab_path="OCR_error_detection/Model/main_i7_frq10.txt",
    dict_min_freq=500,  # 54k types, best FF 0.01%
    max_lexicon_types=None,
    enable_symspell=False
)
verdict = det.verdict("በሸታ", doc_freq=1)

# Option 2: Via adapter (reuses parent's vocab)
class AmharicOCRCorrector:
    def __init__(self):
        # load vocab, symspell, confusion_map, normalizer
        self.suspicion = CorrectorDetectionAdapter(self)  # reuses vocab

    def correct(self, word, context, conf, doc_freq):
        verdict = self.suspicion.verdict(word, context, ocr_conf=conf, doc_freq=doc_freq)
        if verdict.label == "FLAG":
            # correct
            ...
```

## Benefits

- **Sensible**: Parent pipeline 1→2→3→4→5 clear, child self-contained
- **No crash**: Default 50k types no SymSpell, 2.9s init, won't crash PC
- **Single source**: Dictionary main_i7_frq10.txt default for all needs, confusion_matrix.json single source
- **Good practice**: Scripts call dictionary, don't train it; child doesn't train dict, parent doesn't duplicate
- **Easy import**: Parent calls child via adapter, no circular, no double memory
- **Clean tests**: Child tests in OCR_error_detection/tests/, parent tests in tests/ or 5_correction_evaluation/

## Migration Steps (implemented)

1. Create docs/architecture.md, docs/project_structure.md (this file)
2. Move parent corrector files from root to 4_ocr_correction/ (with git mv, update imports)
3. Move eval utils from root to 3_training_evaluation/
4. Move detection-relevant from root to OCR_error_detection/scripts/ or archive/
5. Single source confusion_matrix.json in Model/
6. Archive old detection/ to archive/detection_old/
7. Create READMEs for each folder explaining what it does and how to run
8. Ensure tests pass after moves
