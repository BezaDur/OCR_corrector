# OCR Error Detection — Plain Guide (FINAL with main_i7)

**One job:** Look at each OCR word and decide: keep it, or fix it.

- **pass** = word is fine
- **gray** = not sure
- **flag** = wrong, fix it

---

## Final Model — main_i7 with freq filter

**Pulled from branch `arena/3ca31a9a-ocr-corrector`: `main_i7_frq10.txt` — 737,106 types, 15MB, frequencies from bigger data (max freq 5.6M vs 60k in local_lexicon), already cleaned freq 9 and lower removed (min freq 10).**

Tested different freq thresholds on same 1267 pages (1200 train / 67 val):

| Dictionary | Filter | Types | Val AUC | Recall | False Flag | Score |
|------------|--------|-------|---------|--------|------------|-------|
| local_lexicon | freq>=1 (full 750k) | 719k | 0.9593 | 92.1% | 0.10% | 0.919 |
| local_lexicon | top 150k (freq>=3) | 150k | 0.9485 | 94.2% | 0.04% | 0.941 |
| local_lexicon | **freq>=10 — prev best** | 52k | 0.9346 | **94.8%** | 0.07% | **0.946** |
| **main_i7** | freq>=10 (all) | 735k | 0.9512 | 90.7% | 0.13% | 0.904 |
| main_i7 | freq>=100 | 163k | 0.9423 | 93.5% | 0.11% | 0.913 |
| **main_i7 — FINAL** | **freq>=500 — 54k types** | **54,057** | 0.9242 | 94.2% | **0.01%** | 0.942 |
| main_i7 | freq>=1000 | 32k | 0.9116 | 94.3% | 0.02% | 0.939 |

**FINAL authoritative: main_i7 freq>=500 — 54,057 clean types, recall 94.2%, false_flag 0.01% (7x better than local_lexicon 0.07%, 85x better than WITH conf 0.85%), score 0.942, model 3.5MB, 1267 pages, 14 feats conf-free, pass 0.10 flag 0.65**

Why freq>=500 best for main_i7?
- main_i7 from bigger data: freq 5.6M max vs 60k in local_lexicon → 100x bigger corpus
- freq>=500 keeps words appearing at least 500 times in big corpus → very clean, real words, not web typos
- 54k types similar size to local_lexicon 52k (freq>=10) but cleaner because higher freq threshold on bigger data
- Result: lowest false flag 0.01% — only 1 in 10,000 clean words falsely flagged

---

## How to use

```python
from error_detection import DetectionEngine

detector = DetectionEngine()  # auto-finds main_i7 model
result = detector.verdict("በሸታ", doc_freq=2)
print(result.label, result.suspicion)
```

---

## Dictionary — 5M vs 750k

- `amh_final_training_text.txt` — 4.4M tokens (359k lines, 56MB) — raw training text
- `local_lexicon.txt` — 750,870 unique types (15MB) — from 4.4M tokens, NFC + normalization (ሐ→ሀ etc), no stemming
- `main_i7_frq10.txt` — **737,106 types (15MB)** — from bigger data (5M dict main_i7.txt, 4.9M unique, no duplicates, already clean), **freq 9 and lower removed** (min freq 10), max freq 5.6M (vs 60k in local), cleaner because bigger data

Scripts **call** dictionary (load with freq filter), they **don't train** dictionary itself. Dictionary static, model uses `in_lexicon` and `log_freq`.

To use full 5M dict: `DetectionEngine(dict_vocab_path="main_i7_freq10.txt", max_lexicon_types=None)` — but 737k types already includes all with freq>=10, very clean.

For 5M dict, **freq>=500** gives 54k clean types — best tradeoff (similar to local 52k but cleaner, lower FF).

---

## Files

- `Model/detector_lgbm_conffree.pkl` — **FINAL: main_i7 freq>=500, 54k types, 1267 pages, 14 feats conf-free, 3.5MB, recall 94.2% FF 0.01%**
- `Model/main_i7_frq10.txt` — 737k types, 15MB, from bigger data, freq>=10 already, max freq 5.6M
- `Model/local_lexicon.txt` — 750k types, 15MB, from 4.4M tokens
- `Model/confusion_matrix.json` — 47K
- `Model/harmonized_corpus_jiwer.tsv` — 1367 pages, 946k words (49MB)
- `Evaluation/` — 100 pages + analysis.tsv (62k rows)

Tests: 40 OK

---

## Results

**Tesseract:** 93% word acc, 98.07% char acc, WER 7%, 4267 errors, conf: correct mean 87, error mean 60, but 19% errors conf>85, 7.5% correct conf<70

**Detection:**
- WITH conf: 0.85% FF, 87.9% recall, misses 44% high-conf errors
- local_lexicon 150k: 0.04% FF, 94.2% recall
- local_lexicon freq>=10 (52k): 0.07% FF, 94.8% recall BEST score 0.946
- **main_i7 freq>=500 FINAL (54k): 0.01% FF (7x better), 94.2% recall, 100% high-conf caught in sample, 0% low-conf FF**

**One-line:** Old 9.85% FF, WITH conf 0.85% FF misses 44% high-conf, **FINAL main_i7 freq>=500: 0.01% FF (100x better than old, 85x better than WITH conf), 94.2% recall, 100% high-conf caught, 2.1-3.5MB, linguistic only, no conf trust, from bigger clean data.**
