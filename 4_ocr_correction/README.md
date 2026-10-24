# 4_ocr_correction — confusion matrix and the main corrector

**Parent project, step 4: turn OCR output into corrected text.**

Two jobs live here:

1. **Learn the optical confusion matrix** from aligned OCR errors
   (`03_build_confusion_matrix.py` → `confusion_matrix.json`). The authoritative
   copy is `OCR_error_detection/Model/confusion_matrix.json`; the corrector and
   the detector both read that file.
2. **Run the corrector** (`amharic_ocr_corrector.py`) — a CPU post-processor that
   combines Stage 1 detection, SymSpell/optical candidate generation, a
   morphological gate and a KenLM beam.

---

## The corrector

`amharic_ocr_corrector.py` — **1,817 lines / 88 KB**.

```
line of OCR text
 └─ EthiopicPunctuationManager.normalize_sentence_text   (፡/:: → space/።, keep surface spelling)
 └─ split → decompose_token → (leading, core, trailing)
      └─ Stage 1: suspicion.verdict(word, ctx, ocr_conf, doc_freq)     ← the child detector
            pass  → keep the token, advance the LM state (canonicalize surface if enabled)
            gray  → keep searching, add gray_margin to the original's protection
            flag  → keep searching
      └─ Stage 2: candidates — identity, ghost suffix, SymSpell (d=1),
                  single optical reversals, pairwise reversals, peel-guard vetoes
      └─ Stage 3: KenLM 5-gram beam (width 4)
                  λ_lm·logP(w|h) + λ_opt·(−optical cost) + λ_morph·is_known_word
                  + anchor margin (verified / OOV originals), then </s>
      └─ Stage 4: merge two OOV tokens / split one OOV token
```

Defaults (all constructor arguments; **none of them are tuned on held-out
data**): `beam_width=4`, `max_optical_cost=3.6` (pairwise 1.6×),
`lambda_optical=1.2`, `lambda_lm=1.8`, `lambda_morph=1.5`,
`anchor_margin=3.5`, `oov_anchor_margin=3.0`, `gray_margin=3.0`,
`max_candidates=24`, `score_eos=True`, `stage1_detection=True`. Full table in
`docs/current_state.md` §3.

### Stage 1 is the child detector

```python
# inside AmharicOCRCorrector.__init__
from OCR_error_detection.adapter import CorrectorDetectionAdapter
self.suspicion = CorrectorDetectionAdapter(self)   # reuses the corrector's vocab
```

`CorrectorDetectionAdapter` builds `OCR_error_detection.DetectionEngine`, then
hands it the corrector's vocabulary (`adopt_lexicon`, the model's freq ≥ 500
filter still applies) and delegates every verdict to the LightGBM model. The
adapter is constructed *before* the corrector loads its lexicon, so the handover
actually happens on the first verdict; until then the engine uses the in-repo
54k lexicon it loaded itself. The
engine's payload thresholds (0.10 / 0.65) win over the adapter's constructor
defaults. `4_ocr_correction/suspicion.py` (`SuspicionScorer`) is the rule-based
**fallback** only, used when the engine cannot be constructed. Failures inside
the handover are recorded on the adapter and never propagate into the beam.

### KenLM is required

The Viterbi stage scores every candidate with a word 5-gram, so the corrector
refuses to correct without one — it does not silently fall back.

```bash
export AMHARIC_WORD_KENLM=/path/to/amharic_5gram.binary   # or AMHARIC_DATA_ROOT
python -c "
from amharic_ocr_corrector import AmharicOCRCorrector   # root shim -> this folder
c = AmharicOCRCorrector(verbose=False)
print(c.correct_text('ሰላም በሸታ'))"
```

`AmharicOCRCorrector()` **constructs** fine without the model (the lexicon,
confusion matrix and Stage 1 adapter all load); the first `correct_text()` raises
`ValueError` naming `AMHARIC_WORD_KENLM`, `kenlm_path=` and `lm=`. Inject a fake
via `lm=` (anything with `BeginSentenceWrite`/`BaseScore`/`State`) to test the
decoder without KenLM — that is how `tests/test_head_contract.py` works.

### Morphology

`vendor/segmenter/` is a frozen copy of the Amharic morphological suite. The
corrector calls exactly one method of it (`segment_word()` via `analyze()` /
`is_known_word()`); see `docs/segmenter_integration.md` for what is wired and
what is not. `use_morphology=False` (or `--no-morphology`) turns it off for the
ablation. It needs `pydantic` (now in `requirements.txt`); without it the import
is skipped with a warning and the `lambda_morph` term cancels out.

---

## Files

| File | Purpose |
|---|---|
| `amharic_ocr_corrector.py` | the corrector (stages 1–4 above) |
| `normalizer.py` | parent normalizer: punctuation, numerals, abbreviations, homoglyph keying |
| `amharic_canonical_normalizer.py` | canonicalisation used by metrics |
| `suspicion.py` | rule-based Stage 1 fallback (deprecated, not the default path) |
| `03_build_confusion_matrix.py` | build `confusion_matrix.json` from aligned OCR output |
| `confusion_matrix_patch.json` | curated pairs merged into the matrix |
| `confusion_matrix_coverage_audit.py` | which character pairs the matrix covers |
| `trace_candidate.py` | trace how one token's candidates are scored |
| `test_corrector.py`, `test_page1_fixes.py` | manual smoke runs — both need a real KenLM model |

---

## How to run

```bash
# offline tests for the corrector contract (no KenLM, no data files)
python -m pytest tests/test_head_contract.py -q

# rebuild the confusion matrix from an aligned OCR run
python 4_ocr_correction/03_build_confusion_matrix.py --ocr-dir ... --gt-dir ... \
  --out OCR_error_detection/Model/confusion_matrix.json

# manual smoke run (needs KenLM + the dictionary)
python 4_ocr_correction/test_corrector.py
```

The end-to-end benchmark lives in `5_correction_evaluation/`; see
`docs/two_phase_benchmark.md` and `docs/current_state.md` §5 before quoting any
result — the last committed measurement was produced by an older revision and
was **negative**.
