# detection — OCR error detection

Scores each word for the chance that it is an OCR error and flags the likely ones.
It does not correct anything. A correction step takes the flags as input later.

## Design rules
1. **Engine-neutral.** The detector reads examples (one feature dict per word). It does
   not read any engine's file format. Each engine gets an adapter in `adapters/`.
2. **Tesseract is an input, not a dependency.** Tesseract confidence is one optional
   feature (`ocr_conf`). Models trained without it run without it.
3. **Honest numbers.** Reported accuracy comes from leave-one-document-out, never from
   the data the model was fitted on.
4. **Independent evaluation.** `evaluate.py` scores any model on any labelled examples.
   It needs no correction run.

## Files
| File | Purpose |
|---|---|
| `examples.py` | The example format, reading/writing, validation |
| `detector.py` | `ErrorDetector`: logistic model, `fit / score / flag`, JSON save/load |
| `train.py` | Training with leave-one-document-out and threshold choice |
| `evaluate.py` | Metrics for any model on any labelled examples |
| `build_examples.py` | Build an examples file from a source |
| `adapters/plain_words.py` | Generic word list (any engine) |
| `adapters/tesseract_tsv.py` | Tesseract TSV (the only Tesseract-specific code) |
| `adapters/verdicts.py` | Corrector verdict files |

Shared code outside this folder: `stage1_fit.py` (logistic regression, AUC) and
`tsv_gt.py` (ground-truth alignment for labels). The labelling uses the corrector's
Ethiopic normalisation, so labels depend on that normalisation.

## Example format
One JSON object per line:
```json
{"doc": "page_0001", "index": 12, "text": "ሰላም", "label": "error", "features": {"ocr_conf": 20.0, "word_length": 4.0}}
```
- `doc`: the document the word came from. Splits are made by document.
- `label`: `clean`, `error`, `ambiguous` (excluded from scoring), or `null` (unlabelled).
- `features`: numeric values only. Keys are the model's inputs. `ocr_conf` is present
  only when the engine gave a confidence (0–100).

## Workflow
1. **Build examples** from a source:
   ```bash
   python3 -m detection.build_examples tsv TSV_DIR --gt-dir GT_DIR --out examples.jsonl
   python3 -m detection.build_examples verdicts VERDICT_DIR --out examples.jsonl
   python3 -m detection.build_examples words WORDS.jsonl --out examples.jsonl
   ```
2. **Train** (writes the model and prints out-of-sample results):
   ```bash
   python3 -m detection.train examples.jsonl --out detection/model.json --target 0.85 --report report.json
   ```
   Add `--no-ocr-conf` to train without the engine confidence.
3. **Evaluate** a model on examples. Use examples from documents the model did not train on:
   ```bash
   python3 -m detection.evaluate heldout.jsonl detection/model.json --out eval.json
   ```
4. **Use** in code:
   ```python
   from detection import ErrorDetector
   det = ErrorDetector.load("detection/model.json")
   flags = det.flag(feature_dicts)   # one dict per word, same keys as det.names
   ```

## Reading the numbers
- **AUC**: ranking quality. 0.5 is chance. It does not depend on the threshold.
- **precision**: share of flagged words that are errors. This is the number the
  correction step cares about, since each false flag is a possible over-correction.
- **recall**: share of all errors that get flagged.
- **clean_flagged**: clean words flagged. This is the damage count.
- **threshold**: the lowest one whose out-of-sample precision reaches `--target`.
  If no threshold reaches the target, the report says so and the threshold stays at 0.5.
- Precision on a few flagged words is unstable. Check the flagged count next to it.

## Limits
- The model is logistic regression. It can miss interactions between features.
- Only as good as the labels. Labels come from alignment to ground truth; a token
  matched to the wrong ground-truth word is labelled wrongly.
- The first trained model uses the 10 evaluation pages only, so its numbers are
  optimistic until a larger labelled set exists.
- The `ambiguous` label is excluded from training and scoring.

## Tests
`python3 -m unittest discover -s tests -p 'test_detection.py'`
