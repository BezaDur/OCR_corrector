import sys, os, io, contextlib, tempfile, math
# --- portable paths: tools/analysis/ is two levels below the repo root -------
import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _os.path.join(_REPO, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO, "tools"))
import stage_paths  # noqa: E402
stage_paths.register(_REPO)
DETECTOR = _os.path.join(_REPO, "4_ocr_correction", "error_detection")  # detector package
EVAL = _os.path.join(_REPO, "Evaluation")                              # shared page corpus
DATA = _os.path.join(_REPO, "1_data_preparation", "data")              # corpora
# ----------------------------------------------------------------------------
ROOT = DETECTOR

from error_detection.detector import DetectionEngine

d = tempfile.mkdtemp()
lex = os.path.join(d, 'w.txt')
with open(lex, 'w', encoding='utf-8') as fh:
    for w, f in [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000), ("ትምህርት", 2000), ("ወርቅ", 1000),
                 ("ዓለም", 9000), ("ሀገር", 8000)]:
        fh.write(f'{w} {f}\n')
with contextlib.redirect_stdout(io.StringIO()):
    e = DetectionEngine(dict_vocab_path=lex, confusion_matrix_path=os.path.join(DETECTOR, 'Model', 'confusion_matrix.json'),
                        char_lm_path=None, verbose=False)
print('model loaded:', e.lgbm_model is not None, '| legacy schema:', e._legacy_model)
for tok in ["ቤት", "ቤት፡", "ዓለም", "ሐገር", "ሰላም።"]:
    sig = e._cheap_signals(tok)
    for df in (None, 5):
        v = e.verdict(tok, doc_freq=df)
        print(f"{tok!r:10} doc_freq={str(df):4} in_lex(signal)={sig.in_lexicon!s:5} "
              f"score={v.suspicion:.3f} label={v.label:4} reason={v.reason!r}")
