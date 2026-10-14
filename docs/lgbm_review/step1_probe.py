import sys, os, io, contextlib, tempfile, math
ROOT = '/home/user/OCR_corrector'
sys.path.insert(0, ROOT)
from OCR_error_detection.detector import DetectionEngine

d = tempfile.mkdtemp()
lex = os.path.join(d, 'w.txt')
with open(lex, 'w', encoding='utf-8') as fh:
    for w, f in [("ሰላም", 5000), ("ቤት", 4000), ("ነገር", 3000), ("ትምህርት", 2000), ("ወርቅ", 1000),
                 ("ዓለም", 9000), ("ሀገር", 8000)]:
        fh.write(f'{w} {f}\n')
with contextlib.redirect_stdout(io.StringIO()):
    e = DetectionEngine(dict_vocab_path=lex, confusion_matrix_path=ROOT + '/confusion_matrix.json',
                        char_lm_path=None, verbose=False)
print('model loaded:', e.lgbm_model is not None, '| legacy schema:', e._legacy_model)
for tok in ["ቤት", "ቤት፡", "ዓለም", "ሐገር", "ሰላም።"]:
    sig = e._cheap_signals(tok)
    for df in (None, 5):
        v = e.verdict(tok, doc_freq=df)
        print(f"{tok!r:10} doc_freq={str(df):4} in_lex(signal)={sig.in_lexicon!s:5} "
              f"score={v.suspicion:.3f} label={v.label:4} reason={v.reason!r}")
