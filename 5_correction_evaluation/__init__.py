import os, sys
_root = os.path.dirname(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
_corr = os.path.join(_root, "4_ocr_correction")
_eval = os.path.join(_root, "3_training_evaluation")
for p in [_corr, _eval, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)
