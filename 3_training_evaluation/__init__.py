# Ensure parent corrector and sibling eval can be imported
import os, sys
_root = os.path.dirname(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
_corr = os.path.join(_root, "4_ocr_correction")
if _corr not in sys.path:
    sys.path.insert(0, _corr)
