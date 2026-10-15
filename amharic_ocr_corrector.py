# Wrapper — main corrector moved to 4_ocr_correction/amharic_ocr_corrector.py
#
# Backward-compatibility shim for the old root-level import:
#     from amharic_ocr_corrector import AmharicOCRCorrector
# The real module is loaded from its new home and installed under this module's
# name, so attribute access, `from ... import X` and isinstance checks all see
# the same objects. (The previous version did `from amharic_ocr_corrector
# import *`, which re-imported THIS file and found nothing.)
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "4_ocr_correction", "amharic_ocr_corrector.py")
_spec = spec_from_file_location("amharic_ocr_corrector", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__] = _mod
