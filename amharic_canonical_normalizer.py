# Wrapper — moved to 4_ocr_correction/amharic_canonical_normalizer.py
from importlib.util import spec_from_file_location, module_from_spec
import os, sys
_path = os.path.join(os.path.dirname(__file__), "4_ocr_correction", "amharic_canonical_normalizer.py")
_spec = spec_from_file_location("amharic_canonical_normalizer", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__] = _mod
