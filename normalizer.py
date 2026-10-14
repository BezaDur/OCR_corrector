# Wrapper for backward compatibility — parent normalizer moved to 4_ocr_correction/normalizer.py
from importlib.util import spec_from_file_location, module_from_spec
import os
_path = os.path.join(os.path.dirname(__file__), "4_ocr_correction", "normalizer.py")
_spec = spec_from_file_location("normalizer_parent", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AmharicNormalizer = _mod.AmharicNormalizer
