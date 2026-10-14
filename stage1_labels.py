# Wrapper — moved to 5_correction_evaluation/stage1_labels.py
from importlib.util import spec_from_file_location, module_from_spec
import os, sys
_path = os.path.join(os.path.dirname(__file__), "5_correction_evaluation", "stage1_labels.py")
_spec = spec_from_file_location("stage1_labels", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__] = _mod
