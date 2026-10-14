# Wrapper — moved to 3_training_evaluation/tsv_gt_stats.py
from importlib.util import spec_from_file_location, module_from_spec
import os, sys
_path = os.path.join(os.path.dirname(__file__), "3_training_evaluation", "tsv_gt_stats.py")
_spec = spec_from_file_location("tsv_gt_stats", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__] = _mod
