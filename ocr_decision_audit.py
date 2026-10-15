# Wrapper — decisions audit moved to OCR_error_detection/scripts/audit_decision.py
#
# Backward-compatibility shim for the old root-level import:
#     from ocr_decision_audit import audit_tokens, DECISION_STATUSES
# The real module is loaded from its new home and installed under this module's
# name, so attribute access and `from ... import X` see the same objects.
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "OCR_error_detection", "scripts", "audit_decision.py")
_spec = spec_from_file_location("ocr_decision_audit", _path)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules[__name__] = _mod
