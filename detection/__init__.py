"""OCR error detection: scores each word for the chance it is an OCR error.

Detection only flags words. Correction is a separate step that takes the flags.
"""
from detection.detector import ErrorDetector  # noqa: F401
