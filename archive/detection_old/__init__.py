"""OCR error detection: scores each word for the chance it is an OCR error.

Engine-neutral: the detector reads examples (see detection/examples.py). Engine
formats are handled by adapters in detection/adapters/. Detection only flags words;
correction is a separate step.
"""
from detection.detector import ErrorDetector  # noqa: F401
