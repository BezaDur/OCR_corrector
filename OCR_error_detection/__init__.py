# __init__.py — OCR error detection package.
#
# Checks each OCR word: keep it or fix it.
# Results: "pass" (fine), "gray" (uncertain), "flag" (wrong).
#
# Quick start:
#     from OCR_error_detection import DetectionEngine
#     detector = DetectionEngine(
#         dict_vocab_path="lexicon.txt",
#         confusion_matrix_path="confusion_matrix.json",
#     )
#     result = detector.verdict("በሸታ")
#     print(result.label, result.suspicion)
#

from OCR_error_detection.detector import DetectionEngine
from OCR_error_detection.adapter import CorrectorDetectionAdapter
from OCR_error_detection.signals import TokenSignals, Verdict
from OCR_error_detection.scoring import combine_signals, DEFAULT_WEIGHTS
from OCR_error_detection.confusion import ConfusionMatrix
from OCR_error_detection.normalizer import AmharicNormalizer
from OCR_error_detection.utils import (
    ETHIOPIC, ETHIOPIC_EXT, RARE_GLYPH_CLASSES,
    CONFUSION_CLASSES, ETHIOPIC_PUNCT,
    _is_ethiopic, _confusion_alts,
)

__all__ = [
    "DetectionEngine",
    "CorrectorDetectionAdapter",
    "TokenSignals",
    "Verdict",
    "ConfusionMatrix",
    "AmharicNormalizer",
    "combine_signals",
    "DEFAULT_WEIGHTS",
    "ETHIOPIC",
    "ETHIOPIC_EXT",
    "RARE_GLYPH_CLASSES",
    "CONFUSION_CLASSES",
    "ETHIOPIC_PUNCT",
    "_is_ethiopic",
    "_confusion_alts",
]

__version__ = "0.1.0"