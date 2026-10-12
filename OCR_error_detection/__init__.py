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

from .detector import DetectionEngine
from .adapter import CorrectorDetectionAdapter
from .signals import TokenSignals, Verdict
from .scoring import combine_signals, DEFAULT_WEIGHTS
from .confusion import ConfusionMatrix
from .normalizer import AmharicNormalizer
from .utils import (
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