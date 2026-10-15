# __init__.py — OCR error detection package.
#
# Checks each OCR word: keep it or fix it.
# Results: "pass" (fine), "gray" (uncertain), "flag" (wrong).
#
# Quick start:
#     from error_detection import DetectionEngine
#     detector = DetectionEngine(dict_vocab_path="lexicon.txt")
#     result = detector.verdict("በሸታ")
#     print(result.label, result.suspicion)
#

from error_detection.detector import ConfidencePolicy, DetectionEngine
from error_detection.adapter import CorrectorDetectionAdapter
from error_detection.signals import TokenSignals, Verdict
from error_detection.scoring import combine_signals, DEFAULT_WEIGHTS
from error_detection.confusion import ConfusionMatrix
from error_detection.normalizer import AmharicNormalizer
from error_detection.utils import (
    ETHIOPIC, ETHIOPIC_EXT, RARE_GLYPH_CLASSES,
    CONFUSION_CLASSES, ETHIOPIC_PUNCT,
    _is_ethiopic, _confusion_alts,
)

__all__ = [
    "DetectionEngine",
    "ConfidencePolicy",
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