"""Logistic error detector over per-word features.

A model is a plain JSON file: feature names, standardisation, weights, bias.
Feature dicts use the same keys as the corrector's TokenSignals.as_vector()
plus `ocr_conf` (Tesseract confidence, 0-100) when a TSV is available.
"""
import json
from typing import Dict, List, Sequence

from stage1_fit import fit_logreg, predict


class ErrorDetector:
    def __init__(self, names: Sequence[str], means, stds, weights, bias: float, threshold: float = 0.5):
        self.names = list(names)
        self.means = list(means)
        self.stds = list(stds)
        self.weights = list(weights)
        self.bias = float(bias)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, features: List[Dict[str, float]], labels: List[int], names: Sequence[str],
            l2: float = 1.0, threshold: float = 0.5) -> "ErrorDetector":
        X = [[f[k] for k in names] for f in features]
        m = fit_logreg(X, labels, l2=l2)
        return cls(names, m["means"], m["stds"], m["w"], m["b"], threshold)

    def score(self, features: List[Dict[str, float]]) -> List[float]:
        """Probability that each word is an OCR error."""
        X = [[f[k] for k in self.names] for f in features]
        model = {"w": self.weights, "b": self.bias, "means": self.means, "stds": self.stds}
        return predict(model, X)

    def flag(self, features: List[Dict[str, float]]) -> List[bool]:
        return [p >= self.threshold for p in self.score(features)]

    def to_dict(self) -> dict:
        return dict(names=self.names, means=self.means, stds=self.stds,
                    weights=self.weights, bias=self.bias, threshold=self.threshold)

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorDetector":
        return cls(d["names"], d["means"], d["stds"], d["weights"], d["bias"], d.get("threshold", 0.5))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ErrorDetector":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
