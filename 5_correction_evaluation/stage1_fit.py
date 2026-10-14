"""Learned Stage 1 scoring from labelled verdicts (pure Python, no extra packages).

Inputs are the feature dicts logged per token (TokenSignals.as_vector()) and
the clean/error labels from stage1_labels. Ambiguous tokens are excluded.

  auc(scores, labels)          - how well one score ranks errors above clean tokens
  fit_logreg(X, y)             - L2-regularised logistic regression, plain gradient descent
  leave_one_page_out(rows, ...) - honest out-of-sample scores, one page held out at a time
"""
import math
from typing import Dict, List, Sequence


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability that a random error scores above a random clean token."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _standardise(X):
    cols = list(zip(*X))
    means = [sum(c) / len(c) for c in cols]
    stds = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1.0 for c, m in zip(cols, means)]
    return means, stds


def fit_logreg(X, y, l2: float = 1.0, lr: float = 0.1, iters: int = 400):
    means, stds = _standardise(X)
    Z = [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in X]
    d = len(Z[0]) if Z else 0
    w, b = [0.0] * d, 0.0
    n = len(Z)
    for _ in range(iters):
        gw, gb = [0.0] * d, 0.0
        for row, t in zip(Z, y):
            z = b + sum(wi * xi for wi, xi in zip(w, row))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - t
            gb += err
            for j in range(d):
                gw[j] += err * row[j]
        w = [wj - lr * (gwj / n + l2 * wj / n) for wj, gwj in zip(w, gw)]
        b -= lr * gb / n
    return {"w": w, "b": b, "means": means, "stds": stds}


def predict(model, X) -> List[float]:
    out = []
    for row in X:
        z = model["b"] + sum(
            wj * ((v - m) / s) for wj, v, m, s in zip(model["w"], row, model["means"], model["stds"]))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return out


def feature_names(rows: List[Dict]) -> List[str]:
    """Features present in every row (so no row needs a guessed value)."""
    names = None
    for r in rows:
        keys = set(r.get("features", {}))
        names = keys if names is None else names & keys
    return sorted(names or [])


def leave_one_page_out(rows: List[Dict], names: List[str], l2: float = 1.0) -> List[float]:
    """Out-of-sample probability for every row (rows must have status clean/error)."""
    pages = sorted({r["page"] for r in rows})
    scores = [None] * len(rows)
    for held in pages:
        train = [r for r in rows if r["page"] != held and r["status"] in ("clean", "error")]
        X = [[r["features"][k] for k in names] for r in train]
        y = [1 if r["status"] == "error" else 0 for r in train]
        model = fit_logreg(X, y, l2=l2)
        idx = [i for i, r in enumerate(rows) if r["page"] == held]
        preds = predict(model, [[rows[i]["features"][k] for k in names] for i in idx])
        for i, p in zip(idx, preds):
            scores[i] = p
    return scores
