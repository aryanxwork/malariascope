"""Probability calibration (temperature scaling) and calibration metrics.

Neural nets are often over-confident. Temperature scaling (Guo et al., 2017) learns one scalar T on the
validation set so that sigmoid(logit / T) matches observed frequencies. It never changes the ranking
(ROC-AUC is identical) — it only makes the reported "93% parasitized" mean something.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    z = logits / temperature
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T in [0.5, 10]. Without a bound, perfectly separable validation data drives T -> 0 (infinitely sharp
    probabilities), which is over-fitting the calibration set, not calibration."""
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    res = minimize_scalar(lambda log_t: nll(logits, y, float(np.exp(log_t))), bounds=(np.log(0.5), np.log(10.0)), method="bounded")
    return float(np.exp(res.x))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Equal-width reliability bins + Expected Calibration Error (ECE)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += n / len(p) * abs(acc - conf)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "confidence": conf, "frequency": acc, "count": n})
    return {"ece": float(ece), "bins": bins}


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
