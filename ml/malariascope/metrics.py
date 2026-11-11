"""Classification metrics, operating-threshold selection, and patient-clustered bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def div(a, b):
        return float(a / b) if b else 0.0

    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec = div(tp, tp + fp)
    f1 = div(2 * prec * sens, prec + sens)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "threshold": float(threshold),
        "accuracy": div(tp + tn, len(y)),
        "sensitivity": sens,  # recall for Parasitized
        "specificity": spec,
        "precision": prec,  # PPV
        "npv": div(tn, tn + fn),
        "f1": f1,
        "balanced_accuracy": (sens + spec) / 2,
        "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Best-specificity threshold that still reaches ``target`` sensitivity, placed with a safety margin.

    ``t_hi`` (the highest threshold meeting the target) sits exactly on a positive's score, which is brittle:
    any slightly lower-scoring positive in new data falls below it. Every threshold in ``(t_lo, t_hi]``, where
    ``t_lo`` is the highest negative score below ``t_hi``, gives the *same* validation sensitivity and
    specificity, so we take the midpoint of that plateau (a max-margin choice).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos)))  # need at least k positives at or above the threshold
    t_hi = float(pos[min(k, len(pos)) - 1])
    neg_below = p[(y == 0) & (p < t_hi)]
    t_lo = float(neg_below.max()) if len(neg_below) else 0.0
    return float(np.clip((t_hi + t_lo) / 2, 1e-6, 1 - 1e-6))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[i], 0.0, 1.0))


def _downsample(points: list[dict], n: int = 200) -> list[dict]:
    if len(points) <= n:
        return points
    idx = np.unique(np.linspace(0, len(points) - 1, n).round().astype(int))
    return [points[i] for i in idx]


def curves(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    roc = [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)]
    pr = [{"recall": float(r), "precision": float(q)} for r, q in zip(rec[::-1], prec[::-1])]
    return {"roc": _downsample(roc), "pr": _downsample(pr)}


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def cluster_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    n_iter: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """95% CIs by resampling *patients* (not cells) with replacement.

    Cells from one patient are correlated, so a naive cell-level bootstrap would produce CIs that are too
    narrow. Resampling whole patients gives honest uncertainty for "how would this do on new patients?".
    """
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    index_of = {g: np.where(groups == g)[0] for g in uniq}
    keys = ("accuracy", "sensitivity", "specificity", "f1", "roc_auc")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_iter):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in pick])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        m = binary_metrics(yb, pb, threshold)
        for k in keys[:-1]:
            acc[k].append(m[k])
        acc["roc_auc"].append(float(roc_auc_score(yb, pb)))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in acc.items() if v}
