"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate(y: np.ndarray, s: np.ndarray, prob: bool = True) -> dict:
    """PR-AUC (primary), ROC-AUC, Brier (only if s is a probability), chance level."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    assert len(y) == len(s) and np.isfinite(s).all()
    npos = int(y.sum())
    out = {"n": int(len(y)), "n_pos": npos, "chance_pr_auc": float(y.mean())}
    if 0 < npos < len(y):
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan
    out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1))) if prob else np.nan
    return out


def safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else 0.0
