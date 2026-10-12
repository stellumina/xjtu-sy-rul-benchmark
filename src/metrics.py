"""Evaluation metrics with protocol-mandated RUL filtering."""
from __future__ import annotations

import numpy as np


def _filtered(y_true, y_pred, rul_true=None, truncation=None):
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("length/shape mismatch")
    if rul_true is not None:
        rul = np.asarray(rul_true, dtype=float)
        if rul.shape != truth.shape:
            raise ValueError("rul shape mismatch")
        if truncation is None:
            raise ValueError("truncation is required when rul_true is supplied")
        mask = rul <= float(truncation)
        truth, pred = truth[mask], pred[mask]
    if truth.size == 0:
        raise ValueError("no samples remain after RUL filtering")
    return truth, pred


def rmse(y_true, y_pred, rul_true=None, truncation=None) -> float:
    truth, pred = _filtered(y_true, y_pred, rul_true, truncation)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(y_true, y_pred, rul_true=None, truncation=None) -> float:
    truth, pred = _filtered(y_true, y_pred, rul_true, truncation)
    return float(np.mean(np.abs(pred - truth)))


def phm08_score(y_true, y_pred, rul_true=None, truncation=None) -> float:
    """PHM08 asymmetric score: overprediction uses 10, underprediction 13."""
    truth, pred = _filtered(y_true, y_pred, rul_true, truncation)
    error = pred - truth
    penalties = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(np.sum(penalties))


def compute_metrics(y_true, y_pred, rul_true=None, truncation=None) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred, rul_true, truncation), "mae": mae(y_true, y_pred, rul_true, truncation), "phm08": phm08_score(y_true, y_pred, rul_true, truncation)}


# Existing callers used this name.
def mean_absolute_error(y_true, y_pred):
    return mae(y_true, y_pred)
