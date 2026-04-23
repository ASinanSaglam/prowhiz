"""Evaluation metrics for binding free energy prediction.

Standard metrics used in the protein-ligand affinity literature:
  - Pearson R: Primary metric (correlation with experimental dG).
  - RMSE: Root mean squared error in kcal/mol.
  - MAE: Mean absolute error in kcal/mol.
  - Kendall tau: Rank correlation (useful for virtual screening).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau as _kendalltau  # type: ignore[import-untyped]
from scipy.stats import pearsonr as _pearsonr  # type: ignore[import-untyped]

MetricsDict = dict[str, float]


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Args:
        pred: (N,) predicted dG values.
        target: (N,) ground-truth dG values.

    Returns:
        Pearson R in [-1, 1].
    """
    if len(pred) < 2:
        return float("nan")
    r, _ = _pearsonr(pred.flatten(), target.flatten())
    return float(r)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in kcal/mol."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error in kcal/mol."""
    return float(np.mean(np.abs(pred - target)))


def kendall_tau(pred: np.ndarray, target: np.ndarray) -> float:
    """Kendall rank correlation coefficient.

    Args:
        pred: (N,) predicted dG values.
        target: (N,) ground-truth dG values.

    Returns:
        Kendall tau in [-1, 1].
    """
    if len(pred) < 2:
        return float("nan")
    tau, _ = _kendalltau(pred.flatten(), target.flatten())
    return float(tau)


def docking_success_rate(
    pred: np.ndarray, target: np.ndarray, threshold_kcal: float = 2.0
) -> float:
    """Fraction of predictions within `threshold_kcal` of the true dG.

    Args:
        pred: (N,) predicted dG values.
        target: (N,) ground-truth dG values.
        threshold_kcal: Error threshold in kcal/mol (default 2.0).

    Returns:
        Success rate in [0, 1].
    """
    return float(np.mean(np.abs(pred - target) <= threshold_kcal))


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold_kcal: float = 2.0,
) -> MetricsDict:
    """Compute all standard metrics at once.

    Returns:
        Dictionary with keys: pearson_r, rmse, mae, kendall_tau, success_rate.
    """
    return {
        "pearson_r": pearson_r(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "kendall_tau": kendall_tau(pred, target),
        "success_rate": docking_success_rate(pred, target, threshold_kcal),
    }
