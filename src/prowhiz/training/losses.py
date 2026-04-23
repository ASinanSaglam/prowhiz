"""Loss functions for binding free energy regression.

Available losses:
  - HuberLoss: Robust to outlier dG values (recommended).
  - MSELoss: Standard mean squared error.
  - CombinedLoss: Weighted sum of Huber + (1 - PearsonR), improves ranking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HuberLoss(nn.Module):
    """Huber (smooth L1) loss.

    Less sensitive to extreme dG outliers than MSE. Behaves like MSE when
    |error| < delta and like MAE otherwise.

    Args:
        delta: Transition point between quadratic and linear loss (default 1.0).
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class MSELoss(nn.Module):
    """Standard mean squared error loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)


class CombinedLoss(nn.Module):
    """Weighted combination of Huber loss and correlation-based loss.

    Loss = (1 - alpha) * Huber(pred, target) + alpha * (1 - PearsonR(pred, target))

    The correlation term encourages correct ranking of complexes by binding
    affinity, which is important for virtual screening applications.

    Args:
        delta: Huber loss delta parameter.
        alpha: Weight of the correlation term (0 = pure Huber, 1 = pure correlation).
    """

    def __init__(self, delta: float = 1.0, alpha: float = 0.2) -> None:
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.huber = HuberLoss(delta)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        huber_val = self.huber(pred, target)
        corr_loss = 1.0 - _pearson_r_loss(pred.squeeze(-1), target.squeeze(-1))
        return (1.0 - self.alpha) * huber_val + self.alpha * corr_loss


def _pearson_r_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute differentiable Pearson correlation coefficient.

    Args:
        pred: (N,) predicted values.
        target: (N,) target values.

    Returns:
        Scalar Pearson R in [-1, 1].
    """
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_mean = pred - pred.mean()
    target_mean = target - target.mean()

    num = (pred_mean * target_mean).sum()
    denom = torch.sqrt((pred_mean ** 2).sum() * (target_mean ** 2).sum() + 1e-8)
    return num / denom


def get_loss(name: str, **kwargs: float) -> nn.Module:
    """Factory function to instantiate a loss by name.

    Args:
        name: One of 'huber', 'mse', 'combined'.
        **kwargs: Additional arguments forwarded to the loss constructor.

    Returns:
        Instantiated nn.Module loss function.

    Raises:
        ValueError: If `name` is not recognized.
    """
    losses: dict[str, type[nn.Module]] = {
        "huber": HuberLoss,
        "mse": MSELoss,
        "combined": CombinedLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(losses.keys())}")
    return losses[name](**kwargs)
