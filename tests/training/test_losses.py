"""Tests for training/losses.py."""

from __future__ import annotations

import torch
import pytest

from prowhiz.training.losses import CombinedLoss, HuberLoss, MSELoss, get_loss


def _make_tensors(n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    pred = torch.randn(n, 1)
    target = torch.randn(n, 1)
    return pred, target


class TestHuberLoss:
    def test_returns_scalar(self) -> None:
        loss_fn = HuberLoss(delta=1.0)
        pred, target = _make_tensors()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_zero_for_identical(self) -> None:
        loss_fn = HuberLoss()
        x = torch.tensor([[1.0], [2.0]])
        assert float(loss_fn(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_differentiable(self) -> None:
        loss_fn = HuberLoss()
        pred, target = _make_tensors()
        pred.requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestMSELoss:
    def test_returns_scalar(self) -> None:
        loss_fn = MSELoss()
        pred, target = _make_tensors()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_known_value(self) -> None:
        loss_fn = MSELoss()
        pred = torch.tensor([[1.0]])
        target = torch.tensor([[3.0]])
        assert float(loss_fn(pred, target)) == pytest.approx(4.0, rel=1e-5)


class TestCombinedLoss:
    def test_returns_scalar(self) -> None:
        loss_fn = CombinedLoss()
        pred, target = _make_tensors(16)
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_differentiable(self) -> None:
        loss_fn = CombinedLoss()
        pred, target = _make_tensors(16)
        pred.requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_alpha_zero_equals_huber(self) -> None:
        huber = HuberLoss(delta=1.0)
        combined = CombinedLoss(delta=1.0, alpha=0.0)
        pred, target = _make_tensors()
        assert float(combined(pred, target)) == pytest.approx(float(huber(pred, target)), rel=1e-4)


class TestGetLoss:
    def test_get_huber(self) -> None:
        loss = get_loss("huber")
        assert isinstance(loss, HuberLoss)

    def test_get_mse(self) -> None:
        loss = get_loss("mse")
        assert isinstance(loss, MSELoss)

    def test_get_combined(self) -> None:
        loss = get_loss("combined")
        assert isinstance(loss, CombinedLoss)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("bad_loss")
