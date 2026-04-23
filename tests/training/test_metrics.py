"""Tests for training/metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prowhiz.training.metrics import (
    compute_all_metrics,
    docking_success_rate,
    kendall_tau,
    mae,
    pearson_r,
    rmse,
)


class TestPearsonR:
    def test_perfect_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert pearson_r(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_anti_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        assert pearson_r(x, y) == pytest.approx(-1.0, abs=1e-6)

    def test_returns_nan_for_single_value(self) -> None:
        result = pearson_r(np.array([1.0]), np.array([1.0]))
        assert math.isnan(result)

    def test_range(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        r = pearson_r(x, y)
        assert -1.0 <= r <= 1.0


class TestRMSE:
    def test_zero_for_identical(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert rmse(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self) -> None:
        pred = np.array([0.0, 0.0])
        target = np.array([3.0, 4.0])
        # errors: 3, 4 → MSE = (9+16)/2 = 12.5 → RMSE = sqrt(12.5)
        assert rmse(pred, target) == pytest.approx(math.sqrt(12.5), rel=1e-5)


class TestMAE:
    def test_zero_for_identical(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert mae(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self) -> None:
        pred = np.array([0.0, 1.0])
        target = np.array([2.0, 3.0])
        assert mae(pred, target) == pytest.approx(2.0, rel=1e-5)


class TestKendallTau:
    def test_perfect_concordance(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        tau = kendall_tau(x, x)
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_perfect_discordance(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        tau = kendall_tau(x, y)
        assert tau == pytest.approx(-1.0, abs=1e-6)


class TestDockingSuccessRate:
    def test_all_within_threshold(self) -> None:
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.5, 3.5])  # all errors = 0.5 < 2.0
        assert docking_success_rate(pred, target, threshold_kcal=2.0) == pytest.approx(1.0)

    def test_none_within_threshold(self) -> None:
        pred = np.array([0.0, 0.0])
        target = np.array([5.0, 5.0])  # errors = 5.0 > 2.0
        assert docking_success_rate(pred, target, threshold_kcal=2.0) == pytest.approx(0.0)


class TestComputeAllMetrics:
    def test_returns_all_keys(self) -> None:
        rng = np.random.default_rng(1)
        pred = rng.normal(size=20)
        target = rng.normal(size=20)
        metrics = compute_all_metrics(pred, target)
        assert set(metrics.keys()) == {"pearson_r", "rmse", "mae", "kendall_tau", "success_rate"}

    def test_all_values_are_floats(self) -> None:
        pred = np.linspace(-10, 0, 20)
        target = np.linspace(-10, 0, 20) + np.random.default_rng(0).normal(0, 0.5, 20)
        metrics = compute_all_metrics(pred, target)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is not float"
