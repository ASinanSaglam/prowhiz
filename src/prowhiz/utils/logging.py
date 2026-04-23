"""Experiment tracking factory: MLflow or Weights & Biases.

Usage:
    tracker = get_tracker("mlflow", uri="./mlruns", run_name="egnn_v1", project="prowhiz")
    tracker.log_metrics({"val_pearson_r": 0.78}, step=10)
    tracker.finish()
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Tracker(Protocol):
    """Minimal experiment tracker interface."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class MLflowTracker:
    """MLflow-backed experiment tracker."""

    def __init__(self, uri: str, experiment_name: str, run_name: str) -> None:
        import mlflow  # type: ignore[import-untyped]
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        logger.info("MLflow tracking: %s (run: %s)", uri, run_name)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def finish(self) -> None:
        self._mlflow.end_run()


class WandbTracker:
    """Weights & Biases-backed experiment tracker."""

    def __init__(self, project: str, run_name: str, config: dict[str, Any] | None = None) -> None:
        import wandb  # type: ignore[import-untyped]
        self._run = wandb.init(project=project, name=run_name, config=config or {})
        self._wandb = wandb
        logger.info("W&B tracking: project=%s, run=%s", project, run_name)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        log_dict = dict(metrics)
        if step is not None:
            log_dict["_step"] = step
        self._wandb.log(log_dict, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self._run.config.update(params)

    def finish(self) -> None:
        self._run.finish()


class NullTracker:
    """No-op tracker for runs where tracking is disabled."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def finish(self) -> None:
        pass


def get_tracker(
    backend: str = "mlflow",
    *,
    uri: str = "./mlruns",
    experiment_name: str = "prowhiz",
    run_name: str = "run",
    project: str = "prowhiz",
    config: dict[str, Any] | None = None,
) -> MLflowTracker | WandbTracker | NullTracker:
    """Instantiate an experiment tracker by backend name.

    Args:
        backend: 'mlflow', 'wandb', or 'none'.
        uri: MLflow tracking URI (only used when backend='mlflow').
        experiment_name: MLflow experiment name.
        run_name: Run name shown in the UI.
        project: W&B project name (only used when backend='wandb').
        config: Config dict to log as run parameters.

    Returns:
        A tracker implementing the Tracker protocol.
    """
    if backend == "mlflow":
        return MLflowTracker(uri=uri, experiment_name=experiment_name, run_name=run_name)
    elif backend == "wandb":
        return WandbTracker(project=project, run_name=run_name, config=config)
    elif backend == "none":
        return NullTracker()
    else:
        raise ValueError(f"Unknown tracking backend '{backend}'. Use: mlflow, wandb, none")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a simple format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
