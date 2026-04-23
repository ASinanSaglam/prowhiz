"""Training loop with checkpointing, early stopping, and experiment tracking.

The Trainer class handles:
  - Single-GPU and CPU training
  - Gradient clipping
  - Cosine LR schedule with linear warmup
  - Early stopping on validation Pearson R
  - MLflow / W&B experiment logging
  - Best-checkpoint saving
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from prowhiz.training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Stops training when a metric stops improving.

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: 'max' (higher is better) or 'min' (lower is better).
    """

    def __init__(self, patience: int = 30, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_value = -math.inf if mode == "max" else math.inf
        self.counter = 0

    def __call__(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (value > self.best_value) if self.mode == "max" else (value < self.best_value)
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Orchestrates training and validation loops.

    Args:
        model: PyTorch model (BaselineMLP or BindingGNN).
        loss_fn: Loss function (HuberLoss, MSELoss, CombinedLoss).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        device: 'cuda' or 'cpu'.
        lr: Initial learning rate.
        weight_decay: L2 regularization strength.
        max_epochs: Maximum training epochs.
        warmup_epochs: Number of linear warmup epochs.
        grad_clip: Maximum gradient norm (0 = no clipping).
        patience: Early stopping patience.
        checkpoint_dir: Directory to save best checkpoint.
        tracker: Optional experiment tracker (MLflow or W&B run object).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        warmup_epochs: int = 10,
        grad_clip: float = 1.0,
        patience: int = 30,
        checkpoint_dir: str | Path = "outputs/checkpoints",
        tracker: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.tracker = tracker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # LR schedule: linear warmup → cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        self.early_stopper = EarlyStopper(patience=patience, mode="max")
        self.best_val_r = -math.inf
        self._start_epoch = 1  # overridden by resume()

    def _to_device(self, batch: Batch) -> Batch:
        return batch.to(self.device)

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch. Returns dict of scalar metrics."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            pred: Tensor = self.model(batch).squeeze(-1)
            target: Tensor = batch.y.squeeze(-1)  # type: ignore[attr-defined]

            loss: Tensor = self.loss_fn(pred, target)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        self.scheduler.step()
        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def val_epoch(self) -> dict[str, float]:
        """Run one validation epoch. Returns dict of scalar metrics."""
        self.model.eval()
        all_pred: list[float] = []
        all_target: list[float] = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            pred = self.model(batch).squeeze(-1)
            target = batch.y.squeeze(-1)  # type: ignore[attr-defined]

            loss = self.loss_fn(pred, target)
            total_loss += float(loss.item())
            n_batches += 1

            all_pred.extend(pred.cpu().numpy().tolist())
            all_target.extend(target.cpu().numpy().tolist())

        pred_arr = np.array(all_pred, dtype=np.float32)
        target_arr = np.array(all_target, dtype=np.float32)
        metrics = compute_all_metrics(pred_arr, target_arr)
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore optimizer, scheduler, and stopper state from a checkpoint.

        Call this after constructing Trainer and before calling fit().
        The model weights must already be loaded separately (or will be
        loaded here if the checkpoint contains them).
        """
        ckpt: dict[str, Any] = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_r = float(ckpt.get("val_pearson_r", -math.inf))
        self.early_stopper.best_value = self.best_val_r
        self.early_stopper.counter = int(ckpt.get("stopper_counter", 0))
        self._start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info(
            "Resumed from %s (epoch %d, best val_r=%.4f)",
            checkpoint_path,
            self._start_epoch - 1,
            self.best_val_r,
        )

    def fit(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Best validation metrics dict.
        """
        best_metrics: dict[str, float] = {}

        for epoch in range(self._start_epoch, self.max_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()
            elapsed = time.time() - t0

            val_r = val_metrics.get("pearson_r", 0.0)
            lr_now = self.optimizer.param_groups[0]["lr"]

            log_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            log_metrics["lr"] = lr_now
            log_metrics["epoch"] = float(epoch)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_r=%.4f | val_rmse=%.4f | lr=%.2e | %.1fs",
                epoch,
                self.max_epochs,
                train_metrics["train_loss"],
                val_r,
                val_metrics.get("rmse", float("nan")),
                lr_now,
                elapsed,
            )

            if self.tracker is not None:
                try:
                    self.tracker.log_metrics(log_metrics, step=epoch)
                except Exception:
                    pass

            if val_r > self.best_val_r:
                self.best_val_r = val_r
                best_metrics = val_metrics
                self._save_checkpoint(epoch)

            if self.early_stopper(val_r):
                logger.info("Early stopping at epoch %d (best val_r=%.4f)", epoch, self.best_val_r)
                break

        return best_metrics

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_pearson_r": self.best_val_r,
            "stopper_counter": self.early_stopper.counter,
        }
        path = self.checkpoint_dir / "best.pt"
        torch.save(ckpt, str(path))
        logger.debug("Saved checkpoint to %s (epoch %d, val_r=%.4f)", path, epoch, self.best_val_r)

    @classmethod
    def load_checkpoint(
        cls, model: nn.Module, checkpoint_path: str | Path, device: str = "cpu"
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Load a saved checkpoint into a model.

        Args:
            model: Model instance with the same architecture as the checkpoint.
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to load tensors to.

        Returns:
            (model, checkpoint_dict) tuple.
        """
        ckpt: dict[str, Any] = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt
