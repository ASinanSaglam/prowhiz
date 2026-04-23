"""Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --split data/splits/test.txt \\
        --processed data/processed/ \\
        [--device cpu] \\
        [--out outputs/metrics/test_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from prowhiz.data.dataset import get_dataloader
from prowhiz.training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: object,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a dataloader, return (predictions, targets) arrays."""
    model.eval()
    all_pred: list[float] = []
    all_target: list[float] = []

    for batch in loader:  # type: ignore[union-attr]
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        target = batch.y.squeeze(-1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_target.extend(target.cpu().numpy().tolist())

    return np.array(all_pred), np.array(all_target)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--split", required=True, help="Test split .txt file")
    parser.add_argument("--processed", default="data/processed/", help="Processed data dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="outputs/metrics/test_metrics.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt: dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    logger.info("Loaded checkpoint from epoch %d (val_r=%.4f)", ckpt.get("epoch", -1), ckpt.get("val_pearson_r", float("nan")))

    # The checkpoint doesn't store model architecture — need to infer it.
    # We try BindingGNN first, fall back to BaselineMLP based on key inspection.
    from prowhiz.data.featurizer import EDGE_ATTR_DIM, LIGAND_NODE_DIM, PROTEIN_NODE_DIM
    from prowhiz.models.baseline_mlp import BaselineMLP
    from prowhiz.models.gnn import BindingGNN
    from prowhiz.training.trainer import Trainer

    # Detect model type from state dict keys
    state_dict = ckpt["model_state_dict"]
    if "protein_proj.weight" in state_dict:
        model: torch.nn.Module = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
        )
    else:
        model = BaselineMLP()

    model, _ = Trainer.load_checkpoint(model, args.checkpoint, device=device)
    logger.info("Model loaded: %s", type(model).__name__)

    loader = get_dataloader(
        args.processed,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    pred, target = evaluate(model, loader, device)
    metrics = compute_all_metrics(pred, target)

    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info("  %s = %.4f", k, v)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
