"""Main training entrypoint using Hydra for configuration.

Usage:
    python scripts/train.py run_name=my_experiment
    python scripts/train.py run_name=egnn_v1 model=egnn training.lr=5e-4
    python scripts/train.py +experiment=egnn_v1
    python scripts/train.py model=mlp_baseline run_name=baseline_test

Outputs:
    outputs/checkpoints/best.pt   — best model checkpoint (by val Pearson R)
    outputs/metrics/val_metrics.json — final validation metrics
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_data_dims(processed_dir: str, splits_dir: str) -> dict[str, int | str]:
    """Peek at the first .pt file to get actual node dims and featurizer version.

    Returns a dict with keys: protein_node_dim, ligand_node_dim, featurizer_version.
    Falls back to featurizer constants if no data is found.
    """
    from prowhiz.data.featurizer import EDGE_ATTR_DIM, LIGAND_NODE_DIM, PROTEIN_NODE_DIM

    train_split = Path(splits_dir) / "train.txt"
    defaults = {
        "protein_node_dim": PROTEIN_NODE_DIM,
        "ligand_node_dim": LIGAND_NODE_DIM,
        "edge_attr_dim": EDGE_ATTR_DIM,
        "featurizer_version": "unknown",
        "mlp_input_dim": 10,
    }
    if not train_split.exists():
        return defaults
    pdb_ids = [l.strip() for l in train_split.read_text().splitlines() if l.strip()]
    for pdb_id in pdb_ids:
        pt = Path(processed_dir) / f"{pdb_id}.pt"
        if pt.exists():
            try:
                sample = torch.load(str(pt), weights_only=False)
                n_protein = getattr(sample, "n_protein", None)
                lig_dim = getattr(sample, "ligand_node_dim", None)
                prot_dim = sample.x.shape[1] if hasattr(sample, "x") else PROTEIN_NODE_DIM
                mlp_input_dim = getattr(sample, "mlp_input_dim", 10)
                return {
                    "protein_node_dim": prot_dim,
                    "ligand_node_dim": int(lig_dim) if lig_dim is not None else LIGAND_NODE_DIM,
                    "edge_attr_dim": sample.edge_attr.shape[1] if hasattr(sample, "edge_attr") and sample.edge_attr.shape[0] > 0 else EDGE_ATTR_DIM,
                    "featurizer_version": getattr(sample, "featurizer_version", "unknown"),
                    "mlp_input_dim": int(mlp_input_dim),
                }
            except Exception:
                continue
    return defaults


def _build_model(cfg: DictConfig, data_dims: dict[str, int | str]) -> torch.nn.Module:
    """Instantiate model from Hydra config, using actual data dims where possible."""
    from prowhiz.models.baseline_mlp import BaselineMLP
    from prowhiz.models.gnn import BindingGNN

    model_name: str = cfg.model.get("name", "egnn")

    if model_name == "mlp_baseline":
        return BaselineMLP(
            input_dim=int(data_dims.get("mlp_input_dim", cfg.model.get("input_dim", 10))),
            hidden_dims=list(cfg.model.get("hidden_dims", [64, 32])),
            dropout=cfg.model.get("dropout", 0.1),
        )
    else:  # egnn / default
        return BindingGNN(
            input_node_dim=int(data_dims["protein_node_dim"]),
            ligand_node_dim=int(data_dims["ligand_node_dim"]),
            edge_attr_dim=int(data_dims["edge_attr_dim"]),
            hidden_dim=cfg.model.get("hidden_dim", 128),
            num_layers=cfg.model.get("num_layers", 4),
            dropout=cfg.model.get("dropout", 0.1),
            update_coords=cfg.model.get("update_coords", True),
            head_hidden_dims=list(cfg.model.get("head_hidden_dims", [128, 64])),
            use_batch_norm=cfg.model.get("use_batch_norm", True),
        )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from prowhiz.data.dataset import get_dataloader
    from prowhiz.training.losses import get_loss
    from prowhiz.training.trainer import Trainer
    from prowhiz.utils.logging import get_tracker, setup_logging

    setup_logging("INFO")
    _set_seed(cfg.seed)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Resolve device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Data
    train_loader = get_dataloader(
        cfg.data.processed_dir,
        Path(cfg.data.splits_dir) / "train.txt",
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device == "cuda",
    )
    val_loader = get_dataloader(
        cfg.data.processed_dir,
        Path(cfg.data.splits_dir) / "val.txt",
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device == "cuda",
    )

    # Read actual feature dims from data (handles versioned feature sets automatically)
    data_dims = _read_data_dims(cfg.data.processed_dir, cfg.data.splits_dir)
    logger.info(
        "Data dims — protein: %d, ligand: %d, featurizer: %s",
        data_dims["protein_node_dim"], data_dims["ligand_node_dim"], data_dims["featurizer_version"],
    )

    # Model and loss
    model = _build_model(cfg, data_dims)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | Parameters: %d", cfg.model.get("name", "egnn"), n_params)

    loss_kwargs: dict[str, float] = {}
    if cfg.training.loss == "huber":
        loss_kwargs["delta"] = cfg.training.huber_delta
    elif cfg.training.loss == "combined":
        loss_kwargs["delta"] = cfg.training.huber_delta
        loss_kwargs["alpha"] = cfg.training.combined_pearson_weight

    loss_fn = get_loss(cfg.training.loss, **loss_kwargs)

    # Experiment tracker
    tracker = get_tracker(
        backend=cfg.tracking.backend,
        uri=cfg.tracking.mlflow_uri,
        experiment_name=cfg.project_name,
        run_name=cfg.run_name,
        project=cfg.tracking.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    tracker.log_params({
        "n_params": n_params,
        "model_name": cfg.model.get("name"),
        "featurizer_version": data_dims["featurizer_version"],
        "protein_node_dim": data_dims["protein_node_dim"],
        "ligand_node_dim": data_dims["ligand_node_dim"],
        "mlp_input_dim": data_dims["mlp_input_dim"],
    })

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        grad_clip=cfg.training.grad_clip,
        patience=cfg.training.patience,
        checkpoint_dir=Path(cfg.output_dir) / "checkpoints",
        tracker=tracker,
    )

    resume_path = cfg.get("resume_from", None)
    if resume_path:
        trainer.resume(resume_path)

    best_metrics = trainer.fit()

    # Save metrics
    metrics_dir = Path(cfg.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    logger.info("Best val metrics: %s", best_metrics)
    tracker.log_metrics(best_metrics)
    tracker.finish()


if __name__ == "__main__":
    main()
