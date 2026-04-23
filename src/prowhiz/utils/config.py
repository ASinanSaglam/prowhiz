"""OmegaConf/dataclass configuration schemas.

These dataclasses define the structure of the Hydra config tree and can be
used for type-safe config access throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    cutoff_angstrom: float = 10.5
    contact_zone_buffer: float = 2.0
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    labels_csv: str = "data/external/labels.csv"
    raw_dir: str = "data/raw/"
    processed_dir: str = "data/processed/"
    splits_dir: str = "data/splits/"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    _target_: str = "prowhiz.models.gnn.BindingGNN"
    name: str = "egnn"
    input_node_dim: int = 35
    ligand_node_dim: int = 11
    edge_attr_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    update_coords: bool = True
    head_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    use_batch_norm: bool = True


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 200
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss: str = "huber"
    huber_delta: float = 1.0
    combined_pearson_weight: float = 0.2
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    patience: int = 30
    checkpoint_metric: str = "val_pearson_r"
    checkpoint_mode: str = "max"


@dataclass
class TrackingConfig:
    backend: str = "mlflow"
    mlflow_uri: str = "./mlruns"
    wandb_project: str = "prowhiz"


@dataclass
class ProwhizConfig:
    project_name: str = "prowhiz"
    run_name: str = "default"
    output_dir: str = "outputs/"
    seed: int = 42
    device: str = "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
