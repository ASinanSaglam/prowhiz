"""Baseline MLP model that operates on the 15-dim contact-count feature vector.

This mirrors the original PRODIGY-LIG approach but uses a learned MLP instead
of linear regression. It serves as a fast sanity-check baseline: if contact
feature engineering is correct, this model should approach or exceed PRODIGY-LIG's
Pearson R ≈ 0.74 on CASF-2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data


class BaselineMLP(nn.Module):
    """Contact-count vector → MLP → scalar dG.

    Args:
        input_dim: Size of the contact-count feature vector (default 15).
        hidden_dims: Hidden layer sizes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        prev = input_dim
        for hdim in hidden_dims:
            layers.extend([nn.Linear(prev, hdim), nn.ReLU(), nn.Dropout(dropout)])
            prev = hdim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, data: Data | Batch) -> Tensor:
        """
        Args:
            data: PyG Data or Batch. Must have `contact_counts` (B, 10).
                  Optionally has `ligand_global_feats` (B, G) which are
                  concatenated when present.

        Returns:
            (B, 1) predicted dG values.
        """
        x: Tensor = data.contact_counts  # type: ignore[attr-defined]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.log1p(x)

        # Append ligand global features if the data carries them
        if hasattr(data, "ligand_global_feats"):
            g: Tensor = data.ligand_global_feats  # type: ignore[attr-defined]
            if g.dim() == 1:
                g = g.unsqueeze(0)
            if g.shape[-1] > 0:
                x = torch.cat([x, g], dim=-1)

        return self.net(x)
