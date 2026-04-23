"""Reusable readout modules for graph-level prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_mean_pool


class SumMeanPool(nn.Module):
    """Global pooling that concatenates sum and mean, doubling the feature size.

    Args:
        in_dim: Node feature dimension (output will be 2 * in_dim).
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.out_dim = 2 * in_dim

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            x: (N, in_dim) node features.
            batch: (N,) batch assignment vector.

        Returns:
            (B, 2 * in_dim) graph-level features.
        """
        return torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=-1)


class MLPHead(nn.Module):
    """MLP readout head mapping pooled features to a scalar output.

    Args:
        in_dim: Input feature dimension.
        hidden_dims: List of hidden layer sizes.
        out_dim: Output dimension (1 for regression).
        dropout: Dropout probability applied between hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hdim in hidden_dims:
            layers.extend([nn.Linear(prev, hdim), nn.ReLU(), nn.Dropout(dropout)])
            prev = hdim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
