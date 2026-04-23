"""E(n)-Equivariant Graph Neural Network layer.

Implementation follows Satorras et al. (2021) "E(n) Equivariant Graph Neural Networks"
(https://arxiv.org/abs/2102.09844).

Key properties:
- Equivariant to E(n) transformations (rotations, translations, reflections)
- Message function uses SQUARED distances (||x_i - x_j||²) for E(3)-invariance
- Coordinate update is clamped to prevent numerical explosion during early training
- Uses scatter operations from torch_geometric.nn for efficient aggregation

Usage:
    layer = EGNNLayer(hidden_dim=128, edge_attr_dim=4)
    h_new, pos_new = layer(h, pos, edge_index, edge_attr)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class EGNNLayer(MessagePassing):
    """Single EGNN message-passing layer.

    Args:
        hidden_dim: Node feature dimension (both input and output).
        edge_attr_dim: Edge attribute dimension (0 = no edge attributes).
        update_coords: If True, update atom coordinates (full EGNN).
            If False, only update node features (equivalent to SchNet-style).
        coord_clamp: Maximum absolute value of coordinate update per step.
        residual: If True, add residual connection on node features.
        normalize_messages: If True, normalize aggregated messages by degree.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int = 0,
        update_coords: bool = True,
        coord_clamp: float = 100.0,
        residual: bool = True,
        normalize_messages: bool = False,
    ) -> None:
        super().__init__(aggr="add")  # use add aggregation
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.coord_clamp = coord_clamp
        self.residual = residual
        self.normalize_messages = normalize_messages

        # Message network: φ_e(h_i, h_j, sq_dist, edge_attr) → m_ij
        msg_input_dim = 2 * hidden_dim + 1 + edge_attr_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update network: φ_h(h_i, agg_m) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate update network: φ_x(m_ij) → scalar weight
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),  # bounded output helps coordinate update stability
            )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one EGNN layer.

        Args:
            h: (N, hidden_dim) node features.
            pos: (N, 3) atom coordinates.
            edge_index: (2, E) edge connectivity (source → target).
            edge_attr: (E, edge_attr_dim) optional edge features.

        Returns:
            h_new: (N, hidden_dim) updated node features.
            pos_new: (N, 3) updated coordinates (same as pos if update_coords=False).
        """
        row, col = edge_index[0], edge_index[1]

        # Compute relative positions and squared distances
        rel_pos = pos[row] - pos[col]              # (E, 3)
        sq_dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build message inputs
        if edge_attr is not None and edge_attr.shape[1] > 0:
            msg_input = torch.cat([h[row], h[col], sq_dist, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([h[row], h[col], sq_dist], dim=-1)

        # Compute messages
        m_ij = self.edge_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate messages via propagate (calls message → aggregate)
        agg = self.propagate(edge_index, h=h, m_ij=m_ij, size=(h.size(0), h.size(0)))  # (N, hidden_dim)

        if self.normalize_messages:
            degree = self._compute_degree(edge_index, h.size(0))
            agg = agg / degree.clamp(min=1).unsqueeze(-1).float()

        # Update node features
        h_update = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h + h_update) if self.residual else self.norm(h_update)

        # Update coordinates
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # (E, 1)
            weighted_rel = rel_pos * coord_weights  # (E, 3)
            # Aggregate coordinate updates: Σ_j (x_i - x_j) * φ_x(m_ij)
            coord_update = self._aggregate_coords(edge_index, weighted_rel, h.size(0))
            coord_update = coord_update.clamp(-self.coord_clamp, self.coord_clamp)
            pos_new = pos + coord_update
        else:
            pos_new = pos

        return h_new, pos_new

    def message(self, m_ij: Tensor) -> Tensor:
        return m_ij

    def _aggregate_coords(
        self, edge_index: Tensor, weighted_rel: Tensor, n_nodes: int
    ) -> Tensor:
        """Scatter weighted relative positions to target nodes."""
        col = edge_index[1]
        coord_update = torch.zeros(n_nodes, 3, device=weighted_rel.device, dtype=weighted_rel.dtype)
        coord_update.scatter_add_(0, col.unsqueeze(-1).expand_as(weighted_rel), weighted_rel)
        return coord_update

    def _compute_degree(self, edge_index: Tensor, n_nodes: int) -> Tensor:
        col = edge_index[1]
        degree = torch.zeros(n_nodes, device=col.device, dtype=torch.long)
        degree.scatter_add_(0, col, torch.ones_like(col))
        return degree
