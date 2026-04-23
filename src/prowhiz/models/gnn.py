"""Full BindingGNN model: input projection → stacked EGNN layers → readout.

Architecture:
  1. Separate input linear projections for protein and ligand node features
     into a shared `hidden_dim` space.
  2. N EGNN layers with residual connections and layer normalization.
  3. SumMeanPool global readout.
  4. MLP head mapping to a scalar dG prediction.

The model accepts a homogeneous PyG Data/Batch object where protein and ligand
atoms are concatenated (protein first, then ligand, separated by `n_protein`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data

from prowhiz.models.egnn import EGNNLayer
from prowhiz.models.heads import MLPHead, SumMeanPool


class BindingGNN(nn.Module):
    """E(3)-equivariant GNN for protein-ligand binding free energy prediction.

    Args:
        input_node_dim: Protein node feature dimension (from featurizer.PROTEIN_NODE_DIM).
        ligand_node_dim: Ligand node feature dimension (from featurizer.LIGAND_NODE_DIM).
        edge_attr_dim: Edge feature dimension (from featurizer.EDGE_ATTR_DIM).
        hidden_dim: Hidden dimension for all EGNN layers.
        num_layers: Number of EGNN message-passing layers.
        dropout: Dropout probability in the MLP head.
        update_coords: If True, EGNN updates atom coordinates.
        head_hidden_dims: Hidden layer sizes for the readout MLP head.
        use_batch_norm: If True, apply BatchNorm after input projection.
    """

    def __init__(
        self,
        input_node_dim: int = 35,
        ligand_node_dim: int = 11,
        edge_attr_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        update_coords: bool = True,
        head_hidden_dims: list[int] | None = None,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if head_hidden_dims is None:
            head_hidden_dims = [128, 64]

        # Input projection: different weights for protein vs ligand nodes
        # In the homogeneous graph, we project both to hidden_dim.
        # We handle this by projecting the full combined feature vector;
        # protein and ligand have different dims, so we use the larger dim
        # and mask the rest (or, more cleanly, project separately and scatter).
        # Clean approach: project protein nodes and ligand nodes separately,
        # then build a new combined hidden feature matrix.
        self.protein_proj = nn.Linear(input_node_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_node_dim, hidden_dim)

        if use_batch_norm:
            self.input_bn: nn.Module = nn.BatchNorm1d(hidden_dim)
        else:
            self.input_bn = nn.Identity()

        # EGNN layers
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    update_coords=update_coords,
                    residual=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Readout
        pool_out_dim = 2 * hidden_dim  # SumMeanPool doubles the dimension
        self.pool = SumMeanPool(hidden_dim)
        self.head = MLPHead(pool_out_dim, head_hidden_dims, out_dim=1, dropout=dropout)

    def _project_nodes(self, data: Data | Batch) -> Tensor:
        """Project protein and ligand node features to shared hidden_dim.

        The graph stores all node features in data.x with shape (N_total, max_dim).
        Protein nodes use `input_node_dim` features, ligand nodes use `ligand_node_dim`.
        We need to split, project separately, and recombine.

        The split point is `data.n_protein` (or a batch vector of them).
        """
        x: Tensor = data.x  # type: ignore[attr-defined]
        pos: Tensor = data.pos  # type: ignore[attr-defined]

        # Recover n_protein per sample in the batch
        if hasattr(data, "n_protein"):
            n_prot_per_graph: Tensor | int = data.n_protein  # type: ignore[attr-defined]
        else:
            raise ValueError("data.n_protein is required for BindingGNN")

        # Build mask: True for protein nodes, False for ligand nodes
        n_total = x.shape[0]
        is_protein = torch.zeros(n_total, dtype=torch.bool, device=x.device)

        # Handle both single-graph and batched cases
        if isinstance(data, Batch):
            batch_vec: Tensor = data.batch  # type: ignore[attr-defined]
            # n_protein may be a tensor (one per graph) or scalar
            if isinstance(n_prot_per_graph, Tensor) and n_prot_per_graph.dim() > 0:
                # Build is_protein mask from per-graph n_protein values
                # We use cumulative node offsets from the batch vector
                node_counts = torch.bincount(batch_vec)
                offset = 0
                for graph_idx in range(node_counts.shape[0]):
                    n_prot_i = int(n_prot_per_graph[graph_idx])
                    is_protein[offset : offset + n_prot_i] = True
                    offset += int(node_counts[graph_idx])
            else:
                # All graphs have the same n_protein (unlikely but handle it)
                n_prot_val = int(n_prot_per_graph) if isinstance(n_prot_per_graph, int) else int(n_prot_per_graph.item())
                node_counts = torch.bincount(batch_vec)
                offset = 0
                for graph_idx in range(node_counts.shape[0]):
                    is_protein[offset : offset + n_prot_val] = True
                    offset += int(node_counts[graph_idx])
        else:
            # Single graph
            n_prot_val = int(n_prot_per_graph)
            is_protein[:n_prot_val] = True

        from prowhiz.data.featurizer import LIGAND_NODE_DIM, PROTEIN_NODE_DIM

        # Project protein nodes using first PROTEIN_NODE_DIM features
        prot_mask = is_protein
        lig_mask = ~is_protein

        h = torch.zeros(n_total, self.protein_proj.out_features, device=x.device, dtype=x.dtype)

        if prot_mask.any():
            h[prot_mask] = self.protein_proj(x[prot_mask, :PROTEIN_NODE_DIM])
        if lig_mask.any():
            h[lig_mask] = self.ligand_proj(x[lig_mask, :LIGAND_NODE_DIM])

        h = self.input_bn(h)
        return h

    def forward(self, data: Data | Batch) -> Tensor:
        """Predict dG for a batch of protein-ligand complexes.

        Args:
            data: PyG Data or Batch with x, pos, edge_index, edge_attr, n_protein.

        Returns:
            (B, 1) predicted dG values.
        """
        pos: Tensor = data.pos  # type: ignore[attr-defined]
        edge_index: Tensor = data.edge_index  # type: ignore[attr-defined]
        edge_attr: Tensor | None = getattr(data, "edge_attr", None)

        # Get batch assignment (all zeros for a single graph)
        if hasattr(data, "batch") and data.batch is not None:  # type: ignore[attr-defined]
            batch: Tensor = data.batch  # type: ignore[attr-defined]
        else:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=pos.device)  # type: ignore[attr-defined]

        h = self._project_nodes(data)

        # Message passing
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        # Global readout
        graph_feat = self.pool(h, batch)  # (B, 2 * hidden_dim)
        return self.head(graph_feat)      # (B, 1)
