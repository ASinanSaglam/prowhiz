"""Tests for the EGNNLayer."""

from __future__ import annotations

import torch
import pytest

from prowhiz.models.egnn import EGNNLayer


def _make_random_graph(
    n_nodes: int = 10, n_edges: int = 20, hidden_dim: int = 16, edge_attr_dim: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    h = torch.randn(n_nodes, hidden_dim)
    pos = torch.randn(n_nodes, 3)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(n_edges, edge_attr_dim)
    return h, pos, edge_index, edge_attr


class TestEGNNLayer:
    def test_output_shape_h(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4)
        h, pos, edge_index, edge_attr = _make_random_graph()
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        assert h_new.shape == h.shape

    def test_output_shape_pos(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4, update_coords=True)
        h, pos, edge_index, edge_attr = _make_random_graph()
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        assert pos_new.shape == pos.shape

    def test_pos_unchanged_when_no_coord_update(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4, update_coords=False)
        h, pos, edge_index, edge_attr = _make_random_graph()
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        assert torch.allclose(pos, pos_new)

    def test_no_nan_in_output(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4)
        h, pos, edge_index, edge_attr = _make_random_graph()
        h_new, pos_new = layer(h, pos, edge_index, edge_attr)
        assert not torch.isnan(h_new).any()
        assert not torch.isnan(pos_new).any()

    def test_gradients_flow_through_h(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4)
        h, pos, edge_index, edge_attr = _make_random_graph()
        h.requires_grad_(True)
        h_new, _ = layer(h, pos, edge_index, edge_attr)
        h_new.sum().backward()
        assert h.grad is not None

    def test_no_edge_attr(self) -> None:
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=0)
        h, pos, edge_index, _ = _make_random_graph()
        h_new, pos_new = layer(h, pos, edge_index)
        assert h_new.shape == h.shape

    def test_equivariance_translation(self) -> None:
        """Translating all coordinates should not change node features."""
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4)
        layer.eval()
        h, pos, edge_index, edge_attr = _make_random_graph()

        with torch.no_grad():
            h1, _ = layer(h, pos, edge_index, edge_attr)
            translation = torch.tensor([5.0, 3.0, -2.0])
            h2, _ = layer(h, pos + translation, edge_index, edge_attr)

        assert torch.allclose(h1, h2, atol=1e-5), "Node features should be translation-invariant"

    def test_equivariance_rotation(self) -> None:
        """Rotating coordinates should not change node features (equivariance test)."""
        layer = EGNNLayer(hidden_dim=16, edge_attr_dim=4)
        layer.eval()
        h, pos, edge_index, edge_attr = _make_random_graph()

        # Build a random rotation matrix
        rand_mat = torch.randn(3, 3)
        rot, _ = torch.linalg.qr(rand_mat)
        if torch.det(rot) < 0:
            rot[:, 0] = -rot[:, 0]

        with torch.no_grad():
            h1, pos1 = layer(h, pos, edge_index, edge_attr)
            h2, pos2 = layer(h, pos @ rot.T, edge_index, edge_attr)

        assert torch.allclose(h1, h2, atol=1e-4), "Node features should be rotation-invariant"
