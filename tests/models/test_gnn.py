"""Tests for BindingGNN."""

from __future__ import annotations

import torch
import pytest

from prowhiz.data.cif_parser import StructureData
from prowhiz.data.contacts import compute_contacts
from prowhiz.data.graph_builder import build_graph
from prowhiz.data.featurizer import LIGAND_NODE_DIM, PROTEIN_NODE_DIM, EDGE_ATTR_DIM
from prowhiz.models.gnn import BindingGNN


@pytest.fixture
def mock_data(mock_structure: StructureData) -> object:
    contacts = compute_contacts(
        mock_structure.protein_atoms, mock_structure.ligand_atoms, cutoff=10.5
    )
    return build_graph(
        protein_atoms=mock_structure.protein_atoms,
        ligand_atoms=mock_structure.ligand_atoms,
        contacts=contacts,
        dg=-8.5,
    )


class TestBindingGNN:
    def test_forward_returns_scalar(self, mock_data: object) -> None:
        model = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=32,
            num_layers=2,
        )
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.shape == (1, 1)

    def test_output_dtype(self, mock_data: object) -> None:
        model = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=32,
            num_layers=2,
        )
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, mock_data: object) -> None:
        model = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=32,
            num_layers=2,
        )
        out = model(mock_data)  # type: ignore[arg-type]
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, mock_data: object) -> None:
        model = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=32,
            num_layers=2,
        )
        out = model(mock_data)  # type: ignore[arg-type]
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_no_coord_update_mode(self, mock_data: object) -> None:
        model = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
            hidden_dim=32,
            num_layers=2,
            update_coords=False,
        )
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.shape == (1, 1)
