"""Tests for baseline_mlp.py."""

from __future__ import annotations

import torch
import pytest

from prowhiz.models.baseline_mlp import BaselineMLP
from prowhiz.data.cif_parser import StructureData
from prowhiz.data.contacts import compute_contacts
from prowhiz.data.graph_builder import build_graph


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


class TestBaselineMLP:
    def test_forward_returns_scalar(self, mock_data: object) -> None:
        model = BaselineMLP()
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.shape == (1, 1)

    def test_output_is_float(self, mock_data: object) -> None:
        model = BaselineMLP()
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.dtype == torch.float32

    def test_custom_hidden_dims(self, mock_data: object) -> None:
        model = BaselineMLP(input_dim=15, hidden_dims=[128, 64, 32])
        out = model(mock_data)  # type: ignore[arg-type]
        assert out.shape == (1, 1)

    def test_gradients_flow(self, mock_data: object) -> None:
        model = BaselineMLP()
        out = model(mock_data)  # type: ignore[arg-type]
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_batch_input(self, mock_data: object) -> None:
        """Simulate a batch by stacking contact_counts manually."""
        model = BaselineMLP()
        batch_counts = torch.rand(4, 15)

        class FakeBatch:
            contact_counts = batch_counts

        out = model(FakeBatch())  # type: ignore[arg-type]
        assert out.shape == (4, 1)
