"""Shared pytest fixtures for all tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from prowhiz.data.cif_parser import StructureData, parse_cif

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_CIF = FIXTURES_DIR / "mock_structure.cif"
MOCK_LABELS = FIXTURES_DIR / "mock_labels.csv"


@pytest.fixture(scope="session")
def mock_cif_path() -> Path:
    """Path to the minimal mock mmCIF fixture."""
    assert MOCK_CIF.exists(), f"Mock CIF not found: {MOCK_CIF}"
    return MOCK_CIF


@pytest.fixture(scope="session")
def mock_structure(mock_cif_path: Path) -> StructureData:
    """Parsed StructureData from the mock CIF fixture."""
    return parse_cif(mock_cif_path, pdb_id="MOCK", query_rcsb=False)


@pytest.fixture(scope="session")
def mock_labels_path() -> Path:
    """Path to the mock labels CSV."""
    assert MOCK_LABELS.exists(), f"Mock labels not found: {MOCK_LABELS}"
    return MOCK_LABELS
