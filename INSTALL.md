# Installation

## Requirements

- Python 3.11+
- `uv` (recommended) or `pip`
- CUDA 11.8 / 12.1 / 12.4 (optional, for GPU training)

## Why PyTorch Geometric needs special handling

PyTorch Geometric (PyG) has C++ extensions (`pyg-lib`, `torch-scatter`, `torch-sparse`) that must
match the exact version of PyTorch **and** CUDA installed on your system. They are pre-built wheels
hosted at `https://data.pyg.org/whl/` rather than PyPI, so standard `pip install` / `uv sync` will
not install them automatically.

**The base package (`torch-geometric>=2.5`) works without these extensions** — PyG ships a
pure-Python fallback for its core scatter/pool operations. The C++ extensions improve speed
(~2-3×) but are not required for correctness.

---

## Option A: CPU-only (development / testing)

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies (CPU PyTorch)
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

## Option B: CUDA GPU (recommended for real training)

### Step 1 — Install CUDA-enabled PyTorch first

Pick the command matching your CUDA version from https://pytorch.org/get-started/locally/

```bash
# CUDA 12.1 example
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 example
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 2 — Install the rest of the project

```bash
uv sync --extra dev
uv run pre-commit install
```

### Step 3 (optional) — Install PyG C++ extensions for faster training

```bash
make install-pyg-ext
# This auto-detects your torch+CUDA version and installs matching wheels.
```

---

## Dependency overview

| Package | Why |
|---|---|
| `torch>=2.3` | Core deep learning framework |
| `torch-geometric>=2.5` | Graph neural network primitives |
| `gemmi>=0.6` | Fast mmCIF parser (C++ bindings, ~0.2s per structure) |
| `scipy>=1.12` | `cKDTree` for contact search; `pearsonr`, `kendalltau` for metrics |
| `hydra-core>=1.3` | Hierarchical config management for experiments |
| `mlflow>=2.12` | Local experiment tracking (`mlflow ui` to view) |
| `numpy>=1.26`, `pandas>=2.1` | Array / dataframe handling |
| `tqdm>=4.66` | Progress bars |
| `dvc>=3.50` (dev) | Data version control |
| `ruff>=0.4` (dev) | Linting + formatting |
| `mypy>=1.10` (dev) | Static type checking |
| `pytest>=8.0` (dev) | Testing |
| `aiohttp>=3.9` (dev) | Async RCSB downloads in `scripts/download_pdbs.py` |
| `wandb>=0.17` (optional) | W&B tracking — install via `uv sync --extra wandb` |

---

## Verify installation

```bash
make test     # fast unit tests (no GPU, no real CIF files needed)
make lint
make typecheck
```

All tests should pass on a fresh CPU install.
