.PHONY: install install-torch install-pyg-cpu install-pyg-cuda lint format typecheck test test-all data train evaluate predict clean

# ── Environment setup ─────────────────────────────────────────────────────────

# Standard install (CPU PyTorch + dev tools).
# For GPU, run `make install-torch-cuda` BEFORE this target.
install:
	uv sync --extra dev
	uv run pre-commit install

# Install CPU-only PyTorch (already covered by pyproject.toml, explicit for clarity)
install-torch-cpu:
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install CUDA 12.1 PyTorch (run this first, then `uv sync`)
install-torch-cuda:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install PyG C++ extensions matching the current torch+CUDA version.
# Discovers torch version automatically.
install-pyg-ext:
	@TORCH=$$(python -c "import torch; print(torch.__version__.split('+')[0])"); \
	CUDA=$$(python -c "import torch; v=torch.version.cuda; print('cpu' if v is None else 'cu' + v.replace('.',''))"); \
	echo "Installing PyG extensions for torch=$$TORCH cuda=$$CUDA"; \
	uv pip install pyg-lib torch-scatter torch-sparse \
	  -f https://data.pyg.org/whl/torch-$$TORCH+$$CUDA.html

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

typecheck:
	uv run mypy src/prowhiz/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -m "not slow and not gpu"

test-all:
	uv run pytest tests/

# ── Data pipeline ─────────────────────────────────────────────────────────────

# Convert the bundled PRODIGY_LIG_dataset.csv to the expected labels format
convert-labels:
	uv run python scripts/convert_labels.py \
	  --input data/PRODIGY_LIG_dataset.csv \
	  --out data/external/labels.csv

data:
	dvc repro featurize split

# ── Training ──────────────────────────────────────────────────────────────────

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py

predict:
	uv run python scripts/predict.py --input $(INPUT)

dvc-pipeline:
	dvc repro

# ── Misc ──────────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/ .mypy_cache/ .ruff_cache/ .pytest_cache/ .coverage htmlcov/
