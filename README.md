# prowhiz

Neural network for protein-ligand binding free energy (dG) prediction from crystal structures.

Inspired by [PRODIGY-LIG](https://github.com/haddocking/prodigy-lig), ProWhiz is an exploratory project that aims to replace the linear contact-count model with a learned E(3)-equivariant graph neural network trained on LP-PDBBind dataset. The project will also replicate PRODIGY-LIG initially as a baseline to ensure accuracy so it can be compared correctly. 

---

## Installation

See **[INSTALL.md](INSTALL.md)** for full instructions, including GPU/CUDA setup for PyTorch Geometric.

**TL;DR:**
```bash
# 1. Install dependencies (after setting up PyTorch with the right CUDA version)
pip install torch-geometric hydra-core omegaconf mlflow pytest pytest-cov \
            ruff mypy dvc aiohttp types-tqdm pre-commit ipykernel

# 2. Install this package in editable mode
pip install -e .

# 3. Set up pre-commit hooks
pre-commit install
```

---

## Training Guide

### Step 1 — Prepare the labels file

The bundled dataset (`data/PRODIGY_LIG_dataset.csv`) contains 134 protein-ligand complexes
with experimental dG values. Convert it to the format the pipeline expects:

```bash
python scripts/convert_labels.py \
  --input data/PRODIGY_LIG_dataset.csv \
  --out data/external/labels.csv
```

`data/external/labels.csv` will have two columns: `pdb_id` and `dG_kcal_mol`.

To use your own data instead, create `data/external/labels.csv` directly with those same columns.

---

### Step 2 — Download crystal structures

Fetch the mmCIF files from RCSB PDB for every complex in your labels file:

```bash
python scripts/download_pdbs.py \
  --labels data/external/labels.csv \
  --out data/raw/
```

This downloads up to 10 files concurrently and skips any already present.
Failed downloads are listed in `data/raw/failed_downloads.txt`.

If you already have mmCIF files, place them in `data/raw/` named `{PDB_ID}.cif`
(uppercase IDs, e.g. `1E66.cif`).

---

### Step 3 — Featurize structures

Parse each CIF file, compute protein-ligand atomic contacts (10.5 Å cutoff),
and save a PyTorch Geometric graph as `data/processed/{PDB_ID}.pt`:

```bash
python scripts/prepare_dataset.py \
  --raw data/raw/ \
  --labels data/external/labels.csv \
  --out data/processed/ \
  --cutoff 10.5 \
  --workers 4
  # add --no-rcsb-query to skip network calls (faster, for offline use)
```

- `--cutoff` sets the contact detection distance in Angstroms (default 10.5, matching PRODIGY-LIG)
- `--workers` controls parallelism; use 1 if you hit memory issues
- `--no-rcsb-query` disables the RCSB PDB API call used to disambiguate ligands when a structure contains multiple non-solvent entities; ligand is then picked by heavy-atom count
- Failed structures are logged to `data/processed/failed.txt`
- Ligand selection audit written to `data/processed/ligand_selections.csv` — review this to verify the correct ligand was chosen, especially for rows where `had_ambiguity` is `True` or `selection_method` is `largest_by_heavy_atoms`

---

### Step 4 — Create train/val/test splits

Stratify 134 complexes into train (80%) / val (10%) / test (10%) by dG range:

```bash
python scripts/create_splits.py \
  --processed data/processed/ \
  --out data/splits/ \
  --val-frac 0.1 \
  --test-frac 0.1 \
  --seed 42
```

This writes `data/splits/train.txt`, `val.txt`, `test.txt` — one PDB ID per line.

> **Note on dataset size:** 134 complexes is a small dataset. Expect the model to
> overfit without regularization. The baseline MLP (step 5a) will likely perform
> comparably to the full GNN at this scale. For meaningful GNN training, consider
> supplementing with PDBbind (see [PLAN.md](PLAN.md) for details).

---

### Step 5a — Train the baseline MLP (recommended first)

Before training the full GNN, validate the entire pipeline with the fast MLP baseline.
It runs on CPU in minutes and should match or approach PRODIGY-LIG's performance (R ≈ 0.74):

```bash
python scripts/train.py \
  model=mlp_baseline \
  run_name=baseline_v1 \
  device=cuda \
  training.max_epochs=200
```

Check results in `outputs/metrics/val_metrics.json`. If `pearson_r` is below ~0.5,
there is likely a data pipeline problem to debug before training the GNN.

---

### Step 5b — Train the EGNN graph neural network

```bash
python scripts/train.py \
  model=egnn \
  run_name=egnn_v1 \
  device=cuda
```

Key config overrides you can pass on the command line:

| Override | Default | Notes |
|---|---|---|
| `device=cuda` | `cuda` | Use `cpu` if no GPU |
| `training.batch_size=16` | `32` | Reduce if GPU OOM (RTX 2060 = 6GB) |
| `training.lr=5e-4` | `1e-3` | Try lower LR if loss is unstable |
| `training.max_epochs=300` | `200` | More epochs for small datasets |
| `model.hidden_dim=64` | `128` | Reduce for faster training / less VRAM |
| `model.num_layers=3` | `4` | Fewer layers if overfitting |
| `training.loss=huber` | `huber` | Or `mse`, `combined` |

The model checkpoints to `outputs/checkpoints/best.pt` whenever validation Pearson R improves.
Training stops automatically after 30 epochs without improvement (`training.patience=30`).

---

### Step 6 — Monitor training

Open a second terminal and launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns/artifacts
```

Then open `http://localhost:5000` in your browser to view loss curves, metrics, runs,
and the Model Registry (all features require the SQLite backend).

---

### Step 7 — Evaluate on the test set

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/best.pt \
  --split data/splits/test.txt \
  --processed data/processed/ \
  --device cuda \
  --out outputs/metrics/test_metrics.json
```

Metrics printed and saved: `pearson_r`, `rmse` (kcal/mol), `mae` (kcal/mol),
`kendall_tau`, `success_rate` (fraction within 2 kcal/mol).

---

### Step 8 — Predict on a new structure

```bash
python scripts/predict.py \
  --input path/to/your_complex.cif \
  --checkpoint outputs/checkpoints/best.pt \
  --device cuda
```

Output: `Predicted dG: -9.341 kcal/mol`

The CIF file must contain both the protein (ATOM records) and the ligand (HETATM records)
in the same file, as is standard for PDB crystal structures.

---

## Running the full pipeline with DVC

All steps above are encoded as a reproducible DVC pipeline. After completing
steps 1–2 manually (labels + CIF downloads), you can run everything else with:

```bash
dvc repro
```

DVC tracks which parameters changed and re-runs only the affected stages.
To compare metrics across pipeline runs:

```bash
dvc metrics diff
dvc params diff
```

---

## Bundled Data

`data/PRODIGY_LIG_dataset.csv` — 134 protein-ligand complexes from the PRODIGY-LIG paper.
Columns: `complex` (PDB ID), `DG_exp` (experimental dG, kcal/mol), `DG_pred` (PRODIGY-LIG prediction).

The `DG_pred` column gives you a direct baseline to beat.

---

## Architecture Overview

See [PLAN.md](PLAN.md) for the full design: contact feature engineering, EGNN layer equations,
loss function rationale, data split strategy, and benchmark targets.

---

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
