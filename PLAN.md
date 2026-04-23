# ProWhiz — Project Plan

## What Is This?

ProWhiz is a deep learning system for predicting protein-ligand binding free energy (dG, kcal/mol) from crystal structures. It is inspired by [PRODIGY-LIG](https://github.com/haddocking/prodigy-lig), a linear contact-based predictor, but replaces the linear model with a learned neural network that can capture nonlinear interaction patterns.

**Input**: mmCIF file of a protein-ligand complex + known dG value (for training)  
**Output**: Predicted dG in kcal/mol

---

## Motivation

PRODIGY-LIG achieves R ≈ 0.74 on CASF-2016 using a linear regression over 15 hand-counted contact features. A neural network trained end-to-end on the same contact representation should learn better feature weights and capture interactions that a linear model misses — particularly for charged and aromatic interactions. Using an E(3)-equivariant GNN further ensures the model is physically correct: binding affinity does not depend on the orientation of the crystal in the unit cell.

---

## Data Pipeline

```
mmCIF files  →  gemmi parse  →  StructureData
  →  cKDTree contact search (cutoff = 10.5 Å)
  →  ContactList (atom type + character classification)
  →  Node/edge feature tensors  →  PyG Data object  →  .pt file
  →  Stratified train/val/test split  →  DataLoader
```

### Contact Feature Classification

Atom types: **C** (carbon), **N** (nitrogen), **O** (oxygen), **S** (sulfur), **X** (all other)

Contact-count feature vector (15 dims, for MLP baseline):
```
CC, CN, CO, CS, CX
    NN, NO, NS, NX
        OO, OS, OX
            SS, SX
                XX
```

Character classes (for GNN node features):
- **Apolar** — carbon atoms in non-polar context
- **Polar** — N, O, S atoms in neutral residues
- **Charged** — NZ/NH1/NH2 (Lys/Arg), OD/OE (Asp/Glu), or formal-charge bearing ligand atoms

---

## Model Architecture

### Baseline MLP (`baseline_mlp.py`)
The simplest possible learned model — validates that the contact feature engineering is correct before investing in the GNN.

```
contact_count_vector (15-dim)
  → Linear(15 → 64) → ReLU → Dropout
  → Linear(64 → 32) → ReLU
  → Linear(32 → 1)   →  dG
```

Target: Pearson R ≥ 0.70 on CASF-2016 (matches or exceeds linear PRODIGY-LIG).

### EGNN GNN (`egnn.py` + `gnn.py`)
A 4-layer E(3)-equivariant graph neural network. Equivariance ensures the prediction is invariant to rotation/translation of the crystal structure.

**Graph construction**: All protein atoms within `cutoff + 2.0 Å` of any ligand atom (contact-zone subgraph, ~100–300 protein nodes) + all ligand atoms. Start with a homogeneous graph (all atoms in one graph, `is_protein` flag distinguishes types).

**EGNN layer** (Satorras et al., 2021):
```
m_ij  = φ_e( h_i, h_j, ||x_i − x_j||², edge_attr )   # squared dist for E(3)-invariance
h_i'  = φ_h( h_i, Σ_j m_ij )                          # residual node update
x_i'  = x_i + clamp( Σ_j (x_i − x_j) · φ_x(m_ij) )  # coordinate update (clamped)
```

**Readout**: SumMeanPool on all nodes → MLP head (128 → 64 → 1)

**Loss**: Huber(δ=1.0) — robust to extreme-dG outlier complexes.  
Optional combined loss: `0.8 × Huber + 0.2 × (1 − PearsonR)` for better ranking.

Target: Pearson R ≥ 0.78 on CASF-2016.

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR schedule | Cosine annealing + 10-epoch warmup |
| Early stopping | val Pearson R, patience = 30 |
| Batch size | 32 |
| Max epochs | 200 |

---

## Data Versioning (DVC)

DVC manages all data and model artifacts. The pipeline is defined in `dvc.yaml` with four stages:

1. `download_raw` — fetch mmCIF files from RCSB PDB
2. `featurize` — parse CIFs, compute contacts, save PyG `.pt` files
3. `split` — stratified train/val/test split by dG range
4. `train` — train model, output checkpoint + metrics JSON

To run the full pipeline: `dvc repro`  
To check parameter sensitivity: `dvc params diff`  
To compare experiments: `dvc metrics diff`

---

## Experiment Tracking

- **MLflow** (default): runs stored in `./mlruns/`, view with `mlflow ui`
- **W&B** (optional): set `tracking.backend=wandb` in config or CLI

---

## Tooling

| Tool | Purpose |
|---|---|
| `uv` | Dependency management |
| `hatchling` | Build backend |
| `ruff` | Linting + formatting |
| `mypy --strict` | Type checking |
| `pytest` | Unit tests with coverage |
| `pre-commit` | Enforce quality gates on commit |
| `DVC` | Reproducible data + model pipeline |
| `Hydra` | Hierarchical config + experiment sweeps |
| `GitHub Actions` | CI: lint + test on every push |

### Key Commands

```bash
make install          # set up environment + pre-commit hooks
make test             # run fast unit tests (no GPU, no slow)
make lint             # ruff check
make typecheck        # mypy
make data             # dvc repro featurize split
make train            # python scripts/train.py run_name=my_exp
make evaluate         # evaluate best checkpoint on test set

# Predict dG for a single structure:
prowhiz-predict --input path/to/complex.cif
```

---

## Benchmark

The standard evaluation benchmark is **CASF-2016** (286 protein-ligand complexes from PDBbind v2016 core set). Train on PDBbind v2020 general set minus CASF test set.

| Model | Pearson R (CASF-2016) | RMSE (kcal/mol) |
|---|---|---|
| PRODIGY-LIG (linear) | 0.74 | 1.72 |
| BaselineMLP (target) | ≥ 0.70 | — |
| BindingGNN (target) | ≥ 0.78 | ≤ 1.50 |

---

## Directory Layout

```
prowhiz/
├── .github/workflows/   CI (lint + test)
├── configs/             Hydra config tree
├── data/                DVC-tracked data (raw, processed, splits, labels)
├── notebooks/           Exploratory analysis
├── outputs/             DVC-tracked checkpoints + metrics
├── scripts/             CLI entrypoints (train, evaluate, predict, data prep)
├── src/prowhiz/
│   ├── data/            CIF parsing, contact calculation, featurization, datasets
│   ├── models/          EGNN, GNN, baseline MLP
│   ├── training/        Trainer, losses, metrics
│   └── utils/           Logging, config helpers, registry
└── tests/               Unit tests mirroring src/ structure
```
