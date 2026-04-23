"""Sanity-check contact features and PRODIGY-LIG baseline against processed .pt files.

Checks:
  1. Feature shape — contact_counts must be (10,) per PRODIGY-LIG spec
  2. Ridge CV R² — should be > 0.4 if features match PRODIGY-LIG
  3. PRODIGY-LIG linear formula — Pearson R should be 0.65–0.74 on CASF-2016
  4. Mean-collapse diagnostic — if RMSE is low but R is also low, the model
     is predicting the dataset mean rather than ranking structures
"""

from __future__ import annotations

import glob

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from prowhiz.data.contacts import CONTACT_TYPE_PAIRS
from prowhiz.data.prodigy_baseline import predict_dg_batch

# ── Load processed graphs ─────────────────────────────────────────────────────

files = sorted(glob.glob("data/processed/*.pt"))
if not files:
    raise SystemExit("No .pt files found in data/processed/ — run prepare_dataset.py first.")

counts, ys, pdb_ids = [], [], []
shape_errors: list[str] = []
stale_files: list[str] = []

for f in files:
    d = torch.load(f, weights_only=False)
    if not hasattr(d, "contact_counts"):
        stale_files.append(f)
        continue
    c = d.contact_counts.squeeze().numpy()
    if c.shape != (10,):
        shape_errors.append(f"{f}: shape={c.shape}")
        continue
    counts.append(c)
    ys.append(d.y.item())
    pdb_ids.append(getattr(d, "pdb_id", f.split("/")[-1].replace(".pt", "")))

if stale_files:
    print(f"[WARN] {len(stale_files)} stale .pt files skipped (wrong format — need to regenerate):")
    for f in stale_files[:10]:
        print(f"  {f}")
    if len(stale_files) > 10:
        print(f"  ... and {len(stale_files) - 10} more")
    print("  → Run: rm data/processed/*.pt && python scripts/prepare_dataset.py ...")

if shape_errors:
    print(f"[WARN] {len(shape_errors)} files have wrong contact_counts shape (expected (10,)):")
    for e in shape_errors:
        print(f"  {e}")

counts = np.array(counts, dtype=np.float32)
ys = np.array(ys, dtype=np.float64)

print(f"\nLoaded {len(ys)} structures")
print(f"dG range: {ys.min():.2f} to {ys.max():.2f} kcal/mol  (mean={ys.mean():.2f}, std={ys.std():.2f})")

# ── Contact counts sanity ─────────────────────────────────────────────────────

print("\n── Contact counts (mean per type) ──")
for i, pair in enumerate(CONTACT_TYPE_PAIRS):
    print(f"  {pair[0]}-{pair[1]:1s}: {counts[:, i].mean():6.1f}  (max={counts[:, i].max():.0f})")

zero_contact = (counts.sum(axis=1) == 0).sum()
if zero_contact:
    print(f"\n[WARN] {zero_contact} structures have all-zero contact counts")

# ── Ridge cross-validation ────────────────────────────────────────────────────

n_folds = min(5, len(ys))
r2_scores = cross_val_score(Ridge(), counts, ys, cv=n_folds, scoring="r2")
print(f"\n── Ridge {n_folds}-fold CV ──")
print(f"  R²: {r2_scores.mean():.3f}  folds={r2_scores.round(3).tolist()}")
if r2_scores.mean() < 0.2:
    print("  [WARN] R² < 0.2 — features likely don't match PRODIGY-LIG or data quality issue")
elif r2_scores.mean() > 0.4:
    print("  [OK] R² > 0.4 — features appear consistent with PRODIGY-LIG")

# ── PRODIGY-LIG linear formula ────────────────────────────────────────────────

preds = predict_dg_batch(counts)
r, pval = pearsonr(preds, ys)
rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))
mean_pred_rmse = float(np.sqrt(np.mean((np.full_like(ys, ys.mean()) - ys) ** 2)))

print(f"\n── PRODIGY-LIG formula (ΔG = 0.0355·NN − 0.1278·XX − 0.0072·CN − 5.1923) ──")
print(f"  Pearson R: {r:.3f}  (p={pval:.2e})")
print(f"  RMSE:      {rmse:.3f} kcal/mol")
print(f"  Mean-pred RMSE (baseline): {mean_pred_rmse:.3f} kcal/mol")

if r < 0.4:
    print("  [WARN] R < 0.4 — expected 0.65–0.74 on CASF-2016; check ligand selection and contact geometry")
elif r >= 0.6:
    print("  [OK] R in expected range")

# ── Outlier inspection ────────────────────────────────────────────────────────

residuals = preds - ys
worst_idx = np.argsort(np.abs(residuals))[-5:][::-1]
print("\n── Top-5 outliers (|predicted − actual| kcal/mol) ──")
for i in worst_idx:
    print(f"  {pdb_ids[i]:6s}  predicted={preds[i]:6.2f}  actual={ys[i]:6.2f}  err={residuals[i]:+.2f}"
          f"  NN={counts[i,1]:.0f}  XX={counts[i,3]:.0f}  CN={counts[i,4]:.0f}")
