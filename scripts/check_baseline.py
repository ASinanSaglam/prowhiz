"""Sanity-check contact features and PRODIGY-LIG baseline against processed .pt files.

Checks:
  1. Feature shape — contact_counts must be (10,) per PRODIGY-LIG spec
  2. Ridge CV R² — whether contact features carry any predictive signal
  3. PRODIGY-LIG linear formula — Pearson R (expected 0.65–0.74 on PRODIGY-LIG dataset)
  4. Mean-collapse diagnostic — low RMSE but low R means predicting the mean

Usage:
    python scripts/check_baseline.py
    python scripts/check_baseline.py --processed data/processed/lp_pdbbind/
    python scripts/check_baseline.py --processed data/processed/base/ --split data/splits/train.txt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from prowhiz.data.contacts import CONTACT_TYPE_PAIRS
from prowhiz.data.prodigy_baseline import predict_dg_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="PRODIGY-LIG baseline check on processed graphs")
    parser.add_argument("--processed", default="data/processed/base/", help="Directory of .pt graph files")
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split .txt file to restrict to a subset of PDB IDs",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Ridge CV folds (default 5)")
    args = parser.parse_args()

    processed_dir = Path(args.processed)

    # Optionally restrict to a split
    allowed_ids: set[str] | None = None
    if args.split:
        allowed_ids = {
            line.strip().upper()
            for line in Path(args.split).read_text().splitlines()
            if line.strip()
        }
        print(f"Restricting to {len(allowed_ids)} IDs from {args.split}")

    files = sorted(processed_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found in {processed_dir}")

    counts, ys, pdb_ids = [], [], []
    skipped_format, skipped_split, shape_errors = 0, 0, []
    feat_versions: set[str] = set()

    for f in files:
        try:
            d = torch.load(str(f), weights_only=False)
        except Exception:
            skipped_format += 1
            continue
        if not hasattr(d, "contact_counts"):
            skipped_format += 1
            continue
        pdb_id = str(getattr(d, "pdb_id", f.stem)).upper()
        if allowed_ids is not None and pdb_id not in allowed_ids:
            skipped_split += 1
            continue
        c = d.contact_counts.squeeze().numpy()
        if c.shape != (10,):
            shape_errors.append(f"{f.name}: shape={c.shape}")
            continue
        counts.append(c)
        ys.append(float(d.y.squeeze()))
        pdb_ids.append(pdb_id)
        feat_versions.add(str(getattr(d, "featurizer_version", "unknown")))

    if skipped_format:
        print(f"[WARN] {skipped_format} files skipped (missing contact_counts or unreadable)")
    if skipped_split:
        print(f"[INFO] {skipped_split} files excluded by split filter")
    if shape_errors:
        print(f"[WARN] {len(shape_errors)} files have wrong contact_counts shape:")
        for e in shape_errors[:5]:
            print(f"  {e}")

    if not counts:
        raise SystemExit("No valid graphs loaded.")

    counts = np.array(counts, dtype=np.float32)
    ys = np.array(ys, dtype=np.float64)

    version_str = next(iter(feat_versions)) if len(feat_versions) == 1 else f"MIXED: {feat_versions}"
    print(f"\nFeaturizer version : {version_str}")
    print(f"Structures loaded  : {len(ys)}")
    print(f"dG range           : {ys.min():.2f} to {ys.max():.2f} kcal/mol  (mean={ys.mean():.2f}, std={ys.std():.2f})")

    # ── Contact counts stats ──────────────────────────────────────────────────

    print("\n── Contact counts (mean per type) ──")
    for i, pair in enumerate(CONTACT_TYPE_PAIRS):
        print(f"  {pair[0]}-{pair[1]}: {counts[:, i].mean():7.1f}  (max={counts[:, i].max():.0f})")

    zero_contact = (counts.sum(axis=1) == 0).sum()
    if zero_contact:
        print(f"\n[WARN] {zero_contact} structures have all-zero contact counts")

    # ── Ridge cross-validation ────────────────────────────────────────────────

    n_folds = min(args.cv_folds, len(ys))
    r2_scores = cross_val_score(Ridge(), counts, ys, cv=n_folds, scoring="r2")
    print(f"\n── Ridge {n_folds}-fold CV ──")
    print(f"  R²: {r2_scores.mean():.3f}  folds={r2_scores.round(3).tolist()}")
    if r2_scores.mean() < 0.2:
        print("  [WARN] R² < 0.2 — contact features have weak signal on this dataset")
    elif r2_scores.mean() > 0.4:
        print("  [OK] R² > 0.4 — contact features carry meaningful signal")

    # ── PRODIGY-LIG formula ───────────────────────────────────────────────────

    preds = predict_dg_batch(counts)
    r, pval = pearsonr(preds, ys)
    rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))
    mean_pred_rmse = float(np.sqrt(np.mean((np.full_like(ys, ys.mean()) - ys) ** 2)))

    print(f"\n── PRODIGY-LIG formula (ΔG = 0.0355·NN − 0.1278·XX − 0.0072·CN − 5.192) ──")
    print(f"  Pearson R : {r:.3f}  (p={pval:.2e})")
    print(f"  RMSE      : {rmse:.3f} kcal/mol")
    print(f"  Mean-pred RMSE (naive baseline): {mean_pred_rmse:.3f} kcal/mol")

    if r < 0.3:
        print("  [WARN] R < 0.3 — PRODIGY-LIG formula transfers poorly to this dataset")
    elif r >= 0.5:
        print("  [OK] R ≥ 0.5 — reasonable transfer from PRODIGY-LIG formula")

    # ── Outliers ──────────────────────────────────────────────────────────────

    residuals = preds - ys
    worst_idx = np.argsort(np.abs(residuals))[-5:][::-1]
    print("\n── Top-5 outliers (|predicted − actual|) ──")
    for i in worst_idx:
        print(
            f"  {pdb_ids[i]:6s}  pred={preds[i]:6.2f}  actual={ys[i]:6.2f}  "
            f"err={residuals[i]:+.2f}  NN={counts[i,1]:.0f}  XX={counts[i,3]:.0f}  CN={counts[i,4]:.0f}"
        )


if __name__ == "__main__":
    main()
