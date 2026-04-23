"""Create stratified train/val/test splits from processed data.

Stratification is done by dG value range (quantile-based bins) to ensure
each split has a representative distribution of binding affinities.

Usage:
    python scripts/create_splits.py \\
        --processed data/processed/ \\
        --out data/splits/ \\
        [--val-frac 0.1] \\
        [--test-frac 0.1] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_dg_values(processed_dir: Path) -> dict[str, float]:
    """Load dG values from all processed .pt files."""
    dg_map: dict[str, float] = {}
    for pt_path in sorted(processed_dir.glob("*.pt")):
        if pt_path.stem.startswith("_"):
            continue  # skip cache files
        try:
            graph = torch.load(str(pt_path), weights_only=False)
            pdb_id: str = getattr(graph, "pdb_id", pt_path.stem)
            dg: float = float(graph.y[0, 0])
            dg_map[pdb_id] = dg
        except Exception as exc:
            logger.warning("Could not load %s: %s", pt_path, exc)
    return dg_map


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--processed", required=True, help="Directory of processed .pt files")
    parser.add_argument("--out", required=True, help="Output directory for split .txt files")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dg_map = _load_dg_values(Path(args.processed))
    if not dg_map:
        raise RuntimeError("No processed .pt files found in " + args.processed)

    pdb_ids = np.array(list(dg_map.keys()))
    dg_values = np.array([dg_map[pid] for pid in pdb_ids])

    rng = np.random.default_rng(args.seed)

    # Stratified split: bin dG into quartiles, split within each bin
    n = len(pdb_ids)
    n_bins = 4
    bin_indices = np.argsort(dg_values)
    bin_labels = np.zeros(n, dtype=int)
    for i, idx in enumerate(bin_indices):
        bin_labels[idx] = (i * n_bins) // n

    test_idx: list[int] = []
    val_idx: list[int] = []
    train_idx: list[int] = []

    for bin_id in range(n_bins):
        bin_mask = np.where(bin_labels == bin_id)[0]
        rng.shuffle(bin_mask)
        n_bin = len(bin_mask)
        n_test = max(1, int(n_bin * args.test_frac))
        n_val = max(1, int(n_bin * args.val_frac))
        test_idx.extend(bin_mask[:n_test].tolist())
        val_idx.extend(bin_mask[n_test:n_test + n_val].tolist())
        train_idx.extend(bin_mask[n_test + n_val:].tolist())

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_ids = pdb_ids[indices]
        out_path = out_dir / f"{split_name}.txt"
        out_path.write_text("\n".join(split_ids))
        logger.info("%s: %d complexes", split_name, len(split_ids))

    logger.info("Splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
