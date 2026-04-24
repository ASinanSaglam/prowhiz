"""Convert LP-PDBBind CSV to the prowhiz labels format.

Keeps only entries with exact Kd measurements (Kd=...), excludes covalent
binders, and converts pKd → dG (kcal/mol) using:

    dG = -RT·ln(10)·pKd  =  -1.364 × pKd  (T=298 K)

Entries whose new_split is NaN are assigned to the train set.

Usage:
    python scripts/convert_lp_pdbbind.py \\
        --input /path/to/LP_PDBBind.csv \\
        --out data/external/lp_pdbbind_labels.csv \\
        [--out-splits data/splits/lp_pdbbind/] \\
        [--temperature 298.0]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# R in kcal/(mol·K)
R_KCAL = 1.9872e-3
import math
LN10 = math.log(10)


def pKd_to_dG(pKd: float, T: float = 298.0) -> float:
    return -R_KCAL * T * LN10 * pKd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Convert LP-PDBBind to prowhiz labels format")
    parser.add_argument(
        "--input",
        default="/home/zhedd/LP-PDBBind/dataset/LP_PDBBind.csv",
        help="Path to LP_PDBBind.csv",
    )
    parser.add_argument(
        "--out",
        default="data/external/lp_pdbbind_labels.csv",
        help="Output labels CSV (pdb_id, dG_kcal_mol)",
    )
    parser.add_argument(
        "--out-splits",
        default=None,
        metavar="DIR",
        help="If given, write train.txt / val.txt / test.txt to this directory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature in K for dG conversion (default 298 K)",
    )
    parser.add_argument(
        "--max-resolution",
        type=float,
        default=2.0,
        help="Drop structures with X-ray resolution worse than this (Å). "
             "NMR/missing resolution entries are kept. Default 2.0 Å.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    logger.info("Loaded %d entries from %s", len(df), args.input)

    # 1. Keep only exact Kd= entries (no ~, <, >, <=)
    clean_kd = df["kd/ki"].str.match(r"^Kd=", na=False)
    df = df[clean_kd]
    logger.info("After Kd= filter: %d entries", len(df))

    # 2. Drop covalent binders
    df = df[~df["covalent"].astype(bool)]
    logger.info("After covalent filter: %d entries", len(df))

    # 3. Drop rows with missing pKd value
    df = df.dropna(subset=["value"])
    logger.info("After dropping missing values: %d entries", len(df))

    # 4. Resolution filter — keep NaN (NMR/unknown) but drop poor X-ray structures
    res = pd.to_numeric(df["resolution"], errors="coerce")
    bad_res = res > args.max_resolution
    df = df[~bad_res]
    logger.info("After resolution filter (<= %.1f Å): %d entries", args.max_resolution, len(df))

    # 5. Convert pKd → dG
    df = df.copy()
    df["dG_kcal_mol"] = df["value"].apply(lambda v: pKd_to_dG(v, args.temperature))

    # 6. Normalise PDB ID — the unnamed index column holds the PDB ID (e.g. "6r8o")
    df["pdb_id"] = df.index.str.upper().str.strip()

    # 7. Assign NaN splits to train
    nan_mask = df["new_split"].isna()
    n_reassigned = int(nan_mask.sum())
    df.loc[nan_mask, "new_split"] = "train"
    if n_reassigned:
        logger.info("Assigned %d NaN-split entries to train", n_reassigned)

    # Summary stats
    logger.info(
        "dG range: %.2f to %.2f kcal/mol (mean %.2f)",
        df["dG_kcal_mol"].min(),
        df["dG_kcal_mol"].max(),
        df["dG_kcal_mol"].mean(),
    )
    logger.info("Split distribution:\n%s", df["new_split"].value_counts().to_string())

    # 8. Write labels CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[["pdb_id", "dG_kcal_mol"]].reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Written %d entries to %s", len(out_df), out_path)

    # 9. Optionally write split files
    if args.out_splits:
        splits_dir = Path(args.out_splits)
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            ids = df.loc[df["new_split"] == split, "pdb_id"].tolist()
            split_path = splits_dir / f"{split}.txt"
            split_path.write_text("\n".join(ids) + "\n")
            logger.info("Written %d IDs to %s", len(ids), split_path)


if __name__ == "__main__":
    main()
