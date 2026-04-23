"""Convert the bundled PRODIGY_LIG_dataset.csv to the expected labels format.

The bundled file has columns: complex, DG_pred, DG_score, DG_noelec, DG_exp
Our pipeline expects:         pdb_id,  dG_kcal_mol

Usage:
    python scripts/convert_labels.py \\
        --input data/PRODIGY_LIG_dataset.csv \\
        --out data/external/labels.csv

The `complex` column (PDB ID, lowercase 4-char) is upper-cased and written as
`pdb_id`. The `DG_exp` column (experimental dG in kcal/mol) is used as the
training target.

Complexes with missing experimental dG (NaN) are dropped with a warning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Convert PRODIGY labels to prowhiz format")
    parser.add_argument(
        "--input",
        default="data/PRODIGY_LIG_dataset.csv",
        help="Path to PRODIGY_LIG_dataset.csv",
    )
    parser.add_argument(
        "--out",
        default="data/external/labels.csv",
        help="Output path for converted labels.csv",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    logger.info("Loaded %d rows from %s", len(df), in_path)

    required = {"complex", "DG_exp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Rename and select
    out_df = pd.DataFrame(
        {
            "pdb_id": df["complex"].str.upper().str.strip(),
            "dG_kcal_mol": df["DG_exp"],
        }
    )

    # Drop rows with missing experimental values
    before = len(out_df)
    out_df = out_df.dropna(subset=["dG_kcal_mol"])
    dropped = before - len(out_df)
    if dropped > 0:
        logger.warning("Dropped %d rows with missing DG_exp", dropped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    logger.info("Written %d complexes to %s", len(out_df), out_path)
    logger.info(
        "dG range: %.2f to %.2f kcal/mol",
        float(out_df["dG_kcal_mol"].min()),
        float(out_df["dG_kcal_mol"].max()),
    )


if __name__ == "__main__":
    main()
