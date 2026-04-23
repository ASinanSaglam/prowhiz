"""Featurize raw mmCIF files into PyG Data objects.

Usage:
    python scripts/prepare_dataset.py \\
        --raw data/raw/ \\
        --labels data/external/labels.csv \\
        --out data/processed/ \\
        [--cutoff 10.5] \\
        [--buffer 2.0] \\
        [--workers 4] \\
        [--no-rcsb-query]

Saves one `{PDB_ID}.pt` file per structure in the output directory.
Failed structures are logged to `data/processed/failed.txt`.
Ligand selection audit is written to `data/processed/ligand_selections.csv`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import torch

from prowhiz.data.cif_parser import LigandSelectionInfo, parse_cif
from prowhiz.data.contacts import compute_contacts
from prowhiz.data.featurizer import FeaturizerConfig
from prowhiz.data.graph_builder import build_graph

logger = logging.getLogger(__name__)

_SELECTION_CSV_FIELDS = [
    "pdb_id",
    "selected_comp",
    "entity_name",
    "n_heavy_atoms",
    "n_instances",
    "n_candidates",
    "selection_method",
    "had_ambiguity",
    "rcsb_suggested",
    "all_candidates",
]


def _selection_row(pdb_id: str, info: LigandSelectionInfo) -> dict[str, object]:
    return {
        "pdb_id": pdb_id,
        "selected_comp": info.selected.comp_id,
        "entity_name": info.selected.entity_name,
        "n_heavy_atoms": info.selected.n_heavy_atoms,
        "n_instances": info.selected.n_instances,
        "n_candidates": len(info.all_candidates),
        "selection_method": info.selection_method,
        "had_ambiguity": info.had_ambiguity,
        "rcsb_suggested": info.rcsb_suggested_comp or "",
        "all_candidates": "|".join(c.comp_id for c in info.all_candidates),
    }


def _process_one(
    args: tuple[str, Path, Path, float, float, float, bool, int, str | None],
) -> tuple[str, bool, str, dict[str, object] | None]:
    """Process a single CIF file.

    Returns:
        (pdb_id, success, error_msg, selection_row_or_None)
    """
    pdb_id, cif_path, out_dir, cutoff, buffer, dg, query_rcsb, max_ligand_atoms, force_comp_id = args
    out_path = out_dir / f"{pdb_id}.pt"
    if out_path.exists():
        # Load existing graph to recover selection info if available
        try:
            graph = torch.load(str(out_path), weights_only=False)
            sel_info = getattr(graph, "_selection_info", None)
            return pdb_id, True, "", sel_info
        except Exception:
            return pdb_id, True, "", None

    try:
        struct = parse_cif(cif_path, pdb_id=pdb_id, query_rcsb=query_rcsb, force_comp_id=force_comp_id)
        n_lig_heavy = len(struct.ligand_atoms)
        if n_lig_heavy > max_ligand_atoms and force_comp_id is None:
            return (
                pdb_id,
                False,
                f"ligand too large ({n_lig_heavy} > {max_ligand_atoms} heavy atoms) — "
                f"comp_id={struct.selection_info.selected.comp_id}",
                None,
            )
        contacts = compute_contacts(struct.protein_atoms, struct.ligand_atoms, cutoff=cutoff)
        config = FeaturizerConfig(cutoff=cutoff)
        graph = build_graph(
            protein_atoms=struct.protein_atoms,
            ligand_atoms=struct.ligand_atoms,
            contacts=contacts,
            dg=dg,
            pdb_id=pdb_id,
            config=config,
            contact_zone_cutoff=cutoff,
            contact_zone_buffer=buffer,
        )
        torch.save(graph, str(out_path))
        row = _selection_row(pdb_id, struct.selection_info)
        return pdb_id, True, "", row
    except Exception as exc:
        return pdb_id, False, str(exc), None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Featurize CIF files into PyG graphs")
    parser.add_argument("--raw", required=True, help="Directory of mmCIF files")
    parser.add_argument("--labels", required=True, help="CSV with pdb_id and dG_kcal_mol columns")
    parser.add_argument("--out", required=True, help="Output directory for .pt files")
    parser.add_argument("--cutoff", type=float, default=10.5, help="Contact cutoff (Å)")
    parser.add_argument("--buffer", type=float, default=2.0, help="Contact zone buffer (Å)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--no-rcsb-query",
        action="store_true",
        help="Disable RCSB API query for ligand disambiguation (faster, offline)",
    )
    parser.add_argument(
        "--max-ligand-atoms",
        type=int,
        default=100,
        help="Skip structures whose ligand has more than this many heavy atoms (default 100). "
             "Filters out peptidic/macrocyclic outliers that PRODIGY-LIG features cannot model.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rcsb = not args.no_rcsb_query

    df = pd.read_csv(args.labels)
    if "pdb_id" not in df.columns or "dG_kcal_mol" not in df.columns:
        raise ValueError("labels CSV must have 'pdb_id' and 'dG_kcal_mol' columns")

    has_override_col = "ligand_comp_id" in df.columns

    task_args = []
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        dg = float(row["dG_kcal_mol"])
        force_comp_id: str | None = None
        if has_override_col:
            val = row["ligand_comp_id"]
            if isinstance(val, str) and val.strip():
                force_comp_id = val.strip().upper()
        cif_path = raw_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            cif_path = raw_dir / f"{pdb_id.lower()}.cif"
        if not cif_path.exists():
            logger.warning("CIF not found for %s, skipping", pdb_id)
            continue
        task_args.append((pdb_id, cif_path, out_dir, args.cutoff, args.buffer, dg, query_rcsb, args.max_ligand_atoms, force_comp_id))

    logger.info("Processing %d structures with %d workers", len(task_args), args.workers)

    failed: list[str] = []
    selection_rows: list[dict[str, object]] = []

    def _handle(result: tuple[str, bool, str, dict[str, object] | None]) -> None:
        pdb_id, success, err, sel_row = result
        if success:
            logger.info("OK  %s", pdb_id)
            if sel_row:
                selection_rows.append(sel_row)
        else:
            logger.warning("FAIL %s: %s", pdb_id, err)
            failed.append(f"{pdb_id}: {err}")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for result in pool.imap_unordered(_process_one, task_args):
                _handle(result)
    else:
        for task in task_args:
            _handle(_process_one(task))

    # Write ligand selection audit CSV
    if selection_rows:
        sel_path = out_dir / "ligand_selections.csv"
        with open(sel_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SELECTION_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(selection_rows)
        ambiguous = sum(1 for r in selection_rows if r["had_ambiguity"])
        logger.info(
            "Ligand selections written to %s (%d ambiguous out of %d)",
            sel_path, ambiguous, len(selection_rows),
        )

    logger.info("Done: %d/%d succeeded", len(task_args) - len(failed), len(task_args))
    if failed:
        fail_path = out_dir / "failed.txt"
        fail_path.write_text("\n".join(failed))
        logger.warning("%d failures written to %s", len(failed), fail_path)


if __name__ == "__main__":
    main()
