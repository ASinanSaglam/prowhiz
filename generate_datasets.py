"""Generate featurized datasets for each feature combination.

Runs prepare_dataset.py for every combination of data-affecting feature flags
and places each output in a versioned subdirectory under data/processed/.

Feature combinations generated:
    base                — baseline (no extra features)
    base+arom           — + is_aromatic on ligand nodes
    base+hbd            — + is_hbd / is_hba on ligand nodes
    base+arom+hbd       — both of the above

Architecture-only flags (use_contact_counts_in_gnn, use_separate_pool) do NOT
require separate datasets — they are controlled at training time via model config.

Usage:
    python generate_datasets.py \\
        --raw data/raw/ \\
        --labels data/external/labels.csv \\
        [--workers 4] \\
        [--cutoff 10.5] \\
        [--buffer 2.0] \\
        [--max-ligand-atoms 100] \\
        [--no-rcsb-query] \\
        [--versions base base+arom]   # subset of versions to generate

Output layout:
    data/processed/base/
    data/processed/base+arom/
    data/processed/base+hbd/
    data/processed/base+arom+hbd/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# All data-affecting feature combinations (flags passed to prepare_dataset.py)
FEATURE_COMBINATIONS: list[dict] = [
    {
        "version": "base",
        "flags": [],
    },
    {
        "version": "base+arom",
        "flags": ["--feat-aromaticity"],
    },
    {
        "version": "base+hbd",
        "flags": ["--feat-hbd-hba"],
    },
    {
        "version": "base+arom+hbd",
        "flags": ["--feat-aromaticity", "--feat-hbd-hba"],
    },
]


def run_version(
    version: str,
    flags: list[str],
    raw: str,
    labels: str,
    workers: int,
    cutoff: float,
    buffer: float,
    max_ligand_atoms: int,
    no_rcsb_query: bool,
    out_base: Path,
    dry_run: bool,
) -> bool:
    """Run prepare_dataset.py for one feature combination. Returns True on success."""
    out_dir = out_base / version
    cmd = [
        sys.executable, "scripts/prepare_dataset.py",
        "--raw", raw,
        "--labels", labels,
        "--out", str(out_dir),
        "--workers", str(workers),
        "--cutoff", str(cutoff),
        "--buffer", str(buffer),
        "--max-ligand-atoms", str(max_ligand_atoms),
        *flags,
    ]
    if no_rcsb_query:
        cmd.append("--no-rcsb-query")

    print(f"\n{'='*60}")
    print(f"Version: {version}")
    print(f"Output:  {out_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[dry-run] Skipping execution.")
        return True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("prepare_dataset.py failed for version '%s' (exit %d)", version, result.returncode)
        return False
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    all_versions = [c["version"] for c in FEATURE_COMBINATIONS]

    parser = argparse.ArgumentParser(description="Generate featurized datasets for all feature combinations")
    parser.add_argument("--raw", required=True, help="Directory of mmCIF files")
    parser.add_argument("--labels", required=True, help="CSV with pdb_id and dG_kcal_mol columns")
    parser.add_argument("--out-base", default="data/processed", help="Base output dir (default: data/processed)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=10.5)
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--max-ligand-atoms", type=int, default=100)
    parser.add_argument("--no-rcsb-query", action="store_true")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=all_versions,
        choices=all_versions,
        help=f"Which versions to generate (default: all). Choices: {all_versions}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    out_base = Path(args.out_base)
    selected = {c["version"]: c for c in FEATURE_COMBINATIONS if c["version"] in args.versions}

    print(f"Generating {len(selected)} dataset version(s): {list(selected.keys())}")
    print(f"Base output directory: {out_base}")

    results: dict[str, bool] = {}
    for combo in FEATURE_COMBINATIONS:
        v = combo["version"]
        if v not in selected:
            continue
        ok = run_version(
            version=v,
            flags=combo["flags"],
            raw=args.raw,
            labels=args.labels,
            workers=args.workers,
            cutoff=args.cutoff,
            buffer=args.buffer,
            max_ligand_atoms=args.max_ligand_atoms,
            no_rcsb_query=args.no_rcsb_query,
            out_base=out_base,
            dry_run=args.dry_run,
        )
        results[v] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for v, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {status:6s}  data/processed/{v}/")
    print(f"{'='*60}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
