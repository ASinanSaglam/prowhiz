"""Download mmCIF files from RCSB PDB for a given list of PDB IDs.

Usage:
    python scripts/download_pdbs.py \\
        --labels data/external/labels.csv \\
        --out data/raw/ \\
        [--workers 10]

The input CSV must have a `pdb_id` column. Already-downloaded files are
skipped. Failed downloads are logged but do not abort the run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
DEFAULT_WORKERS = 10
TIMEOUT_SECONDS = 30


async def _download_one(
    session: object,  # aiohttp.ClientSession
    pdb_id: str,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[str, bool]:
    """Download a single CIF file. Returns (pdb_id, success)."""
    import aiohttp  # type: ignore[import-untyped]

    out_path = out_dir / f"{pdb_id.upper()}.cif"
    if out_path.exists():
        return pdb_id, True

    url = RCSB_CIF_URL.format(pdb_id=pdb_id.lower())
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as resp:  # type: ignore[attr-defined]
                if resp.status != 200:
                    logger.warning("HTTP %d for %s", resp.status, pdb_id)
                    return pdb_id, False
                content = await resp.read()
                out_path.write_bytes(content)
                return pdb_id, True
        except Exception as exc:
            logger.warning("Failed to download %s: %s", pdb_id, exc)
            return pdb_id, False


async def _download_all(pdb_ids: list[str], out_dir: Path, max_workers: int) -> dict[str, bool]:
    """Download all CIF files concurrently."""
    try:
        import aiohttp  # type: ignore[import-untyped]
    except ImportError:
        logger.error("aiohttp is required for async downloads: pip install aiohttp")
        sys.exit(1)

    semaphore = asyncio.Semaphore(max_workers)
    results: dict[str, bool] = {}

    async with aiohttp.ClientSession() as session:
        tasks = [
            _download_one(session, pdb_id, out_dir, semaphore)
            for pdb_id in pdb_ids
        ]
        for coro in asyncio.as_completed(tasks):
            pdb_id, success = await coro
            results[pdb_id] = success
            status = "OK" if success else "FAIL"
            logger.info("[%s] %s", status, pdb_id)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Download mmCIF files from RCSB PDB")
    parser.add_argument("--labels", required=True, help="CSV with pdb_id column")
    parser.add_argument("--out", required=True, help="Output directory for CIF files")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Max concurrent downloads")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels)
    if "pdb_id" not in df.columns:
        logger.error("CSV must have a 'pdb_id' column")
        sys.exit(1)

    pdb_ids = df["pdb_id"].str.upper().unique().tolist()
    logger.info("Downloading %d structures to %s", len(pdb_ids), out_dir)

    results = asyncio.run(_download_all(pdb_ids, out_dir, max_workers=args.workers))
    n_ok = sum(v for v in results.values())
    n_fail = len(results) - n_ok
    logger.info("Done: %d succeeded, %d failed", n_ok, n_fail)

    if n_fail > 0:
        failed = [k for k, v in results.items() if not v]
        fail_path = out_dir / "failed_downloads.txt"
        fail_path.write_text("\n".join(failed))
        logger.warning("Failed IDs written to %s", fail_path)


if __name__ == "__main__":
    main()
