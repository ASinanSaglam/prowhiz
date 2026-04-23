"""prowhiz-train entry point.

Delegates to scripts/train.py where the @hydra.main decorator lives.
Hydra resolves config_path relative to scripts/train.py's location,
so the configs/ directory is found correctly regardless of CWD.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable when invoked as an installed entry point
_scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from train import main  # noqa: E402  (scripts/train.py)

__all__ = ["main"]
