"""Predict binding free energy (dG) for a single mmCIF file.

Usage:
    prowhiz-predict --input path/to/complex.cif --checkpoint outputs/checkpoints/best.pt
    python scripts/predict.py --input 1abc.cif [--device cpu]

Output: predicted dG in kcal/mol, printed to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from prowhiz.data.cif_parser import parse_cif
from prowhiz.data.contacts import compute_contacts
from prowhiz.data.featurizer import FeaturizerConfig
from prowhiz.data.graph_builder import build_graph

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF = 10.5
DEFAULT_BUFFER = 2.0


def predict_single(
    cif_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
    cutoff: float = DEFAULT_CUTOFF,
    buffer: float = DEFAULT_BUFFER,
) -> float:
    """Predict dG for a single CIF file.

    Args:
        cif_path: Path to the mmCIF file.
        checkpoint_path: Path to the model checkpoint (.pt).
        device: 'cpu' or 'cuda'.
        cutoff: Contact detection cutoff in Angstroms.
        buffer: Contact zone buffer in Angstroms.

    Returns:
        Predicted dG in kcal/mol.
    """
    cif_path = Path(cif_path)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    from prowhiz.data.featurizer import EDGE_ATTR_DIM, LIGAND_NODE_DIM, PROTEIN_NODE_DIM
    from prowhiz.models.baseline_mlp import BaselineMLP
    from prowhiz.models.gnn import BindingGNN
    from prowhiz.training.trainer import Trainer

    state_dict = ckpt["model_state_dict"]
    if "protein_proj.weight" in state_dict:
        model: torch.nn.Module = BindingGNN(
            input_node_dim=PROTEIN_NODE_DIM,
            ligand_node_dim=LIGAND_NODE_DIM,
            edge_attr_dim=EDGE_ATTR_DIM,
        )
    else:
        model = BaselineMLP()

    model, _ = Trainer.load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    struct = parse_cif(cif_path)
    contacts = compute_contacts(struct.protein_atoms, struct.ligand_atoms, cutoff=cutoff)
    config = FeaturizerConfig(cutoff=cutoff)
    graph = build_graph(
        protein_atoms=struct.protein_atoms,
        ligand_atoms=struct.ligand_atoms,
        contacts=contacts,
        dg=None,
        pdb_id=struct.pdb_id,
        config=config,
        contact_zone_cutoff=cutoff,
        contact_zone_buffer=buffer,
    )

    graph = graph.to(device)

    with torch.no_grad():
        pred: torch.Tensor = model(graph)

    return float(pred.squeeze().item())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Predict dG for a single mmCIF file")
    parser.add_argument("--input", required=True, help="Path to mmCIF file")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    dg = predict_single(
        cif_path=args.input,
        checkpoint_path=args.checkpoint,
        device=device,
        cutoff=args.cutoff,
        buffer=args.buffer,
    )

    print(f"Predicted dG: {dg:.3f} kcal/mol")


if __name__ == "__main__":
    main()
