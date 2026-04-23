"""prowhiz-predict entry point — predict dG for a single mmCIF file.

Usage:
    prowhiz-predict --input path/to/complex.cif --ligand ATP --checkpoint best.pt
    python scripts/predict.py --input 1abc.cif --ligand ATP --model-index 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_CUTOFF = 10.5
DEFAULT_BUFFER = 2.0


def _reconstruct_featurizer_config(
    ckpt_cfg: dict[str, Any], cutoff_override: float | None
) -> "FeaturizerConfig":
    from prowhiz.data.featurizer import FeaturizerConfig

    cutoff = cutoff_override if cutoff_override is not None else float(ckpt_cfg.get("cutoff", DEFAULT_CUTOFF))
    version = str(ckpt_cfg.get("featurizer_version", "base"))
    return FeaturizerConfig(
        cutoff=cutoff,
        use_aromaticity="arom" in version,
        use_hbd_hba="hbd" in version,
    )


def _detect_model_type(state_dict: dict[str, Any]) -> str:
    if any("protein_proj" in k or "egnn_layers" in k or "gnn_layers" in k for k in state_dict):
        return "gnn"
    return "mlp"


def _build_model_from_checkpoint(
    ckpt: dict[str, Any],
    feat_config: "FeaturizerConfig",
    device: str,
) -> torch.nn.Module:
    from prowhiz.data.featurizer import EDGE_ATTR_DIM, PROTEIN_NODE_DIM
    from prowhiz.models.baseline_mlp import BaselineMLP
    from prowhiz.models.gnn import BindingGNN

    state_dict = ckpt["model_state_dict"]
    model_type = _detect_model_type(state_dict)

    if model_type == "mlp":
        first_weight_key = next(k for k in state_dict if k.endswith(".weight"))
        input_dim = state_dict[first_weight_key].shape[1]
        model: torch.nn.Module = BaselineMLP(input_dim=input_dim)
    else:
        ckpt_cfg: dict[str, Any] = ckpt.get("featurizer_config", {})
        model = BindingGNN(
            input_node_dim=int(ckpt_cfg.get("protein_node_dim", PROTEIN_NODE_DIM)),
            ligand_node_dim=int(ckpt_cfg.get("ligand_node_dim", feat_config.ligand_node_dim)),
            edge_attr_dim=int(ckpt_cfg.get("edge_attr_dim", EDGE_ATTR_DIM)),
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(
    cif_path: str | Path,
    checkpoint_path: str | Path,
    ligand_comp_id: str,
    protein_chain: str | None = None,
    model_index: int = 0,
    device: str = "cpu",
    cutoff_override: float | None = None,
    buffer: float = DEFAULT_BUFFER,
) -> float:
    """Predict dG for a single CIF file.

    Returns:
        Predicted dG in kcal/mol.
    """
    import gemmi

    from prowhiz.data.cif_parser import parse_cif
    from prowhiz.data.contacts import compute_contacts
    from prowhiz.data.graph_builder import build_graph

    cif_path = Path(cif_path)
    ckpt: dict[str, Any] = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ckpt_cfg: dict[str, Any] = ckpt.get("featurizer_config", {})

    feat_config = _reconstruct_featurizer_config(ckpt_cfg, cutoff_override)
    logger.info(
        "Checkpoint: epoch=%s  val_r=%.4f  featurizer=%s",
        ckpt.get("epoch", "?"),
        float(ckpt.get("val_pearson_r", float("nan"))),
        ckpt_cfg.get("featurizer_version", "unknown"),
    )

    model = _build_model_from_checkpoint(ckpt, feat_config, device)

    _st = gemmi.read_structure(str(cif_path))
    n_models = len(_st)
    if n_models > 1:
        logger.warning(
            "Structure has %d models — using model index %d. "
            "Pass --model-index N to select a different one.",
            n_models,
            model_index,
        )
    del _st

    struct = parse_cif(cif_path, force_comp_id=ligand_comp_id, model_index=model_index)

    n_protein = len(struct.protein_atoms)
    n_ligand = len(struct.ligand_atoms)
    logger.info(
        "Selected: ligand=%s  protein_atoms=%d  ligand_atoms=%d  pdb_id=%s",
        ligand_comp_id,
        n_protein,
        n_ligand,
        struct.pdb_id,
    )

    if n_ligand == 0:
        raise ValueError(
            f"No ligand atoms found for comp_id={ligand_comp_id!r}. "
            "Check that the comp_id matches what is in the CIF file."
        )
    if n_protein == 0:
        raise ValueError("No protein atoms found. Check the CIF file.")

    contacts = compute_contacts(struct.protein_atoms, struct.ligand_atoms, cutoff=feat_config.cutoff)
    logger.info(
        "Contacts: total=%d  types=%s",
        contacts.num_contacts,
        contacts.contact_counts.astype(int).tolist(),
    )

    graph = build_graph(
        protein_atoms=struct.protein_atoms,
        ligand_atoms=struct.ligand_atoms,
        contacts=contacts,
        dg=None,
        pdb_id=struct.pdb_id,
        config=feat_config,
        contact_zone_cutoff=feat_config.cutoff,
        contact_zone_buffer=buffer,
    )
    graph = graph.to(device)

    with torch.no_grad():
        pred: torch.Tensor = model(graph)

    return float(pred.squeeze().item())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Predict dG for a single mmCIF structure")
    parser.add_argument("--input", required=True, help="Path to mmCIF file")
    parser.add_argument("--ligand", required=True, metavar="COMP_ID",
                        help="Ligand comp_id (e.g. ATP). Bypasses size/exclusion filters.")
    parser.add_argument("--protein", default=None, metavar="CHAIN",
                        help="Protein chain ID (optional)")
    parser.add_argument("--model-index", type=int, default=0, metavar="N",
                        help="Model index for multi-model structures (default 0)")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="Override contact cutoff in Å (default: from checkpoint)")
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    try:
        dg = predict(
            cif_path=args.input,
            checkpoint_path=args.checkpoint,
            ligand_comp_id=args.ligand,
            protein_chain=args.protein,
            model_index=args.model_index,
            device=device,
            cutoff_override=args.cutoff,
            buffer=args.buffer,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\nPredicted dG: {dg:.3f} kcal/mol")


if __name__ == "__main__":
    main()
