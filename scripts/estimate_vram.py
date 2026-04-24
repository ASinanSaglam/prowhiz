"""Estimate peak VRAM for one training step via a real forward+backward pass.

Usage:
    python scripts/estimate_vram.py --processed data/processed/base/ --batch-size 32
    python scripts/estimate_vram.py --processed data/processed/base/ --batch-size 32 --model egnn
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch_geometric.data import Batch

from prowhiz.models.gnn import BindingGNN
from prowhiz.models.baseline_mlp import BaselineMLP


def fmt(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate peak VRAM for one training step")
    parser.add_argument("--processed", default="data/processed/base/", help="Directory of .pt graph files")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="egnn", choices=["egnn", "mlp_baseline"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of batches to measure (takes max)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (no VRAM estimate possible)")
        return

    # Load graphs
    pt_files = sorted(Path(args.processed).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {args.processed}")

    graphs = []
    for p in pt_files:
        try:
            g = torch.load(str(p), weights_only=False)
            if hasattr(g, "x"):  # skip cache files / non-graph objects
                graphs.append(g)
        except Exception:
            pass
    print(f"Loaded {len(graphs)} graphs from {args.processed}")

    sample = graphs[0]
    protein_dim = int(sample.x.shape[1])
    ligand_dim  = int(sample.get("ligand_node_dim", protein_dim))
    edge_dim    = int(sample.edge_attr.shape[1]) if sample.edge_attr is not None else 4

    n_nodes = [int(g.x.shape[0]) for g in graphs]
    n_edges = [int(g.edge_index.shape[1]) for g in graphs]
    print(f"Nodes/graph  — min {min(n_nodes)}  max {max(n_nodes)}  mean {sum(n_nodes)/len(n_nodes):.0f}")
    print(f"Edges/graph  — min {min(n_edges)}  max {max(n_edges)}  mean {sum(n_edges)/len(n_edges):.0f}")

    if args.model == "egnn":
        model = BindingGNN(
            input_node_dim=protein_dim,
            ligand_node_dim=ligand_dim,
            edge_attr_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.1,
            update_coords=True,
            head_hidden_dims=[args.hidden_dim, args.hidden_dim // 2],
            use_batch_norm=True,
        )
    else:
        mlp_dim = int(getattr(sample, "contact_counts", torch.zeros(10)).shape[0])
        model = BaselineMLP(input_dim=mlp_dim, hidden_dims=[128, 64], dropout=0.1)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params: {n_params:,}  hidden_dim: {args.hidden_dim}  layers: {args.num_layers}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.HuberLoss()
    model.train()

    peak_bytes: list[int] = []

    for trial in range(args.n_trials):
        batch_graphs = random.sample(graphs, min(args.batch_size, len(graphs)))
        batch = Batch.from_data_list(batch_graphs).to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            peak_bytes.append(peak)
            print(f"  Trial {trial + 1}: peak {fmt(peak)}")

    if peak_bytes:
        p = max(peak_bytes)
        weights_bytes = n_params * 4 * 3  # params + Adam m + Adam v
        print(f"\nPeak VRAM (batch_size={args.batch_size}): {fmt(p)}")
        print(f"  weights + optimizer state : {fmt(weights_bytes)}")
        print(f"  activations (approx)      : {fmt(max(0, p - weights_bytes))}")


if __name__ == "__main__":
    main()
