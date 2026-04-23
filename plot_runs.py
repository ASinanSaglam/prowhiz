"""Plot training curves from MLflow runs.

Usage:
    # All runs in the experiment
    python plot_runs.py

    # Specific runs by name (partial match)
    python plot_runs.py --runs mlp_base egnn_light

    # Different metric
    python plot_runs.py --metric val_rmse

    # Save to file instead of showing interactively
    python plot_runs.py --out curves.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import mlflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="prowhiz")
    parser.add_argument("--metric", default="val_pearson_r")
    parser.add_argument("--runs", nargs="*", help="Run name substrings to include (default: all)")
    parser.add_argument("--out", help="Save plot to this path instead of showing")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.db)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"No experiment named '{args.experiment}' in {args.db}")

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
    )

    if args.runs:
        all_runs = [
            r for r in all_runs
            if any(s in (r.info.run_name or "") for s in args.runs)
        ]

    if not all_runs:
        raise SystemExit("No runs found matching criteria.")

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for run in all_runs:
        history = client.get_metric_history(run.info.run_id, args.metric)
        if not history:
            print(f"  [skip] {run.info.run_name} — no history for '{args.metric}'")
            continue

        steps = [m.step for m in history]
        values = [m.value for m in history]
        feat_ver = run.data.params.get("featurizer_version", "")
        label = run.info.run_name or run.info.run_id[:8]
        if feat_ver:
            label = f"{label} [{feat_ver}]"

        ax.plot(steps, values, label=label, linewidth=1.5)
        peak = max(values)
        peak_step = steps[values.index(peak)]
        ax.annotate(
            f"{peak:.3f}",
            xy=(peak_step, peak),
            fontsize=7,
            ha="center",
            va="bottom",
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No runs had metric history for '{args.metric}'.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{args.metric} — {args.experiment}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
