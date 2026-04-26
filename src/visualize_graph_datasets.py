"""
Visualize dataset statistics for MUTAG and PROTEINS.

Generates per-dataset figures with:
- class distribution
- node count histogram
- edge count histogram
- graph density histogram
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.load_mutag import load_mutag
from data.load_proteins import load_proteins


def load_by_name(name: str):
    name = name.lower()
    if name == "mutag":
        return load_mutag()
    if name in ("proteins", "protein"):
        return load_proteins()
    raise ValueError(f"Unsupported dataset: {name}")


def summarize_graphs(graphs):
    labels = np.array([int(g.y.item()) for g in graphs], dtype=int)
    num_nodes = np.array([int(g.num_nodes) for g in graphs], dtype=int)
    num_edges = np.array([int(g.edge_index.shape[1]) for g in graphs], dtype=int)
    # Treat as undirected estimate for density denominator.
    max_undirected_edges = np.maximum(num_nodes * (num_nodes - 1) / 2.0, 1.0)
    density = (num_edges / 2.0) / max_undirected_edges

    return {
        "n_graphs": int(len(graphs)),
        "feature_dim": int(graphs[0].x.size(1)) if graphs else 0,
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "nodes_mean": float(num_nodes.mean()),
        "nodes_std": float(num_nodes.std()),
        "edges_mean": float(num_edges.mean()),
        "edges_std": float(num_edges.std()),
        "density_mean": float(density.mean()),
        "density_std": float(density.std()),
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
    }


def plot_dataset(name: str, stats: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{name.upper()} Dataset Visualization", fontsize=14, fontweight="bold")

    # Class distribution
    cls = sorted(int(k) for k in stats["class_counts"].keys())
    cnts = [stats["class_counts"][str(k)] for k in cls]
    axes[0, 0].bar([str(c) for c in cls], cnts, color="#4C72B0", edgecolor="black")
    axes[0, 0].set_title("Class Distribution")
    axes[0, 0].set_xlabel("Class")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis="y", alpha=0.25)

    # Nodes per graph
    axes[0, 1].hist(stats["num_nodes"], bins=30, color="#55A868", edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Nodes per Graph")
    axes[0, 1].set_xlabel("# Nodes")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(alpha=0.25)

    # Edges per graph
    axes[1, 0].hist(stats["num_edges"], bins=30, color="#DD8452", edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Edges per Graph")
    axes[1, 0].set_xlabel("# Directed Edges")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(alpha=0.25)

    # Density
    axes[1, 1].hist(stats["density"], bins=30, color="#C44E52", edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Estimated Graph Density")
    axes[1, 1].set_xlabel("Density")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / f"{name.lower()}_dataset_visualization.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MUTAG/PROTEINS graph datasets")
    parser.add_argument("--datasets", nargs="+", default=["mutag", "proteins"])
    parser.add_argument("--output-dir", default="dataset_visualizations")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in args.datasets:
        graphs = load_by_name(name)
        stats = summarize_graphs(graphs)
        fig_path = plot_dataset(name, stats, out_dir)
        summary[name.lower()] = {
            k: v for k, v in stats.items()
            if k not in ("labels", "num_nodes", "num_edges", "density")
        }
        summary[name.lower()]["figure"] = str(fig_path)
        print(f"[+] {name.upper()}: figure saved to {fig_path}")

    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
