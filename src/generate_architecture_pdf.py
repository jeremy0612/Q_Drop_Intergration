"""
Generate a PDF diagram for the full Q-Drop-Integration training architecture.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, xy, w, h, title, body, fc="#F7F7F7", ec="#333333"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + 0.015, y + h - 0.09, body, ha="left", va="top", fontsize=9, family="monospace")


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#222222"),
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Q-Drop-Integration: Full Training Architecture (MUTAG + PROTEINS)",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )

    add_box(
        ax, (0.03, 0.73), 0.22, 0.2,
        "1) Dataset Input",
        "src/data/load_mutag.py\nsrc/data/load_proteins.py\n\nHuggingFace graphs-datasets\n- MUTAG\n- PROTEINS",
        fc="#EAF3FF",
    )
    add_box(
        ax, (0.30, 0.73), 0.22, 0.2,
        "2) Split Protocol",
        "StratifiedKFold(n_splits)\n\nPer fold:\n- train_graphs\n- test_graphs",
        fc="#FFF5E6",
    )
    add_box(
        ax, (0.57, 0.73), 0.38, 0.2,
        "3) Model: QGCN (HQGC-style quantum path)",
        "src/models/Quantum_GCN.py\n  -> QGCNConv layers\nsrc/models/GCNConv_Layers/QGCNConv.py\n  -> quantum_net(...)\nsrc/models/QNN_Node_Embedding.py\n  -> AngleEmbedding + BasicEntanglerLayers",
        fc="#EEFCEA",
    )

    add_box(
        ax, (0.03, 0.43), 0.28, 0.23,
        "4) Core Training Loop",
        "train_quantum_models.py\n\nfor epoch:\n  forward -> BCEWithLogitsLoss\n  backward -> gradient clipping\n  optimizer.step (AdamW)\n  scheduler.step (OneCycleLR opt.)\n  validation + early stopping",
        fc="#F4EEFF",
    )
    add_box(
        ax, (0.35, 0.43), 0.28, 0.23,
        "5) Q-Drop Algorithm Hook",
        "src/utils/torch_qdrop.py\nTorchQDropManager.apply()\n\nModes:\n- baseline\n- pruning\n- dropout\n- both\n\nTargets: layers.*.qc.weights",
        fc="#FFEFF3",
    )
    add_box(
        ax, (0.67, 0.43), 0.28, 0.23,
        "6) Q-Drop Details",
        "Pruning:\n- accumulate_window\n- prune_window\n- prune_ratio (+ schedule)\n\nDropout:\n- drop_prob\n- n_drop_wires\n- wire-level gradient masking",
        fc="#FFF9E8",
    )

    add_box(
        ax, (0.03, 0.10), 0.28, 0.24,
        "7) Metrics Per Fold",
        "accuracy, precision, recall,\nf1, roc_auc, pr_auc\n\nStored per fold + curves",
        fc="#EFFFF7",
    )
    add_box(
        ax, (0.35, 0.10), 0.28, 0.24,
        "8) Aggregation",
        "mean/std over folds\nfor each dataset\n(MUTAG, PROTEINS)",
        fc="#F0F8FF",
    )
    add_box(
        ax, (0.67, 0.10), 0.28, 0.24,
        "9) Artifacts",
        "training_results/\nquantum_graph_training_<timestamp>/\n  mutag/metrics.json\n  proteins/metrics.json\n  summary.json\n\n+ dataset_visualizations/*.png",
        fc="#FDF1FF",
    )

    arrow(ax, (0.25, 0.83), (0.30, 0.83))
    arrow(ax, (0.52, 0.83), (0.57, 0.83))
    arrow(ax, (0.57, 0.73), (0.31, 0.54))
    arrow(ax, (0.31, 0.54), (0.35, 0.54))
    arrow(ax, (0.63, 0.54), (0.67, 0.54))
    arrow(ax, (0.49, 0.43), (0.49, 0.34))
    arrow(ax, (0.49, 0.34), (0.17, 0.34))
    arrow(ax, (0.17, 0.34), (0.17, 0.24))
    arrow(ax, (0.31, 0.22), (0.35, 0.22))
    arrow(ax, (0.63, 0.22), (0.67, 0.22))

    out_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_training_architecture.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
