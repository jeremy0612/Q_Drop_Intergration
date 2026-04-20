"""
QGCN Architecture Visualizations for MUTAG and PROTEINS experiments.
Outputs:
  1. architecture_overview.png   - matplotlib full architecture diagram
  2. quantum_circuit.png         - PennyLane circuit drawer
  3. architecture.md             - Mermaid diagram source
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import pennylane as qml

OUT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# 1. PennyLane quantum circuit diagram
# ─────────────────────────────────────────────

def draw_quantum_circuit():
    n_qubits = 8
    n_layers = 1
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    inputs  = np.zeros(n_qubits)
    weights = np.zeros((n_layers, n_qubits))

    fig, ax = qml.draw_mpl(circuit, decimals=None, style="pennylane")(inputs, weights)
    fig.set_size_inches(14, 5)
    fig.suptitle(
        "Quantum Circuit: AngleEmbedding + BasicEntanglerLayers\n"
        "8 qubits · 1 variational layer · PauliZ expectation values → 8-dim node embedding",
        fontsize=11, fontweight='bold', y=1.04
    )
    out = os.path.join(OUT, "quantum_circuit.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] {out}")


# ─────────────────────────────────────────────
# 2. Full architecture matplotlib diagram
# ─────────────────────────────────────────────

COLORS = {
    'input':    '#AED6F1',
    'linear':   '#A9DFBF',
    'quantum':  '#F9E79F',
    'gcn':      '#FAD7A0',
    'pool':     '#D2B4DE',
    'classify': '#F1948A',
    'output':   '#FDFEFE',
    'bg_conv':  '#FDFAF6',
}

def box(ax, x, y, w, h, label, sublabel='', color='white', fontsize=9, radius=0.04):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          linewidth=1.2, edgecolor='#555', facecolor=color, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.08, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=4)
        ax.text(x, y - 0.10, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#444', zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=4)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.4),
                zorder=5)

def brace_label(ax, x, y1, y2, text, side='left'):
    xo = -0.05 if side == 'left' else 0.05
    ax.annotate('', xy=(x + xo, y2), xytext=(x + xo, y1),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=1.0))
    ax.text(x + xo * 3, (y1 + y2) / 2, text, ha='center', va='center',
            fontsize=7.5, color='#666', rotation=90)


def draw_qgcnconv_block(ax, x_center, y_top, dataset_label, in_dim, has_reduction):
    """Draw one QGCNConv block starting at y_top, return y_bottom."""
    bh = 0.55   # box height
    gap = 0.22  # gap between boxes
    w   = 2.8

    # count rows to size outer frame
    n_rows = 4 + (1 if has_reduction else 0)
    block_h = bh * n_rows + gap * (n_rows + 1) + 0.1
    rect = FancyBboxPatch((x_center - w/2 - 0.1, y_top - block_h), w + 0.2, block_h,
                          boxstyle="round,pad=0.04", linewidth=1.2,
                          edgecolor='#CCA800', facecolor='#FFFDE7',
                          linestyle='--', zorder=2)
    ax.add_patch(rect)
    ax.text(x_center, y_top - 0.06, f'QGCNConv  [{dataset_label}]',
            ha='center', va='top', fontsize=10, color='#7D6608',
            fontstyle='italic', fontweight='bold', zorder=4)

    y = y_top - gap - 0.05

    if has_reduction:
        y -= bh/2
        box(ax, x_center, y, w, bh,
            f'Linear({in_dim} → 8)', 'feature dimension reduction', color=COLORS['linear'], fontsize=10)
        arrow(ax, x_center, y - bh/2, x_center, y - bh/2 - gap)
        y -= bh + gap

    y -= bh/2
    box(ax, x_center, y, w, bh,
        'AngleEmbedding', 'encodes x_i ∈ ℝ⁸ as rotation angles → |ψ⟩',
        color=COLORS['quantum'], fontsize=10)
    arrow(ax, x_center, y - bh/2, x_center, y - bh/2 - gap)
    y -= bh + gap

    y -= bh/2
    box(ax, x_center, y, w, bh,
        'BasicEntanglerLayers', 'variational circuit  (8 qubits · depth=1)',
        color=COLORS['quantum'], fontsize=10)
    arrow(ax, x_center, y - bh/2, x_center, y - bh/2 - gap)
    y -= bh + gap

    y -= bh/2
    box(ax, x_center, y, w, bh,
        '⟨Z⟩ Measurement', 'PauliZ expectation → 8-dim real vector',
        color=COLORS['quantum'], fontsize=10)
    arrow(ax, x_center, y - bh/2, x_center, y - bh/2 - gap)
    y -= bh + gap

    y -= bh/2
    box(ax, x_center, y, w, bh,
        'GCN Propagation', 'D⁻½ A D⁻½ · h  +  bias',
        color=COLORS['gcn'], fontsize=10)

    return y - bh/2  # y_bottom


def draw_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(22, 20))
    datasets = [
        ('MUTAG',    7, '187 graphs · 7-dim node features\ntask: mutagenic vs non-mutagenic',  '#2E86C1'),
        ('PROTEINS', 3, '1,113 graphs · 3-dim node features\ntask: enzyme vs non-enzyme',      '#1E8449'),
    ]

    for ax, (dname, in_dim, desc, dcolor) in zip(axes, datasets):
        xlim = 1.7
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-17.5, 1.5)
        ax.axis('off')
        ax.set_title(f'QGCN Architecture — {dname}', fontsize=15, fontweight='bold',
                     color=dcolor, pad=14)

        bh  = 0.55
        w   = 2.8
        gap = 0.25

        # ── Input
        y = 1.0
        box(ax, 0, y, w, bh,
            f'Input Graph  G',
            f'x ∈ ℝ^(N×{in_dim})   ·   A ∈ {{0,1}}^(N×N)',
            color=COLORS['input'], fontsize=11)
        arrow(ax, 0, y - bh/2, 0, y - bh/2 - gap)
        y -= bh + gap

        # ── QGCNConv layer 1
        y = draw_qgcnconv_block(ax, 0, y, f'Layer 1 · {in_dim}→8', in_dim,
                                has_reduction=(in_dim != 8))
        arrow(ax, 0, y, 0, y - gap)
        y -= gap

        # ── Activation 1
        y -= bh/2
        box(ax, 0, y, w, bh,
            'LeakyReLU(0.2)', 'element-wise non-linearity',
            color='#EAFAF1', fontsize=11)
        arrow(ax, 0, y - bh/2, 0, y - bh/2 - gap)
        y -= bh + gap

        # ── QGCNConv layer 2
        y = draw_qgcnconv_block(ax, 0, y, 'Layer 2 · 8→8', 8, has_reduction=False)
        arrow(ax, 0, y, 0, y - gap)
        y -= gap

        # ── Activation 2
        y -= bh/2
        box(ax, 0, y, w, bh,
            'LeakyReLU(0.2)', 'element-wise non-linearity',
            color='#EAFAF1', fontsize=11)
        arrow(ax, 0, y - bh/2, 0, y - bh/2 - gap)
        y -= bh + gap

        # ── Global mean pool
        y -= bh/2
        box(ax, 0, y, w, bh,
            'global_mean_pool',
            'h_G = mean(h_i for i in graph)  →  ℝ⁸',
            color=COLORS['pool'], fontsize=11)
        arrow(ax, 0, y - bh/2, 0, y - bh/2 - gap)
        y -= bh + gap

        # ── Classifier
        y -= bh/2
        box(ax, 0, y, w, bh,
            'Linear(8 → 1)', 'graph-level logit',
            color=COLORS['classify'], fontsize=11)
        arrow(ax, 0, y - bh/2, 0, y - bh/2 - gap)
        y -= bh + gap

        # ── Output
        y -= bh/2
        box(ax, 0, y, w, bh,
            'BCEWithLogitsLoss  →  ŷ',
            'σ(logit) > 0.5  →  class prediction',
            color=COLORS['output'], fontsize=11)

        # ── Dataset note badge
        ax.text(0, y - bh/2 - 0.3, desc,
                ha='center', va='top', fontsize=10,
                color=dcolor, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=dcolor+'18',
                          edgecolor=dcolor, lw=1.0))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'],    edgecolor='#555', label='Input'),
        mpatches.Patch(facecolor=COLORS['linear'],   edgecolor='#555', label='Classical linear'),
        mpatches.Patch(facecolor=COLORS['quantum'],  edgecolor='#555', label='Quantum circuit'),
        mpatches.Patch(facecolor=COLORS['gcn'],      edgecolor='#555', label='GCN aggregation'),
        mpatches.Patch(facecolor=COLORS['pool'],     edgecolor='#555', label='Readout / pooling'),
        mpatches.Patch(facecolor=COLORS['classify'], edgecolor='#555', label='Classifier'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle('QGCN: Quantum Graph Convolutional Network\n'
                 'AngleEmbedding + BasicEntanglerLayers · 8 qubits · 2 conv layers · global mean pool',
                 fontsize=14, fontweight='bold', y=0.995)

    out = os.path.join(OUT, "architecture_overview.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] {out}")


# ─────────────────────────────────────────────
# 3. Mermaid diagram
# ─────────────────────────────────────────────

MERMAID = """\
# QGCN Architecture — Mermaid Diagrams

## MUTAG (7-dim node features)

```mermaid
flowchart TD
    A["Input Graph G\\nx ∈ ℝ^(N×7)  ·  A ∈ {0,1}^(N×N)"] --> B

    subgraph CONV1["QGCNConv Layer 1  (7 → 8)"]
        B["Linear(7→8)\\nfeature reduction"] --> C
        C["AngleEmbedding\\nx_i ∈ ℝ⁸ → |ψ⟩"] --> D
        D["BasicEntanglerLayers\\n8 qubits · depth=1"] --> E
        E["PauliZ Measurement\\n⟨Z⟩_i → 8-dim real vector"] --> F
        F["GCN Propagation\\nD⁻½ A D⁻½ · h + bias"]
    end

    F --> G["LeakyReLU(0.2)"]

    subgraph CONV2["QGCNConv Layer 2  (8 → 8)"]
        G --> H
        H["AngleEmbedding\\nx_i ∈ ℝ⁸ → |ψ⟩"] --> I
        I["BasicEntanglerLayers\\n8 qubits · depth=1"] --> J
        J["PauliZ Measurement\\n⟨Z⟩_i → 8-dim real vector"] --> K
        K["GCN Propagation\\nD⁻½ A D⁻½ · h + bias"]
    end

    K --> L["LeakyReLU(0.2)"]
    L --> M["global_mean_pool\\nh_G = mean(h_i)  →  ℝ⁸"]
    M --> N["Linear(8→1)\\ngraph-level logit"]
    N --> O["BCEWithLogitsLoss\\nσ(logit) > 0.5 → mutagenic?"]

    style CONV1 fill:#fffbe6,stroke:#d4b000
    style CONV2 fill:#fffbe6,stroke:#d4b000
    style O fill:#fde8e8,stroke:#c0392b
```

## PROTEINS (3-dim node features)

```mermaid
flowchart TD
    A["Input Graph G\\nx ∈ ℝ^(N×3)  ·  A ∈ {0,1}^(N×N)"] --> B

    subgraph CONV1["QGCNConv Layer 1  (3 → 8)"]
        B["Linear(3→8)\\nfeature reduction"] --> C
        C["AngleEmbedding\\nx_i ∈ ℝ⁸ → |ψ⟩"] --> D
        D["BasicEntanglerLayers\\n8 qubits · depth=1"] --> E
        E["PauliZ Measurement\\n⟨Z⟩_i → 8-dim real vector"] --> F
        F["GCN Propagation\\nD⁻½ A D⁻½ · h + bias"]
    end

    F --> G["LeakyReLU(0.2)"]

    subgraph CONV2["QGCNConv Layer 2  (8 → 8)"]
        G --> H
        H["AngleEmbedding\\nx_i ∈ ℝ⁸ → |ψ⟩"] --> I
        I["BasicEntanglerLayers\\n8 qubits · depth=1"] --> J
        J["PauliZ Measurement\\n⟨Z⟩_i → 8-dim real vector"] --> K
        K["GCN Propagation\\nD⁻½ A D⁻½ · h + bias"]
    end

    K --> L["LeakyReLU(0.2)"]
    L --> M["global_mean_pool\\nh_G = mean(h_i)  →  ℝ⁸"]
    M --> N["Linear(8→1)\\ngraph-level logit"]
    N --> O["BCEWithLogitsLoss\\nσ(logit) > 0.5 → enzyme?"]

    style CONV1 fill:#fffbe6,stroke:#d4b000
    style CONV2 fill:#fffbe6,stroke:#d4b000
    style O fill:#e8f8e8,stroke:#27ae60
```

## Dataset Comparison Table

| Property | MUTAG | PROTEINS |
|---|---|---|
| Graphs | 187 | 1,113 |
| Node feat dim | 7 | 3 |
| Feature reduction | Linear(7→8) | Linear(3→8) |
| Qubits | 8 | 8 |
| Task | mutagenic? | enzyme? |
| Batch size | 16 | 32 |
"""


def write_mermaid():
    out = os.path.join(OUT, "architecture.md")
    with open(out, 'w') as f:
        f.write(MERMAID)
    print(f"[+] {out}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating QGCN architecture visualizations...")
    draw_quantum_circuit()
    draw_architecture()
    write_mermaid()
    print("\nDone. Outputs:")
    print(f"  {OUT}/quantum_circuit.png      - PennyLane circuit (8Q)")
    print(f"  {OUT}/architecture_overview.png - full model per dataset")
    print(f"  {OUT}/architecture.md           - Mermaid flowchart source")
