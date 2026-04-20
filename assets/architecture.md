# QGCN Architecture — Mermaid Diagrams

## MUTAG (7-dim node features)

```mermaid
flowchart TD
    A["Input Graph G\nx ∈ ℝ^(N×7)  ·  A ∈ {0,1}^(N×N)"] --> B

    subgraph CONV1["QGCNConv Layer 1  (7 → 8)"]
        B["Linear(7→8)\nfeature reduction"] --> C
        C["AngleEmbedding\nx_i ∈ ℝ⁸ → |ψ⟩"] --> D
        D["BasicEntanglerLayers\n8 qubits · depth=1"] --> E
        E["PauliZ Measurement\n⟨Z⟩_i → 8-dim real vector"] --> F
        F["GCN Propagation\nD⁻½ A D⁻½ · h + bias"]
    end

    F --> G["LeakyReLU(0.2)"]

    subgraph CONV2["QGCNConv Layer 2  (8 → 8)"]
        G --> H
        H["AngleEmbedding\nx_i ∈ ℝ⁸ → |ψ⟩"] --> I
        I["BasicEntanglerLayers\n8 qubits · depth=1"] --> J
        J["PauliZ Measurement\n⟨Z⟩_i → 8-dim real vector"] --> K
        K["GCN Propagation\nD⁻½ A D⁻½ · h + bias"]
    end

    K --> L["LeakyReLU(0.2)"]
    L --> M["global_mean_pool\nh_G = mean(h_i)  →  ℝ⁸"]
    M --> N["Linear(8→1)\ngraph-level logit"]
    N --> O["BCEWithLogitsLoss\nσ(logit) > 0.5 → mutagenic?"]

    style CONV1 fill:#fffbe6,stroke:#d4b000
    style CONV2 fill:#fffbe6,stroke:#d4b000
    style O fill:#fde8e8,stroke:#c0392b
```

## PROTEINS (3-dim node features)

```mermaid
flowchart TD
    A["Input Graph G\nx ∈ ℝ^(N×3)  ·  A ∈ {0,1}^(N×N)"] --> B

    subgraph CONV1["QGCNConv Layer 1  (3 → 8)"]
        B["Linear(3→8)\nfeature reduction"] --> C
        C["AngleEmbedding\nx_i ∈ ℝ⁸ → |ψ⟩"] --> D
        D["BasicEntanglerLayers\n8 qubits · depth=1"] --> E
        E["PauliZ Measurement\n⟨Z⟩_i → 8-dim real vector"] --> F
        F["GCN Propagation\nD⁻½ A D⁻½ · h + bias"]
    end

    F --> G["LeakyReLU(0.2)"]

    subgraph CONV2["QGCNConv Layer 2  (8 → 8)"]
        G --> H
        H["AngleEmbedding\nx_i ∈ ℝ⁸ → |ψ⟩"] --> I
        I["BasicEntanglerLayers\n8 qubits · depth=1"] --> J
        J["PauliZ Measurement\n⟨Z⟩_i → 8-dim real vector"] --> K
        K["GCN Propagation\nD⁻½ A D⁻½ · h + bias"]
    end

    K --> L["LeakyReLU(0.2)"]
    L --> M["global_mean_pool\nh_G = mean(h_i)  →  ℝ⁸"]
    M --> N["Linear(8→1)\ngraph-level logit"]
    N --> O["BCEWithLogitsLoss\nσ(logit) > 0.5 → enzyme?"]

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
