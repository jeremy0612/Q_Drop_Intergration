# Q-Drop Integration: Quantum Machine Learning with Statistical Pruning & Dynamic Dropout

## Overview

This project integrates two leading quantum ML innovations:

1. **Q-Drop Algorithms** (from Q-Drop-Implementation)
   - Scheduled Gradient Pruning: Probabilistic pruning of low-magnitude quantum parameters
   - Dynamic Quantum Dropout: Wire-level dropout for quantum circuits

2. **HQGC Components** (from HQGC project)
   - Quantum Neural Networks using AngleEmbedding + BasicEntanglerLayers
   - Message-passing inspired quantum circuits

## Architecture

The integrated model combines:

```
MNIST Input (4x4 images)
  ↓
Classical Pre-processing (Flatten → Dense)
  ↓
Quantum Circuit (HQGC-style: AngleEmbedding + BasicEntanglerLayers)
  ↓ (with Q-Drop Algorithms applied)
Classical Post-processing (Dense → Output)
  ↓
Binary Classification (class 3 vs class 6)
```

## Project Structure

```
Q_Drop_Intergration/
├── code/
│   ├── models/
│   │   ├── integrated_model.py      # IntegratedQDropHQGCModel class
│   │   └── __init__.py
│   ├── utils/
│   │   ├── pruning.py               # ScheduledGradientPruning
│   │   ├── dropout.py               # QuantumDynamicDropoutManager
│   │   └── __init__.py
│   ├── train_mnist.py               # MNIST training script
│   └── __init__.py
├── docs/
│   └── current_research.md          # Architecture documentation
├── README.md
└── LICENSE
```

## Installation

### Requirements

```bash
pip install tensorflow pennylane numpy matplotlib scikit-learn
```

### Environment Setup

```bash
conda env create -f penny_env.yml  # or create a new conda environment
conda activate Penny2
```

## Quick Start

Run the complete training pipeline:

```bash
cd code
python train_mnist.py
```

Train quantum graph models on MUTAG and PROTEINS (unified script):

```bash
cd Q-Drop-Integration/src
python train_quantum_models.py --datasets mutag proteins
```

Run with Q-Drop modes on HQGC quantum weights:

```bash
python train_quantum_models.py --datasets mutag proteins --algorithm both
```

Run vulnerability QGAT pipeline from this folder:

```bash
cd Q-Drop-Integration
python src/train_vulnerability_qgat.py
```

Generate vulnerability IEEE figures from this folder:

```bash
cd Q-Drop-Integration
python src/plot_vulnerability_ieee.py
```

This will:
1. Load and preprocess MNIST (binary classification: digit 3 vs digit 6)
2. Train with **Scheduled Gradient Pruning** only
3. Train with **Dynamic Quantum Dropout** only
4. Train with **both algorithms combined**
5. Compare accuracies and save plots

## Algorithms

### Scheduled Gradient Pruning

**Two-phase training:**

**Phase 1: Accumulation (10 steps)**
- Sum quantum gradients across multiple steps
- Update all parameters normally

**Phase 2: Pruning (8 steps)**
1. Min-max normalize accumulated gradients
2. Create sampling logits: `log(norm_grad + ε)`
3. Sample `k = prune_ratio × n_params` indices (weighted by magnitude)
4. Only selected parameters receive updates

**Prune Ratio Schedule:**
- Starts at 0.8 → gradually increases to 1.0
- Updates every 5 pruning steps by factor `e^0.1`

### Dynamic Quantum Dropout

**Wire-level dropout:**
- Predefined masks map parameters to qubits
- Each epoch: 50% chance to enable dropout
- Forward pass: Zero measurements on dropped wires
- Backward pass: Zero gradients for dropped parameters

**Supports 1-wire or 2-wire dropout**

## Usage

### Custom Training

```python
from code.models.integrated_model import IntegratedQDropHQGCModel
from tensorflow.keras.optimizers import Adam

# Create model
model = IntegratedQDropHQGCModel(
    n_qubits=4,
    n_layers=2,
    algorithm='pruning',  # 'pruning', 'dropout', or 'both'
    apply_dropout=False
)

# Train
model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
```

### Configuration

Edit `TrainingConfig` in `code/train_mnist.py`:

```python
class TrainingConfig:
    epochs = 20                # Training epochs
    initial_lr = 0.3          # Initial learning rate
    batch_size = 32           # Batch size
    n_qubits = 4              # Number of qubits
    n_layers = 2              # Circuit depth
    algorithm = 'pruning'     # Q-Drop algorithm
    train_samples = 500       # Training set size
```

## Expected Results

Performance on MNIST (digit 3 vs 6):

| Algorithm | Test Accuracy |
|-----------|---------------|
| Pruning Only | ~65-75% |
| Dropout Only | ~60-70% |
| Combined | ~68-78% |

## Key Files

- **`code/utils/pruning.py`** – `ScheduledGradientPruning` class
- **`code/utils/dropout.py`** – `QuantumDynamicDropoutManager` class
- **`code/models/integrated_model.py`** – `IntegratedQDropHQGCModel` class
- **`code/train_mnist.py`** – Main training script

## Citation

1. **Q-Drop (Pruning & Dropout):**
   ```bibtex
   @INPROCEEDINGS{11161668,
     author={Nguyen, Pham Thai Quang and others},
     booktitle={ICC 2025 - IEEE International Conference on Communications},
     title={Q-Drop: Optimizing Quantum Orthogonal Networks with Statistic Pruning and Dynamic Dropout},
     year={2025}
   }
   ```

## Troubleshooting

**CUDA/cuDNN warnings:** Safe to ignore (PennyLane warnings)

**NaN losses:** Gradient sanitization is automatic in `train_step`

**Memory errors:** Reduce `batch_size` or `n_qubits`

**Slow training:** Use GPU (ensure CUDA/cuDNN installed)

## Future Work

- Scale to larger circuits (6-8 qubits)
- Apply to real quantum hardware
- Extend to other datasets
- Hybrid quantum GCN training
- Comparative studies with classical methods

---

**Project Status:** Integration complete • Training pipeline ready • MNIST benchmarks available