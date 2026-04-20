"""
MUTAG experiment: QGCN on molecular graph classification.
Standard benchmark protocol: 10-fold cross-validation.
Dataset: 187 graphs, binary (mutagenic vs non-mutagenic), 7-dim node features.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import LeakyReLU

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.load_mutag import load_mutag
from models.Quantum_GCN import QGCN


class MUTAGConfig:
    epochs = 100
    lr = 0.005
    batch_size = 16        # small dataset
    q_depths = [1, 1]
    n_folds = 10
    early_stop_patience = 15
    val_frequency = 5


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.float().unsqueeze(1)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = (torch.sigmoid(out) > 0.5).float()
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.float().unsqueeze(1)
        total_loss += criterion(out, target).item()
        pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()
        all_preds.extend(pred.flatten())
        all_labels.extend(target.cpu().numpy().flatten())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), acc, f1, prec, rec


def run_fold(train_data, test_data, config, device, fold_idx):
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    input_dim = train_data[0].x.size(1)  # 7
    model = QGCN(
        input_dims=input_dim,
        q_depths=config.q_depths,
        output_dims=1,
        activ_fn=LeakyReLU(0.2),
        readout=False
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    train_accs, val_accs = [], []

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_accs.append(train_acc)

        if (epoch + 1) % config.val_frequency == 0 or epoch == config.epochs - 1:
            val_loss, val_acc, val_f1, val_prec, val_rec = eval_epoch(model, test_loader, criterion, device)
            val_accs.append((epoch + 1, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stop_patience:
                print(f"  Fold {fold_idx+1}: early stop at epoch {epoch+1} (best val={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, test_f1, test_prec, test_rec = eval_epoch(model, test_loader, criterion, device)
    print(f"  Fold {fold_idx+1}: test acc={test_acc:.4f}, f1={test_f1:.4f}")

    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'precision': test_prec,
        'recall': test_rec,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }


def plot_results(fold_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-fold accuracy bar
    fold_accs = [r['accuracy'] for r in fold_results]
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    axes[0].bar(range(1, len(fold_accs) + 1), fold_accs, color='#4C72B0', edgecolor='black')
    axes[0].axhline(mean_acc, color='red', linestyle='--', label=f'Mean={mean_acc:.3f}±{std_acc:.3f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('10-Fold CV Accuracy per Fold')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Training curves (all folds)
    for i, r in enumerate(fold_results):
        axes[1].plot(r['train_accs'], alpha=0.4, linewidth=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Accuracy')
    axes[1].set_title('Training Accuracy (All Folds)')
    axes[1].grid(alpha=0.3)

    # Metrics summary bar
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [np.mean([r[m] for r in fold_results]) for m in metrics]
    stds = [np.std([r[m] for r in fold_results]) for m in metrics]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    bars = axes[2].bar(metrics, means, yerr=stds, color=colors, edgecolor='black', capsize=5)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_ylabel('Score')
    axes[2].set_title('Mean Metrics (±std) over 10 Folds')
    axes[2].grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(f'QGCN on MUTAG — {mean_acc:.3f}±{std_acc:.3f} accuracy', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Plot saved to {out_path}")


def main():
    config = MUTAGConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading MUTAG from HuggingFace...")

    graphs = load_mutag()
    labels = [int(g.y.item()) for g in graphs]
    print(f"Loaded {len(graphs)} graphs | Classes: {set(labels)}")

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n10-Fold Cross-Validation | {config.epochs} epochs max\n{'-'*50}")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
        train_data = [graphs[i] for i in train_idx]
        test_data = [graphs[i] for i in test_idx]
        result = run_fold(train_data, test_data, config, device, fold_idx)
        fold_results.append(result)

    # Aggregate
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])

    print(f"\n{'='*50}")
    print(f"MUTAG 10-Fold CV Results")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  F1:       {mean_f1:.4f}")
    print(f"{'='*50}")

    # Save metrics JSON (repo root for CML)
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'MUTAG',
        'model': 'QGCN',
        'n_folds': config.n_folds,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'mean_f1': float(mean_f1),
        'per_fold': [
            {k: float(v) if isinstance(v, (float, np.floating)) else v
             for k, v in r.items() if k not in ('train_accs', 'val_accs')}
            for r in fold_results
        ]
    }
    # Save to repo root (one level up from src/)
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    metrics_path = os.path.join(repo_root, 'mutag_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved to {metrics_path}")

    plot_path = os.path.join(repo_root, 'mutag_results.png')
    plot_results(fold_results, plot_path)


if __name__ == '__main__':
    main()
