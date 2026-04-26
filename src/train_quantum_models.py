"""
Unified quantum model trainer for graph benchmarks in Q-Drop-Integration.

This script integrates the optimization style from the top-level
train_quantum_models.py into this repository and trains QGCN on:
  - MUTAG
  - PROTEINS

Default protocol follows existing project scripts:
  - 10-fold stratified cross-validation
  - BCEWithLogitsLoss for binary graph classification
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.nn import LeakyReLU
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.load_mutag import load_mutag
from data.load_proteins import load_proteins
from models.Quantum_GCN import QGCN
from utils.torch_qdrop import TorchQDropConfig, TorchQDropManager


@dataclass
class TrainConfig:
    datasets: Sequence[str]
    epochs: int = 100
    lr: float = 5e-3
    weight_decay: float = 1e-3
    batch_size: int = 32
    q_depths: Tuple[int, int] = (1, 1)
    n_folds: int = 10
    early_stop_patience: int = 15
    val_frequency: int = 5
    grad_clip: float = 1.0
    use_scheduler: bool = True
    use_class_weights: bool = True
    algorithm: str = "baseline"
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    qdrop_schedule: bool = True
    drop_prob: float = 0.5
    n_drop_wires: int = 1
    output_dir: str = "training_results"
    seed: int = 42


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weight(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    counter = Counter(labels)
    n_pos = counter.get(1, 1)
    n_neg = counter.get(0, 1)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  Class distribution: {dict(counter)}")
    print(f"  Positive class weight: {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float, device=device)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    y_prob_np = np.asarray(y_prob).reshape(-1)

    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_prob_np))
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_np, y_prob_np))
    except ValueError:
        metrics["pr_auc"] = 0.0
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer = None,
    scheduler: OneCycleLR = None,
    grad_clip: float = 1.0,
    qdrop_manager: TorchQDropManager = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)

            target = batch.y.float()
            loss = criterion(logits, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if qdrop_manager is not None:
                    qdrop_manager.apply()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            total_loss += float(loss.item())
            all_labels.extend(target.long().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    return total_loss / max(len(loader), 1), compute_metrics(all_labels, all_preds, all_probs)


def train_fold(
    train_graphs: Sequence,
    test_graphs: Sequence,
    config: TrainConfig,
    device: torch.device,
    dataset_name: str,
    fold_idx: int,
) -> Dict:
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

    model = QGCN(
        input_dims=train_graphs[0].x.size(1),
        q_depths=list(config.q_depths),
        output_dims=1,
        activ_fn=LeakyReLU(0.2),
        readout=False,
    ).to(device)

    pos_weight = None
    if config.use_class_weights:
        labels = [int(g.y.item()) for g in train_graphs]
        pos_weight = compute_class_weight(labels, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = None
    if config.use_scheduler:
        total_steps = max(len(train_loader) * config.epochs, 1)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr * 5.0,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=50.0,
        )

    qdrop_manager = TorchQDropManager(
        model=model,
        config=TorchQDropConfig(
            algorithm=config.algorithm,
            accumulate_window=config.accumulate_window,
            prune_window=config.prune_window,
            prune_ratio=config.prune_ratio,
            schedule=config.qdrop_schedule,
            drop_prob=config.drop_prob,
            n_drop_wires=config.n_drop_wires,
        ),
    )
    if config.algorithm != "baseline":
        n_qparams = len(qdrop_manager.quantum_params)
        print(f"  Q-Drop mode: {config.algorithm} | quantum tensors: {n_qparams}")

    stopper = EarlyStopping(config.early_stop_patience)
    train_curve = []
    val_curve = []

    print(f"  Fold {fold_idx + 1}: training...")
    for epoch in tqdm(range(1, config.epochs + 1), leave=False, desc=f"{dataset_name}-F{fold_idx+1}"):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=config.grad_clip,
            qdrop_manager=qdrop_manager,
        )
        train_curve.append({"epoch": epoch, "loss": train_loss, **train_metrics})

        if epoch % config.val_frequency != 0 and epoch != config.epochs:
            continue

        val_loss, val_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        val_record = {"epoch": epoch, "loss": val_loss, **val_metrics}
        val_curve.append(val_record)

        if stopper.step(val_metrics["accuracy"], model):
            print(f"    Early stopping at epoch {epoch} (best val acc: {stopper.best_score:.4f})")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
    print(
        f"    Fold {fold_idx + 1} test: "
        f"acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )

    return {
        "fold": fold_idx + 1,
        "test_loss": test_loss,
        **test_metrics,
        "train_curve": train_curve,
        "val_curve": val_curve,
    }


def load_dataset_by_name(name: str):
    name_lower = name.lower()
    if name_lower == "mutag":
        return load_mutag()
    if name_lower in ("proteins", "protein"):
        return load_proteins()
    raise ValueError(f"Unsupported dataset: {name}")


def aggregate_fold_results(fold_results: Sequence[Dict]) -> Dict:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in fold_results]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals))
    return out


def train_dataset(dataset_name: str, config: TrainConfig, device: torch.device, base_output: Path) -> Dict:
    print("\n" + "=" * 72)
    print(f"Training QGCN on {dataset_name.upper()}")
    print("=" * 72)

    graphs = load_dataset_by_name(dataset_name)
    labels = [int(g.y.item()) for g in graphs]
    print(f"Loaded {len(graphs)} graphs | Classes: {set(labels)} | Feature dim: {graphs[0].x.size(1)}")

    splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(graphs, labels)):
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        fold_result = train_fold(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            config=config,
            device=device,
            dataset_name=dataset_name.upper(),
            fold_idx=fold_idx,
        )
        fold_results.append(fold_result)

    metrics = aggregate_fold_results(fold_results)
    print(
        f"{dataset_name.upper()} results: "
        f"acc={metrics['mean_accuracy']:.4f}±{metrics['std_accuracy']:.4f}, "
        f"f1={metrics['mean_f1']:.4f}±{metrics['std_f1']:.4f}"
    )

    dataset_dir = base_output / dataset_name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name.upper(),
        "config": {
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "q_depths": list(config.q_depths),
            "n_folds": config.n_folds,
            "early_stop_patience": config.early_stop_patience,
            "val_frequency": config.val_frequency,
            "grad_clip": config.grad_clip,
            "use_scheduler": config.use_scheduler,
            "use_class_weights": config.use_class_weights,
            "seed": config.seed,
        },
        "summary": metrics,
        "folds": fold_results,
    }

    out_path = dataset_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved metrics to: {out_path}")
    return result_payload


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train quantum models on MUTAG/PROTEINS")
    parser.add_argument("--datasets", nargs="+", default=["mutag", "proteins"], help="Datasets to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-depths", nargs="+", type=int, default=[1, 1])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--val-frequency", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--disable-scheduler", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="baseline",
        choices=["baseline", "pruning", "dropout", "both"],
        help="Q-Drop algorithm mode for HQGC quantum weights",
    )
    parser.add_argument("--accumulate-window", type=int, default=10)
    parser.add_argument("--prune-window", type=int, default=8)
    parser.add_argument("--prune-ratio", type=float, default=0.8)
    parser.add_argument("--disable-qdrop-schedule", action="store_true")
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--n-drop-wires", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="training_results")
    args = parser.parse_args()

    return TrainConfig(
        datasets=args.datasets,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        q_depths=tuple(args.q_depths),
        n_folds=args.folds,
        early_stop_patience=args.early_stop_patience,
        val_frequency=args.val_frequency,
        grad_clip=args.grad_clip,
        use_scheduler=not args.disable_scheduler,
        use_class_weights=not args.disable_class_weights,
        algorithm=args.algorithm,
        accumulate_window=args.accumulate_window,
        prune_window=args.prune_window,
        prune_ratio=args.prune_ratio,
        qdrop_schedule=not args.disable_qdrop_schedule,
        drop_prob=args.drop_prob,
        n_drop_wires=args.n_drop_wires,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(config.output_dir) / f"quantum_graph_training_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    print("Unified quantum training started")
    print(f"Device: {device}")
    print(f"Datasets: {[d.upper() for d in config.datasets]}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Output: {base_output.resolve()}")

    all_results = {}
    for dataset_name in config.datasets:
        all_results[dataset_name.lower()] = train_dataset(dataset_name, config, device, base_output)

    summary_path = base_output / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved global summary to: {summary_path}")


if __name__ == "__main__":
    main()
