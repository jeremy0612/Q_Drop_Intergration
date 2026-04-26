"""
PyTorch implementation of Q-Drop-style gradient controls for quantum parameters.

Supports:
- Scheduled gradient pruning (accumulate + prune phases)
- Dynamic quantum dropout (wire-level gradient masking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class TorchQDropConfig:
    algorithm: str = "baseline"  # baseline | pruning | dropout | both
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    schedule: bool = True
    drop_prob: float = 0.5
    n_drop_wires: int = 1


class TorchQDropManager:
    """
    Manage Q-Drop masking over quantum gradients in a PyTorch model.

    Quantum parameters are inferred by parameter names containing ".qc.weights".
    """

    def __init__(self, model: torch.nn.Module, config: TorchQDropConfig):
        self.config = config
        self.global_step = 0
        self.accumulate_phase = True
        self.acc_count = config.accumulate_window
        self.prune_count = config.prune_window

        self.quantum_params: Dict[str, torch.nn.Parameter] = {
            name: p for name, p in model.named_parameters() if ".qc.weights" in name
        }
        self.accumulated_grads: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param) for name, param in self.quantum_params.items()
        }
        self.current_prune_ratio = float(config.prune_ratio)

    def _step_schedule(self) -> None:
        if self.accumulate_phase:
            self.acc_count -= 1
            if self.acc_count <= 0:
                self.accumulate_phase = False
                self.acc_count = self.config.accumulate_window
        else:
            self.prune_count -= 1
            if self.prune_count <= 0:
                self.accumulate_phase = True
                self.prune_count = self.config.prune_window

    def _wire_dropout_mask(self, grad: torch.Tensor) -> torch.Tensor:
        # For qlayer weights [n_layers, n_qubits]: drop full qubit columns.
        if grad.ndim != 2:
            return torch.ones_like(grad, dtype=torch.bool)
        _, n_qubits = grad.shape
        n_drop = max(1, min(self.config.n_drop_wires, n_qubits))
        drop_idx = torch.randperm(n_qubits, device=grad.device)[:n_drop]
        keep_mask = torch.ones_like(grad, dtype=torch.bool)
        keep_mask[:, drop_idx] = False
        return keep_mask

    def _pruning_mask(self, grad: torch.Tensor) -> torch.Tensor:
        flat = grad.flatten()
        numel = flat.numel()
        k = max(1, int(self.current_prune_ratio * numel))
        if k >= numel:
            return torch.ones_like(grad, dtype=torch.bool)
        # Keep top-k by absolute magnitude
        topk_idx = torch.topk(flat.abs(), k=k, largest=True).indices
        keep_flat = torch.zeros(numel, dtype=torch.bool, device=grad.device)
        keep_flat[topk_idx] = True
        return keep_flat.view_as(grad)

    def apply(self) -> None:
        if self.config.algorithm == "baseline" or not self.quantum_params:
            return

        self.global_step += 1
        apply_dropout = self.config.algorithm in ("dropout", "both")
        apply_pruning = self.config.algorithm in ("pruning", "both")

        for name, param in self.quantum_params.items():
            if param.grad is None:
                continue
            grad = param.grad

            if apply_pruning:
                if self.accumulate_phase:
                    self.accumulated_grads[name].add_(grad.detach())
                else:
                    pr_mask = self._pruning_mask(self.accumulated_grads[name])
                    grad = torch.where(pr_mask, self.accumulated_grads[name], torch.zeros_like(grad))
                    self.accumulated_grads[name].zero_()

            if apply_dropout and torch.rand(1, device=grad.device).item() < self.config.drop_prob:
                keep_mask = self._wire_dropout_mask(grad)
                grad = torch.where(keep_mask, grad, torch.zeros_like(grad))

            param.grad.copy_(grad)

        if apply_pruning:
            self._step_schedule()
            if self.config.schedule and (self.global_step % 5 == 0):
                self.current_prune_ratio = min(1.0, self.current_prune_ratio * 1.105170918)  # exp(0.1)
