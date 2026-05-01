"""Research-faithful Torch runtime for Q-Drop gradient pruning and dropout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .contract import DiscoveredQuantumLayer, QuantumDropoutState, QuantumParameterMetadata, discover_quantum_layers

SUPPORTED_QDROP_ALGORITHMS = {"baseline", "pruning", "dropout", "both"}


@dataclass
class TorchQDropConfig:
    algorithm: str = "baseline"  # baseline | pruning | dropout | both
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    schedule: bool = True
    dropout_prob: float = 0.5
    n_drop_wires: int = 1
    enable_forward_mask: bool = True
    sanitize_quantum_gradients: bool = True
    sanitize_quantum_parameters: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_QDROP_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. "
                f"Expected one of {sorted(SUPPORTED_QDROP_ALGORITHMS)}."
            )


class TorchQDropManager:
    """
    Manage Q-Drop over discovered quantum layers inside a Torch model.

    The runtime mirrors the TensorFlow research implementation closely:
    - epoch-level dropout decisions
    - accumulation/pruning phase switching
    - pruning via categorical sampling over normalized accumulated gradients
    """

    def __init__(self, model: torch.nn.Module, config: TorchQDropConfig):
        self.config = config
        self.quantum_layers: List[DiscoveredQuantumLayer] = discover_quantum_layers(model)
        self.quantum_parameters: Dict[str, QuantumParameterMetadata] = {}
        self.accumulated_grads: Dict[str, torch.Tensor] = {}
        self.active_dropout_states: Dict[str, QuantumDropoutState] = {}

        for layer in self.quantum_layers:
            for spec in layer.parameter_specs:
                parameter_key = self._parameter_key(layer.layer_name, spec.parameter_name)
                self.quantum_parameters[parameter_key] = spec
                self.accumulated_grads[parameter_key] = torch.zeros_like(spec.parameter.data)

        self.accumulate_phase = True
        self.accumulate_count = config.accumulate_window
        self.prune_count = config.prune_window
        self.current_prune_ratio = float(config.prune_ratio)
        self.pruning_step_count = 0
        self.current_epoch = 0
        self.dropout_enabled = False

    @property
    def quantum_param_count(self) -> int:
        """Return how many quantum tensors are currently managed."""
        return len(self.quantum_parameters)

    @staticmethod
    def _parameter_key(layer_name: str, parameter_name: str) -> str:
        return f"{layer_name}.{parameter_name}" if layer_name else parameter_name

    def clear_dropout(self) -> None:
        """Disable forward masking on every discovered quantum layer."""
        self.dropout_enabled = False
        self.active_dropout_states = {}
        for layer in self.quantum_layers:
            layer.layer.set_qdrop_dropout_state(None)

    def start_epoch(self, epoch_index: int) -> None:
        """Sample the dropout state for the upcoming training epoch."""
        self.current_epoch = epoch_index
        self.clear_dropout()

        if self.config.algorithm not in {"dropout", "both"}:
            return
        if epoch_index <= 1:
            return

        self.dropout_enabled = bool(torch.rand(1).item() < self.config.dropout_prob)
        if not self.dropout_enabled:
            return

        for layer in self.quantum_layers:
            num_wires = max((spec.num_wires for spec in layer.parameter_specs), default=0)
            if num_wires <= 0:
                continue

            n_drop = max(1, min(self.config.n_drop_wires, num_wires))
            dropped_wires = tuple(sorted(torch.randperm(num_wires)[:n_drop].tolist()))
            dropout_state = QuantumDropoutState(enabled=True, dropped_wires=dropped_wires)
            self.active_dropout_states[layer.layer_name] = dropout_state

            if self.config.enable_forward_mask:
                layer.layer.set_qdrop_dropout_state(dropout_state)

    def _sanitize_quantum_gradients(self) -> None:
        for spec in self.quantum_parameters.values():
            if spec.parameter.grad is None:
                continue
            spec.parameter.grad = torch.nan_to_num(
                spec.parameter.grad,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    def _wire_mask_for_parameter(
        self,
        parameter_spec: QuantumParameterMetadata,
        dropout_state: QuantumDropoutState,
    ) -> torch.Tensor:
        mask = torch.zeros_like(parameter_spec.parameter.data, dtype=torch.bool)
        for wire_index in dropout_state.dropped_wires:
            wire_mask = parameter_spec.wire_masks.get(wire_index)
            if wire_mask is None:
                continue
            mask = torch.logical_or(mask, wire_mask.to(device=mask.device))
        return mask

    def _apply_dropout_to_gradients(self) -> None:
        if not self.dropout_enabled or self.config.algorithm not in {"dropout", "both"}:
            return

        for layer in self.quantum_layers:
            dropout_state = self.active_dropout_states.get(layer.layer_name)
            if dropout_state is None or not dropout_state.enabled:
                continue

            for parameter_spec in layer.parameter_specs:
                if parameter_spec.parameter.grad is None:
                    continue

                drop_mask = self._wire_mask_for_parameter(parameter_spec, dropout_state)
                parameter_spec.parameter.grad = torch.where(
                    drop_mask,
                    torch.zeros_like(parameter_spec.parameter.grad),
                    parameter_spec.parameter.grad,
                )

    def _update_phase(self) -> None:
        if self.accumulate_count == 0:
            self.accumulate_count = self.config.accumulate_window
            self.accumulate_phase = False
        elif self.prune_count == 0:
            self.prune_count = self.config.prune_window
            self.accumulate_phase = True

    def _build_pruning_mask(self, accumulated_grad: torch.Tensor) -> Tuple[torch.Tensor, int]:
        flat_grad = accumulated_grad.reshape(-1)
        num_params = flat_grad.numel()
        if num_params == 0:
            return torch.zeros_like(accumulated_grad, dtype=torch.bool), 0

        epsilon = torch.tensor(1e-8, device=flat_grad.device, dtype=flat_grad.dtype)
        grad_min = flat_grad.min()
        grad_max = flat_grad.max()
        normalized_grad = (flat_grad - grad_min) / (grad_max - grad_min + epsilon)
        logits = torch.log(normalized_grad + epsilon)

        num_samples = max(1, int(self.current_prune_ratio * num_params))
        distribution = torch.distributions.Categorical(logits=logits)
        sampled_indices = distribution.sample((num_samples,))

        keep_mask = torch.zeros(num_params, dtype=torch.bool, device=flat_grad.device)
        keep_mask[sampled_indices] = True
        return keep_mask.view_as(accumulated_grad), num_samples

    def _apply_pruning_phase(self) -> None:
        self.prune_count -= 1

        for parameter_key, parameter_spec in self.quantum_parameters.items():
            pruning_mask, _ = self._build_pruning_mask(self.accumulated_grads[parameter_key])
            parameter_spec.parameter.grad = torch.where(
                pruning_mask,
                self.accumulated_grads[parameter_key],
                torch.zeros_like(self.accumulated_grads[parameter_key]),
            )
            self.accumulated_grads[parameter_key].zero_()

        self.pruning_step_count += 1
        if self.config.schedule and self.pruning_step_count % 5 == 0:
            self.current_prune_ratio = min(1.0, self.current_prune_ratio * math.exp(0.1))

    def _apply_accumulation_phase(self) -> None:
        self.accumulate_count -= 1

        for parameter_key, parameter_spec in self.quantum_parameters.items():
            if parameter_spec.parameter.grad is None:
                continue
            self.accumulated_grads[parameter_key].add_(parameter_spec.parameter.grad.detach())

    def apply(self) -> None:
        """Apply gradient controls after ``loss.backward()`` and before ``optimizer.step()``."""
        if not self.quantum_parameters:
            return

        if self.config.sanitize_quantum_gradients:
            self._sanitize_quantum_gradients()

        self._apply_dropout_to_gradients()

        if self.config.algorithm not in {"pruning", "both"}:
            return

        self._update_phase()
        if self.accumulate_phase:
            self._apply_accumulation_phase()
            return

        self._apply_pruning_phase()

    def sanitize_parameters(self) -> None:
        """Replace NaNs/Infs in managed quantum parameters after optimizer steps."""
        if not self.config.sanitize_quantum_parameters:
            return

        for parameter_spec in self.quantum_parameters.values():
            parameter_spec.parameter.data.copy_(
                torch.nan_to_num(
                    parameter_spec.parameter.data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
