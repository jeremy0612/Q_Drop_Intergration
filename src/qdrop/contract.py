"""Contracts for Torch modules that participate in Q-Drop training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantumDropoutState:
    """Per-layer dropout decision for the current training epoch."""

    enabled: bool
    dropped_wires: Tuple[int, ...] = ()


@dataclass(frozen=True)
class QuantumParameterMetadata:
    """Q-Drop metadata for a trainable quantum tensor."""

    parameter_name: str
    parameter: nn.Parameter
    num_wires: int
    wire_masks: Dict[int, torch.Tensor]


class QuantumDropCompatible(ABC):
    """Mixin for modules that expose quantum parameters to the Q-Drop runtime."""

    @abstractmethod
    def get_qdrop_parameters(self) -> Sequence[QuantumParameterMetadata]:
        """Return the quantum parameters and wire metadata owned by this layer."""

    @abstractmethod
    def set_qdrop_dropout_state(self, dropout_state: Optional[QuantumDropoutState]) -> None:
        """Apply or clear the active forward-mask state for this layer."""


@dataclass(frozen=True)
class DiscoveredQuantumLayer:
    """Concrete quantum layer discovered inside a model."""

    layer_name: str
    layer: QuantumDropCompatible
    parameter_specs: List[QuantumParameterMetadata]


def discover_quantum_layers(model: nn.Module) -> List[DiscoveredQuantumLayer]:
    """Collect all Q-Drop-compatible quantum layers from a model."""
    discovered_layers: List[DiscoveredQuantumLayer] = []

    for layer_name, module in model.named_modules():
        if not isinstance(module, QuantumDropCompatible):
            continue

        parameter_specs = list(module.get_qdrop_parameters())
        if not parameter_specs:
            continue

        discovered_layers.append(
            DiscoveredQuantumLayer(
                layer_name=layer_name,
                layer=module,
                parameter_specs=parameter_specs,
            )
        )

    return discovered_layers
