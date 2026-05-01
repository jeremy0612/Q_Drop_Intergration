"""Reusable Torch-side Q-Drop contract and runtime."""

from .contract import (
    DiscoveredQuantumLayer,
    QuantumDropCompatible,
    QuantumDropoutState,
    QuantumParameterMetadata,
    discover_quantum_layers,
)
from .runtime import TorchQDropConfig, TorchQDropManager

__all__ = [
    "DiscoveredQuantumLayer",
    "QuantumDropCompatible",
    "QuantumDropoutState",
    "QuantumParameterMetadata",
    "TorchQDropConfig",
    "TorchQDropManager",
    "discover_quantum_layers",
]
