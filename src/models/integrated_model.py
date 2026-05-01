"""Integrated TensorFlow model for Q-Drop and HQGC MNIST experiments."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
import tensorflow as tf
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers

from utils.dropout import QuantumDynamicDropoutManager
from utils.pruning import ScheduledGradientPruning

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DEFAULT_ALGORITHM_PARAMS = {
    "accumulate_window": 10,
    "prune_window": 8,
    "prune_ratio": 0.8,
    "schedule": True,
}
SUPPORTED_ALGORITHMS = {"baseline", "pruning", "dropout", "both"}
LEGACY_WIRE_0_MASK_PATTERN = (1, 0, 1, 0)
LEGACY_WIRE_1_MASK_PATTERN = (0, 1, 0, 0)


def _set_random_seed(random_seed: int) -> None:
    """Seed every RNG used by the integrated model."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    qml.numpy.random.seed(random_seed)


def _resolve_algorithm_params(
    algorithm_params: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Merge caller overrides with the default Q-Drop settings."""
    resolved_params = DEFAULT_ALGORITHM_PARAMS.copy()
    if algorithm_params is not None:
        resolved_params.update(algorithm_params)
    return resolved_params


def _build_dropout_mask(pattern: Sequence[int], total_params: int) -> tf.Tensor:
    """
    Expand a legacy flat mask pattern to the current quantum weight size.

    The project currently relies on a small hand-authored mask pattern. We pad
    the pattern instead of changing its semantics during a naming-only refactor.
    """
    clipped_pattern = list(pattern[:total_params])
    padded_pattern = clipped_pattern + [0] * max(total_params - len(clipped_pattern), 0)
    return tf.constant(padded_pattern, dtype=tf.int32)


class IntegratedQDropHQGCModel(tf.keras.Model):
    """
    Integrated model combining:
    - Q-Drop scheduled gradient pruning and dynamic dropout
    - HQGC quantum components (AngleEmbedding + BasicEntanglerLayers)
    - MNIST image classification
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        algorithm: str = "pruning",
        algorithm_params: Optional[Dict[str, object]] = None,
        apply_dropout: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()

        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Expected one of: {sorted(SUPPORTED_ALGORITHMS)}"
            )

        _set_random_seed(random_seed)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.algorithm = algorithm
        self.random_seed = random_seed
        self.algorithm_params = _resolve_algorithm_params(algorithm_params)
        self.apply_dropout_flag = apply_dropout

        self.flatten_layer = tf.keras.layers.Flatten()
        self.preprocess_dense = tf.keras.layers.Dense(
            n_qubits,
            activation="relu",
            dtype=tf.float32,
        )
        self.quantum_weights = self.add_weight(
            shape=(n_layers, n_qubits),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
            name="quantum_weights",
        )
        self.quantum_device = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = self._build_quantum_circuit()
        self.postprocess_dense = tf.keras.layers.Dense(32, activation="relu", dtype=tf.float32)
        self.output_dense = tf.keras.layers.Dense(2, activation="softmax", dtype=tf.float32)

        self.pruning_algorithm = self._build_pruning_algorithm()
        self.dropout_algorithm = self._build_dropout_algorithm()

        # Backward-compatible aliases for existing scripts and docs.
        self.flatten = self.flatten_layer
        self.dense_preprocess = self.preprocess_dense
        self.dense_postprocess = self.postprocess_dense
        self.output_layer = self.output_dense
        self.dev = self.quantum_device
        self.pruning_algo = self.pruning_algorithm
        self.dropout_algo = self.dropout_algorithm

    def _build_quantum_circuit(self):
        """Create the PennyLane circuit used during forward passes."""

        @qml.qnode(self.quantum_device, interface="numpy", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="X")
            BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wire_index)) for wire_index in range(self.n_qubits)]

        return quantum_circuit

    def _build_pruning_algorithm(self) -> Optional[ScheduledGradientPruning]:
        """Initialize scheduled gradient pruning when enabled."""
        if self.algorithm not in {"pruning", "both"}:
            return None

        return ScheduledGradientPruning(
            self.quantum_weights,
            accumulate_window=self.algorithm_params.get("accumulate_window", 10),
            prune_window=self.algorithm_params.get("prune_window", 8),
            prune_ratio=self.algorithm_params.get("prune_ratio", 0.8),
            seed=self.random_seed,
            dtype=tf.float32,
            schedule=self.algorithm_params.get("schedule", True),
        )

    def _build_dropout_algorithm(self) -> Optional[QuantumDynamicDropoutManager]:
        """Initialize wire-level quantum dropout when enabled."""
        if self.algorithm not in {"dropout", "both"}:
            return None

        total_quantum_params = self.n_qubits * self.n_layers
        wire_0_mask = _build_dropout_mask(LEGACY_WIRE_0_MASK_PATTERN, total_quantum_params)
        wire_1_mask = _build_dropout_mask(LEGACY_WIRE_1_MASK_PATTERN, total_quantum_params)

        # The current implementation keeps the original one-wire drop behavior.
        self.dropout_enabled = tf.Variable(self.apply_dropout_flag, trainable=False)
        self.drop_flag = self.dropout_enabled

        return QuantumDynamicDropoutManager(
            self.quantum_weights,
            wire_0_mask,
            wire_1_mask,
            tf.constant(1, dtype=tf.int32),
            self.dropout_enabled,
        )

    @staticmethod
    def _replace_nan_values(tensor: tf.Tensor) -> tf.Tensor:
        """Replace NaN values with zeros to keep training numerically stable."""
        return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

    @staticmethod
    def _to_numpy(value):
        """Convert TensorFlow values to NumPy in eager-safe fashion."""
        return value.numpy() if tf.executing_eagerly() else np.asarray(value)

    def _run_quantum_batch(self, embedded_inputs: tf.Tensor) -> tf.Tensor:
        """Execute the quantum circuit for each sample in the batch."""
        batch_size = tf.shape(embedded_inputs)[0]
        batch_quantum_outputs: List[tf.Tensor] = []
        quantum_weights_np = self._to_numpy(self.quantum_weights)

        for batch_index in tf.range(batch_size):
            sample_inputs = embedded_inputs[batch_index]
            circuit_output = np.asarray(
                self.quantum_circuit(self._to_numpy(sample_inputs), quantum_weights_np),
                dtype=np.float32,
            )
            batch_quantum_outputs.append(tf.constant(circuit_output, dtype=tf.float32))

        return tf.stack(batch_quantum_outputs)

    def _get_quantum_weight_index(self) -> int:
        """Locate the trainable variable index for the quantum weight tensor."""
        for variable_index, variable in enumerate(self.trainable_variables):
            if variable is self.quantum_weights or "quantum_weights" in variable.name:
                return variable_index

        raise ValueError(
            "Quantum weights variable not found in trainable_variables: "
            f"{[variable.name for variable in self.trainable_variables]}"
        )

    def _sanitize_gradients(self, gradients: List[Optional[tf.Tensor]]) -> List[tf.Tensor]:
        """Replace missing or NaN gradients with zeros."""
        sanitized_gradients: List[tf.Tensor] = []
        for gradient, variable in zip(gradients, self.trainable_variables):
            if gradient is None:
                sanitized_gradients.append(tf.zeros_like(variable))
            else:
                sanitized_gradients.append(self._replace_nan_values(gradient))
        return sanitized_gradients

    def _apply_dropout_if_enabled(self, gradients: List[tf.Tensor]) -> List[tf.Tensor]:
        """Apply quantum dropout to gradients when the configured mode requires it."""
        if self.algorithm not in {"dropout", "both"} or self.dropout_algorithm is None:
            return gradients
        return self.dropout_algorithm.apply_dropout(gradients, self.trainable_variables)

    def _apply_optimizer_step(
        self,
        gradients: List[tf.Tensor],
        can_use_pruning: bool,
    ) -> None:
        """
        Apply a single optimizer step using either pruning-aware or standard logic.

        The previous implementation could update weights twice per batch when
        dropout was enabled. This keeps the gradient path explicit and unified.
        """
        if self.algorithm in {"pruning", "both"} and self.pruning_algorithm is not None and can_use_pruning:
            quantum_weight_index = self._get_quantum_weight_index()
            quantum_gradient = gradients[quantum_weight_index]
            self.pruning_algorithm.apply(
                quantum_gradient,
                self.optimizer,
                gradients,
                self.trainable_variables,
            )
            return

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def _sanitize_trainable_variables(self) -> None:
        """Remove NaNs from model weights after each optimization step."""
        for variable in self.trainable_variables:
            variable.assign(self._replace_nan_values(variable))

    def call(self, inputs, training=None):
        del training

        inputs = tf.cast(inputs, tf.float32)
        flattened_inputs = self.flatten_layer(inputs)
        embedded_inputs = self.preprocess_dense(flattened_inputs)
        quantum_outputs = self._run_quantum_batch(embedded_inputs)
        sanitized_outputs = self._replace_nan_values(quantum_outputs)
        postprocessed_outputs = self.postprocess_dense(sanitized_outputs)
        return self.output_dense(postprocessed_outputs)

    def train_step(self, data):
        inputs, labels = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(
                labels,
                predictions,
                regularization_losses=self.losses,
            )

        raw_gradients = tape.gradient(loss, self.trainable_variables)
        quantum_weight_index = self._get_quantum_weight_index()
        raw_quantum_gradient = raw_gradients[quantum_weight_index]

        sanitized_gradients = self._sanitize_gradients(raw_gradients)
        masked_gradients = self._apply_dropout_if_enabled(sanitized_gradients)
        self._apply_optimizer_step(
            masked_gradients,
            can_use_pruning=raw_quantum_gradient is not None,
        )
        self._sanitize_trainable_variables()

        self.compiled_metrics.update_state(labels, predictions)
        metric_results = {metric.name: metric.result() for metric in self.metrics}
        metric_results["loss"] = loss
        return metric_results
