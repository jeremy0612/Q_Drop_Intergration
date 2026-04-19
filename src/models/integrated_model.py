import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pennylane as qml
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers
import random as rd
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.pruning import ScheduledGradientPruning
from utils.dropout import QuantumDynamicDropoutManager


class IntegratedQDropHQGCModel(tf.keras.Model):
    """
    Integrated model combining:
    - Q-Drop Scheduled Gradient Pruning & Dynamic Dropout
    - HQGC Quantum Components (AngleEmbedding + BasicEntanglerLayers)
    - MNIST image classification
    """

    def __init__(self,
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 algorithm: str = 'pruning',  # 'pruning', 'dropout', or 'both'
                 algorithm_params: dict = None,
                 apply_dropout: bool = False,
                 random_seed: int = 42):
        super(IntegratedQDropHQGCModel, self).__init__()

        # Set seeds for reproducibility
        rd.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        qml.numpy.random.seed(random_seed)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.algorithm = algorithm
        self.apply_dropout_flag = apply_dropout

        # Classical pre-processing layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense_preprocess = tf.keras.layers.Dense(n_qubits, activation='relu', dtype=tf.float32)

        # Quantum weights for BasicEntanglerLayers
        # Shape: (n_layers, n_qubits) for RX rotations + CNOT gates
        self.quantum_weights = self.add_weight(
            shape=(n_layers, n_qubits),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32,
            name='quantum_weights'
        )

        # Quantum device with n_qubits wires
        self.dev = qml.device('default.qubit', wires=n_qubits)

        # Build the quantum node (QNode) using HQGC-style circuits
        # Use interface='numpy' for stability with eager execution
        @qml.qnode(self.dev, interface='numpy', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights):
            # Angle embedding: each input -> RX rotation
            AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            # Variational layers with entanglement
            BasicEntanglerLayers(weights, wires=range(n_qubits))
            # Measure Pauli-Z on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

        # Classical post-processing
        self.dense_postprocess = tf.keras.layers.Dense(32, activation='relu', dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax', dtype=tf.float32)

        # Initialize Q-Drop algorithms
        if algorithm_params is None:
            algorithm_params = {
                'accumulate_window': 10,
                'prune_window': 8,
                'prune_ratio': 0.8,
                'schedule': True
            }

        if algorithm in ['pruning', 'both']:
            self.pruning_algo = ScheduledGradientPruning(
                self.quantum_weights,
                accumulate_window=algorithm_params.get('accumulate_window', 10),
                prune_window=algorithm_params.get('prune_window', 8),
                prune_ratio=algorithm_params.get('prune_ratio', 0.8),
                seed=random_seed,
                dtype=tf.float32,
                schedule=algorithm_params.get('schedule', True)
            )
        else:
            self.pruning_algo = None

        if algorithm in ['dropout', 'both']:
            # Define wire-level dropout masks
            # Map which parameters belong to which wires
            theta_wire_0 = tf.constant([1, 0, 1, 0] + [0]*(n_qubits*n_layers-4), dtype=tf.int32)
            theta_wire_1 = tf.constant([0, 1, 0, 0] + [0]*(n_qubits*n_layers-4), dtype=tf.int32)
            n_drop = tf.constant(1, dtype=tf.int32)
            self.drop_flag = tf.Variable(apply_dropout, trainable=False)

            self.dropout_algo = QuantumDynamicDropoutManager(
                self.quantum_weights,
                theta_wire_0,
                theta_wire_1,
                n_drop,
                self.drop_flag
            )
        else:
            self.dropout_algo = None

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        flattened = self.flatten(inputs)
        preprocessed = self.dense_preprocess(flattened)

        # Run quantum circuit for each preprocessed output
        batch_size = tf.shape(preprocessed)[0]
        quantum_outputs_list = []

        for i in tf.range(batch_size):
            x = preprocessed[i]
            w = self.quantum_weights

            # Convert to numpy for quantum circuit (numpy interface)
            x_np = x.numpy() if tf.executing_eagerly() else np.asarray(x)
            w_np = w.numpy() if tf.executing_eagerly() else np.asarray(w)

            # Execute quantum circuit
            result = np.array(self.quantum_circuit(x_np, w_np), dtype=np.float32)
            quantum_outputs_list.append(tf.constant(result, dtype=tf.float32))

        quantum_outputs = tf.stack(quantum_outputs_list)

        # Handle NaN values
        quantum_outputs = tf.where(tf.math.is_nan(quantum_outputs),
                                   tf.zeros_like(quantum_outputs), quantum_outputs)

        # Post-process
        postprocessed = self.dense_postprocess(quantum_outputs)
        output = self.output_layer(postprocessed)

        return output

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Find quantum weights gradient by exact name
        quantum_grad = None
        quantum_weights_found = False
        for idx, var in enumerate(self.trainable_variables):
            if 'quantum_weights' in var.name:
                quantum_weights_found = True
                quantum_grad = gradients[idx]  # may be None if tape lost track through numpy ops
                break

        if not quantum_weights_found:
            raise ValueError(
                f"Quantum weights variable not found in trainable_variables: "
                f"{[v.name for v in self.trainable_variables]}"
            )

        # Apply Q-Drop algorithms
        # quantum_grad may be None when the circuit runs via numpy interface (tape can't
        # differentiate through .numpy() calls); fall back to plain gradient update in that case.
        if self.algorithm in ['pruning', 'both'] and self.pruning_algo is not None and quantum_grad is not None:
            self.pruning_algo.apply(quantum_grad, self.optimizer, gradients, self.trainable_variables)
        else:
            clean = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(gradients, self.trainable_variables)]
            self.optimizer.apply_gradients(zip(clean, self.trainable_variables))

        if self.algorithm in ['dropout', 'both'] and self.dropout_algo is not None:
            gradients = self.dropout_algo.apply_dropout(gradients, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Sanitize weights: replace NaNs with zeros
        for var in self.trainable_variables:
            sanitized_var = tf.where(tf.math.is_nan(var), tf.zeros_like(var), var)
            var.assign(sanitized_var)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
