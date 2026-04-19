import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pennylane as qml
import random as rd

# =============================================================================
# Dynamic Quantum Dropout
# From Q-Drop: Optimizing Quantum Orthogonal Networks with Statistical Pruning
# =============================================================================

class QuantumDynamicDropoutManager:
    """
    Wire-level dropout for quantum circuits.
    Stochastically disables entire qubit pathways during training.
    """
    def __init__(self, quantum_weights, theta_wire_0, theta_wire_1, n_drop, drop_flag):
        """
        Args:
            quantum_weights: The trainable quantum weight variable.
            theta_wire_0: A tensor mask for dropping parameters on wire 0.
            theta_wire_1: A tensor mask for dropping parameters on wire 1.
            n_drop: A tf.constant indicating how many wires to drop.
            drop_flag: A tf.Variable (bool) that turns dropout on/off.
        """
        self.quantum_weights = quantum_weights
        self.theta_wire_0 = theta_wire_0
        self.theta_wire_1 = theta_wire_1
        self.n_drop = n_drop
        self.drop_flag = drop_flag

    @tf.function
    def sanitize_gradients(self, gradients):
        """Replace any NaN values in gradients with zeros."""
        sanitized = []
        for grad in gradients:
            if grad is not None:
                sanitized.append(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad))
            else:
                sanitized.append(grad)
        return sanitized

    @tf.function
    def apply_dropout(self, gradients, trainable_variables):
        """
        Applies the quantum dropout mask to the gradient corresponding to the
        quantum weights.
        """
        # Find the index corresponding to the quantum weights variable
        quantum_index = None
        for i, var in enumerate(trainable_variables):
            if var is self.quantum_weights:
                quantum_index = i
                break
        if quantum_index is None:
            raise ValueError("Quantum weights variable not found in trainable_variables.")

        # Gradient is None when the tape can't differentiate through numpy ops in the
        # quantum circuit; use zeros so the mask operations below don't crash.
        raw_grad = gradients[quantum_index]
        quantum_grad = raw_grad if raw_grad is not None else tf.zeros_like(self.quantum_weights)

        # Define dropout functions
        def one_wire_drop():
            return tf.where(self.theta_wire_0 == 1, 0.0, quantum_grad)

        def two_wire_drop():
            dropped = tf.where(self.theta_wire_0 == 1, 0.0, quantum_grad)
            return tf.where(self.theta_wire_1 == 1, 0.0, dropped)

        # Choose which dropout to apply
        def dropout_fn():
            return tf.cond(
                tf.equal(self.n_drop, 1),
                lambda: one_wire_drop(),
                lambda: tf.cond(
                    tf.equal(self.n_drop, 2),
                    two_wire_drop,
                    lambda: quantum_grad
                )
            )

        new_quantum_grad = tf.cond(self.drop_flag, dropout_fn, lambda: quantum_grad)
        new_gradients = list(gradients)
        new_gradients[quantum_index] = new_quantum_grad
        return new_gradients
