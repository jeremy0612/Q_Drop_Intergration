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
# Scheduled Gradient Pruning Algorithm
# From Q-Drop: Optimizing Quantum Orthogonal Networks with Statistical Pruning
# =============================================================================

class ScheduledGradientPruning:
    """
    Two-phase gradient pruning for quantum parameters:
    - Accumulation phase: sum gradients over multiple steps
    - Pruning phase: probabilistically select high-magnitude accumulated gradients
    """
    def __init__(self,
                 quantum_weights : tf.Variable,
                 accumulate_window : int=10,
                 prune_window : int=8,
                 prune_ratio : float=0.8,
                 seed : int=42,
                 dtype : tf.dtypes.DType=tf.float64,
                 schedule : bool = False):
        self.quantum_weights = quantum_weights
        self.dtype = dtype

        # Initialize the accumulated gradient variable
        self.accumulated_grads = tf.Variable(tf.zeros_like(quantum_weights),
                                             trainable=False, dtype=dtype)
        # Boolean flag: True for accumulation phase, False for pruning phase
        self.accumulate_flag = tf.Variable(True, trainable=False)

        # Window lengths for accumulation and pruning phases
        self.accumulate_window = tf.constant(accumulate_window)
        self.prune_window = tf.constant(prune_window)

        # Pruning ratio (can be updated dynamically)
        self.prune_ratio = tf.Variable(prune_ratio, trainable=False, dtype=dtype)

        # Counters for how many steps remain in each phase
        self.accumulate_count = tf.Variable(accumulate_window, dtype=tf.int32, trainable=False)
        self.prune_count = tf.Variable(prune_window, dtype=tf.int32, trainable=False)

        # Set seeds for reproducibility
        rd.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        qml.numpy.random.seed(seed)

        # Initialize the PruneScheduler
        self.scheduler = PruneScheduler(self) if schedule else None

    def update_phase(self):
        """Update the phase flags and counters based on current counts."""
        accumulate_count_val = int(self.accumulate_count.numpy()) if hasattr(self.accumulate_count, 'numpy') else int(self.accumulate_count)
        prune_count_val = int(self.prune_count.numpy()) if hasattr(self.prune_count, 'numpy') else int(self.prune_count)

        if accumulate_count_val == 0:
            self.accumulate_count.assign(self.accumulate_window)
            self.accumulate_flag.assign(False)
        elif prune_count_val == 0:
            self.prune_count.assign(self.prune_window)
            self.accumulate_flag.assign(True)

    def apply(self, quantum_grad, optimizer, gradients, trainable_variables):
        """
        Applies the accumulation or pruning algorithm based on the phase.
        """
        self.update_phase()

        accumulate_flag_val = bool(self.accumulate_flag.numpy()) if hasattr(self.accumulate_flag, 'numpy') else bool(self.accumulate_flag)

        if accumulate_flag_val:
            # Accumulation Phase
            self.accumulate_count.assign_sub(1)
            if quantum_grad is not None:
                self.accumulated_grads.assign_add(quantum_grad)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            # Pruning Phase
            self.prune_count.assign_sub(1)
            epsilon = tf.constant(1e-8, dtype=self.dtype)

            # Flatten for categorical sampling (weights may be multi-dimensional)
            original_shape = self.quantum_weights.shape
            flat_accum = tf.reshape(self.accumulated_grads, [-1])

            grad_min = tf.reduce_min(flat_accum)
            grad_max = tf.reduce_max(flat_accum)
            # Normalize the accumulated gradients
            norm_grads = (flat_accum - grad_min) / (grad_max - grad_min + epsilon)
            norm_grads_with_epsilon = norm_grads + epsilon
            logits = tf.math.log(norm_grads_with_epsilon)

            num_params = tf.size(self.quantum_weights)
            # Compute the number of parameters to sample based on prune_ratio
            num_samples = tf.maximum(1, tf.cast(self.prune_ratio * tf.cast(num_params, self.dtype), tf.int32))

            # Draw random indices according to logits (needs 2D input: [1, num_params])
            indices = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=num_samples)
            indices = tf.clip_by_value(indices, 0, tf.cast(num_params - 1, tf.int64))
            indices = tf.cast(tf.reshape(indices, [-1, 1]), tf.int32)

            # Create mask on flattened weights, then reshape back
            flat_mask = tf.zeros([tf.size(self.quantum_weights)], dtype=tf.bool)
            updates = tf.ones([tf.shape(indices)[0]], dtype=tf.bool)
            flat_mask = tf.tensor_scatter_nd_update(flat_mask, indices, updates)
            mask = tf.reshape(flat_mask, original_shape)

            # Apply pruned gradient only to quantum weights
            pruned_grad = tf.where(mask, self.accumulated_grads, tf.zeros_like(self.accumulated_grads))
            optimizer.apply_gradients([(pruned_grad, self.quantum_weights)])

            # Apply gradients for other variables
            other_gradients = []
            other_variables = []
            for grad, var in zip(gradients, trainable_variables):
                if var is not self.quantum_weights and grad is not None:
                    other_gradients.append(grad)
                    other_variables.append(var)
            optimizer.apply_gradients(zip(other_gradients, other_variables))

            # Reset accumulated gradients
            self.accumulated_grads.assign(tf.zeros_like(self.accumulated_grads))

            # Schedule prune ratio update
            self.scheduler.on_train_batch_end() if self.scheduler is not None else None


class PruneScheduler:
    """Exponentially increases prune ratio over time"""
    def __init__(self, gradient_pruning):
        self.gradient_pruning = gradient_pruning
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.global_step.assign(0)

    def on_train_batch_end(self):
        self.global_step.assign_add(1)
        should_update = tf.equal(tf.math.floormod(self.global_step, 5), 0)

        def update():
            current_prune_ratio = self.gradient_pruning.prune_ratio
            new_prune_ratio = tf.minimum(current_prune_ratio * math.exp(0.1),
                                         tf.constant(1.0, dtype=self.gradient_pruning.dtype))
            self.gradient_pruning.prune_ratio.assign(new_prune_ratio)
            tf.print("Updated prune_ratio to", new_prune_ratio)
            return tf.constant(0)

        def no_update():
            return tf.constant(0)

        tf.cond(should_update, update, no_update)
