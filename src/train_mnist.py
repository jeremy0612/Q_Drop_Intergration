"""
Training script: Q-Drop Integration with HQGC on MNIST
Combines Scheduled Gradient Pruning & Dynamic Quantum Dropout
with Quantum GCN components for MNIST image classification.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.run_functions_eagerly(True)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import random as rd
import json
from datetime import datetime

from models.integrated_model import IntegratedQDropHQGCModel


class TrainingConfig:
    """Configuration for integrated Q-Drop + HQGC training"""
    def __init__(self):
        self.epochs = 5
        self.initial_lr = 0.3
        self.final_lr = 0.03
        self.batch_size = 32
        self.n_qubits = 4
        self.n_layers = 2
        self.algorithm = 'pruning'  # Options: 'pruning', 'dropout', 'both'
        self.random_seed = 42
        self.monte_carlo_runs = 5
        self.train_samples = 500
        self.test_samples = 300
        # Binary classification classes
        self.class_1 = 3
        self.class_2 = 6


def load_and_preprocess_mnist(config):
    """Load MNIST and preprocess for 2-class binary classification"""
    print("[*] Loading MNIST dataset...")

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Filter for only the specified classes
    train_filter = np.where((y_train == config.class_1) | (y_train == config.class_2))
    test_filter = np.where((y_test == config.class_1) | (y_test == config.class_2))

    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    # Use subset for faster training
    x_train, y_train = x_train[:config.train_samples], y_train[:config.train_samples]
    x_test, y_test = x_test[:config.test_samples], y_test[:config.test_samples]

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Expand dimensions for channel
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Preprocess: center crop and downsample
    def preprocess_images(images):
        images_cropped = tf.image.central_crop(images, central_fraction=24/28)
        images_downsampled = tf.image.resize(images_cropped, size=(4, 4),
                                            method=tf.image.ResizeMethod.BILINEAR)
        return images_downsampled

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    # Binary labels
    y_train_binary = np.where(y_train == config.class_1, 0, 1)
    y_test_binary = np.where(y_test == config.class_1, 0, 1)

    # One-hot encoding
    y_train_onehot = to_categorical(y_train_binary, 2)
    y_test_onehot = to_categorical(y_test_binary, 2)

    print(f"[+] Training samples: {x_train.shape[0]} (classes {config.class_1} vs {config.class_2})")
    print(f"[+] Test samples: {x_test.shape[0]}")
    print(f"[+] Input shape: {x_train.shape}")

    return x_train, y_train_onehot, x_test, y_test_onehot


def create_model(config, algorithm, apply_dropout=False):
    """Create integrated model"""
    algorithm_params = {
        'accumulate_window': 10,
        'prune_window': 8,
        'prune_ratio': 0.8,
        'schedule': True
    }

    model = IntegratedQDropHQGCModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        algorithm=algorithm,
        algorithm_params=algorithm_params,
        apply_dropout=apply_dropout,
        random_seed=config.random_seed
    )

    return model


def train_model(model, x_train, y_train, x_test, y_test, config, algorithm):
    """Train the model"""
    print(f"\n[*] Training with algorithm: {algorithm}")

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.initial_lr,
        decay_steps=config.epochs,
        alpha=config.final_lr / config.initial_lr
    )

    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        x_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[+] Test Accuracy: {test_acc:.4f}")

    return history, test_acc


def plot_results(histories, algorithms):
    """Plot training results: curves + accuracy bar chart"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Accuracy curves
    for hist, algo in zip(histories, algorithms):
        axes[0].plot(hist.history['accuracy'], label=f'{algo} (train)', linestyle='-')
        axes[0].plot(hist.history['val_accuracy'], label=f'{algo} (val)', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy')
    axes[0].legend()
    axes[0].grid()

    # Loss curves
    for hist, algo in zip(histories, algorithms):
        axes[1].plot(hist.history['loss'], label=f'{algo} (train)', linestyle='-')
        axes[1].plot(hist.history['val_loss'], label=f'{algo} (val)', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid()

    # Final val accuracy bar chart
    val_accs = [hist.history['val_accuracy'][-1] for hist in histories]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bars = axes[2].bar(algorithms, val_accs, color=colors[:len(algorithms)], edgecolor='black')
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel('Val Accuracy')
    axes[2].set_title('Final Validation Accuracy by Algorithm')
    axes[2].grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, val_accs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('qd_hqgc_mnist_training.png', dpi=150)
    print("[+] Plot saved to qd_hqgc_mnist_training.png")
    plt.close()


def main():
    config = TrainingConfig()
    device = tf.config.list_physical_devices('GPU')
    print(f"[*] GPU Available: {len(device) > 0}")
    if device:
        print(f"[+] GPU Device: {device[0].name}")

    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist(config)

    # Test different algorithms
    algorithms = []
    accuracies = []
    histories = []

    print("\n" + "="*60)
    print("INTEGRATED Q-DROP + HQGC ON MNIST")
    print("="*60)

    # Algorithm 1: Pruning only
    print("\n[Stage 1] Scheduled Gradient Pruning")
    print("-"*60)
    rd.seed(config.random_seed)
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    model_prune = create_model(config, 'pruning', apply_dropout=False)
    hist_prune, acc_prune = train_model(model_prune, x_train, y_train, x_test, y_test,
                                        config, "Pruning")
    algorithms.append("Pruning")
    accuracies.append(acc_prune)
    histories.append(hist_prune)

    # Algorithm 2: Dropout only
    print("\n[Stage 2] Dynamic Quantum Dropout")
    print("-"*60)
    rd.seed(config.random_seed)
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    model_dropout = create_model(config, 'dropout', apply_dropout=True)
    hist_dropout, acc_dropout = train_model(model_dropout, x_train, y_train, x_test, y_test,
                                            config, "Dropout")
    algorithms.append("Dropout")
    accuracies.append(acc_dropout)
    histories.append(hist_dropout)

    # Algorithm 3: Combined
    print("\n[Stage 3] Combined Pruning + Dropout")
    print("-"*60)
    rd.seed(config.random_seed)
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    model_combined = create_model(config, 'both', apply_dropout=True)
    hist_combined, acc_combined = train_model(model_combined, x_train, y_train, x_test, y_test,
                                              config, "Combined")
    algorithms.append("Combined")
    accuracies.append(acc_combined)
    histories.append(hist_combined)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for algo, acc in zip(algorithms, accuracies):
        print(f"{algo:20s}: {acc:.4f}")

    # Save metrics as JSON for CML/DVC
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "algorithms": {},
        "best_algorithm": max(zip(algorithms, accuracies), key=lambda x: x[1])[0],
        "best_accuracy": max(accuracies)
    }

    for algo, acc, hist in zip(algorithms, accuracies, histories):
        metrics["algorithms"][algo] = {
            "accuracy": float(acc),
            "final_loss": float(hist.history['loss'][-1]),
            "final_val_loss": float(hist.history['val_loss'][-1]),
            "final_val_accuracy": float(hist.history['val_accuracy'][-1])
        }

    # Save to parent directory (accessible from CI)
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Metrics saved to {metrics_path}")

    # Plot
    plot_results(histories, algorithms)

    print("\n[+] Training Complete!")


if __name__ == "__main__":
    main()
