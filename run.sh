#!/bin/bash
# Q-Drop Integration: Quick Start Script
# Trains integrated Q-Drop + HQGC model on MNIST

set -e

echo "=================================="
echo "Q-Drop Integration Setup"
echo "=================================="

# Check Python
echo "[*] Checking Python..."
python3 --version

# Check required packages
echo "[*] Checking dependencies..."
pip list | grep -E "tensorflow|pennylane|numpy" || echo "[!] Some dependencies may be missing"

# Navigate to code directory
cd /Users/quangnguyen/Documents/quang_nguyen/2026_CISlab/Q_Drop_Intergration/src

# Run training
echo ""
echo "Starting training..."
echo "=================================="
python train_mnist.py

echo ""
echo "=================================="
echo "Training complete!"
echo "[+] Results saved to:"
echo "    - qd_hqgc_mnist_training.png"
echo "=================================="
