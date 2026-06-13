#!/bin/bash

echo "==========================================="
echo "   VIGA Real-World Performance Benchmark   "
echo "==========================================="
echo ""

# 1. Run Python Baseline
echo "-------------------------------------------"
echo "Running Python (Numpy + ONNXRuntime) Baseline..."
echo "-------------------------------------------"
cd python_baseline
source ../.venv/bin/activate
python main.py
cd ..

echo ""

# 2. Run Rust GA Pipeline
echo "-------------------------------------------"
echo "Running Rust (garust + ort) Pipeline..."
echo "-------------------------------------------"
cargo run --release --bin bench_e2e

echo ""
echo "==========================================="
echo "               Benchmark Complete          "
echo "==========================================="
