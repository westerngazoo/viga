#!/bin/bash

# Ensure we're in the right directory
cd /Users/goose/projects/VIGA

echo "Running Rust Benchmark..."
cargo build --release --bin bench_e2e > /dev/null 2>&1
./target/release/bench_e2e

echo ""
echo "Setting up Python environment..."
# Check if uv is installed, if not, use python3 -m venv
if command -v uv &> /dev/null
then
    uv venv .venv > /dev/null 2>&1
    source .venv/bin/activate
    uv pip install -r python_baseline/requirements.txt > /dev/null 2>&1
else
    python3 -m venv .venv > /dev/null 2>&1
    source .venv/bin/activate
    pip install -r python_baseline/requirements.txt > /dev/null 2>&1
fi

echo "Running Python Benchmark..."
python python_baseline/main.py
