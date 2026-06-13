# VIGA Usage Guide

This repository contains the VIGA core engine (Rust) and the baseline clients/benchmarks (Python).

## Prerequisites

1. **Rust Toolchain**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. **Python 3**: Ensure you have Python 3 installed.
3. **Model Weights**: You must have the YOLOv8 ONNX weights located at `models/yolov8n.onnx`.

## 1. Running the Live Safety Demo

The Live Demo uses your laptop's webcam, pipes the video frames into the VIGA engine, and visually draws the output 3D geometry and safety violations.

```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Run the live viewer
python live_demo.py
```
*(Press `q` to quit the live feed).*

## 2. Running the VIGA Engine via CLI

If you want to integrate VIGA into your own external application, you can use the raw `viga_cli` microservice. It accepts image file paths via standard input and outputs pure JSON.

```bash
# Compile and run the CLI
echo "tests/test_data/zidane.jpg" | cargo run --release --bin viga_cli
```

**Example JSON Output:**
```json
{
  "alerts": [
    {
      "rule_id": "PPE_VIOLATION",
      "severity": 10,
      "message": "Person detected without required safety gear",
      "timestamp": "2026-06-13T12:00:00Z"
    }
  ],
  "geometry": {
    "tracked_objects": [ ... ]
  }
}
```

## 3. Running the Performance Benchmarks

VIGA includes a rigorous benchmarking suite comparing Geometric Algebra math against Python/NumPy Affine Matrices and Norfair 2D trackers.

```bash
# Execute the E2E and Math tracking race
./run_comparison.sh
```
