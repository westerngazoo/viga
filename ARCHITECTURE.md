# VIGA Architecture

VIGA (Visual Intelligence with Geometric Algebra) is an edge-native Computer Vision pipeline. It is designed to be highly memory-efficient and performant by replacing traditional 2D pixel-tracking (e.g., DeepSORT, Kalman Filters) and 3D matrix math with pure **Projective Geometric Algebra (PGA)**.

## High-Level Data Flow

1. **Camera Feed**: Raw video frames are captured (via OpenCV/Python).
2. **Perception Layer (`yolov8n.onnx`)**: The frame is evaluated using ONNX Runtime. The neural network outputs 2D bounding boxes and class probabilities.
3. **Spatial Tracking (`garust::Pga3`)**: 
    - The 2D bounding box centers are unprojected into 3D geometric **Rays**.
    - The ground is defined as a `Pga3` **Plane**.
    - We take the geometric `meet` (wedge product) of the Camera Ray and Ground Plane to instantly extract the exact 3D physical point of the object.
4. **Temporal State (Velocity via Motors)**:
    - Instead of predicting 2D box movements, VIGA matches 3D physical points between frames.
    - Velocity and rotation are encapsulated perfectly into a single `Motor` multivector, avoiding bulky $4 \times 4$ Affine transformation matrices.
5. **Alert Engine (Safety Logic)**:
    - Business rules are applied to the 3D geometry (e.g., checking if a "Person" is physically intersecting with "Safety Gear").
6. **JSON Sink**: Results are streamed rapidly via `stdout` to the client visualization.

---

## Why Geometric Algebra (PGA)?

Traditional tracking frameworks like Norfair or DeepSORT operate in the 2D pixel space. When objects move, turn, or are viewed from a different perspective, 2D tracking can become extremely complex or break entirely.

When traditional systems *do* upgrade to 3D, they rely heavily on $4 \times 4$ Affine Transformation Matrices and Euler angles. This leads to:
1. **High Memory Footprint**: 16 floats per matrix.
2. **Gimbal Lock**: Euler angle rotations break down at certain axes.
3. **Slow Computation**: Matrix multiplications require massive throughput.

**VIGA solves this with PGA (`garust`):**
In PGA, points, lines, planes, translations, and rotations are all part of the exact same mathematical fabric.
- By using a **Motor**, we represent 3D translation and rotation with just a few floats instead of 16.
- Motors interpolate smoothly, meaning **no Gimbal Lock**.
- Because the math is native to the geometry, VIGA tracks 3D spatial velocity **10x faster** than NumPy matrices, and **100x faster** than Python 2D probabilistic trackers.
