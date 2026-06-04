import time
import numpy as np

def run_perception_layer():
    # Simulate ONNX returning bounding boxes [x, y, w, h, confidence, class]
    # In a real scenario: session.run(None, {input_name: img_data})
    # We allocate a new numpy array to simulate the overhead of receiving tensor outputs.
    return np.array([
        [100.0, 100.0, 50.0, 150.0, 0.95, 0.0],
        [150.0, 120.0, 45.0, 160.0, 0.85, 0.0]
    ])

def run_spatial_tracking(detections):
    # Standard Python/numpy tracking logic using standard matrix math.
    tracked_objects = []
    for det in detections:
        is_real = det[4] > 0.90
        # Numpy operations to simulate casting a 3D ray and evaluating it.
        # This mocks standard linear algebra (e.g., matrix mults) typical in Python backends
        ray_dir = np.array([1.0, 1.0, 1.0, 1.0])
        transform = np.eye(4)
        projected = transform @ ray_dir
        
        tracked_objects.append({
            "bbox": det,
            "ray": projected,
            "is_real_person": is_real
        })
    return tracked_objects

def benchmark(num_frames=100000):
    start_time = time.time()
    
    for _ in range(num_frames):
        detections = run_perception_layer()
        tracked = run_spatial_tracking(detections)
        
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    latency_ms = (total_time / num_frames) * 1000
    
    print("=== Python Baseline Benchmark ===")
    print(f"Frames processed: {num_frames}")
    print(f"Total time:       {total_time:.4f} s")
    print(f"Throughput:       {fps:.2f} FPS")
    print(f"Latency/frame:    {latency_ms:.4f} ms")

if __name__ == "__main__":
    # Warmup
    benchmark(100)
    # Actual bench
    benchmark(100000)
