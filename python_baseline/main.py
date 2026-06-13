import time
import cv2
import numpy as np
import onnxruntime as ort

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

class PerceptionLayer:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    def detect_objects(self, image_path):
        # 1. Load image and resize to 640x640
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        
        # 2. Convert to shape [1, 3, 640, 640] and normalize to 0.0-1.0
        input_tensor = resized.transpose((2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
        
        # 3. Run inference
        outputs = self.session.run(None, {"images": input_tensor})
        output_tensor = outputs[0] # shape [1, 84, 8400]
        
        # 4. Parse YOLOv8 output
        boxes = []
        view = output_tensor[0] # shape [84, 8400]
        
        cx = view[0, :]
        cy = view[1, :]
        w = view[2, :]
        h = view[3, :]
        classes = view[4:84, :]
        
        max_class_probs = np.max(classes, axis=0)
        best_class_ids = np.argmax(classes, axis=0)
        
        mask = max_class_probs > 0.5
        valid_indices = np.where(mask)[0]
        
        for idx in valid_indices:
            x_min = cx[idx] - w[idx] / 2.0
            y_min = cy[idx] - h[idx] / 2.0
            boxes.append({
                "x": x_min,
                "y": y_min,
                "width": w[idx],
                "height": h[idx],
                "confidence": max_class_probs[idx],
                "class_id": best_class_ids[idx]
            })
            
        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        nms_boxes = []
        for b in boxes:
            keep = True
            for kept in nms_boxes:
                box_b = [b["x"], b["y"], b["width"], b["height"]]
                box_kept = [kept["x"], kept["y"], kept["width"], kept["height"]]
                if iou(box_b, box_kept) > 0.45:
                    keep = False
                    break
            if keep:
                nms_boxes.append(b)
                
        return nms_boxes

class SpatialTracker:
    def __init__(self):
        self.focal_length = 320.0
        self.cx = 320.0
        self.cy = 320.0
        self.tracked_objects = []
        self.next_id = 0
        
    def update_tracking(self, detections):
        new_tracked_objects = []
        for i, det in enumerate(detections):
            px = det["x"] + det["width"] / 2.0
            py = det["y"] + det["height"] / 2.0
            
            # Standard analytical 3D projection (direction vector)
            nx = (px - self.cx) / self.focal_length
            ny = (py - self.cy) / self.focal_length
            nz = 1.0
            
            direction = np.array([nx, ny, nz])
            
            # Plane intersection (y=2.0)
            # t = 2.0 / ny
            # Avoid divide by zero
            t = 2.0 / ny if ny != 0 else float('inf')
            
            point = np.array([nx * t, 2.0, nz * t])
            
            # Temporal ID matching
            matched_id = self.next_id
            closest_dist = float('inf')
            velocity_matrix = None
            
            for old_obj in self.tracked_objects:
                dist = np.linalg.norm(point - old_obj["point"])
                if dist < 2.0 and dist < closest_dist:
                    closest_dist = dist
                    matched_id = old_obj["id"]
                    
                    # Create 4x4 Affine Matrix for translation
                    tx = point[0] - old_obj["point"][0]
                    ty = point[1] - old_obj["point"][1]
                    tz = point[2] - old_obj["point"][2]
                    velocity_matrix = np.array([
                        [1.0, 0.0, 0.0, tx],
                        [0.0, 1.0, 0.0, ty],
                        [0.0, 0.0, 1.0, tz],
                        [0.0, 0.0, 0.0, 1.0]
                    ])

            if matched_id == self.next_id:
                self.next_id += 1
                
            new_tracked_objects.append({
                "id": matched_id,
                "box": det,
                "ray_vector": direction,
                "point": point,
                "velocity": velocity_matrix,
                "is_person": det["class_id"] == 0
            })
            
        self.tracked_objects = new_tracked_objects

def main():
    model_path = "../models/yolov8n.onnx"
    image_path = "../tests/test_data/zidane.jpg"
    
    try:
        perception = PerceptionLayer(model_path)
    except Exception as e:
        print(f"Failed to load ONNX model. {e}")
        return
        
    tracker = SpatialTracker()
    
    print("==================================================")
    print("                PYTHON BASELINE                   ")
    print("==================================================")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print("Libraries: onnxruntime (inference), cv2 (vision), numpy (linear algebra tracking)")
    print("--------------------------------------------------")
    print("Starting E2E Benchmark...")
    
    # Warmup
    initial_detections = perception.detect_objects(image_path)
    tracker.update_tracking(initial_detections)
    
    # 1. E2E Benchmark
    e2e_iterations = 100
    start_e2e = time.time()
    for _ in range(e2e_iterations):
        detections = perception.detect_objects(image_path)
        tracker.update_tracking(detections)
    e2e_duration = time.time() - start_e2e
    e2e_fps = e2e_iterations / e2e_duration
    
    print("[1] End-to-End Pipeline (Inference + Matrix Tracking)")
    print(f"    Total time for {e2e_iterations} frames: {e2e_duration:.4f}s")
    print(f"    Throughput: {e2e_fps:.2f} FPS")
    
    # 2. Math-only benchmark
    math_iterations = 100000
    start_math = time.time()
    for _ in range(math_iterations):
        # We pass initial_detections again and again to simulate the tracking across frames
        tracker.update_tracking(initial_detections)
    math_duration = time.time() - start_math
    math_fps = math_iterations / math_duration
    
    print("[2] Spatial Tracking Math Only (Numpy 4x4 Matrices)")
    print(f"    Total time for {math_iterations} updates: {math_duration:.4f}s")
    print(f"    Throughput: {math_fps:.2f} Updates/sec")
    
    # 3. Norfair tracking benchmark
    try:
        from norfair import Tracker, Detection
        # Norfair needs a distance function, we'll just use euclidean on center points
        tracker_norfair = Tracker(distance_function="euclidean", distance_threshold=50)
        
        # Convert initial_detections to Norfair Detections
        norfair_detections = []
        for det in initial_detections:
            px = det["x"] + det["width"] / 2.0
            py = det["y"] + det["height"] / 2.0
            norfair_detections.append(Detection(points=np.array([[px, py]])))
            
        start_norfair = time.time()
        for _ in range(math_iterations):
            tracker_norfair.update(detections=norfair_detections)
        norfair_duration = time.time() - start_norfair
        norfair_fps = math_iterations / norfair_duration
        
        print("[3] Norfair 2D Tracking (Kalman Filter)")
        print(f"    Total time for {math_iterations} updates: {norfair_duration:.4f}s")
        print(f"    Throughput: {norfair_fps:.2f} Updates/sec")
    except ImportError:
        print("[3] Norfair not installed. Skipping.")
        
    print("==================================================\n")

if __name__ == "__main__":
    main()
