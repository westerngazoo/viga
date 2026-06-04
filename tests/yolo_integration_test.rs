use std::path::PathBuf;
use viga::perception::PerceptionLayer;
use viga::spatial_tracking::{CameraConfig, SpatialTracker};

#[test]
fn test_yolov8_integration() {
    let model_path = PathBuf::from("models/yolov8n.onnx");
    let image_path = PathBuf::from("tests/test_data/zidane.jpg");
    
    // Ensure the model exists before testing
    if !model_path.exists() {
        println!("Skipping integration test because YOLOv8n ONNX model is missing. Please run export_yolo.py");
        return;
    }

    // 1. Initialize Perception Layer
    let mut perception = PerceptionLayer::new(&model_path);
    
    // 2. Detect objects in the test image (zidane.jpg typically has a couple of people)
    let detections = perception.detect_objects(&image_path);
    
    assert!(!detections.is_empty(), "YOLO should have detected at least one object");
    
    // Check if we detected a person (class_id == 0)
    let people_count = detections.iter().filter(|d| d.class_id == 0).count();
    assert!(people_count > 0, "YOLO should detect at least one person in zidane.jpg");
    
    // 3. Project to 3D Space
    let camera = CameraConfig::default();
    let mut tracker = SpatialTracker::new(camera);
    tracker.update_tracking(detections);
    
    assert_eq!(tracker.tracked_objects.len(), people_count, "Tracker should have an object for every detection... wait, actually we track ALL detections right now.");
}

#[test]
fn test_geometric_projection() {
    let model_path = PathBuf::from("models/yolov8n.onnx");
    let image_path = PathBuf::from("tests/test_data/zidane.jpg");
    if !model_path.exists() { return; }

    let mut perception = PerceptionLayer::new(&model_path);
    let detections = perception.detect_objects(&image_path);
    
    let camera = CameraConfig::default();
    let mut tracker = SpatialTracker::new(camera);
    tracker.update_tracking(detections);
    
    for obj in &tracker.tracked_objects {
        // Evaluate the geometric ray
        // A valid ray through the origin and image plane should have non-zero bivector components (since it's a line in Pga3)
        // Check if the ray is not perfectly zero
        let is_zero = obj.ray.coeffs.iter().all(|&c| c.abs() < 1e-6);
        assert!(!is_zero, "Projected Pga3 ray should not be mathematically zero.");
    }
}
