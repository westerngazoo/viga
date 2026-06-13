use std::path::PathBuf;
use std::time::Instant;
use viga::perception::{BoundingBox, PerceptionLayer};
use viga::spatial_tracking::{CameraConfig, SpatialTracker};
use garust::Pga3;

fn main() {
    let model_path = PathBuf::from("models/yolov8n.onnx");
    let image_path = PathBuf::from("tests/test_data/zidane.jpg");

    if !model_path.exists() {
        println!("ONNX model not found at {:?}. Please run export script.", model_path);
        return;
    }

    let mut perception = PerceptionLayer::new(&model_path);
    let camera = CameraConfig::default();
    let mut tracker = SpatialTracker::new(camera);

    println!("==================================================");
    println!("                  RUST BASELINE                   ");
    println!("==================================================");
    println!("Model: {:?}", model_path);
    println!("Image: {:?}", image_path);
    println!("Libraries: ort (inference), image (vision), garust (Geometric Algebra tracking)");
    println!("--------------------------------------------------");
    println!("Starting E2E Benchmark...");
    
    // Warmup
    let initial_detections = perception.detect_objects(&image_path);
    tracker.update_tracking(initial_detections.clone());

    // 1. End-to-end benchmark (Inference + NMS + GA Projection)
    let e2e_iterations = 100;
    let start_e2e = Instant::now();
    for _ in 0..e2e_iterations {
        let detections = perception.detect_objects(&image_path);
        tracker.update_tracking(detections);
    }
    let e2e_duration = start_e2e.elapsed();
    let e2e_fps = e2e_iterations as f64 / e2e_duration.as_secs_f64();
    
    println!("[1] End-to-End Pipeline (Inference + garust Motor Tracking)");
    println!("    Total time for {} frames: {:?}", e2e_iterations, e2e_duration);
    println!("    Throughput: {:.2} FPS", e2e_fps);

    // 2. Math-only benchmark (GA Projection only)
    let math_iterations = 100_000;
    let start_math = Instant::now();
    for _ in 0..math_iterations {
        tracker.update_tracking(initial_detections.clone());
    }
    let math_duration = start_math.elapsed();
    let math_fps = math_iterations as f64 / math_duration.as_secs_f64();
    
    println!("[2] Spatial Tracking Math Only (garust Motors)");
    println!("    Total time for {} updates: {:?}", math_iterations, math_duration);
    println!("    Throughput: {:.2} Updates/sec", math_fps);
    println!("==================================================\n");
}
