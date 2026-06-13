use std::path::PathBuf;
use std::io::{self, BufRead};
use viga::perception::PerceptionLayer;
use viga::spatial_tracking::{CameraConfig, SpatialTracker};
use viga::alert_engine::{AlertEngine, SafetyAlert};

fn main() {
    let model_path = PathBuf::from("models/yolov8n.onnx");
    if !model_path.exists() {
        eprintln!("ONNX model not found at {:?}", model_path);
        std::process::exit(1);
    }

    let mut perception = PerceptionLayer::new(&model_path);
    let camera = CameraConfig::default();
    let mut tracker = SpatialTracker::new(camera);
    let engine = AlertEngine::new();

    // Read image paths line-by-line from stdin
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let image_path = PathBuf::from(line.trim());
        if !image_path.exists() {
            eprintln!("Image not found: {:?}", image_path);
            continue;
        }

        let detections = perception.detect_objects(&image_path);
        tracker.update_tracking(detections);
        let alerts = engine.check_safety(&tracker.tracked_objects);
        
        let json = serde_json::to_string(&alerts).unwrap();
        println!("{}", json);
    }
}
