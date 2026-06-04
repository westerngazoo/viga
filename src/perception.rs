use std::path::Path;
use image::{imageops::FilterType, GenericImageView};
use ndarray::{Array, IxDyn};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub class_id: usize,
}

pub struct PerceptionLayer {
    session: Session,
}

impl PerceptionLayer {
    pub fn new(model_path: &Path) -> Self {
        // Initialize ort environment (usually automatic in v2)
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        Self { session }
    }

    pub fn detect_objects(&mut self, image_path: &Path) -> Vec<BoundingBox> {
        // 1. Load image and resize to 640x640
        let img = image::open(image_path).unwrap();
        let resized = img.resize_exact(640, 640, FilterType::Nearest);
        
        // 2. Convert to ndarray shape [1, 3, 640, 640] and normalize to 0.0-1.0
        let mut input_tensor = Array::zeros(IxDyn(&[1, 3, 640, 640]));
        for (x, y, pixel) in resized.pixels() {
            input_tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32) / 255.0;
            input_tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32) / 255.0;
            input_tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32) / 255.0;
        }

        // 3. Run inference
        let tensor = Tensor::from_array(input_tensor).unwrap();
        let inputs = ort::inputs!["images" => tensor];
        let outputs = self.session.run(inputs).unwrap();
        
        // 4. Parse YOLOv8 output
        // The output shape is typically [1, 84, 8400] for YOLOv8
        // Rows 0-3 are cx, cy, w, h. Rows 4-83 are class probabilities.
        let output_tensor = outputs["output0"].try_extract_tensor::<f32>().unwrap();
        let slice = output_tensor.1; // slice is &[f32] of length 84 * 8400
        
        let mut boxes = Vec::new();
        // shape is [1, 84, 8400]
        for col in 0..8400 {
            let mut max_class_prob = 0.0;
            let mut best_class_id = 0;
            
            // YOLOv8 has 80 COCO classes starting at row 4
            for class_idx in 0..80 {
                let prob = slice[(class_idx + 4) * 8400 + col];
                if prob > max_class_prob {
                    max_class_prob = prob;
                    best_class_id = class_idx;
                }
            }
            
            // Confidence threshold
            if max_class_prob > 0.5 {
                let cx = slice[0 * 8400 + col];
                let cy = slice[1 * 8400 + col];
                let w = slice[2 * 8400 + col];
                let h = slice[3 * 8400 + col];
                
                boxes.push(BoundingBox {
                    x: cx - w / 2.0,
                    y: cy - h / 2.0,
                    width: w,
                    height: h,
                    confidence: max_class_prob,
                    class_id: best_class_id,
                });
            }
        }
        
        // Simple NMS (Non-Maximum Suppression)
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut nms_boxes = Vec::new();
        for b in boxes {
            let mut keep = true;
            for kept in &nms_boxes {
                if iou(&b, kept) > 0.45 {
                    keep = false;
                    break;
                }
            }
            if keep {
                nms_boxes.push(b);
            }
        }
        
        nms_boxes
    }
}

fn iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = f32::max(a.x, b.x);
    let y1 = f32::max(a.y, b.y);
    let x2 = f32::min(a.x + a.width, b.x + b.width);
    let y2 = f32::min(a.y + a.height, b.y + b.height);
    
    let inter_area = f32::max(0.0, x2 - x1) * f32::max(0.0, y2 - y1);
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    
    inter_area / (area_a + area_b - inter_area)
}
