use crate::perception::BoundingBox;
use crate::spatial_tracking::TrackedObject;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SafetyAlert {
    pub person_id: usize,
    pub missing_gear: String,
    pub bounding_box: BoundingBox,
    pub ray_components: Vec<f64>, // The PGA3 geometric ray representation
}

pub struct AlertEngine {
    pub safety_gear_classes: Vec<usize>,
}

impl AlertEngine {
    pub fn new() -> Self {
        Self {
            // Class 24 (backpack) as mock safety gear for the demo
            safety_gear_classes: vec![24],
        }
    }

    pub fn check_safety(&self, tracked_objects: &[TrackedObject]) -> Vec<SafetyAlert> {
        let mut alerts = Vec::new();
        
        let persons: Vec<&TrackedObject> = tracked_objects.iter().filter(|o| o.is_person).collect();
        let gears: Vec<&TrackedObject> = tracked_objects.iter().filter(|o| self.safety_gear_classes.contains(&o.bounding_box.class_id)).collect();

        for person in persons {
            let mut has_gear = false;
            
            // Check if any safety gear intersects with this person
            for gear in &gears {
                if iou(&person.bounding_box, &gear.bounding_box) > 0.1 || is_inside(&gear.bounding_box, &person.bounding_box) {
                    has_gear = true;
                    break;
                }
            }

            if !has_gear {
                alerts.push(SafetyAlert {
                    person_id: person.id,
                    missing_gear: "Backpack/SafetyVest".to_string(),
                    bounding_box: person.bounding_box.clone(),
                    ray_components: person.ray.coeffs.to_vec(),
                });
            }
        }

        alerts
    }
}

// Helper: Intersection over Union
fn iou(box_a: &BoundingBox, box_b: &BoundingBox) -> f32 {
    let x_a = box_a.x.max(box_b.x);
    let y_a = box_a.y.max(box_b.y);
    let x_b = (box_a.x + box_a.width).min(box_b.x + box_b.width);
    let y_b = (box_a.y + box_a.height).min(box_b.y + box_b.height);

    let inter_area = (x_b - x_a).max(0.0) * (y_b - y_a).max(0.0);
    if inter_area <= 0.0 {
        return 0.0;
    }

    let box_a_area = box_a.width * box_a.height;
    let box_b_area = box_b.width * box_b.height;

    inter_area / (box_a_area + box_b_area - inter_area)
}

// Helper: Check if box_a is fully or mostly inside box_b
fn is_inside(box_a: &BoundingBox, box_b: &BoundingBox) -> bool {
    let center_x = box_a.x + box_a.width / 2.0;
    let center_y = box_a.y + box_a.height / 2.0;
    
    center_x >= box_b.x && center_x <= (box_b.x + box_b.width) &&
    center_y >= box_b.y && center_y <= (box_b.y + box_b.height)
}
