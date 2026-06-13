use garust::Pga3;
use garust::pga::{Line, Plane, Point};
use garust::Motor;
use crate::perception::BoundingBox;

#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub focal_length: f64, // in pixels
    pub cx: f64,           // principal point X
    pub cy: f64,           // principal point Y
}

impl Default for CameraConfig {
    fn default() -> Self {
        // Assume 640x640 image, 90 deg FOV -> f = 640 / (2*tan(45)) = 320
        Self {
            focal_length: 320.0,
            cx: 320.0,
            cy: 320.0,
        }
    }
}

pub struct TrackedObject {
    pub id: usize,
    pub bounding_box: BoundingBox,
    pub ray: Pga3, // Kept for backwards compatibility
    pub pga_line: Line<f64>,
    pub pga_point: Point<f64>, // The exact 3D location of the person
    pub velocity: Option<Motor<f64>>, // The movement from previous frame
    pub is_person: bool,
}

pub struct SpatialTracker {
    pub tracked_objects: Vec<TrackedObject>,
    pub camera: CameraConfig,
    pub ground_plane: Plane<f64>,
    next_id: usize,
}

impl SpatialTracker {
    pub fn new(camera: CameraConfig) -> Self {
        Self {
            tracked_objects: Vec::new(),
            camera,
            // Assuming camera is 2 meters above ground, and Y is down.
            // Ground plane is y - 2.0 = 0 -> a=0, b=1, c=0, d=-2.0
            ground_plane: Plane::new(0.0, 1.0, 0.0, -2.0),
            next_id: 0,
        }
    }

    pub fn update_tracking(&mut self, detections: Vec<BoundingBox>) {
        let mut new_tracked_objects = Vec::new();
        let camera_origin = Point::new(0.0, 0.0, 0.0);
        
        for det in detections {
            let px = det.x as f64 + det.width as f64 / 2.0;
            let py = det.y as f64 + det.height as f64 / 2.0;
            
            let nx = (px - self.camera.cx) / self.camera.focal_length;
            let ny = (py - self.camera.cy) / self.camera.focal_length;
            let nz = 1.0; 
            
            let image_plane_point = Point::new(nx, ny, nz);
            
            // The ray is the geometric join of the camera origin and the image plane point
            let pga_line = camera_origin.join(&image_plane_point);
            
            // Extract the exact 3D point by meeting the line with the ground plane
            let pga_point = pga_line.meet(&self.ground_plane);
            
            let is_person = det.class_id == 0;
            
            // Simple temporal ID matching
            let mut matched_id = self.next_id;
            let mut best_motor: Option<Motor<f64>> = None;
            let mut closest_dist = f64::MAX;

            for old_obj in &self.tracked_objects {
                // Find distance using euclidean coordinates
                let (x1, y1, z1) = old_obj.pga_point.to_euclidean();
                let (x2, y2, z2) = pga_point.to_euclidean();
                let dist = ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt();
                
                // If it's the same object (less than 2 meters movement)
                if dist < 2.0 && dist < closest_dist {
                    closest_dist = dist;
                    matched_id = old_obj.id;
                    
                    // Construct a Motor that translates from old_point to new_point
                    // Motor is ratio of two points or lines. For pure translation between points, 
                    // it is 1 + 0.5 * (new_point - old_point) * e0, or we can use Motor::translator
                    best_motor = Some(Motor::translator(x2 - x1, y2 - y1, z2 - z1));
                }
            }

            if matched_id == self.next_id {
                self.next_id += 1;
            }
            
            new_tracked_objects.push(TrackedObject {
                id: matched_id,
                bounding_box: det,
                ray: pga_line.multivector(),
                pga_line,
                pga_point,
                velocity: best_motor,
                is_person,
            });
        }
        
        self.tracked_objects = new_tracked_objects;
    }
}
