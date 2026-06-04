use garust::Pga3;
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
    pub ray: Pga3, // The Pga3 line extending from the camera to the object
    pub is_person: bool,
}

pub struct SpatialTracker {
    pub tracked_objects: Vec<TrackedObject>,
    pub camera: CameraConfig,
}

impl SpatialTracker {
    pub fn new(camera: CameraConfig) -> Self {
        Self {
            tracked_objects: Vec::new(),
            camera,
        }
    }

    pub fn update_tracking(&mut self, detections: Vec<BoundingBox>) {
        self.tracked_objects.clear();
        
        // Pinhole camera origin in 3D: (0, 0, 0)
        let camera_origin = Pga3::point(0.0, 0.0, 0.0);
        
        for (i, det) in detections.into_iter().enumerate() {
            // Find center of bounding box in pixel coordinates
            let px = det.x as f64 + det.width as f64 / 2.0;
            let py = det.y as f64 + det.height as f64 / 2.0;
            
            // Map pixel coordinates to normalized device coordinates (3D vector from camera)
            // Assuming Z is forward, X is right, Y is down
            let nx = (px - self.camera.cx) / self.camera.focal_length;
            let ny = (py - self.camera.cy) / self.camera.focal_length;
            let nz = 1.0; 
            
            // Create a Pga3 point on the image plane at Z=1
            let image_plane_point = Pga3::point(nx, ny, nz);
            
            // The ray is the geometric join of the camera origin and the image plane point
            let ray = camera_origin.line_through(&image_plane_point);
            
            // Real YOLOv8 has person class = 0
            let is_person = det.class_id == 0;
            
            self.tracked_objects.push(TrackedObject {
                id: i,
                bounding_box: det,
                ray,
                is_person,
            });
        }
    }
}
