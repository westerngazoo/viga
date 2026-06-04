use garust::{Pga3, Motor3};

/// Apply a rigid body transformation to a point using a PGA Motor.
pub fn transform_point_ga(point: &Pga3, motor: &Motor3) -> Pga3 {
    motor.apply(point)
}

/// Find the intersection of three planes to find a point using the wedge product.
pub fn intersect_planes_ga(p1: &Pga3, p2: &Pga3, p3: &Pga3) -> Pga3 {
    p1.wedge(p2).wedge(p3)
}
