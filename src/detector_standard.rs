use nalgebra::{Point3, Isometry3, Vector3, Matrix3};

/// Apply a rigid body transformation to a point using nalgebra's Isometry3.
pub fn transform_point_standard(point: &Point3<f64>, isometry: &Isometry3<f64>) -> Point3<f64> {
    isometry * point
}

/// Find the intersection of three planes (given by normal vectors and distances to origin)
/// by solving a 3x3 linear system.
pub fn intersect_planes_standard(
    n1: &Vector3<f64>, d1: f64,
    n2: &Vector3<f64>, d2: f64,
    n3: &Vector3<f64>, d3: f64,
) -> Option<Point3<f64>> {
    let m = Matrix3::from_columns(&[*n1, *n2, *n3]).transpose();
    let d = Vector3::new(d1, d2, d3);
    m.try_inverse().map(|inv| {
        let p = inv * d;
        Point3::from(p)
    })
}
