use criterion::{black_box, criterion_group, criterion_main, Criterion};
use viga::detector_ga::{transform_point_ga, intersect_planes_ga};
use viga::detector_standard::{transform_point_standard, intersect_planes_standard};
use garust::{Pga3, Motor3};
use nalgebra::{Point3, Isometry3, Vector3, Translation3, UnitQuaternion};

fn bench_transform_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform Point");

    // Setup GA
    let p_ga = Pga3::point(1.0, 2.0, 3.0);
    // Create a motor: translate by 10,20,30, rotate by 90deg about X
    let r = Motor3::rotor(std::f64::consts::FRAC_PI_2, Pga3::basis(0b0110)); // e23
    let t = Motor3::translator(10.0, 20.0, 30.0);
    let m_ga = t * r;

    group.bench_function("Geometric Algebra", |b| {
        b.iter(|| {
            transform_point_ga(black_box(&p_ga), black_box(&m_ga))
        })
    });

    // Setup Standard
    let p_std = Point3::new(1.0, 2.0, 3.0);
    let iso_std = Isometry3::from_parts(
        Translation3::new(10.0, 20.0, 30.0),
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2)
    );

    group.bench_function("Standard Math", |b| {
        b.iter(|| {
            transform_point_standard(black_box(&p_std), black_box(&iso_std))
        })
    });

    group.finish();
}

fn bench_intersect_planes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Intersect Planes");

    // Setup GA
    let p1_ga = Pga3::plane(1.0, 0.0, 0.0, -1.0); // x = 1
    let p2_ga = Pga3::plane(0.0, 1.0, 0.0, -2.0); // y = 2
    let p3_ga = Pga3::plane(0.0, 0.0, 1.0, -3.0); // z = 3

    group.bench_function("Geometric Algebra", |b| {
        b.iter(|| {
            intersect_planes_ga(black_box(&p1_ga), black_box(&p2_ga), black_box(&p3_ga))
        })
    });

    // Setup Standard
    let n1_std = Vector3::new(1.0, 0.0, 0.0);
    let n2_std = Vector3::new(0.0, 1.0, 0.0);
    let n3_std = Vector3::new(0.0, 0.0, 1.0);
    let d1_std = 1.0;
    let d2_std = 2.0;
    let d3_std = 3.0;

    group.bench_function("Standard Math", |b| {
        b.iter(|| {
            intersect_planes_standard(
                black_box(&n1_std), black_box(d1_std),
                black_box(&n2_std), black_box(d2_std),
                black_box(&n3_std), black_box(d3_std)
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_transform_point, bench_intersect_planes);
criterion_main!(benches);
