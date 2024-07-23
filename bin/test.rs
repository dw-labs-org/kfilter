use kfilter::system::{LinearSystem, System};
use nalgebra::{Matrix2, Matrix2x1, SMatrix};

fn main() {
    let mut a = LinearSystem::new(
        Matrix2::new(1.0, 0.5, 0.0, 1.0),
        SMatrix::identity(),
        Matrix2x1::identity(),
    );
    println!("{}", std::mem::size_of::<LinearSystem<f64, 2, 0>>());
    println!("{}", a.step());
}
