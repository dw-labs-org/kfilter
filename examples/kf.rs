use kfilter::kalman::{Kalman1M, KalmanPredict};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, SMatrix};

fn main() {
    // Create a new 2 state kalman filter
    let mut k = Kalman1M::new(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        SMatrix::identity(),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
        SMatrix::zeros(),
    );

    for i in 0..100 {
        k.predict();
        k.update(Matrix1::new(i as f64));
    }
}
