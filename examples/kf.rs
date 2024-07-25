use kfilter::kalman::{Kalman1M, KalmanFilter, KalmanPredict};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Matrix2x1, SMatrix};

fn main() {
    // Create a new 2 state kalman filter
    let mut k = Kalman1M::new(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        SMatrix::identity(),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
    );

    for i in 0..100 {
        k.predict();
        k.update(Matrix1::new(i as f64));
    }

    // Create a new 2 state kalman filter with 1 input
    let mut k = Kalman1M::new_with_input(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        SMatrix::identity(),
        Matrix2x1::new(0.0, 1.0),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
    );
}
