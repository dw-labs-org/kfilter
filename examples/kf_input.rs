use kfilter::{kalman::Kalman1M, KalmanFilter, KalmanPredictInput};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Matrix2x1, SMatrix};
use rand::thread_rng;
use rand_distr::Distribution;

fn main() {
    // Create a new 2 state kalman filter
    // Model has acceleration input via the B matrix
    let mut k = Kalman1M::new_with_input(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        SMatrix::identity(),
        Matrix2x1::new(0.0, 1.0),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
        SMatrix::zeros(),
    );

    // Create a normal distribution to corrupt the position reading
    let noise = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    // Assuming x0 = v0 = 0 and a = 1, x = 0.5 * t^2
    for i in 0..100 {
        let x_real = 0.5 * ((i as f64) / 10.0).powi(2);
        // constant acceleration input
        let x_predicted = k.predict(Matrix1::new(1.0)).x;
        let x_measured = x_real + noise.sample(&mut rng);
        k.update(Matrix1::new(x_measured));
        let x_updated = k.state().x;
        println!("{x_real}, {x_measured}, {x_predicted}, {x_updated}");
    }
}
