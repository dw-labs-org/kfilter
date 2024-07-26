use kfilter::{
    kalman::{Kalman1M, KalmanPredict},
    KalmanFilter,
};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, SMatrix};
use rand::thread_rng;
use rand_distr::Distribution;

fn main() {
    // Create a new 2 state kalman filter. Constant velocity, 0.1s timestep
    // noise matrices = identity
    let mut k = Kalman1M::new(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        SMatrix::identity(),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
        SMatrix::zeros(),
    );

    // Create a normal distribution to corrupt the position reading
    let noise = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    // Run for 100 timesteps
    for i in 0..100 {
        let x_real = i as f64;
        let x_predicted = k.predict().x;
        let x_measured = x_real + noise.sample(&mut rng);
        k.update(Matrix1::new(x_measured));
        let x_updated = k.state().x;
        println!("{x_real}, {x_measured}, {x_predicted}, {x_updated}");
    }
}
