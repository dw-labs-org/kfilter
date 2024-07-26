use kfilter::{kalman::Kalman1M, system::StepReturn};
use nalgebra::{Matrix1x2, Matrix2, SMatrix, SVector, Vector2};

// Example of creating an Kalman filter for a van der pol oscillator
fn main() {
    // Create a non-linear KF (EKF)
    let _k = Kalman1M::new_ekf_with_input(
        step_fn,
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
        SMatrix::zeros(),
        SMatrix::identity(),
    );
}

// Van der pol oscillator
fn step_fn(x: SVector<f64, 2>, _u: SVector<f64, 0>) -> StepReturn<f64, 2> {
    let dt = 0.1;
    let mu = 0.5;
    StepReturn {
        state: Vector2::new(
            x.x + x.y * dt,
            x.y + dt * ((mu * (1.0 - x.x.powi(2)) * x.y) - x.x),
        ),
        jacobian: Matrix2::new(
            1.0,
            dt,
            dt * mu * (1.0 - 2.0 * x.x) * x.y - 1.0,
            1.0 + dt * mu * (1.0 - x.x.powi(2)),
        ),
        covariance: SMatrix::identity(),
    }
}
