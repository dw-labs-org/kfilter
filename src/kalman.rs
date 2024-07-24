use std::usize;

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    measurement::Measurement,
    system::{LinearSystemNoInput, System},
};

/// Representation of Kalman filter
#[derive(Debug)]
struct Kalman<T, const N: usize, const U: usize, S> {
    /// Covariance
    pub P: SMatrix<T, N, N>,
    /// The associated system
    pub system: S,
}

impl<T: RealField + Copy, const N: usize, const U: usize, S: System<T, N, U>> Kalman<T, N, U, S> {
    pub fn predict(&mut self) {
        self.system.step();
        let F = self.system.transition();
        self.P = F * self.P * F.transpose() + self.system.covariance();
    }

    pub fn predict_with_input(&mut self, u: SVector<T, U>) {
        self.system.step_with_input(u);
        let F = self.system.transition();
        self.P = F * self.P * F.transpose() + self.system.covariance();
    }

    pub fn new(system: S) -> Self {
        Kalman {
            P: SMatrix::zeros(),
            system,
        }
    }
}

trait KalmanUpdate<T, const N: usize, const M: usize> {
    fn update(&mut self, measurement: &Measurement<T, N, M>);
}

impl<T: RealField + Copy, const N: usize, const M: usize, const U: usize, S: System<T, N, U>>
    KalmanUpdate<T, N, M> for Kalman<T, N, U, S>
{
    fn update(&mut self, measurement: &Measurement<T, N, M>) {
        let H_transpose = measurement.H.transpose();
        // innovation
        let y = measurement.z - (measurement.H * self.system.state());
        // innovation covariance
        let S = measurement.H * self.P * H_transpose + measurement.R;
        // Optimal gain
        let K = self.P * H_transpose * S.try_inverse().unwrap();
        // state update
        *self.system.state_mut() += K * y;
        // covariance update
        self.P = (SMatrix::identity() - K * measurement.H) * self.P;
    }
}

type KalmanLinearNoInput<T, const N: usize> = Kalman<T, N, 0, LinearSystemNoInput<T, N>>;

impl<T: RealField + Copy, const N: usize> KalmanLinearNoInput<T, N> {
    pub fn new_linear_no_input(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>) -> Self {
        Kalman::new(LinearSystemNoInput::new(F, Q, SMatrix::zeros()))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix1x2, Matrix2, SMatrix, Vector1, Vector2};

    use crate::measurement::Measurement;

    use super::{KalmanLinearNoInput, KalmanUpdate};

    #[test]
    fn different_measurments() {
        let mut kf = KalmanLinearNoInput::new_linear_no_input(
            Matrix2::new(1.0, 0.1, 0.0, 1.0),
            SMatrix::identity(),
        );
        let m1 = Measurement {
            z: Vector1::new(5.0),
            H: Matrix1x2::new(1.0, 0.0),
            R: SMatrix::identity(),
        };
        let m2 = Measurement {
            z: Vector2::new(5.0, 1.4),
            H: Matrix2::identity(),
            R: SMatrix::identity(),
        };
        kf.predict();
        kf.update(&m1);
        kf.predict();
        kf.update(&m2);
    }
}
