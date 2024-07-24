use std::usize;

use nalgebra::{Matrix, RealField, SMatrix, SVector};

use crate::{
    measurement::{LinearMeasurement, Measurement},
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

trait KalmanUpdate<T, const N: usize, const M: usize, ME: Measurement<T, N, M>> {
    fn update(&mut self, measurement: &ME);
}

impl<
        T: RealField + Copy,
        const N: usize,
        const M: usize,
        const U: usize,
        S: System<T, N, U>,
        ME: Measurement<T, N, M>,
    > KalmanUpdate<T, N, M, ME> for Kalman<T, N, U, S>
{
    fn update(&mut self, measurement: &ME) {
        let H_transpose = measurement.observation().transpose();
        // innovation
        let y = measurement.innovation(self.system.state());
        // innovation covariance
        let S = measurement.observation() * self.P * H_transpose + measurement.covariance();
        // Optimal gain
        let K = self.P * H_transpose * S.try_inverse().unwrap();
        // state update
        *self.system.state_mut() += K * y;
        // covariance update
        self.P = (SMatrix::identity() - K * measurement.observation()) * self.P;
    }
}

/// Kalman filter with a fixed shape measurement
struct Kalman1M<
    T,
    const N: usize,
    const U: usize,
    const M: usize,
    S: System<T, N, U>,
    ME: Measurement<T, N, M>,
> {
    kalman: Kalman<T, N, U, S>,
    measurement: ME,
}

impl<T: RealField + Copy, const N: usize, const M: usize>
    Kalman1M<T, N, 0, M, LinearSystemNoInput<T, N>, LinearMeasurement<T, N, M>>
{
    pub fn new_linear_no_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
    ) -> Self {
        Kalman1M {
            kalman: Kalman::new_linear_no_input(F, Q),
            measurement: LinearMeasurement {
                z: SMatrix::zeros(),
                H,
                R,
            },
        }
    }

    pub fn predict(&mut self) {
        self.kalman.predict();
    }

    pub fn update(&mut self, z: SVector<T, M>) {
        self.measurement.z = z;
        self.kalman.update(&self.measurement);
    }
}

type KalmanLinearNoInput<T, const N: usize> = Kalman<T, N, 0, LinearSystemNoInput<T, N>>;

impl<T: RealField + Copy, const N: usize> KalmanLinearNoInput<T, N> {
    pub fn new_linear_no_input(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>) -> Self {
        Kalman::new(LinearSystemNoInput::new(F, Q, SMatrix::zeros()))
    }
}
