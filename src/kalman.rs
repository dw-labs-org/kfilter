use nalgebra::{RealField, SMatrix, SVector};

use crate::system::{LinearSystemNoInput, System};

type KalmanLinearNoInput<T, const N: usize, const M: usize> =
    Kalman<T, N, M, 0, LinearSystemNoInput<T, N>>;

impl<T: RealField + Copy, const N: usize, const M: usize> KalmanLinearNoInput<T, N, M> {
    pub fn new_linear_no_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
    ) -> Self {
        Kalman::new(LinearSystemNoInput::new(F, Q, SMatrix::zeros()), H, R)
    }
}

/// Representation of Kalman filter
#[derive(Debug)]
struct Kalman<T, const N: usize, const M: usize, const U: usize, S: System<T, N, U>> {
    /// Covariance
    pub P: SMatrix<T, N, N>,
    /// The associated system
    pub system: S,
    /// Observation Matrix
    H: SMatrix<T, M, N>,
    /// Measurment covariance
    R: SMatrix<T, M, M>,
}

impl<T: RealField + Copy, const N: usize, const M: usize, const U: usize, S: System<T, N, U>>
    Kalman<T, N, M, U, S>
{
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

    pub fn update(&mut self, z: &SVector<T, M>) {
        // innovation
        let y = z - (self.H * self.system.state());
        // innovation covariance
        let S = self.H * self.P * self.H.transpose() + self.R;
        // Optimal gain
        let K = self.P * self.H.transpose() * S.try_inverse().unwrap();
        // state update
        *self.system.state_mut() += K * y;
        // covariance update
        self.P = (SMatrix::identity() - K * self.H) * self.P;
    }

    pub fn new(system: S, H: SMatrix<T, M, N>, R: SMatrix<T, M, M>) -> Self {
        Kalman {
            P: SMatrix::zeros(),
            system,
            H,
            R,
        }
    }
}
