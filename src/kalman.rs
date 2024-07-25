use std::usize;

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    measurement::Measurement,
    system::{InputSystem, NoInputSystem, System},
};

/// Trait for a prediction of the next state for a system with no input
trait KalmanPredict<T, const N: usize> {
    /// predict the next state and return it. Also update covariance
    fn predict(&mut self) -> &SVector<T, N>;
}

/// Trait for a prediction of the next state for a system with input
trait KalmanPredictInput<T, const N: usize, const U: usize> {
    /// predict the next state with an input vector and return it. Also update covariance
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N>;
}

/// Trait for a kalman filter to update the state and covariance based on a measurement
trait KalmanUpdate<T, const N: usize, const M: usize, ME: Measurement<T, N, M>> {
    /// Optimally update state and covariance based on the measurement
    fn update(&mut self, measurement: &ME);
}

/// Representation of Kalman filter
#[derive(Debug)]
#[allow(non_snake_case)]
struct Kalman<T, const N: usize, const U: usize, S: System<T, N, U>> {
    /// Covariance
    pub P: SMatrix<T, N, N>,
    /// The associated system containing the state vector x
    pub system: S,
}

/// Implement the predict stage for a system with no input
impl<T: RealField + Copy, const N: usize, S: NoInputSystem<T, N>> KalmanPredict<T, N>
    for Kalman<T, N, 0, S>
{
    fn predict(&mut self) -> &SVector<T, N> {
        self.system.step();
        self.P = self.system.transition() * self.P * self.system.transition_transpose()
            + self.system.covariance();
        self.system.state()
    }
}

/// Implement the predict stage for a system with input
impl<T: RealField + Copy, const N: usize, const U: usize, S: InputSystem<T, N, U>>
    KalmanPredictInput<T, N, U> for Kalman<T, N, U, S>
{
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        self.system.step(u);
        self.P = self.system.transition() * self.P * self.system.transition_transpose()
            + self.system.covariance();
        self.system.state()
    }
}

/// Implement the update stage
impl<
        T: RealField + Copy,
        const N: usize,
        const M: usize,
        const U: usize,
        S: System<T, N, U>,
        ME: Measurement<T, N, M>,
    > KalmanUpdate<T, N, M, ME> for Kalman<T, N, U, S>
{
    #[allow(non_snake_case)]
    fn update(&mut self, measurement: &ME) {
        // innovation
        let y = measurement.innovation(self.system.state());
        // innovation covariance
        let S = measurement.observation() * self.P * measurement.observation_transpose()
            + measurement.covariance();
        // Optimal gain
        let K = self.P * measurement.observation_transpose() * S.try_inverse().unwrap();
        // state update
        *self.system.state_mut() += K * y;
        // covariance update
        self.P = (SMatrix::identity() - K * measurement.observation()) * self.P;
    }
}

// / Kalman filter with a fixed shape measurement
// struct Kalman1M<
//     T,
//     const N: usize,
//     const U: usize,
//     const M: usize,
//     S: System<T, N, U>,
//     ME: Measurement<T, N, M>,
// > {
//     kalman: Kalman<T, N, U, S>,
//     measurement: ME,
// }

// impl<T: RealField + Copy, const N: usize, const M: usize>
//     Kalman1M<T, N, 0, M, LinearSystemNoInput<T, N>, LinearMeasurement<T, N, M>>
// {
//     pub fn new_linear_no_input(
//         F: SMatrix<T, N, N>,
//         Q: SMatrix<T, N, N>,
//         H: SMatrix<T, M, N>,
//         R: SMatrix<T, M, M>,
//     ) -> Self {
//         Kalman1M {
//             kalman: Kalman::new_linear_no_input(F, Q),
//             measurement: LinearMeasurement::new(H, R, SMatrix::zeros()),
//         }
//     }

//     pub fn predict(&mut self) {
//         self.kalman.predict();
//     }

//     pub fn update(&mut self, z: SVector<T, M>) {
//         self.measurement.z = z;
//         self.kalman.update(&self.measurement);
//     }
// }

// type KalmanLinearNoInput<T, const N: usize> = Kalman<T, N, 0, LinearSystemNoInput<T, N>>;

// impl<T: RealField + Copy, const N: usize> KalmanLinearNoInput<T, N> {
//     pub fn new_linear_no_input(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>) -> Self {
//         Kalman::new(LinearSystemNoInput::new(F, Q, SMatrix::zeros()))
//     }
// }
