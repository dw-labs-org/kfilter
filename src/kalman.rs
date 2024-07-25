use core::usize;

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    measurement::{LinearMeasurement, Measurement},
    system::{InputSystem, LinearNoInputSystem, LinearSystem, NoInputSystem, System},
};

/// Base trait for Kalman or wrappers around it
pub trait KalmanFilter<T, const N: usize> {
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a reference to the covariance
    fn covariance(&self) -> &SMatrix<T, N, N>;
}

/// Trait for a prediction of the next state for a system with no input
pub trait KalmanPredict<T, const N: usize> {
    /// predict the next state and return it. Also update covariance
    fn predict(&mut self) -> &SVector<T, N>;
}

/// Trait for a prediction of the next state for a system with input
pub trait KalmanPredictInput<T, const N: usize, const U: usize> {
    /// predict the next state with an input vector and return it. Also update covariance
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N>;
}

/// Trait for a kalman filter to update the state and covariance based on a measurement
pub trait KalmanUpdate<T, const N: usize, const M: usize, ME: Measurement<T, N, M>> {
    /// Optimally update state and covariance based on the measurement
    fn update(&mut self, measurement: &ME);
}

/// Representation of Kalman filter
#[derive(Debug)]
#[allow(non_snake_case)]
struct Kalman<T, const N: usize, const U: usize, S> {
    /// Covariance
    pub P: SMatrix<T, N, N>,
    /// The associated system containing the state vector x
    pub system: S,
}

/// Implement [KalmanFilter] for state and covariance access
impl<T, const N: usize, const U: usize, S> KalmanFilter<T, N> for Kalman<T, N, U, S>
where
    S: System<T, N, U>,
{
    fn state(&self) -> &SVector<T, N> {
        self.system.state()
    }

    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.P
    }
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

/// Linear Kalman constructor
impl<T, const N: usize, const U: usize> Kalman<T, N, U, LinearSystem<T, N, U>>
where
    T: RealField + Copy,
{
    #[allow(non_snake_case)]
    pub fn new_with_input(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>, B: SMatrix<T, N, U>) -> Self {
        Self {
            P: SMatrix::zeros(),
            system: LinearSystem::new(F, Q, B),
        }
    }
}

/// Linear Kalman constructor without inputs
impl<T, const N: usize> Kalman<T, N, 0, LinearNoInputSystem<T, N>>
where
    T: RealField + Copy,
{
    #[allow(non_snake_case)]
    pub fn new(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>) -> Self {
        Self {
            P: SMatrix::zeros(),
            system: LinearNoInputSystem::new(F, Q),
        }
    }
}

/// Kalman filter with a fixed shape measurement
pub struct Kalman1M<T, const N: usize, const U: usize, const M: usize, S, ME> {
    kalman: Kalman<T, N, U, S>,
    measurement: ME,
}

impl<T, const N: usize, const U: usize, const M: usize, S, ME> KalmanFilter<T, N>
    for Kalman1M<T, N, U, M, S, ME>
where
    S: System<T, N, U>,
{
    fn state(&self) -> &SVector<T, N> {
        self.kalman.state()
    }

    fn covariance(&self) -> &SMatrix<T, N, N> {
        self.kalman.covariance()
    }
}

/// Implement predict for Kalman1M with input
impl<T, const N: usize, const U: usize, const M: usize, S, ME> KalmanPredictInput<T, N, U>
    for Kalman1M<T, N, U, M, S, ME>
where
    T: RealField + Copy,
    S: InputSystem<T, N, U>,
    ME: Measurement<T, N, M>,
{
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        self.kalman.predict(u)
    }
}

/// Implement predict for Kalman1M with no input
impl<T, const N: usize, const M: usize, S, ME> KalmanPredict<T, N> for Kalman1M<T, N, 0, M, S, ME>
where
    T: RealField + Copy,
    S: NoInputSystem<T, N>,
    ME: Measurement<T, N, M>,
{
    fn predict(&mut self) -> &SVector<T, N> {
        self.kalman.predict()
    }
}

impl<T, const N: usize, const U: usize, const M: usize, S, ME> Kalman1M<T, N, U, M, S, ME>
where
    T: RealField + Copy,
    S: System<T, N, U>,
    ME: Measurement<T, N, M>,
{
    /// Update the state with a new measurement
    pub fn update(&mut self, z: SVector<T, M>) {
        self.measurement.set_measurement(z);
        self.kalman.update(&self.measurement);
    }
}

/// Linear system with input
impl<T, const N: usize, const U: usize, const M: usize>
    Kalman1M<T, N, U, M, LinearSystem<T, N, U>, LinearMeasurement<T, N, M>>
where
    T: RealField + Copy,
{
    /// Constructor for a linear kalman filter
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
    ) -> Self {
        Self {
            kalman: Kalman::new_with_input(F, Q, B),
            measurement: LinearMeasurement::new(H, R, SMatrix::zeros()),
        }
    }
}

/// Linear system without input
impl<T, const N: usize, const M: usize>
    Kalman1M<T, N, 0, M, LinearNoInputSystem<T, N>, LinearMeasurement<T, N, M>>
where
    T: RealField + Copy,
{
    /// Constructor for a linear kalman filter
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
    ) -> Self {
        Self {
            kalman: Kalman::new(F, Q),
            measurement: LinearMeasurement::new(H, R, SMatrix::zeros()),
        }
    }
}
