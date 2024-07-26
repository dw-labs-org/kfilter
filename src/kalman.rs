//! Implementation of the Kalman and Extended Kalman Filter. The [Kalman] type
//! performs state predictions based on a generic [System] and updates the state based on
//! a [Measurement].

use core::usize;

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    measurement::{LinearMeasurement, Measurement},
    system::{
        InputSystem, LinearNoInputSystem, LinearSystem, NoInputSystem, NonLinearSystem,
        StepFunction, System,
    },
};

/// Base trait for [Kalman] or wrappers around it. Allows viewing the state and covariance
/// and modifying the covariance.
/// Modifying the covariance can be necessary if it becomes non symmetric.
pub trait KalmanFilter<T, const N: usize> {
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a reference to the covariance
    fn covariance(&self) -> &SMatrix<T, N, N>;
    /// Get a mutable reference to the covariance
    fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N>;
}

/// Trait for a prediction of the next state for a system with no input.
/// Uses the underlying [System] to perform the prediction and updates
/// the covariance according to P = F * P * F_t + Q.
pub trait KalmanPredict<T, const N: usize> {
    /// predict the next state and return it. Also update covariance
    fn predict(&mut self) -> &SVector<T, N>;
}

/// Trait for a prediction of the next state for a system with input.
/// Uses the underlying [System] to perform the prediction and updates
/// the covariance according to P = F * P * F_t + Q.
pub trait KalmanPredictInput<T, const N: usize, const U: usize> {
    /// predict the next state with an input vector and return it. Also update covariance
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N>;
}

/// Trait for a kalman filter to update the state and covariance based on a measurement.
/// Optimally updates using the Kalman gain.
pub trait KalmanUpdate<T, const N: usize, const M: usize, ME: Measurement<T, N, M>> {
    /// Optimally update state and covariance based on the measurement
    fn update(&mut self, measurement: &ME);
}

/// Representation of the Kalman filter. This is the base type that can be interacted
/// with directly for full control or using a wrapper like [Kalman1M] that simplifies
/// the measurement update.
///
/// ## Usage
/// There are a few constructors that can create the underlying system automatically:
/// - Linear system without inputs: [Kalman::new].
/// - Linear system with inputs: [Kalman::new_with_input].
/// - Non-linear system with inputs: [Kalman::new_ekf_with_input].
///
/// Or for a more configurable setup: [Kalman::new_custom].
///
/// Then the state can be predicted with or without input ([predict](#method.predict)
/// or [predict](#method.predict-1)) and updated using a [Measurement].
/// ```
/// use nalgebra::{SMatrix, Matrix2, Matrix1, Matrix1x2};
/// use kfilter::{
/// kalman::{Kalman, KalmanPredict, KalmanUpdate},
/// measurement::{LinearMeasurement,Measurement},
/// };
/// // Create a linear Kalman filter with Q = I and zero initial state and covariance.
/// let mut kalman = Kalman::new(Matrix2::new(1.0,0.1,0.0,1.0), SMatrix::identity(), SMatrix::zeros(), SMatrix::zeros());
/// // Create a new linear measurement
/// let mut m = LinearMeasurement::new(
///     Matrix1x2::new(1.0, 0.0),
///     SMatrix::identity(),
///     Matrix1::new(0.0),
/// );
/// let predicted_x = kalman.predict();
/// let updated_x = kalman.update(&m);
/// // Run 10 timesteps
/// for i in 0..10 {
///     kalman.predict();
///     m.set_measurement(Matrix1::new(i as f64));
///     kalman.update(&m);
/// }
/// ```
///
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct Kalman<T, const N: usize, const U: usize, S> {
    /// Covariance
    P: SMatrix<T, N, N>,
    /// The associated [System] containing the state vector x.
    /// This is public so changes can be made on the fly, which may be useful
    /// for custom systems.
    pub system: S,
}

impl<T, const N: usize, const U: usize, S> Kalman<T, N, U, S>
where
    S: System<T, N, U>,
{
    /// Create a new Kalman Filter using a [System]
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>) -> Self {
        Self {
            P: initial_covariance,
            system,
        }
    }
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

    fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        &mut self.P
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
    /// Create a new linear kalman filter with a system model with inputs.
    /// State transition F, process noise Q, control B and state x
    /// and covariance P initial conditions.
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            P: P_initial,
            system: LinearSystem::new(F, Q, B, x_initial),
        }
    }
}

/// Linear Kalman constructor without inputs
impl<T, const N: usize> Kalman<T, N, 0, LinearNoInputSystem<T, N>>
where
    T: RealField + Copy,
{
    #[allow(non_snake_case)]
    /// Create a new linear kalman filter with a system model without inputs.
    /// State transition F, process noise Q and state x
    /// and covariance P initial conditions.
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            P: P_initial,
            system: LinearNoInputSystem::new(F, Q, x_initial),
        }
    }
}

/// NonLinear Kalman constructor
impl<T, const N: usize, const U: usize> Kalman<T, N, U, NonLinearSystem<T, N, U>>
where
    T: RealField + Copy,
{
    /// Create a new EKF with a non-linear system
    #[allow(non_snake_case)]
    pub fn new_ekf_with_input(
        step_fn: StepFunction<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            P: P_initial,
            system: NonLinearSystem::new(step_fn, x_initial),
        }
    }
}

/// Kalman filter with a fixed shape measurement. Useful for systems with a single sensor,
/// or multiple sensors sampled at the same rate that perform a single update step. Use
/// [Kalman] for sensors with different sample rates.
///
/// ## Usage
/// There are a few constuctors that simplify the creation of the underlying [Kalman] filter,
/// its [System] and the fixed [Measurement]:
/// - Linear system without inputs, linear measurement: [Kalman1M::new].
/// - Linear system with inputs, linear measurement: [Kalman1M::new_with_input].
/// - Non-linear system with inputs, linear measurement: [Kalman1M::new_ekf_with_input].
///
/// Or for more configuration, use [Kalman1M::new_custom].
///
/// The [precict](Kalman1M::predict) and [update](Kalman1M::update) are then used to run
/// the filter.
/// ```
/// use kfilter::kalman::{Kalman1M, KalmanPredict};
/// use nalgebra::{Matrix1, Matrix1x2, Matrix2, SMatrix};
/// // Create a new 2 state kalman filter
/// let mut k = Kalman1M::new(
///     Matrix2::new(1.0, 0.1, 0.0, 1.0),   // F
///     SMatrix::identity(),                // Q
///     Matrix1x2::new(1.0, 0.0),           // H
///     SMatrix::identity(),                // R
///     SMatrix::zeros(),                   // x
/// );
/// // Run 100 timesteps
/// for i in 0..100 {
///     k.predict();
///     k.update(Matrix1::new(i as f64));
/// }
/// ```
pub struct Kalman1M<T, const N: usize, const U: usize, const M: usize, S, ME> {
    kalman: Kalman<T, N, U, S>,
    measurement: ME,
}

/// Custom system and measurement
impl<T, const N: usize, const U: usize, const M: usize, S, ME> Kalman1M<T, N, U, M, S, ME>
where
    S: System<T, N, U>,
    ME: Measurement<T, N, M>,
{
    /// Create a new Kalman filter based on supplied [System] and [Measurement] types.
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>, measurement: ME) -> Self {
        Self {
            kalman: Kalman::new_custom(system, initial_covariance),
            measurement,
        }
    }
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

    fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        self.kalman.covariance_mut()
    }
}

/// Implement predict for [Kalman1M] with input
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

/// Implement predict for [Kalman1M] with no input
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
    /// Initial covariance is set to Q
    #[allow(non_snake_case)]
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
        x_initial: SVector<T, N>,
    ) -> Self {
        Self {
            kalman: Kalman::new_with_input(F, Q, B, x_initial, Q),
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
    /// Initial covariance is set to Q
    #[allow(non_snake_case)]
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
        x_initial: SVector<T, N>,
    ) -> Self {
        Self {
            kalman: Kalman::new(F, Q, x_initial, Q),
            measurement: LinearMeasurement::new(H, R, SMatrix::zeros()),
        }
    }
}

/// Non-Linear system with input
impl<T, const N: usize, const U: usize, const M: usize>
    Kalman1M<T, N, U, M, NonLinearSystem<T, N, U>, LinearMeasurement<T, N, M>>
where
    T: RealField + Copy,
{
    /// An EKF with a nonlinear system defined by step_fn but with a linear measurement
    /// Use [Kalman1M::new_custom] for a nonlinear measurement.
    #[allow(non_snake_case)]
    pub fn new_ekf_with_input(
        step_fn: StepFunction<T, N, U>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            kalman: Kalman::new_ekf_with_input(step_fn, x_initial, P_initial),
            measurement: LinearMeasurement::new(H, R, SMatrix::zeros()),
        }
    }
}
