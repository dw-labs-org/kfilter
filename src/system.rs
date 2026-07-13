//! A system performs the modelling role in the Kalman filter, predicting the next state
//! based on the current state and any inputs. A system must implement the [System] trait
//! and either [InputSystem] or [NoInputSystem] accordingly.
//!
//! Typically one of
//! [LinearSystem], [LinearNoInputSystem] or [NonLinearSystem] will be used and these will
//! be created automatically in one of the [Kalman](crate::kalman::Kalman) or
//! [Kalman1M](crate::kalman::Kalman1M) constructors.

use nalgebra::{RealField, SMatrix, SVector};

// ================================== Traits =================================

/// Base trait for a system which must also implement [InputSystem] or [NoInputSystem].
pub trait System<T, const N: usize, const U: usize> {
    /// Get a reference to the process covariance matrix
    fn covariance(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a mutable reference to the state
    fn state_mut(&mut self) -> &mut SVector<T, N>;
}

/// A System which has (or can generate) a transition matrix/Jacobian
pub trait LinearisableSystem<T, const N: usize, const U: usize>: System<T, N, U> {
    /// Get the transition matrix (Jacobian)
    fn transition(&self) -> &SMatrix<T, N, N>;
    /// Get the transpose of the transition matrix
    fn transition_transpose(&self) -> &SMatrix<T, N, N>;
}

/// A System with an input.
pub trait InputSystem<T, const N: usize, const U: usize>: System<T, N, U> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        *self.state_mut() = self.predict(self.state(), &u);
        self.state()
    }
    /// Predict the next state based on the given state and input
    fn predict(&self, x: &SVector<T, N>, u: &SVector<T, U>) -> SVector<T, N>;
}

// ========================== Linear Systems =================================
/// A System without an input.
pub trait NoInputSystem<T, const N: usize>: System<T, N, 0> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self) -> &SVector<T, N> {
        *self.state_mut() = self.predict(self.state());
        self.state()
    }
    /// Predict the next state based on the given state
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, N>;
}

/// A linear system with an input.
/// Defined by the transition matrix F, control matrix B and covariance matrix Q.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
// #[cfg_attr(feature = "defmt", defmt(bound(T: ::defmt::Format, [T;N]: ::defmt::Format)))]
pub struct LinearSystem<T: RealField, const N: usize, const U: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    F_t: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
    B: SMatrix<T, N, U>,
}

#[allow(non_snake_case)]
impl<T: RealField + Copy, const N: usize, const U: usize> LinearSystem<T, N, U> {
    /// Create new [LinearSystem] from the transition matrix F, process covariance Q
    /// and control matrix B.
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
    ) -> Self {
        LinearSystem {
            x: x_initial,
            F,
            F_t: F.transpose(),
            Q,
            B,
        }
    }
    /// Set a new transition matrix, also updating the transpose
    pub fn set_transition(&mut self, transition: SMatrix<T, N, N>) {
        self.F_t = transition.transpose();
        self.F = transition;
    }
    /// Get a mutable reference to the process covariance matrix
    pub fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        &mut self.Q
    }
}

/// Implement [System] for [LinearSystem]
impl<T: RealField + Copy, const N: usize, const U: usize> System<T, N, U>
    for LinearSystem<T, N, U>
{
    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.Q
    }

    fn state(&self) -> &SVector<T, N> {
        &self.x
    }

    fn state_mut(&mut self) -> &mut SVector<T, N> {
        &mut self.x
    }
}

impl<T: RealField + Copy, const N: usize, const U: usize> LinearisableSystem<T, N, U>
    for LinearSystem<T, N, U>
{
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }
    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }
}

/// impl [InputSystem] for [LinearSystem]
impl<T: RealField + Copy, const N: usize, const U: usize> InputSystem<T, N, U>
    for LinearSystem<T, N, U>
{
    fn predict(&self, x: &SVector<T, N>, u: &SVector<T, U>) -> SVector<T, N> {
        self.F * x + self.B * u
    }
}

/// A linear system with no input.
/// Defined by the transition matrix F and covariance matrix Q.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct LinearNoInputSystem<T: RealField, const N: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    F_t: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
}

#[allow(non_snake_case)]
impl<T: RealField + Copy, const N: usize> LinearNoInputSystem<T, N> {
    /// Create new [LinearNoInputSystem] from the transition matrix F and process covariance Q.
    pub fn new(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>, x_initial: SVector<T, N>) -> Self {
        LinearNoInputSystem {
            x: x_initial,
            F,
            F_t: F.transpose(),
            Q,
        }
    }

    /// Set a new transition matrix, also updating the transpose
    pub fn set_transition(&mut self, transition: SMatrix<T, N, N>) {
        self.F_t = transition.transpose();
        self.F = transition;
    }
    /// Get a mutable reference to the process covariance matrix
    pub fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        &mut self.Q
    }
}

impl<T: RealField + Copy, const N: usize> System<T, N, 0> for LinearNoInputSystem<T, N> {
    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.Q
    }

    fn state(&self) -> &SVector<T, N> {
        &self.x
    }

    fn state_mut(&mut self) -> &mut SVector<T, N> {
        &mut self.x
    }
}

impl<T: RealField + Copy, const N: usize> LinearisableSystem<T, N, 0>
    for LinearNoInputSystem<T, N>
{
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }
    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }
}

impl<T: RealField + Copy, const N: usize> NoInputSystem<T, N> for LinearNoInputSystem<T, N> {
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, N> {
        self.F * x
    }
}

// ========================== Non-Linear Systems ==============================

/// Type returned from [StepFunction].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct StepReturn<T: RealField, const N: usize> {
    /// The new state (x).
    pub state: SVector<T, N>,
    /// The jacobian of the transition (F).
    pub jacobian: SMatrix<T, N, N>,
    /// The process covariance (Q).
    pub covariance: SMatrix<T, N, N>,
}

/// A function that takes the current state and input,
/// returning the next state, its covariance and the jacobian.
/// Used for the state transition in a [NonLinearSystem].
pub type StepFunction<T, const N: usize, const U: usize> =
    fn(SVector<T, N>, SVector<T, U>) -> StepReturn<T, N>;

/// A non-linear system with an input.
/// Defined by a [StepFunction] that performs state transition and jacobian and covariance calculation.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct NonLinearSystem<T: RealField, const N: usize, const U: usize> {
    /// System state
    x: SVector<T, N>,
    /// Process Covariance, updated after step_fn() call.
    Q: SMatrix<T, N, N>,
    /// Jacobian, updated after jacobian() call.
    F: SMatrix<T, N, N>,
    /// Jacobian transpose, updated after jacobian() call.
    F_t: SMatrix<T, N, N>,
    /// Function that steps from current state to next with an input
    /// Returns the new state, the jacobian and the process covariance
    // #[cfg_attr(feature = "serde", serde(skip))]
    step_fn: StepFunction<T, N, U>,
}

impl<T: RealField, const N: usize, const U: usize> NonLinearSystem<T, N, U> {
    /// Create a new [NonLinearSystem] using a [StepFunction]
    pub fn new(step_fn: StepFunction<T, N, U>, x_initial: SVector<T, N>) -> Self {
        Self {
            x: x_initial,
            Q: SMatrix::zeros(),
            F: SMatrix::zeros(),
            F_t: SMatrix::zeros(),
            step_fn,
        }
    }
}

impl<T: RealField + Copy, const N: usize, const U: usize> System<T, N, U>
    for NonLinearSystem<T, N, U>
{
    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.Q
    }

    fn state(&self) -> &SVector<T, N> {
        &self.x
    }

    fn state_mut(&mut self) -> &mut SVector<T, N> {
        &mut self.x
    }
}

impl<T: RealField + Copy, const N: usize, const U: usize> LinearisableSystem<T, N, U>
    for NonLinearSystem<T, N, U>
{
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }
    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }
}

impl<T: RealField + Copy, const N: usize, const U: usize> InputSystem<T, N, U>
    for NonLinearSystem<T, N, U>
{
    fn step(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        // Get updated state, jacobian and process covariance
        let r = (self.step_fn)(self.x, u);
        self.x = r.state;
        self.F = r.jacobian;
        self.F_t = self.F.transpose();
        self.Q = r.covariance;
        // Return state
        self.state()
    }

    fn predict(&self, x: &SVector<T, N>, u: &SVector<T, U>) -> SVector<T, N> {
        let r = (self.step_fn)(*x, *u);
        r.state
    }
}

/// Convert a state matrix from continuous time (A) to discrete time (F) using the
/// exact zero-order-hold transform: `F = exp(A * dt)`.
///
/// The matrix exponential is computed with scaling-and-squaring: `A * dt` is halved
/// until its norm is small, a truncated Taylor series is evaluated on the scaled
/// matrix, and the result is squared back up. This is exact in the limit of the
/// series and accurate to within a few ULPs in practice, at the cost of more
/// computation than [euler]. Prefer [euler] where the extra accuracy isn't needed.
pub fn zero_order_hold<T: RealField + Copy, const N: usize>(
    state_matrix: SMatrix<T, N, N>,
    timestep: T,
) -> SMatrix<T, N, N> {
    matrix_exp(state_matrix * timestep)
}

/// Convert a state matrix from continuous time (A) to discrete time (F) using a
/// first-order (Euler) approximation of the zero-order-hold transform: `F = I + A * dt`.
///
/// Cheaper than [zero_order_hold] but only accurate for small `dt` relative to the
/// magnitude of `A`; prefer [zero_order_hold] for fast dynamics or larger timesteps.
pub fn euler<T: RealField, const N: usize>(
    state_matrix: SMatrix<T, N, N>,
    timestep: T,
) -> SMatrix<T, N, N> {
    SMatrix::identity() + state_matrix * timestep
}

/// Maximum number of scaling-and-squaring doublings, bounding the loop below even
/// for pathologically large inputs.
const MATRIX_EXP_MAX_SCALE: u32 = 64;
/// Number of Taylor series terms evaluated on the scaled matrix. After scaling,
/// the matrix norm is <= 0.5, so this is accurate to well beyond `f64` precision.
const MATRIX_EXP_TAYLOR_TERMS: usize = 12;

/// Compute the matrix exponential `exp(a)` via scaling-and-squaring.
fn matrix_exp<T: RealField + Copy, const N: usize>(a: SMatrix<T, N, N>) -> SMatrix<T, N, N> {
    let half = T::from_f64(0.5).unwrap();

    // Find s such that ||a|| / 2^s <= 0.5, capped to avoid unbounded looping.
    let mut scale_power = 0u32;
    let mut scaled_norm = a.norm();
    while scaled_norm > half && scale_power < MATRIX_EXP_MAX_SCALE {
        scaled_norm *= half;
        scale_power += 1;
    }
    let two = T::from_f64(2.0).unwrap();
    let mut scale = T::one();
    for _ in 0..scale_power {
        scale *= two;
    }
    let scaled = a / scale;

    // Taylor series for exp(scaled) = sum_{k=0}^{K} scaled^k / k!
    let mut term = SMatrix::<T, N, N>::identity();
    let mut result = SMatrix::<T, N, N>::identity();
    for k in 1..=MATRIX_EXP_TAYLOR_TERMS {
        term = term * scaled / T::from_usize(k).unwrap();
        result += term;
    }

    // Undo the scaling by repeated squaring.
    for _ in 0..scale_power {
        result *= result;
    }
    result
}

#[cfg(test)]
mod discretization_tests {
    use super::*;
    use nalgebra::Matrix2;

    #[test]
    fn zero_order_hold_matches_analytic_rotation() {
        // A skew-symmetric generator produces an exact rotation under exp(A*dt),
        // which the first-order Euler approximation cannot reproduce exactly.
        let omega = 1.3_f64;
        let dt = 0.2_f64;
        let a = Matrix2::new(0.0, -omega, omega, 0.0);
        let f = zero_order_hold(a, dt);
        let (s, c) = (omega * dt).sin_cos();
        let expected = Matrix2::new(c, -s, s, c);
        assert!((f - expected).norm() < 1e-9);
        // A true rotation matrix is orthogonal with determinant 1.
        assert!((f.determinant() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_order_hold_matches_scalar_exp() {
        let a = nalgebra::Matrix1::new(-0.7_f64);
        let dt = 0.5_f64;
        let f = zero_order_hold(a, dt);
        assert!((f[(0, 0)] - (-0.7_f64 * 0.5).exp()).abs() < 1e-9);
    }

    #[test]
    fn euler_matches_first_order_expansion() {
        let a = Matrix2::new(1.0, 2.0, 0.5, -1.0);
        let dt = 0.05_f64;
        let f = euler(a, dt);
        let expected = Matrix2::identity() + a * dt;
        assert_eq!(f, expected);
    }
}
