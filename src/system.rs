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
    /// Get the transition matrix (Jacobian)
    fn transition(&self) -> &SMatrix<T, N, N>;
    /// Get the transpose of the transition matrix
    fn transition_transpose(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the process covariance matrix
    fn covariance(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a mutable reference to the state
    fn state_mut(&mut self) -> &mut SVector<T, N>;
}

/// A System with an input.
pub trait InputSystem<T, const N: usize, const U: usize>: System<T, N, U> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self, u: SVector<T, U>) -> &SVector<T, N>;
}

// ========================== Linear Systems =================================
/// A System without an input.
pub trait NoInputSystem<T, const N: usize>: System<T, N, 0> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self) -> &SVector<T, N>;
}

/// A linear system with an input.
/// Defined by the transition matrix F, control matrix B and covariance matrix Q.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct LinearSystem<T, const N: usize, const U: usize> {
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
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }
    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }

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

/// impl [InputSystem] for [LinearSystem]
impl<T: RealField + Copy, const N: usize, const U: usize> InputSystem<T, N, U>
    for LinearSystem<T, N, U>
{
    fn step(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        self.x = self.F * self.x + self.B * u;
        &self.x
    }
}

/// A linear system with no input.
/// Defined by the transition matrix F and covariance matrix Q.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct LinearNoInputSystem<T, const N: usize> {
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
}

impl<T: RealField + Copy, const N: usize, const U: usize> System<T, N, U>
    for LinearNoInputSystem<T, N>
{
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }
    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }

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

impl<T: RealField + Copy, const N: usize> NoInputSystem<T, N> for LinearNoInputSystem<T, N> {
    fn step(&mut self) -> &SVector<T, N> {
        self.x = self.F * self.x;
        &self.x
    }
}

// ========================== Non-Linear Systems ==============================

/// Type returned from [StepFunction].
#[derive(Debug, Clone)]
pub struct StepReturn<T, const N: usize> {
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
pub struct NonLinearSystem<T, const N: usize, const U: usize> {
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
    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
    }

    fn transition_transpose(&self) -> &SMatrix<T, N, N> {
        &self.F_t
    }

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
}
