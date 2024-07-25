use nalgebra::{RealField, SMatrix, SVector};

pub type LinearSystemNoInput<T, const N: usize> = LinearSystem<T, N, 0>;

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

/// A Linear system with an input matrix B
#[allow(non_snake_case)]
pub struct LinearSystem<T, const N: usize, const U: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    F_t: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
    B: SMatrix<T, N, U>,
}

#[allow(non_snake_case)]
impl<T: RealField + Copy, const N: usize, const U: usize> LinearSystem<T, N, U> {
    pub fn new(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>, B: SMatrix<T, N, U>) -> Self {
        LinearSystem {
            x: SMatrix::zeros(),
            F,
            F_t: F.transpose(),
            Q,
            B,
        }
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

/// A Linear system with no input
#[allow(non_snake_case)]
pub struct LinearNoInputSystem<T, const N: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    F_t: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
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

/// Type returned from a call to [StepInput] and [Step] functions.
pub struct StepReturn<T, const N: usize> {
    pub state: SVector<T, N>,
    pub jacobian: SMatrix<T, N, N>,
    pub covariance: SMatrix<T, N, N>,
}

/// A function that takes the current state and input, returning the next state and its covariance
type StepInput<T, const N: usize, const U: usize> =
    fn(SVector<T, N>, SVector<T, U>) -> StepReturn<T, N>;

/// A non-linear system with inputs
#[allow(non_snake_case)]
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
    step_fn: StepInput<T, N, U>,
}

impl<T: RealField, const N: usize, const U: usize> NonLinearSystem<T, N, U> {
    pub fn new(step_fn: StepInput<T, N, U>) -> Self {
        Self {
            x: SMatrix::zeros(),
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
