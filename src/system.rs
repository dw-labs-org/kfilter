use nalgebra::{RealField, SMatrix, SVector};

pub type LinearSystemNoInput<T, const N: usize> = LinearSystem<T, N, 0>;

pub trait System<T, const N: usize, const U: usize> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self) -> &SVector<T, N>;
    /// transition to the next state, returning a reference to it
    fn step_with_input(&mut self, u: SVector<T, U>) -> &SVector<T, N>;
    /// Get a reference to the transition matrix (Jacobian)
    fn transition(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the process covariance matrix
    fn covariance(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a mutable reference to the state
    fn state_mut(&mut self) -> &mut SVector<T, N>;
}

pub struct LinearSystem<T, const N: usize, const U: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
    B: SMatrix<T, N, U>,
}

impl<T: RealField + Copy, const N: usize, const U: usize> LinearSystem<T, N, U> {
    pub fn new(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>, B: SMatrix<T, N, U>) -> Self {
        LinearSystem {
            x: SMatrix::zeros(),
            F,
            Q,
            B,
        }
    }
}

impl<T: RealField + Copy, const N: usize> LinearSystem<T, N, 0> {
    pub fn new_no_input(F: SMatrix<T, N, N>, Q: SMatrix<T, N, N>) -> Self {
        LinearSystem {
            x: SMatrix::zeros(),
            F,
            Q,
            B: SMatrix::zeros(),
        }
    }
}

impl<T: RealField + Copy, const N: usize, const U: usize> System<T, N, U>
    for LinearSystem<T, N, U>
{
    fn step(&mut self) -> &SVector<T, N> {
        self.x = self.F * self.x;
        &self.x
    }
    fn step_with_input(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        self.x = self.F * self.x + self.B * u;
        &self.x
    }

    fn transition(&self) -> &SMatrix<T, N, N> {
        &self.F
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
