use nalgebra::{RealField, SMatrix, SVector};

pub trait System<T, const N: usize> {
    /// transition to the next state, returning a reference to it
    fn step(&mut self) -> &SVector<T, N>;
    /// Get a reference to the transition matrix (Jacobian)
    fn transition(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the process covariance matrix
    fn covariance(&self) -> &SMatrix<T, N, N>;
    /// Get a reference to the state
    fn state(&self) -> &SVector<T, N>;
    /// Get a mutable reference to the state
    fn state_mut(&mut self) -> &mut SVector<T, N>;
}

struct LinearSystem<T, const N: usize> {
    x: SVector<T, N>,
    F: SMatrix<T, N, N>,
    Q: SMatrix<T, N, N>,
}

impl<T: RealField + Copy, const N: usize> System<T, N> for LinearSystem<T, N> {
    fn step(&mut self) -> &SVector<T, N> {
        self.x = self.F * self.x;
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
