use nalgebra::{SMatrix, SVector};

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
