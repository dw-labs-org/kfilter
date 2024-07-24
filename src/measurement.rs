use nalgebra::{RealField, SMatrix, SVector};

pub trait Measurement<T, const N: usize, const M: usize> {
    /// Calculate the innovation (y) based on the measurement and the observation mapping
    fn innovation(&self, x: &SVector<T, N>) -> SVector<T, M>;
    /// Get the measurement covariance
    fn covariance(&self) -> &SMatrix<T, M, M>;
    /// Get the observation matrix
    fn observation(&self) -> &SMatrix<T, M, N>;
    /// Get the observation matrix transpose
    fn observation_transpose(&self) -> &SMatrix<T, N, M>;
}

pub struct LinearMeasurement<T, const N: usize, const M: usize> {
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    pub R: SMatrix<T, M, M>,
}

impl<T: RealField, const N: usize, const M: usize> LinearMeasurement<T, N, M> {
    pub fn new(H: SMatrix<T, M, N>, R: SMatrix<T, M, M>, z: SVector<T, M>) -> Self {
        Self {
            z,
            H_t: H.transpose(),
            H,
            R,
        }
    }

    pub fn set_observation(&mut self, H: SMatrix<T, M, N>) {
        self.H_t = H.transpose();
        self.H = H;
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> Measurement<T, N, M>
    for LinearMeasurement<T, N, M>
{
    fn innovation(&self, x: &SVector<T, N>) -> SVector<T, M> {
        self.z - (self.H * x)
    }

    fn covariance(&self) -> &SMatrix<T, M, M> {
        &self.R
    }

    fn observation(&self) -> &SMatrix<T, M, N> {
        &self.H
    }

    fn observation_transpose(&self) -> &SMatrix<T, N, M> {
        &self.H_t
    }
}
