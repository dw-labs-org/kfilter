use nalgebra::{RealField, SMatrix, SVector};

pub trait Measurement<T, const N: usize, const M: usize> {
    /// Calculate the innovation (y) based on the measurement and the observation mapping
    fn innovation(&self, x: &SVector<T, N>) -> SVector<T, M>;
    /// Get the measurement covariance
    fn covariance(&self) -> &SMatrix<T, M, M>;
    /// Get the observation matrix
    fn observation(&self) -> &SMatrix<T, M, N>;
}

pub struct LinearMeasurement<T, const N: usize, const M: usize> {
    pub z: SVector<T, M>,
    pub H: SMatrix<T, M, N>,
    pub R: SMatrix<T, M, M>,
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
}
