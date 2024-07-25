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
    /// Set the measurment (z) value
    fn set_measurement(&mut self, z: SVector<T, M>);
}

#[allow(non_snake_case)]
pub struct LinearMeasurement<T, const N: usize, const M: usize> {
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    pub R: SMatrix<T, M, M>,
}

impl<T: RealField, const N: usize, const M: usize> LinearMeasurement<T, N, M> {
    #[allow(non_snake_case)]
    pub fn new(H: SMatrix<T, M, N>, R: SMatrix<T, M, M>, z: SVector<T, M>) -> Self {
        Self {
            z,
            H_t: H.transpose(),
            H,
            R,
        }
    }

    #[allow(non_snake_case)]
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

    fn set_measurement(&mut self, z: SVector<T, M>) {
        self.z = z;
    }
}

/// Function that returns the predicted value of the measurment based on the current state
pub type Prediction<T, const N: usize, const M: usize> = fn(&SVector<T, N>) -> SVector<T, M>;

/// A non linear measurement that uses a prediction function to calculate h(x)
/// H and R must be updated before being passed to kalman filter
#[allow(non_snake_case)]
pub struct NonLinearMeasurement<T, const N: usize, const M: usize> {
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    pub R: SMatrix<T, M, M>,
    /// Calculates the predicted value of z. i.e h(x)
    prediction_fn: Prediction<T, N, M>,
}

impl<T: RealField, const N: usize, const M: usize> NonLinearMeasurement<T, N, M> {
    pub fn new(prediction_fn: Prediction<T, N, M>) -> Self {
        Self {
            z: SMatrix::zeros(),
            H: SMatrix::zeros(),
            H_t: SMatrix::zeros(),
            R: SMatrix::zeros(),
            prediction_fn,
        }
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> Measurement<T, N, M>
    for NonLinearMeasurement<T, N, M>
{
    fn innovation(&self, x: &SVector<T, N>) -> SVector<T, M> {
        self.z - (self.prediction_fn)(x)
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

    fn set_measurement(&mut self, z: SVector<T, M>) {
        self.z = z;
    }
}
