//! A measurement is the input to the Kalman filter during the update stage.
//! It is defined by the measurement vector z (sensor readings), the observation
//! matrix H (or function h(x)) and the measurment noise R.
//!
//! For use with the Kalman filter, [Measurement] must be implemented. The
//! vast majority of filters can use [LinearMeasurement] or even [NonLinearMeasurement]
//! if required.

use nalgebra::{RealField, SMatrix, SVector};

/// Trait that defines the functionality for the [Kalman](crate::kalman::Kalman) filter
/// to interact with a measurement.
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
    /// Set the observation matrix. Ensures transpose is updated.
    #[allow(non_snake_case)]
    fn set_observation(&mut self, H: SMatrix<T, M, N>);
}

#[allow(non_snake_case)]
/// A linear measurement defined by the observation matrix H and noise matrix R.
/// Implements the innovation function y = z - H * x.
pub struct LinearMeasurement<T, const N: usize, const M: usize> {
    /// Observation / measurement. Can be modifed directly to set new value.
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    /// Measurement noise. Can be modifed directly to set new value.
    pub R: SMatrix<T, M, M>,
}

impl<T: RealField, const N: usize, const M: usize> LinearMeasurement<T, N, M> {
    #[allow(non_snake_case)]
    /// Create new measurement with observation matrix H, noise matrix R and measurement z.
    pub fn new(H: SMatrix<T, M, N>, R: SMatrix<T, M, M>, z: SVector<T, M>) -> Self {
        Self {
            z,
            H_t: H.transpose(),
            H,
            R,
        }
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
    #[allow(non_snake_case)]
    fn set_observation(&mut self, H: SMatrix<T, M, N>) {
        self.H_t = H.transpose();
        self.H = H;
    }
}

/// Function that returns the predicted value of the measurment based on the current state
pub type Prediction<T, const N: usize, const M: usize> = fn(&SVector<T, N>) -> SVector<T, M>;

/// A non linear measurement that uses a prediction function to calculate h(x)
/// H and R must be updated before being passed to [Kalman](crate::kalman::Kalman) filter
#[allow(non_snake_case)]
pub struct NonLinearMeasurement<T, const N: usize, const M: usize> {
    /// Observation / measurement. Can be modifed directly to set new value.
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    /// Measurement noise. Can be modifed directly to set new value.
    pub R: SMatrix<T, M, M>,
    /// Calculates the predicted value of z. i.e h(x)
    prediction_fn: Prediction<T, N, M>,
}

impl<T: RealField, const N: usize, const M: usize> NonLinearMeasurement<T, N, M> {
    /// Create new [NonLinearMeasurement].
    #[allow(non_snake_case)]
    pub fn new(prediction_fn: Prediction<T, N, M>, R: SMatrix<T, M, M>, z: SVector<T, M>) -> Self {
        Self {
            z,
            H: SMatrix::zeros(),
            H_t: SMatrix::zeros(),
            R,
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
    #[allow(non_snake_case)]
    fn set_observation(&mut self, H: SMatrix<T, M, N>) {
        self.H_t = H.transpose();
        self.H = H;
    }
}
