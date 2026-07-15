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
    /// Get the measurement covariance
    fn covariance(&self) -> &SMatrix<T, M, M>;
    /// Get the current measurement vector
    fn measurement(&self) -> &SVector<T, M>;
    /// Set the measurement (z) value
    fn set_measurement(&mut self, z: SVector<T, M>);
    /// Predict the measurement (z) based on the current state (x)
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, M>;
}

/// Linear or linearisable measurement that can provide the Jacobian matrix of the
/// measurement function
pub trait LinearisableMeasurement<T, const N: usize, const M: usize>: Measurement<T, N, M> {
    /// Get the observation matrix
    fn observation(&self) -> &SMatrix<T, M, N>;
    /// Get the observation matrix transpose
    fn observation_transpose(&self) -> &SMatrix<T, N, M>;
}

#[allow(non_snake_case)]
/// A linear measurement defined by the observation matrix H and noise matrix R.
/// Implements the innovation function y = z - H * x.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct LinearMeasurement<T: RealField, const N: usize, const M: usize> {
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
    fn covariance(&self) -> &SMatrix<T, M, M> {
        &self.R
    }

    fn set_measurement(&mut self, z: SVector<T, M>) {
        self.z = z;
    }

    fn predict(&self, x: &SVector<T, N>) -> SVector<T, M> {
        self.H * x
    }

    fn measurement(&self) -> &SVector<T, M> {
        &self.z
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> LinearisableMeasurement<T, N, M>
    for LinearMeasurement<T, N, M>
{
    fn observation(&self) -> &SMatrix<T, M, N> {
        &self.H
    }

    fn observation_transpose(&self) -> &SMatrix<T, N, M> {
        &self.H_t
    }
}

/// Function that returns the predicted value of the measurment based on the current state
pub type Prediction<T, const N: usize, const M: usize> = fn(&SVector<T, N>) -> SVector<T, M>;

/// A non linear measurement that uses a prediction function to calculate h(x)
/// H (via [Self::set_observation]) and R must be updated before being passed to
/// [Kalman](crate::kalman::Kalman) filter
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct NonLinearMeasurement<T, const N: usize, const M: usize> {
    /// Observation / measurement. Can be modifed directly to set new value.
    pub z: SVector<T, M>,
    H: SMatrix<T, M, N>,
    H_t: SMatrix<T, N, M>,
    /// Measurement noise. Can be modifed directly to set new value.
    pub R: SMatrix<T, M, M>,
    /// Calculates the predicted value of z. i.e h(x)
    // #[cfg_attr(feature = "serde", serde(skip))]
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

    /// Set a new observation matrix (the Jacobian of the prediction function,
    /// typically evaluated at the current state estimate), also updating its
    /// cached transpose. Must be called before each [Kalman](crate::kalman::Kalman)
    /// `update()` when using the (non-UKF) EKF update path, since [Self::observation]
    /// otherwise stays at its zero initial value forever.
    #[allow(non_snake_case)]
    pub fn set_observation(&mut self, H: SMatrix<T, M, N>) {
        self.H_t = H.transpose();
        self.H = H;
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> Measurement<T, N, M>
    for NonLinearMeasurement<T, N, M>
{
    fn covariance(&self) -> &SMatrix<T, M, M> {
        &self.R
    }

    fn set_measurement(&mut self, z: SVector<T, M>) {
        self.z = z;
    }

    fn predict(&self, x: &SVector<T, N>) -> SVector<T, M> {
        (self.prediction_fn)(x)
    }

    fn measurement(&self) -> &SVector<T, M> {
        &self.z
    }
}

impl<T: RealField + Copy, const N: usize, const M: usize> LinearisableMeasurement<T, N, M>
    for NonLinearMeasurement<T, N, M>
{
    fn observation(&self) -> &SMatrix<T, M, N> {
        &self.H
    }

    fn observation_transpose(&self) -> &SMatrix<T, N, M> {
        &self.H_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix1, Matrix1x2, Vector1, Vector2};

    fn identity_h(x: &Vector2<f64>) -> Vector1<f64> {
        Vector1::new(x[0])
    }

    #[test]
    fn observation_starts_zero() {
        let m = NonLinearMeasurement::<f64, 2, 1>::new(identity_h, Matrix1::identity(), Vector1::new(0.0));
        assert_eq!(*m.observation(), Matrix1x2::zeros());
        assert_eq!(*m.observation_transpose(), SMatrix::<f64, 2, 1>::zeros());
    }

    #[test]
    fn set_observation_updates_h_and_transpose() {
        let mut m =
            NonLinearMeasurement::<f64, 2, 1>::new(identity_h, Matrix1::identity(), Vector1::new(0.0));
        let h = Matrix1x2::new(2.0, -1.0);

        m.set_observation(h);

        assert_eq!(*m.observation(), h);
        assert_eq!(*m.observation_transpose(), h.transpose());
    }
}
