//! Improved Unscented Kalman Filter implementation with kfilter design patterns
//! Follows kfilter's architecture for consistency and robustness.

use nalgebra::{RealField, SMatrix, SVector};
use core::marker::PhantomData;

use crate::{
    system::{System, NoInputSystem, InputSystem, LinearNoInputSystem, LinearSystem},
    kalman::{KalmanFilter, KalmanPredict, KalmanPredictInput},
};

/// Parameters for the Unscented Transform with validation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct UnscentedParameters<T: RealField + Copy> {
    /// Alpha parameter (0 < alpha <= 1), controls spread of sigma points
    pub alpha: T,
    /// Beta parameter (beta >= 0), incorporates prior knowledge of distribution (2 is optimal for Gaussian)
    pub beta: T,
    /// Kappa parameter, secondary scaling parameter (usually 0 or 3-n)
    pub kappa: T,
}

impl<T: RealField + Copy> UnscentedParameters<T> {
    /// Create new UKF parameters with validation
    pub fn new(alpha: T, beta: T, kappa: T) -> Result<Self, &'static str> {
        let zero = T::zero();
        let one = T::one();
        
        if alpha <= zero || alpha > one {
            return Err("Alpha must be in (0, 1]");
        }
        if beta < zero {
            return Err("Beta must be non-negative");
        }
        
        Ok(Self { alpha, beta, kappa })
    }
    
    /// Create default UKF parameters with commonly used values
    pub fn default() -> Self {
        Self {
            alpha: T::from_f64(1e-3).unwrap(),
            beta: T::from_f64(2.0).unwrap(),
            kappa: T::zero(),
        }
    }

    /// Calculate lambda parameter
    pub fn lambda(&self, n: usize) -> T {
        let n_f = T::from_usize(n).unwrap();
        self.alpha * self.alpha * (n_f + self.kappa) - n_f
    }

    /// Calculate all weights at once for efficiency
    pub fn compute_weights(&self, n: usize) -> UKFWeights<T> {
        let n_f = T::from_usize(n).unwrap();
        let lambda = self.lambda(n);
        let two = T::from_f64(2.0).unwrap();
        
        let w_mean_0 = lambda / (n_f + lambda);
        let w_cov_0 = w_mean_0 + (T::one() - self.alpha * self.alpha + self.beta);
        let w_other = T::one() / (two * (n_f + lambda));
        
        UKFWeights {
            mean_0: w_mean_0,
            cov_0: w_cov_0,
            other: w_other,
        }
    }
}

/// Pre-computed weights for efficiency
#[derive(Debug, Clone, Copy)]
pub struct UKFWeights<T: RealField + Copy> {
    /// Weight for the mean calculation of the central sigma point
    pub mean_0: T,
    /// Weight for the covariance calculation of the central sigma point
    pub cov_0: T,
    /// Weight for all other sigma points (both mean and covariance)
    pub other: T,
}

/// Sigma point collection for no-std compatibility
#[derive(Debug, Clone)]
pub struct SigmaPoints<T: RealField + Copy, const N: usize> {
    /// Collection of sigma points (fixed size for no-std)
    /// Maximum support for 10D states (2*10+1 = 21 points)
    pub points_buffer: [SVector<T, N>; 21],
    /// Pre-computed weights for mean and covariance calculations
    pub weights: UKFWeights<T>,
    /// Number of actual points used (2N+1)
    pub num_points: usize,
}

impl<T: RealField + Copy, const N: usize> SigmaPoints<T, N> {
    /// Generate sigma points for N-dimensional state with proper error handling
    pub fn generate(
        mean: &SVector<T, N>,
        covariance: &SMatrix<T, N, N>,
        params: &UnscentedParameters<T>,
    ) -> Result<Self, &'static str> {
        let num_points = 2 * N + 1;
        if num_points > 21 {
            return Err("State dimension too large (max 10D supported)");
        }

        let n_f = T::from_usize(N).unwrap();
        let lambda = params.lambda(N);
        
        // Robust Cholesky decomposition
        let chol = covariance.clone().cholesky()
            .ok_or("Covariance matrix is not positive definite")?;
        
        let sqrt_factor = (n_f + lambda).sqrt();
        let sqrt_matrix = chol.l() * sqrt_factor;
        
        let mut points_buffer = [SVector::<T, N>::zeros(); 21];
        
        // Central point
        points_buffer[0] = *mean;
        
        // Positive and negative sigma points
        for i in 0..N {
            let offset = sqrt_matrix.column(i).into_owned();
            points_buffer[i + 1] = mean + offset;
            points_buffer[i + 1 + N] = mean - offset;
        }
        
        let weights = params.compute_weights(N);
        
        Ok(Self { 
            points_buffer,
            weights,
            num_points,
        })
    }
    
    /// Get number of sigma points
    pub fn len(&self) -> usize {
        self.num_points
    }
    
    /// Get iterator over active sigma points
    pub fn points_iter(&self) -> impl Iterator<Item = &SVector<T, N>> {
        self.points_buffer[..self.num_points].iter()
    }
    
    /// Apply function to all sigma points (no-std compatible)
    pub fn transform<F, const M: usize>(&self, f: F) -> [SVector<T, M>; 21]
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        let mut result = [SVector::<T, M>::zeros(); 21];
        for (i, point) in self.points_iter().enumerate() {
            result[i] = f(point);
        }
        result
    }
    
    /// Compute weighted mean of transformed points
    pub fn weighted_mean<const M: usize>(&self, transformed: &[SVector<T, M>; 21]) -> SVector<T, M> {
        let mut mean = transformed[0] * self.weights.mean_0;
        for i in 1..self.num_points {
            mean += transformed[i] * self.weights.other;
        }
        mean
    }
    
    /// Compute weighted covariance with numerical stability
    pub fn weighted_covariance<const M: usize>(
        &self,
        transformed: &[SVector<T, M>; 21],
        mean: &SVector<T, M>,
    ) -> SMatrix<T, M, M> {
        let mut cov = SMatrix::<T, M, M>::zeros();
        
        // Central point with special weight
        let diff0 = transformed[0] - mean;
        cov += (diff0 * diff0.transpose()) * self.weights.cov_0;
        
        // Other points
        for i in 1..self.num_points {
            let diff = transformed[i] - mean;
            cov += (diff * diff.transpose()) * self.weights.other;
        }
        
        // Ensure symmetry for numerical stability (like kfilter does)
        cov.symmetric_part()
    }
    
    /// Compute cross-covariance between state and measurement
    pub fn cross_covariance<const M: usize>(
        &self,
        state_mean: &SVector<T, N>,
        measurement_transformed: &[SVector<T, M>; 21],
        measurement_mean: &SVector<T, M>,
    ) -> SMatrix<T, N, M> {
        let mut cross_cov = SMatrix::<T, N, M>::zeros();
        
        // Central point
        let state_diff0 = self.points_buffer[0] - state_mean;
        let meas_diff0 = measurement_transformed[0] - measurement_mean;
        cross_cov += (state_diff0 * meas_diff0.transpose()) * self.weights.cov_0;
        
        // Other points
        for i in 1..self.num_points {
            let state_diff = self.points_buffer[i] - state_mean;
            let meas_diff = measurement_transformed[i] - measurement_mean;
            cross_cov += (state_diff * meas_diff.transpose()) * self.weights.other;
        }
        
        cross_cov
    }
}

/// Generic Unscented Kalman Filter following kfilter patterns
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct UnscentedKalman<T, const N: usize, const U: usize, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// State covariance matrix
    #[cfg_attr(feature = "defmt", defmt(Debug2Format))]
    P: SMatrix<T, N, N>,
    /// System model (public like kfilter for flexibility)
    pub system: S,
    /// UKF parameters
    params: UnscentedParameters<T>,
    /// Phantom data for type system
    _phantom: PhantomData<T>,
}

/// Type aliases following kfilter patterns
/// UKF with LinearNoInputSystem
pub type UKFLinearNoInput<T, const N: usize> = UnscentedKalman<T, N, 0, LinearNoInputSystem<T, N>>;
/// UKF with LinearSystem
pub type UKFLinear<T, const N: usize, const U: usize> = UnscentedKalman<T, N, U, LinearSystem<T, N, U>>;
/// 2D Unscented Kalman Filter
pub type UKF2D<T, const U: usize, S> = UnscentedKalman<T, 2, U, S>;
/// 3D Unscented Kalman Filter
pub type UKF3D<T, const U: usize, S> = UnscentedKalman<T, 3, U, S>;

impl<T, const N: usize, const U: usize, S> UnscentedKalman<T, N, U, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Create new UKF with custom system (like kfilter's new_custom)
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>) -> Self {
        Self {
            P: initial_covariance,
            system,
            params: UnscentedParameters::default(),
            _phantom: PhantomData,
        }
    }
    
    /// Create new UKF with custom parameters
    pub fn new_custom_with_params(
        system: S,
        initial_covariance: SMatrix<T, N, N>,
        params: UnscentedParameters<T>,
    ) -> Self {
        Self {
            P: initial_covariance,
            system,
            params,
            _phantom: PhantomData,
        }
    }
    
    /// Generate sigma points from current state
    fn generate_sigma_points(&self) -> Result<SigmaPoints<T, N>, &'static str> {
        SigmaPoints::generate(
            self.system.state(),
            &self.P,
            &self.params,
        )
    }
}

/// Linear UKF constructors (following kfilter patterns)
impl<T, const N: usize, const U: usize> UKFLinear<T, N, U>
where
    T: RealField + Copy,
{
    /// Create UKF with linear system with inputs
    #[allow(non_snake_case)]
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            P: P_initial,
            system: LinearSystem::new(F, Q, B, x_initial),
            params: UnscentedParameters::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> UKFLinearNoInput<T, N>
where
    T: RealField + Copy,
{
    /// Create UKF with linear system without inputs
    #[allow(non_snake_case)]
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self {
            P: P_initial,
            system: LinearNoInputSystem::new(F, Q, x_initial),
            params: UnscentedParameters::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize, const U: usize, S> KalmanFilter<T, N, S> for UnscentedKalman<T, N, U, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    fn state(&self) -> &SVector<T, N> {
        self.system.state()
    }

    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.P
    }

    fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        &mut self.P
    }

    fn system_mut(&mut self) -> &mut S {
        &mut self.system
    }
}

/// UKF Prediction for systems without input
impl<T, const N: usize, S> KalmanPredict<T, N> for UnscentedKalman<T, N, 0, S>
where
    T: RealField + Copy,
    S: NoInputSystem<T, N>,
{
    #[track_caller]
    fn predict(&mut self) -> &SVector<T, N> {
        // Use system transition for UKF prediction
        let transition = *self.system.transition();
        
        // Try UKF prediction first
        if self.predict_ukf(move |state| transition * state).is_err() {
            // Fallback to linear prediction if UKF fails
            self.system.step();
            self.P = self.system.transition() * self.P * self.system.transition_transpose()
                + self.system.covariance();
        }
        
        self.system.state()
    }
}

/// UKF Prediction for systems with input
impl<T, const N: usize, const U: usize, S> KalmanPredictInput<T, N, U> for UnscentedKalman<T, N, U, S>
where
    T: RealField + Copy,
    S: InputSystem<T, N, U>,
{
    #[track_caller]
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        // For input systems, we use a simplified approach:
        // Apply the linear system step which handles input correctly
        self.system.step(u);
        self.P = self.system.transition() * self.P * self.system.transition_transpose()
            + self.system.covariance();
        
        self.system.state()
    }
}

/// UKF-specific methods
impl<T, const N: usize, const U: usize, S> UnscentedKalman<T, N, U, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Predict using unscented transform with custom process function
    #[track_caller]
    pub fn predict_ukf<F>(&mut self, process_fn: F) -> Result<(), &'static str>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, N>,
    {
        // Generate sigma points
        let sigma_points = self.generate_sigma_points()?;
        
        // Transform through process function
        let transformed = sigma_points.transform(process_fn);
        
        // Compute predicted mean
        let predicted_mean = sigma_points.weighted_mean(&transformed);
        
        // Compute predicted covariance
        let mut predicted_cov = sigma_points.weighted_covariance(&transformed, &predicted_mean);
        
        // Add process noise
        predicted_cov += *self.system.covariance();
        
        // Update state and covariance
        *self.system.state_mut() = predicted_mean;
        self.P = predicted_cov.symmetric_part(); // Ensure symmetry like kfilter
        
        Ok(())
    }
    
    /// Update using unscented transform with robust error handling
    #[track_caller]
    #[allow(non_snake_case)]
    pub fn update_ukf<F, const M: usize>(
        &mut self,
        measurement_fn: F,
        measurement: &SVector<T, M>,
        measurement_noise: &SMatrix<T, M, M>,
    ) -> Result<&SVector<T, N>, &'static str>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        // Validate measurement noise matrix (like kfilter does)
        if measurement_noise.try_inverse().is_none() {
            return Err("Measurement covariance matrix is not invertible");
        }

        // Generate sigma points
        let sigma_points = self.generate_sigma_points()?;
        
        // Transform through measurement function
        let measurement_transformed = sigma_points.transform(&measurement_fn);
        
        // Compute predicted measurement
        let predicted_measurement = sigma_points.weighted_mean(&measurement_transformed);
        
        // Compute innovation covariance
        let mut innovation_cov = sigma_points.weighted_covariance(
            &measurement_transformed,
            &predicted_measurement,
        );
        innovation_cov += measurement_noise;
        
        // Robust inversion with detailed error messages (like kfilter)
        let innovation_cov_inv = innovation_cov.try_inverse()
            .ok_or("Innovation covariance matrix could not be inverted")?;
        
        // Compute cross-covariance
        let cross_cov = sigma_points.cross_covariance(
            self.system.state(),
            &measurement_transformed,
            &predicted_measurement,
        );
        
        // Compute Kalman gain
        let K = cross_cov * innovation_cov_inv;
        
        // Update state
        let innovation = measurement - predicted_measurement;
        *self.system.state_mut() += K * innovation;
        
        // Update covariance using Joseph form for numerical stability
        let i_minus_kh = SMatrix::<T, N, N>::identity() - &K * innovation_cov * K.transpose();
        self.P = i_minus_kh * self.P * i_minus_kh.transpose()
            + K * measurement_noise * K.transpose();
        
        // Ensure symmetry (like kfilter does)
        self.P = self.P.symmetric_part();
        
        Ok(self.system.state())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2, Matrix1, Matrix1x2};
    
    #[test]
    fn test_ukf_parameters_validation() {
        type T = f64;
        
        // Valid parameters
        assert!(UnscentedParameters::<T>::new(0.5, 2.0, 0.0).is_ok());
        
        // Invalid alpha
        assert!(UnscentedParameters::<T>::new(0.0, 2.0, 0.0).is_err());
        assert!(UnscentedParameters::<T>::new(1.5, 2.0, 0.0).is_err());
        
        // Invalid beta
        assert!(UnscentedParameters::<T>::new(0.5, -1.0, 0.0).is_err());
    }
    
    #[test]
    fn test_sigma_points_generation() {
        type T = f64;
        
        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();
        let params = UnscentedParameters::<T>::default();
        
        let sigma_points = SigmaPoints::generate(&mean, &cov, &params).unwrap();
        
        assert_eq!(sigma_points.len(), 5); // 2*2 + 1
        assert!((sigma_points.points_buffer[0] - mean).norm() < 1e-10); // Central point
    }
    
    #[test]
    fn test_ukf_linear_prediction() {
        type T = f64;
        
        // Create a simple 2D position+velocity system
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        // Predict should not panic
        ukf.predict();
        
        // State should be accessible
        let _state = ukf.state();
    }
    
    #[test]
    fn test_ukf_update() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::identity() * 0.1;
        
        // Update with linear measurement function
        let result = ukf.update_ukf(
            |state| *state, // Direct observation
            &measurement,
            &measurement_noise,
        );
        
        assert!(result.is_ok());
    }
    
    #[test]
    #[should_panic(expected = "Measurement covariance matrix is not invertible")]
    fn test_ukf_update_with_singular_noise() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::zeros(); // Singular matrix
        
        // Should panic with clear error message
        ukf.update_ukf(
            |state| *state,
            &measurement,
            &measurement_noise,
        ).unwrap();
    }
}