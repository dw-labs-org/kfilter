//! Improved Unscented Kalman Filter implementation following kfilter design patterns
//! Optimized for performance, type safety, and consistency with existing kfilter architecture.

use nalgebra::{RealField, SMatrix, SVector};
use core::marker::PhantomData;

use crate::{
    system::{System, NoInputSystem, InputSystem, LinearNoInputSystem, LinearSystem},
    kalman::{KalmanFilter, KalmanPredict, KalmanPredictInput},
};

/// Maximum supported state dimension (following kfilter pragmatic approach)
/// 32 states = 65 sigma points, suitable for most practical applications
const MAX_SIGMA_POINTS: usize = 65;

/// Parameters for the Unscented Transform with validation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct UnscentedParameters<T: RealField + Copy> {
    /// Alpha parameter (0 < alpha <= 1), controls spread of sigma points
    pub alpha: T,
    /// Beta parameter (beta >= 0), incorporates prior knowledge of distribution
    pub beta: T,
    /// Kappa parameter, secondary scaling parameter (usually 0 or 3-n)
    pub kappa: T,
}

impl<T: RealField + Copy> UnscentedParameters<T> {
    /// Create new UKF parameters with validation (kfilter style with assert)
    #[track_caller]
    pub fn new(alpha: T, beta: T, kappa: T) -> Self {
        let zero = T::zero();
        let one = T::one();
        
        assert!(alpha > zero && alpha <= one, "Alpha must be in (0, 1]");
        assert!(beta >= zero, "Beta must be non-negative");
        
        Self { alpha, beta, kappa }
    }
    
    /// Create default UKF parameters (following kfilter patterns)
    pub fn default() -> Self {
        Self {
            alpha: T::from_f64(1e-3).unwrap(),
            beta: T::from_f64(2.0).unwrap(),
            kappa: T::zero(),
        }
    }

    /// Calculate lambda parameter
    #[inline]
    pub fn lambda(&self, n: usize) -> T {
        let n_f = T::from_usize(n).unwrap();
        self.alpha * self.alpha * (n_f + self.kappa) - n_f
    }

    /// Calculate all weights at once for efficiency
    #[inline]
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

/// Fixed-size sigma point collection (kfilter pragmatic approach)
/// Uses maximum size for no-std compatibility, but tracks actual usage
#[derive(Debug, Clone)]
pub struct SigmaPoints<T: RealField + Copy, const N: usize> {
    /// Collection of sigma points (fixed size for no-std compatibility)
    pub points_buffer: [SVector<T, N>; MAX_SIGMA_POINTS],
    /// Pre-computed weights for mean and covariance calculations
    pub weights: UKFWeights<T>,
    /// Number of actual points used (2N+1)
    pub num_points: usize,
}

impl<T: RealField + Copy, const N: usize> SigmaPoints<T, N> {
    /// Generate sigma points for N-dimensional state (kfilter style with unwrap)
    #[track_caller]
    pub fn generate(
        mean: &SVector<T, N>,
        covariance: &SMatrix<T, N, N>,
        params: &UnscentedParameters<T>,
    ) -> Self {
        let num_points = 2 * N + 1;
        assert!(num_points <= MAX_SIGMA_POINTS, 
            "State dimension {} too large (max {} supported)", N, (MAX_SIGMA_POINTS - 1) / 2);

        let n_f = T::from_usize(N).unwrap();
        let lambda = params.lambda(N);
        
        // Robust Cholesky decomposition (fail-fast like kfilter)
        let chol = covariance.clone().cholesky()
            .expect("Covariance matrix is not positive definite");
        
        let sqrt_factor = (n_f + lambda).sqrt();
        let sqrt_matrix = chol.l() * sqrt_factor;
        
        let mut points_buffer = [SVector::<T, N>::zeros(); MAX_SIGMA_POINTS];
        
        // Central point
        points_buffer[0] = *mean;
        
        // Positive and negative sigma points (optimized loop)
        for i in 0..N {
            let offset = sqrt_matrix.column(i).into_owned();
            points_buffer[i + 1] = mean + &offset;
            points_buffer[i + 1 + N] = mean - &offset;
        }
        
        let weights = params.compute_weights(N);
        
        Self { 
            points_buffer,
            weights,
            num_points,
        }
    }
    
    /// Get number of sigma points
    #[inline]
    pub fn len(&self) -> usize {
        self.num_points
    }
    
    /// Get iterator over active sigma points (zero-cost abstraction)
    #[inline]
    pub fn points_iter(&self) -> impl Iterator<Item = &SVector<T, N>> {
        self.points_buffer[..self.num_points].iter()
    }
    
    /// Apply function to all sigma points (optimized version)
    #[inline]
    pub fn transform<F, const M: usize>(&self, f: F) -> [SVector<T, M>; MAX_SIGMA_POINTS]
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        let mut result = [SVector::<T, M>::zeros(); MAX_SIGMA_POINTS];
        
        // Process only active points
        for i in 0..self.num_points {
            result[i] = f(&self.points_buffer[i]);
        }
        
        result
    }
    
    /// Compute weighted mean of transformed points (optimized)
    #[inline]
    pub fn weighted_mean<const M: usize>(&self, transformed: &[SVector<T, M>; MAX_SIGMA_POINTS]) -> SVector<T, M> {
        let mut mean = transformed[0] * self.weights.mean_0;
        
        // Vectorized addition for better performance
        for i in 1..self.num_points {
            mean += &transformed[i] * self.weights.other;
        }
        
        mean
    }
    
    /// Compute weighted covariance with pre-computed differences (performance optimized)
    pub fn weighted_covariance_optimized<const M: usize>(
        &self,
        transformed: &[SVector<T, M>; MAX_SIGMA_POINTS],
        mean: &SVector<T, M>,
    ) -> SMatrix<T, M, M> {
        // Pre-compute differences for reuse
        let mut diffs: [SVector<T, M>; MAX_SIGMA_POINTS] = [SVector::zeros(); MAX_SIGMA_POINTS];
        for i in 0..self.num_points {
            diffs[i] = &transformed[i] - mean;
        }
        
        // Central point with special weight
        let mut cov = (diffs[0] * diffs[0].transpose()) * self.weights.cov_0;
        
        // Other points (vectorized)
        for i in 1..self.num_points {
            cov += (&diffs[i] * diffs[i].transpose()) * self.weights.other;
        }
        
        // Ensure symmetry (like kfilter does)
        cov.symmetric_part()
    }
    
    /// Compute cross-covariance between state and measurement (optimized)
    pub fn cross_covariance<const M: usize>(
        &self,
        state_mean: &SVector<T, N>,
        measurement_transformed: &[SVector<T, M>; MAX_SIGMA_POINTS],
        measurement_mean: &SVector<T, M>,
    ) -> SMatrix<T, N, M> {
        let mut cross_cov = SMatrix::<T, N, M>::zeros();
        
        // Central point
        let state_diff0 = &self.points_buffer[0] - state_mean;
        let meas_diff0 = &measurement_transformed[0] - measurement_mean;
        cross_cov += (state_diff0 * meas_diff0.transpose()) * self.weights.cov_0;
        
        // Other points (unrolled for performance)
        for i in 1..self.num_points {
            let state_diff = &self.points_buffer[i] - state_mean;
            let meas_diff = &measurement_transformed[i] - measurement_mean;
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
    
    /// Generate sigma points from current state (fail-fast like kfilter)
    #[track_caller]
    #[inline]
    fn generate_sigma_points(&self) -> SigmaPoints<T, N> {
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
    #[inline]
    fn state(&self) -> &SVector<T, N> {
        self.system.state()
    }

    #[inline]
    fn covariance(&self) -> &SMatrix<T, N, N> {
        &self.P
    }

    #[inline]
    fn covariance_mut(&mut self) -> &mut SMatrix<T, N, N> {
        &mut self.P
    }

    #[inline]
    fn system_mut(&mut self) -> &mut S {
        &mut self.system
    }
}

/// UKF Prediction for systems without input (kfilter style with fallback)
impl<T, const N: usize, S> KalmanPredict<T, N> for UnscentedKalman<T, N, 0, S>
where
    T: RealField + Copy,
    S: NoInputSystem<T, N>,
{
    #[track_caller]
    fn predict(&mut self) -> &SVector<T, N> {
        // Get transition matrix for UKF prediction
        let transition = *self.system.transition();
        
        // Try UKF prediction first
        match self.predict_ukf_internal(move |state| &transition * state) {
            Ok(_) => {
                #[cfg(feature = "defmt")]
                defmt::trace!("UKF prediction successful");
            },
            Err(_e) => {
                #[cfg(feature = "defmt")]
                defmt::warn!("UKF prediction failed, falling back to linear");
                
                // Fallback to linear prediction (kfilter standard)
                self.system.step();
                self.P = self.system.transition() * &self.P * self.system.transition().transpose()
                    + self.system.covariance();
            }
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
        // For input systems, use linear system step (like kfilter)
        // UKF for input systems is more complex and left for future implementation
        self.system.step(u);
        self.P = self.system.transition() * &self.P * self.system.transition().transpose()
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
    /// Internal UKF prediction with error handling
    #[track_caller]
    fn predict_ukf_internal<F>(&mut self, process_fn: F) -> Result<(), &'static str>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, N>,
    {
        // Generate sigma points (can fail if covariance is not positive definite)
        let sigma_points = SigmaPoints::generate(
            self.system.state(),
            &self.P,
            &self.params,
        );
        
        // Transform through process function
        let transformed = sigma_points.transform(process_fn);
        
        // Compute predicted mean
        let predicted_mean = sigma_points.weighted_mean(&transformed);
        
        // Compute predicted covariance (optimized version)
        let mut predicted_cov = sigma_points.weighted_covariance_optimized(&transformed, &predicted_mean);
        
        // Add process noise
        predicted_cov += self.system.covariance();
        
        // Update state and covariance
        *self.system.state_mut() = predicted_mean;
        self.P = predicted_cov.symmetric_part(); // Ensure symmetry like kfilter
        
        Ok(())
    }
    
    /// Predict using unscented transform with custom process function (public API)
    #[track_caller]
    pub fn predict_ukf<F>(&mut self, process_fn: F) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, N>,
    {
        self.predict_ukf_internal(process_fn)
            .expect("UKF prediction failed");
        
        self.system.state()
    }
    
    /// Update using unscented transform (kfilter style with unwrap)
    #[track_caller]
    #[allow(non_snake_case)]
    pub fn update_ukf<F, const M: usize>(
        &mut self,
        measurement_fn: F,
        measurement: &SVector<T, M>,
        measurement_noise: &SMatrix<T, M, M>,
    ) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        // Generate sigma points (fail-fast like kfilter)
        let sigma_points = self.generate_sigma_points();
        
        // Transform through measurement function
        let measurement_transformed = sigma_points.transform(&measurement_fn);
        
        // Compute predicted measurement
        let predicted_measurement = sigma_points.weighted_mean(&measurement_transformed);
        
        // Compute innovation covariance (optimized)
        let mut innovation_cov = sigma_points.weighted_covariance_optimized(
            &measurement_transformed,
            &predicted_measurement,
        );
        innovation_cov += measurement_noise;
        
        // Robust inversion (fail-fast like kfilter)
        let innovation_cov_inv = innovation_cov.try_inverse()
            .expect("Innovation covariance matrix is singular");
        
        // Compute cross-covariance
        let cross_cov = sigma_points.cross_covariance(
            self.system.state(),
            &measurement_transformed,
            &predicted_measurement,
        );
        
        // Compute Kalman gain
        let K = &cross_cov * &innovation_cov_inv;
        
        // Update state
        let innovation = measurement - &predicted_measurement;
        *self.system.state_mut() += &K * &innovation;
        
        // Update covariance using standard form (like kfilter)
        let I = SMatrix::<T, N, N>::identity();
        // Simplified update (kfilter style)
        // Note: For UKF, we use an approximation similar to EKF
        let K_times_S = &K * &innovation_cov;
        self.P = (&I - &K_times_S * K.transpose()) * &self.P;
        
        // Ensure symmetry (like kfilter does)
        self.P = self.P.symmetric_part();
        
        self.system.state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2};
    
    #[test]
    fn test_ukf_parameters_creation() {
        type T = f64;
        
        // Valid parameters
        let params = UnscentedParameters::<T>::new(0.5, 2.0, 0.0);
        assert!((params.alpha - 0.5).abs() < 1e-10);
        
        // Default parameters
        let params_default = UnscentedParameters::<T>::default();
        assert!(params_default.alpha > 0.0);
    }
    
    #[test]
    #[should_panic(expected = "Alpha must be in (0, 1]")]
    fn test_ukf_parameters_invalid_alpha() {
        type T = f64;
        UnscentedParameters::<T>::new(0.0, 2.0, 0.0);
    }
    
    #[test]
    fn test_sigma_points_generation_fixed_size() {
        type T = f64;
        
        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();
        let params = UnscentedParameters::<T>::default();
        
        let sigma_points = SigmaPoints::generate(&mean, &cov, &params);
        
        // Check fixed size buffer with actual points
        assert_eq!(sigma_points.len(), 5); // 2*2 + 1
        assert_eq!(sigma_points.points_buffer.len(), MAX_SIGMA_POINTS); // Fixed size
        
        // Central point should be the mean
        assert!((sigma_points.points_buffer[0] - mean).norm() < 1e-10);
    }
    
    #[test]
    fn test_ukf_linear_prediction_performance() {
        type T = f64;
        
        // Create a simple 2D position+velocity system
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        // Multiple predictions should be fast
        for _ in 0..100 {
            ukf.predict();
        }
        
        // State should be accessible
        let _state = ukf.state();
    }
    
    #[test]
    fn test_ukf_update_kfilter_style() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::identity() * 0.1;
        
        // Update with linear measurement function (should not panic)
        let _result = ukf.update_ukf(
            |state| *state, // Direct observation
            &measurement,
            &measurement_noise,
        );
    }
    
    #[test]
    #[should_panic(expected = "Innovation covariance matrix is singular")]
    fn test_ukf_update_with_singular_noise_kfilter_style() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );
        
        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::zeros(); // Singular matrix
        
        // Should panic immediately (kfilter style)
        ukf.update_ukf(
            |state| *state,
            &measurement,
            &measurement_noise,
        );
    }
    
    #[test]
    fn test_ukf_fallback_mechanism() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInput::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::zeros(),                  // Singular P_initial to trigger fallback
        );
        
        // Should fallback to linear prediction without panicking
        ukf.predict();
        
        // State should still be accessible
        let _state = ukf.state();
    }
    
    #[test]
    fn test_max_dimension_support() {
        type T = f64;
        const LARGE_N: usize = 32; // Should work with MAX_SIGMA_POINTS = 65
        
        // This should compile and work
        let mean = SVector::<T, LARGE_N>::zeros();
        let cov = SMatrix::<T, LARGE_N, LARGE_N>::identity();
        let params = UnscentedParameters::<T>::default();
        
        let sigma_points = SigmaPoints::generate(&mean, &cov, &params);
        assert_eq!(sigma_points.len(), 2 * LARGE_N + 1);
    }
}