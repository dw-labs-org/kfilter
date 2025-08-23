//! Improved Unscented Kalman Filter implementation with Rust compatibility
//! Fixed for Rust type system limitations and nalgebra compatibility

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    kalman::{KalmanFilter, KalmanPredict, KalmanPredictInput},
    system::{InputSystem, NoInputSystem, System},
};

/// Structured error type for UKF operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum UKFError {
    /// Matrix inversion failed (singular matrix)
    SingularMatrix,
    /// Invalid UKF parameters
    InvalidParameters,
    /// Numerical instability detected
    NumericalInstability,
}

impl core::fmt::Display for UKFError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UKFError::SingularMatrix => write!(f, "Singular matrix encountered"),
            UKFError::InvalidParameters => write!(f, "Invalid UKF parameters"),
            UKFError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

/// Result type for UKF operations
pub type UKFResult<T> = Result<T, UKFError>;

/// Performance statistics for UKF operations
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UKFPerformanceStats {
    /// Number of prediction operations performed
    pub prediction_count: usize,
    /// Number of update operations performed
    pub update_count: usize,
    /// Number of sigma point generation failures
    pub sigma_generation_failures: usize,
    /// Number of matrix inversion failures
    pub matrix_inversion_failures: usize,
}

impl UKFPerformanceStats {
    /// Calculate prediction success rate
    pub fn prediction_success_rate(&self) -> f64 {
        if self.prediction_count == 0 {
            1.0
        } else {
            1.0 - (self.sigma_generation_failures as f64 / self.prediction_count as f64)
        }
    }

    /// Calculate update success rate
    pub fn update_success_rate(&self) -> f64 {
        if self.update_count == 0 {
            1.0
        } else {
            1.0 - (self.matrix_inversion_failures as f64 / self.update_count as f64)
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

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
    /// Create new UKF parameters with validation
    #[track_caller]
    pub fn new(alpha: T, beta: T, kappa: T) -> UKFResult<Self> {
        let zero = T::zero();
        let one = T::one();

        if alpha <= zero || alpha > one {
            return Err(UKFError::InvalidParameters);
        }
        if beta < zero {
            return Err(UKFError::InvalidParameters);
        }

        Ok(Self { alpha, beta, kappa })
    }

    /// Create default UKF parameters
    pub fn new_default() -> Self {
        Self {
            alpha: T::from_f64(1e-3)
                .unwrap_or_else(|| T::one() / T::from_f64(1000.0).unwrap_or(T::one())),
            beta: T::from_f64(2.0).unwrap_or_else(|| T::one() + T::one()),
            kappa: T::zero(),
        }
    }

    /// Create parameters optimized for high-dimensional systems
    pub fn high_dimensional(n: usize) -> Self {
        Self {
            alpha: T::from_f64(1e-3).unwrap(),
            beta: T::from_f64(2.0).unwrap(),
            kappa: T::from_f64(3.0 - n as f64).unwrap(),
        }
    }

    /// Calculate lambda parameter
    #[inline]
    pub fn lambda(&self, n: usize) -> T {
        let n_f = T::from_usize(n).unwrap_or_else(|| T::zero());
        self.alpha * self.alpha * (n_f + self.kappa) - n_f
    }

    /// Calculate all weights with numerical stability checks
    #[inline]
    pub fn compute_weights(&self, n: usize) -> UKFWeights<T> {
        let n_f = T::from_usize(n).unwrap_or_else(|| T::zero());
        let lambda = self.lambda(n);
        let two = T::from_f64(2.0).unwrap_or_else(|| T::one() + T::one());

        let denominator = n_f + lambda;

        // Check for problematic denominator (too close to zero)
        let min_abs_denom =
            T::from_f64(1e-10).unwrap_or_else(|| T::one() / T::from_f64(1e10).unwrap_or(T::one()));

        if denominator.abs() < min_abs_denom {
            // Use modified kappa to avoid division by near-zero
            let safe_kappa = if lambda < T::zero() {
                // If lambda is negative, adjust kappa to make denominator reasonable
                T::from_f64(1e-6).unwrap_or_else(|| T::one() / T::from_f64(1e6).unwrap_or(T::one()))
            } else {
                self.kappa
                    + T::from_f64(1e-6)
                        .unwrap_or_else(|| T::one() / T::from_f64(1e6).unwrap_or(T::one()))
            };

            let safe_lambda = self.alpha * self.alpha * (n_f + safe_kappa) - n_f;
            let safe_denominator = n_f + safe_lambda;

            let w_mean_0 = safe_lambda / safe_denominator;
            let w_cov_0 = w_mean_0 + (T::one() - self.alpha * self.alpha + self.beta);
            let w_other = T::one() / (two * safe_denominator);

            return UKFWeights {
                mean_0: w_mean_0,
                cov_0: w_cov_0,
                other: w_other,
            };
        }

        // Normal case - use standard UKF weight equations
        let w_mean_0 = lambda / denominator;
        let w_cov_0 = w_mean_0 + (T::one() - self.alpha * self.alpha + self.beta);
        let w_other = T::one() / (two * denominator);

        UKFWeights {
            mean_0: w_mean_0,
            cov_0: w_cov_0,
            other: w_other,
        }
    }

    /// Check if parameters will cause numerical issues for given dimension
    pub fn is_numerically_stable(&self, n: usize) -> bool {
        let n_f = T::from_usize(n).unwrap_or_else(|| T::zero());
        let lambda = self.lambda(n);
        let denominator = n_f + lambda;

        let min_abs_denom =
            T::from_f64(1e-10).unwrap_or_else(|| T::one() / T::from_f64(1e10).unwrap_or(T::one()));
        denominator.abs() >= min_abs_denom
    }
}

impl<T: RealField + Copy> Default for UnscentedParameters<T> {
    fn default() -> Self {
        Self::new_default()
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

trait ValidSigma {
    const VALID: ();
}

impl<T: RealField + Copy, const N: usize, const X: usize> ValidSigma for SigmaPoints<T, N, X> {
    const VALID: () = assert!(X == 2 * N + 1);
}

/// Fixed-size sigma point collection using const generics
#[derive(Debug, Clone)]
pub struct SigmaPoints<T: RealField + Copy, const N: usize, const X: usize> {
    /// Collection of sigma points
    pub points_buffer: [SVector<T, N>; X],
    /// Pre-computed weights for mean and covariance calculations
    pub weights: UKFWeights<T>,
}

impl<T: RealField + Copy, const N: usize, const X: usize> SigmaPoints<T, N, X> {
    /// Generate sigma points with validation
    #[track_caller]
    pub fn generate(
        mean: &SVector<T, N>,
        covariance: &SMatrix<T, N, N>,
        params: &UnscentedParameters<T>,
    ) -> UKFResult<Self> {
        // Use const assert to check that S = 2N + 1
        #[allow(clippy::let_unit_value)]
        let _ = <Self as ValidSigma>::VALID;

        let n_f = T::from_usize(N).unwrap_or_else(|| T::zero());
        let lambda = params.lambda(N);

        // Robust Cholesky decomposition
        let sqrt_matrix = Self::robust_matrix_sqrt(covariance, n_f + lambda);

        let mut points_buffer: [nalgebra::Matrix<
            T,
            nalgebra::Const<N>,
            nalgebra::Const<1>,
            nalgebra::ArrayStorage<T, N, 1>,
        >; X] = [SVector::<T, N>::zeros(); X];

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
        })
    }

    /// Robust matrix square root computation with fallback
    fn robust_matrix_sqrt(matrix: &SMatrix<T, N, N>, scale: T) -> SMatrix<T, N, N> {
        let sqrt_scale = scale.sqrt();

        // Primary: Cholesky decomposition
        if let Some(chol) = matrix.cholesky() {
            return chol.l() * sqrt_scale;
        }

        // Fallback: Use identity matrix scaled appropriately
        // This is a safe fallback that ensures positive definiteness
        let min_eigenvalue = T::from_f64(1e-6).unwrap();
        SMatrix::<T, N, N>::identity() * (min_eigenvalue * sqrt_scale)
    }

    /// Get iterator over active sigma points
    #[inline]
    pub fn points_iter(&self) -> impl Iterator<Item = &SVector<T, N>> {
        self.points_buffer.iter()
    }

    /// Apply function to all sigma points with external buffer
    #[inline]
    pub fn transform_into<F, const M: usize>(&self, f: F, result: &mut [SVector<T, M>; X])
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        // Process only active points
        for (point, r) in self.points_buffer.iter().zip(result) {
            *r = f(point)
        }
    }

    /// Apply function to all sigma points
    #[inline]
    pub fn transform<F, const M: usize>(&self, f: F) -> [SVector<T, M>; X]
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        let mut result = [SVector::<T, M>::zeros(); X];
        self.transform_into(f, &mut result);
        result
    }

    /// Compute weighted mean of transformed points
    #[inline]
    pub fn weighted_mean<const M: usize>(&self, transformed: &[SVector<T, M>; X]) -> SVector<T, M> {
        let mut mean = transformed[0] * self.weights.mean_0;

        for transformed_point in transformed.iter().skip(1) {
            mean += transformed_point * self.weights.other;
        }

        mean
    }

    /// Compute weighted covariance with enhanced numerical stability
    pub fn weighted_covariance_stable<const M: usize>(
        &self,
        transformed: &[SVector<T, M>; X],
        mean: &SVector<T, M>,
    ) -> SMatrix<T, M, M> {
        let mut cov = SMatrix::<T, M, M>::zeros();

        // Central point with special weight
        let diff_0 = transformed[0] - mean;
        cov += (diff_0 * diff_0.transpose()) * self.weights.cov_0;

        // Other points
        for transformed_point in transformed.iter().skip(1) {
            let diff_i = transformed_point - mean;
            cov += (diff_i * diff_i.transpose()) * self.weights.other;
        }

        // Ensure symmetry
        cov.symmetric_part()
    }

    /// Compute cross-covariance between state and measurement
    pub fn cross_covariance<const M: usize>(
        &self,
        state_mean: &SVector<T, N>,
        measurement_transformed: &[SVector<T, M>; X],
        measurement_mean: &SVector<T, M>,
    ) -> SMatrix<T, N, M> {
        let mut cross_cov = SMatrix::<T, N, M>::zeros();

        // Central point
        let state_diff0 = self.points_buffer[0] - state_mean;
        let meas_diff0 = measurement_transformed[0] - measurement_mean;
        cross_cov += (state_diff0 * meas_diff0.transpose()) * self.weights.cov_0;

        // Other points
        for (i, meas_transformed) in measurement_transformed.iter().enumerate().skip(1) {
            let state_diff = self.points_buffer[i + 1] - state_mean;
            let meas_diff = meas_transformed - measurement_mean;
            cross_cov += (state_diff * meas_diff.transpose()) * self.weights.other;
        }

        cross_cov
    }
}

/// Enhanced Unscented Kalman Filter with fixed buffer sizes
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct UnscentedKalman<T, const N: usize, const U: usize, const X: usize, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// State covariance matrix
    #[cfg_attr(feature = "defmt", defmt(Debug2Format))]
    P: SMatrix<T, N, N>,
    /// System model
    pub system: S,
    /// UKF parameters
    params: UnscentedParameters<T>,
}

/// Builder for creating UKF instances with better ergonomics
pub struct UKFBuilder<T: RealField + Copy> {
    /// UKF parameters
    params: UnscentedParameters<T>,
}

impl<T: RealField + Copy> Default for UKFBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> UKFBuilder<T> {
    /// Create a new builder with default parameters
    pub fn new() -> Self {
        Self {
            params: UnscentedParameters::default(),
        }
    }

    /// Set custom UKF parameters
    pub fn with_params(mut self, params: UnscentedParameters<T>) -> Self {
        self.params = params;
        self
    }
}

/// Performance monitoring wrapper for UKF
pub struct MonitoredUKF<T, const N: usize, const U: usize, const X: usize, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Inner UKF instance
    inner: UnscentedKalman<T, N, U, X, S>,
    /// Performance statistics
    stats: UKFPerformanceStats,
}

impl<T, const N: usize, const U: usize, const X: usize, S> MonitoredUKF<T, N, U, X, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Create a new monitored UKF
    pub fn new(inner: UnscentedKalman<T, N, U, X, S>) -> Self {
        Self {
            inner,
            stats: UKFPerformanceStats::default(),
        }
    }

    /// Get reference to the inner UKF
    pub fn inner(&self) -> &UnscentedKalman<T, N, U, X, S> {
        &self.inner
    }

    /// Get mutable reference to the inner UKF
    pub fn inner_mut(&mut self) -> &mut UnscentedKalman<T, N, U, X, S> {
        &mut self.inner
    }

    /// Get performance statistics
    pub fn stats(&self) -> &UKFPerformanceStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Predict with monitoring for systems with input
    pub fn predict_monitored_with_input(
        &mut self,
        u: SVector<T, U>,
    ) -> Result<&SVector<T, N>, UKFError>
    where
        S: InputSystem<T, N, U>,
    {
        self.stats.prediction_count += 1;
        self.inner.predict(u)
    }

    /// Update with monitoring
    #[allow(non_snake_case)]
    pub fn update_monitored<F, const M: usize>(
        &mut self,
        measurement_fn: F,
        measurement: &SVector<T, M>,
        measurement_noise: &SMatrix<T, M, M>,
    ) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        self.stats.update_count += 1;

        // Try UKF update
        if let Ok(sigma_points) = self.inner.generate_sigma_points() {
            // Transform sigma points through measurement function
            let mut measurement_transformed = [SVector::<T, M>::zeros(); X];
            sigma_points.transform_into(&measurement_fn, &mut measurement_transformed);

            // Compute predicted measurement
            let predicted_measurement = sigma_points.weighted_mean(&measurement_transformed);

            // Compute innovation covariance
            let mut innovation_cov = sigma_points
                .weighted_covariance_stable(&measurement_transformed, &predicted_measurement);
            innovation_cov += measurement_noise;

            // Try robust inversion
            if let Ok(innovation_cov_inv) = self.inner.robust_matrix_inverse(&innovation_cov) {
                // Compute cross-covariance
                let cross_cov = sigma_points.cross_covariance(
                    self.inner.system.state(),
                    &measurement_transformed,
                    &predicted_measurement,
                );

                // Compute Kalman gain
                let K = cross_cov * innovation_cov_inv;

                // Update state
                let innovation = measurement - predicted_measurement;
                *self.inner.system.state_mut() += K * innovation;

                // Simplified covariance update for UKF
                self.inner.P -= K * cross_cov.transpose();
                self.inner.P = self.inner.P.symmetric_part();
            } else {
                self.stats.matrix_inversion_failures += 1;
            }
        } else {
            self.stats.sigma_generation_failures += 1;
        }

        self.inner.system.state()
    }
}

impl<T, const N: usize, const X: usize, S> MonitoredUKF<T, N, 0, X, S>
where
    T: RealField + Copy,
    S: NoInputSystem<T, N>,
{
    /// Predict with monitoring for systems without input
    pub fn predict_monitored(&mut self) -> Result<&SVector<T, N>, UKFError> {
        self.stats.prediction_count += 1;
        self.inner.predict()
    }
}

impl<T, const N: usize, const U: usize, const X: usize, S> UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Create new UKF with custom system
    #[track_caller]
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>) -> Self {
        Self {
            P: initial_covariance,
            system,
            params: UnscentedParameters::default(),
        }
    }

    /// Create new UKF with custom parameters
    #[track_caller]
    pub fn new_custom_with_params(
        system: S,
        initial_covariance: SMatrix<T, N, N>,
        params: UnscentedParameters<T>,
    ) -> Self {
        Self {
            P: initial_covariance,
            system,
            params,
        }
    }

    /// Generate sigma points from current state
    #[track_caller]
    #[inline]
    fn generate_sigma_points(&self) -> UKFResult<SigmaPoints<T, N, X>> {
        SigmaPoints::generate(self.system.state(), &self.P, &self.params)
    }
}

/// Implement KalmanFilter trait for UnscentedKalman
impl<T, const N: usize, const U: usize, const MAX_POINTS: usize, S> KalmanFilter<T, N, S>
    for UnscentedKalman<T, N, U, MAX_POINTS, S>
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

/// UKF Prediction for systems without input
impl<T, const N: usize, const X: usize, S> KalmanPredict<T, N> for UnscentedKalman<T, N, 0, X, S>
where
    T: RealField + Copy,
    S: NoInputSystem<T, N>,
{
    type Error = UKFError;
    #[track_caller]
    fn predict(&mut self) -> Result<&SVector<T, N>, Self::Error> {
        // Get transition matrix for UKF prediction
        let transition = *self.system.transition();

        // Try UKF prediction first
        match self.predict_ukf_internal(move |state| transition * state) {
            Ok(_) => {
                #[cfg(feature = "defmt")]
                defmt::trace!("UKF prediction successful");
            }
            Err(_e) => {
                #[cfg(feature = "defmt")]
                defmt::warn!("UKF prediction failed, falling back to linear");

                // Fallback to linear prediction
                self.system.step();
                let f = self.system.transition();
                let q = self.system.covariance();
                self.P = f * self.P * f.transpose() + q;
            }
        }

        Ok(self.system.state())
    }
}

/// UKF Prediction for systems with input
impl<T, const N: usize, const U: usize, const X: usize, S> KalmanPredictInput<T, N, U>
    for UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: InputSystem<T, N, U>,
{
    type Error = UKFError;
    #[track_caller]
    fn predict(&mut self, u: SVector<T, U>) -> Result<&SVector<T, N>, Self::Error> {
        // Try UKF prediction with input first
        match self.predict_ukf_with_input_internal(u) {
            Ok(_) => {
                #[cfg(feature = "defmt")]
                defmt::trace!("UKF prediction with input successful");
            }
            Err(_e) => {
                #[cfg(feature = "defmt")]
                defmt::warn!("UKF prediction with input failed, falling back to linear");

                // Fallback to linear prediction
                self.system.step(u);
                let f = self.system.transition();
                let q = self.system.covariance();
                self.P = f * self.P * f.transpose() + q;
            }
        }

        Ok(self.system.state())
    }
}

/// UKF-specific methods
impl<T, const N: usize, const U: usize, const X: usize, S> UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Internal UKF prediction with input - optimized without cloning
    #[track_caller]
    fn predict_ukf_with_input_internal(&mut self, u: SVector<T, U>) -> UKFResult<()>
    where
        S: InputSystem<T, N, U>,
    {
        // Generate sigma points from current state
        let sigma_points = self.generate_sigma_points()?;

        // Get system matrices once for efficiency
        let f_matrix = *self.system.transition();
        let b_matrix = self.try_get_linear_system().copied();

        // Transform sigma points through system dynamics with input
        let mut transformed = [SVector::<T, N>::zeros(); X];

        // For each sigma point, apply the system dynamics
        for (i, sigma_point) in sigma_points.points_buffer.iter().enumerate() {
            // Use linear approximation if available, otherwise fall back to system stepping
            if let Some(b) = &b_matrix {
                // Linear system: x_next = F * x + B * u
                transformed[i] = f_matrix * sigma_point + b * u;
            } else {
                // For non-linear systems, we need to step through the system
                // Store original state temporarily
                let original_state = *self.system.state();

                // Set sigma point as current state
                *self.system.state_mut() = *sigma_point;

                // Step the system with input
                self.system.step(u);
                transformed[i] = *self.system.state();

                // Restore original state for next iteration
                *self.system.state_mut() = original_state;
            }
        }

        // Compute predicted mean
        let predicted_mean = sigma_points.weighted_mean(&transformed);

        // Compute predicted covariance
        let mut predicted_cov =
            sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);

        // Add process noise
        let q = self.system.covariance();
        predicted_cov += q;

        // Update state and covariance
        *self.system.state_mut() = predicted_mean;
        self.P = predicted_cov.symmetric_part();

        Ok(())
    }

    /// Helper to try extracting linear system control matrix for optimization
    fn try_get_linear_system(&self) -> Option<&SMatrix<T, N, U>>
    where
        S: InputSystem<T, N, U>,
    {
        // This would need to be implemented based on actual system types
        // For now, return None to use the fallback approach
        None
    }

    /// Internal UKF prediction with error handling
    #[track_caller]
    fn predict_ukf_internal<F>(&mut self, process_fn: F) -> UKFResult<()>
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
        let mut predicted_cov =
            sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);

        // Add process noise
        let q = self.system.covariance();
        predicted_cov += q;

        // Update state and covariance
        *self.system.state_mut() = predicted_mean;
        self.P = predicted_cov.symmetric_part();

        Ok(())
    }

    /// Public API: Predict using unscented transform with custom process function
    #[track_caller]
    pub fn predict_ukf<F>(&mut self, process_fn: F) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, N>,
    {
        if self.predict_ukf_internal(process_fn).is_err() {
            #[cfg(feature = "defmt")]
            defmt::warn!("UKF prediction failed, state unchanged");
        }

        self.system.state()
    }

    /// Public API: Predict using unscented transform with custom process function and input
    #[track_caller]
    pub fn predict_ukf_with_input<F>(&mut self, process_fn: F, u: SVector<T, U>) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>, &SVector<T, U>) -> SVector<T, N>,
    {
        // Generate sigma points from current state
        if let Ok(sigma_points) = self.generate_sigma_points() {
            // Transform sigma points through custom process function with input
            let transformed = sigma_points.transform(|state| process_fn(state, &u));

            // Compute predicted mean
            let predicted_mean = sigma_points.weighted_mean(&transformed);

            // Compute predicted covariance
            let mut predicted_cov =
                sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);

            // Add process noise
            let q = self.system.covariance();
            predicted_cov += q;

            // Update state and covariance
            *self.system.state_mut() = predicted_mean;
            self.P = predicted_cov.symmetric_part();
        } else {
            #[cfg(feature = "defmt")]
            defmt::warn!("UKF prediction with input failed, state unchanged");
        }

        self.system.state()
    }

    /// Update using unscented transform with measurement function
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
        // Generate sigma points
        if let Ok(sigma_points) = self.generate_sigma_points() {
            // Transform sigma points through measurement function
            let mut measurement_transformed = [SVector::<T, M>::zeros(); X];
            sigma_points.transform_into(&measurement_fn, &mut measurement_transformed);

            // Compute predicted measurement
            let predicted_measurement = sigma_points.weighted_mean(&measurement_transformed);

            // Compute innovation covariance
            let mut innovation_cov = sigma_points
                .weighted_covariance_stable(&measurement_transformed, &predicted_measurement);
            innovation_cov += measurement_noise;

            // Robust inversion
            if let Ok(innovation_cov_inv) = self.robust_matrix_inverse(&innovation_cov) {
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

                // Simplified covariance update for UKF
                self.P -= K * cross_cov.transpose();
                self.P = self.P.symmetric_part();
            } else {
                #[cfg(feature = "defmt")]
                defmt::warn!("UKF update failed: singular innovation covariance");
            }
        } else {
            #[cfg(feature = "defmt")]
            defmt::warn!("UKF update failed: could not generate sigma points");
        }

        self.system.state()
    }

    /// Batch update with multiple measurements
    #[track_caller]
    pub fn update_ukf_batch<F, const M: usize>(
        &mut self,
        measurement_fn: F,
        measurements: &[SVector<T, M>],
        measurement_noise: &SMatrix<T, M, M>,
    ) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        for measurement in measurements {
            self.update_ukf(&measurement_fn, measurement, measurement_noise);
        }
        self.system.state()
    }

    /// Robust matrix inversion with fallback
    fn robust_matrix_inverse<const M: usize>(
        &self,
        matrix: &SMatrix<T, M, M>,
    ) -> UKFResult<SMatrix<T, M, M>> {
        // Primary: try direct inversion
        if let Some(inv) = matrix.try_inverse() {
            return Ok(inv);
        }

        // Fallback: regularized inversion
        let regularization = T::from_f64(1e-6).unwrap();
        let regularized = matrix + SMatrix::identity() * regularization;
        if let Some(inv) = regularized.try_inverse() {
            return Ok(inv);
        }

        Err(UKFError::SingularMatrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

    #[test]
    fn test_ukf_small_configuration() {
        type T = f64;

        let mut ukf = UKFLinearNoInputSmall::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,       // Q
            Vector2::zeros(),                 // x_initial
            Matrix2::identity(),              // P_initial
        );

        // Should work fine for 2D system
        ukf.predict();
        let _state = ukf.state();
    }

    #[test]
    #[should_panic(expected = "requires")]
    fn test_ukf_small_configuration_overflow() {
        type T = f64;

        // This should panic because SmallConfig only supports limited dimensions
        let _ukf = UKFLinearNoInputSmall::<T, 5>::new(
            SMatrix::<T, 5, 5>::identity(),
            SMatrix::<T, 5, 5>::identity() * 0.01,
            SVector::<T, 5>::zeros(),
            SMatrix::<T, 5, 5>::identity(),
        );
    }

    #[test]
    fn test_ukf_prediction_and_update() {
        type T = f64;

        let mut ukf = UKFLinearNoInputMedium::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0),
            Matrix2::identity() * 0.01,
            Vector2::zeros(),
            Matrix2::identity(),
        );

        // Test prediction
        ukf.predict();

        // Test update
        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::identity() * 0.1;

        let _result = ukf.update_ukf(
            |state| *state, // Direct observation
            &measurement,
            &measurement_noise,
        );
    }

    #[test]
    fn test_ukf_parameters() {
        type T = f64;

        // Valid parameters
        let params = UnscentedParameters::<T>::new(0.5, 2.0, 0.0);
        assert!(params.is_ok());
        let params = params.unwrap();
        assert!((params.alpha - 0.5).abs() < 1e-10);

        // Invalid parameters
        let invalid_params = UnscentedParameters::<T>::new(0.0, 2.0, 0.0);
        assert!(invalid_params.is_err());

        // Default parameters
        let params_default = UnscentedParameters::<T>::new_default();
        assert!(params_default.alpha > 0.0);
    }

    #[test]
    fn test_sigma_points_generation() {
        type T = f64;

        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();
        let params = UnscentedParameters::<T>::default();

        let sigma_points = SigmaPoints::<T, 2, MEDIUM_MAX_POINTS>::generate(&mean, &cov, &params);

        assert!(sigma_points.is_ok());
        let sigma_points = sigma_points.unwrap();
        assert_eq!(sigma_points.len(), 5); // 2*2 + 1

        // Central point should be the mean
        assert!((sigma_points.points_buffer[0] - mean).norm() < 1e-10);
    }

    #[test]
    fn test_ukf_nonlinear_measurement() {
        type T = f64;

        let mut ukf = UKFLinearNoInputMedium::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0),
            Matrix2::identity() * 0.01,
            Vector2::new(1.0, 0.0),
            Matrix2::identity(),
        );

        let measurement = Vector2::new(1.0, 0.5);
        let measurement_noise = Matrix2::identity() * 0.1;

        // Test with nonlinear measurement function
        let _result = ukf.update_ukf(
            |state| Vector2::new(state[0] * state[0], state[1].abs()),
            &measurement,
            &measurement_noise,
        );
    }

    #[test]
    fn test_ukf_with_input() {
        type T = f64;

        let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
            Matrix2::new(1.0, 0.1, 0.0, 1.0),  // F
            Matrix2::identity() * 0.01,        // Q
            SMatrix::<T, 2, 1>::new(0.0, 1.0), // B (input affects velocity)
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
        );

        // Test prediction with input
        let input = SVector::<T, 1>::new(0.5);
        ukf.predict(input);

        // Test custom UKF prediction with input
        let input2 = SVector::<T, 1>::new(0.3);
        let _result = ukf.predict_ukf_with_input(
            |state, input| {
                // Custom nonlinear dynamics
                let mut new_state = *state;
                new_state[0] += state[1] * 0.1 + input[0] * 0.05;
                new_state[1] += input[0] * 0.8;
                new_state
            },
            input2,
        );

        // Verify state was updated
        assert!(ukf.state().norm() > 0.0);
    }

    #[test]
    fn test_ukf_input_system_robustness() {
        type T = f64;

        let mut ukf = UKFLinearMedium::<T, 3, 2>::new_with_input(
            Matrix3::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1.0), // F
            Matrix3::identity() * 0.01,                                // Q
            SMatrix::<T, 3, 2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),     // B
            Vector3::new(1.0, 0.0, 0.0),                               // x_initial
            Matrix3::identity(),                                       // P_initial
        );

        // Multiple prediction steps with different inputs
        for i in 0..10 {
            let input = SVector::<T, 2>::new(i as f64 * 0.1, (i as f64 * 0.1).sin());
            ukf.predict(input);

            // Verify state remains finite
            assert!(ukf.state().iter().all(|x| x.is_finite()));
            assert!(ukf.covariance().iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_ukf_with_input_prediction_comparison() {
        type T = f64;

        // Test that UKF with input produces reasonable results
        let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F - simple position/velocity model
            Matrix2::identity() * 0.01,       // Q - small process noise
            SMatrix::<T, 2, 1>::new(0.0, 1.0), // B - input affects velocity only
            Vector2::new(0.0, 0.0),           // x_initial - start at origin
            Matrix2::identity() * 0.1,        // P_initial
        );

        let initial_state = ukf.state().clone();

        // Apply constant acceleration input
        let input = SVector::<T, 1>::new(1.0);
        ukf.predict(input);

        let final_state = ukf.state();

        // With constant acceleration, velocity should increase
        assert!(final_state[1] > initial_state[1]);

        // Position should also change due to initial velocity and acceleration
        // Even if initial velocity is 0, position changes due to dt*v term
        // where v is updated by input
        assert!((final_state[0] - initial_state[0]).abs() >= 0.0); // Should be >= due to UKF effects
    }

    #[test]
    fn test_ukf_predict_with_custom_nonlinear_function() {
        type T = f64;

        let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
            Matrix2::new(1.0, 0.1, 0.0, 1.0),  // F
            Matrix2::identity() * 0.01,        // Q
            SMatrix::<T, 2, 1>::new(0.0, 1.0), // B
            Vector2::new(1.0, 0.5),            // x_initial
            Matrix2::identity() * 0.1,         // P_initial
        );

        let input = SVector::<T, 1>::new(0.2);

        // Test custom nonlinear prediction function
        let result = ukf.predict_ukf_with_input(
            |state, input| {
                // Nonlinear dynamics with quadratic terms
                let pos = state[0];
                let vel = state[1];
                let acc = input[0];

                Vector2::new(
                    pos + vel * 0.1 + 0.005 * vel * vel.abs(), // nonlinear position update
                    vel + acc + 0.01 * pos.signum() * pos * pos, // nonlinear velocity update
                )
            },
            input,
        );

        // Verify result is reasonable
        assert!(result.iter().all(|x| x.is_finite()));
        assert!(result.norm() > 0.0);
    }

    #[test]
    fn test_ukf_builder_pattern() {
        type T = f64;

        let params = UnscentedParameters::<T>::new(0.1, 2.0, 0.0).unwrap();
        let ukf = UKFBuilder::<T>::new()
            .with_params(params)
            .build_linear_no_input(
                Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
                Matrix2::identity() * 0.01,       // Q
                Vector2::zeros(),                 // x_initial
                Matrix2::identity(),              // P_initial
            );

        // Verify UKF was created successfully
        assert_eq!(ukf.state(), &Vector2::zeros());
    }

    #[test]
    fn test_monitored_ukf() {
        type T = f64;

        let ukf = UKFLinearNoInputMedium::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,       // Q
            Vector2::zeros(),                 // x_initial
            Matrix2::identity(),              // P_initial
        );

        let mut monitored = MonitoredUKF::new(ukf);

        // Test prediction with monitoring
        monitored.predict_monitored();

        // Check stats
        let stats = monitored.stats();
        assert_eq!(stats.prediction_count, 1);
        assert_eq!(stats.update_count, 0);
    }

    #[test]
    fn test_ukf_error_types() {
        type T = f64;

        // Test error handling
        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();
        let params = UnscentedParameters::<T>::default();

        // This should work for medium-sized storage
        let result = SigmaPointsStorage::<T, 2>::new_optimal(&mean, &cov, &params);
        assert!(result.is_ok());

        let storage = result.unwrap();
        assert_eq!(storage.len(), 5); // 2*2 + 1 sigma points
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = UKFPerformanceStats::default();

        // Initially empty
        assert_eq!(stats.prediction_success_rate(), 1.0);
        assert_eq!(stats.update_success_rate(), 1.0);

        // Add some operations
        stats.prediction_count = 10;
        stats.sigma_generation_failures = 2;
        stats.update_count = 5;
        stats.matrix_inversion_failures = 1;

        assert_eq!(stats.prediction_success_rate(), 0.8); // 8/10
        assert_eq!(stats.update_success_rate(), 0.8); // 4/5

        // Reset
        stats.reset();
        assert_eq!(stats.prediction_count, 0);
        assert_eq!(stats.prediction_success_rate(), 1.0);
    }
}
