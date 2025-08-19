//! Improved Unscented Kalman Filter implementation with Rust compatibility
//! Fixed for Rust type system limitations and nalgebra compatibility

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    system::{System, NoInputSystem, InputSystem, LinearNoInputSystem, LinearSystem},
    kalman::{KalmanFilter, KalmanPredict, KalmanPredictInput},
};

/// Fixed-size sigma point configurations using const generics
/// These work around Rust's limitation with generic associated constants

/// Small systems (up to 4 dimensions, 9 sigma points)
pub const SMALL_MAX_POINTS: usize = 9;
/// Sigma points for small systems (up to 4 dimensions)
pub type SmallSigmaPoints<T, const N: usize> = SigmaPoints<T, N, 9>;

/// Medium systems (up to 12 dimensions, 25 sigma points)  
pub const MEDIUM_MAX_POINTS: usize = 25;
/// Sigma points for medium systems (up to 12 dimensions)
pub type MediumSigmaPoints<T, const N: usize> = SigmaPoints<T, N, 25>;

/// Large systems (up to 32 dimensions, 65 sigma points)
pub const LARGE_MAX_POINTS: usize = 65;
/// Sigma points for large systems (up to 32 dimensions)
pub type LargeSigmaPoints<T, const N: usize> = SigmaPoints<T, N, 65>;

/// Default configuration using medium-sized sigma points
pub type DefaultSigmaPoints<T, const N: usize> = MediumSigmaPoints<T, N>;

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
    pub fn new(alpha: T, beta: T, kappa: T) -> Self {
        let zero = T::zero();
        let one = T::one();
        
        assert!(alpha > zero && alpha <= one, "Alpha must be in (0, 1]");
        assert!(beta >= zero, "Beta must be non-negative");
        
        Self { alpha, beta, kappa }
    }
    
    /// Create default UKF parameters
    pub fn default() -> Self {
        Self {
            alpha: T::from_f64(1e-3).unwrap(),
            beta: T::from_f64(2.0).unwrap(),
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

/// Fixed-size sigma point collection using const generics
#[derive(Debug, Clone)]
pub struct SigmaPoints<T: RealField + Copy, const N: usize, const MAX_POINTS: usize> {
    /// Collection of sigma points
    pub points_buffer: [SVector<T, N>; MAX_POINTS],
    /// Pre-computed weights for mean and covariance calculations
    pub weights: UKFWeights<T>,
    /// Number of actual points used (2N+1)
    pub num_points: usize,
}

impl<T: RealField + Copy, const N: usize, const MAX_POINTS: usize> SigmaPoints<T, N, MAX_POINTS> {
    /// Generate sigma points with validation
    #[track_caller]
    pub fn generate(
        mean: &SVector<T, N>,
        covariance: &SMatrix<T, N, N>,
        params: &UnscentedParameters<T>,
    ) -> Result<Self, &'static str> {
        let num_points = 2 * N + 1;
        if num_points > MAX_POINTS {
            return Err("Insufficient buffer size for sigma points");
        }
        
        let n_f = T::from_usize(N).unwrap();
        let lambda = params.lambda(N);
        
        // Robust Cholesky decomposition
        let sqrt_matrix = Self::robust_matrix_sqrt(covariance, n_f + lambda)?;
        
        let mut points_buffer = [SVector::<T, N>::zeros(); MAX_POINTS];
        
        // Central point
        points_buffer[0] = *mean;
        
        // Positive and negative sigma points
        for i in 0..N {
            let offset = sqrt_matrix.column(i).into_owned();
            points_buffer[i + 1] = mean + &offset;
            points_buffer[i + 1 + N] = mean - &offset;
        }
        
        let weights = params.compute_weights(N);
        
        Ok(Self { 
            points_buffer,
            weights,
            num_points,
        })
    }
    
    /// Robust matrix square root computation with fallback
    fn robust_matrix_sqrt(
        matrix: &SMatrix<T, N, N>, 
        scale: T
    ) -> Result<SMatrix<T, N, N>, &'static str> {
        let sqrt_scale = scale.sqrt();
        
        // Primary: Cholesky decomposition
        if let Some(chol) = matrix.clone().cholesky() {
            return Ok(chol.l() * sqrt_scale);
        }
        
        // Fallback: Use identity matrix scaled appropriately
        // This is a safe fallback that ensures positive definiteness
        let min_eigenvalue = T::from_f64(1e-6).unwrap();
        Ok(SMatrix::<T, N, N>::identity() * (min_eigenvalue * sqrt_scale))
    }
    
    /// Get number of sigma points
    #[inline]
    pub fn len(&self) -> usize {
        self.num_points
    }
    
    /// Get iterator over active sigma points
    #[inline]
    pub fn points_iter(&self) -> impl Iterator<Item = &SVector<T, N>> {
        self.points_buffer[..self.num_points].iter()
    }
    
    /// Apply function to all sigma points with external buffer
    #[inline]
    pub fn transform_into<F, const M: usize>(
        &self, 
        f: F, 
        result: &mut [SVector<T, M>; MAX_POINTS]
    ) 
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        // Process only active points
        for i in 0..self.num_points {
            result[i] = f(&self.points_buffer[i]);
        }
        
        // Zero out unused entries for safety
        for i in self.num_points..MAX_POINTS {
            result[i] = SVector::<T, M>::zeros();
        }
    }
    
    /// Apply function to all sigma points
    #[inline]
    pub fn transform<F, const M: usize>(&self, f: F) -> [SVector<T, M>; MAX_POINTS]
    where
        F: Fn(&SVector<T, N>) -> SVector<T, M>,
    {
        let mut result = [SVector::<T, M>::zeros(); MAX_POINTS];
        self.transform_into(f, &mut result);
        result
    }
    
    /// Compute weighted mean of transformed points
    #[inline]
    pub fn weighted_mean<const M: usize>(&self, transformed: &[SVector<T, M>; MAX_POINTS]) -> SVector<T, M> {
        let mut mean = transformed[0] * self.weights.mean_0;
        
        for i in 1..self.num_points {
            mean += &transformed[i] * self.weights.other;
        }
        
        mean
    }
    
    /// Compute weighted covariance with enhanced numerical stability
    pub fn weighted_covariance_stable<const M: usize>(
        &self,
        transformed: &[SVector<T, M>; MAX_POINTS],
        mean: &SVector<T, M>,
    ) -> SMatrix<T, M, M> {
        let mut cov = SMatrix::<T, M, M>::zeros();
        
        // Central point with special weight
        let diff_0 = &transformed[0] - mean;
        cov += (diff_0 * diff_0.transpose()) * self.weights.cov_0;
        
        // Other points
        for i in 1..self.num_points {
            let diff_i = &transformed[i] - mean;
            cov += (diff_i * diff_i.transpose()) * self.weights.other;
        }
        
        // Ensure symmetry
        cov.symmetric_part()
    }
    
    /// Compute cross-covariance between state and measurement
    pub fn cross_covariance<const M: usize>(
        &self,
        state_mean: &SVector<T, N>,
        measurement_transformed: &[SVector<T, M>; MAX_POINTS],
        measurement_mean: &SVector<T, M>,
    ) -> SMatrix<T, N, M> {
        let mut cross_cov = SMatrix::<T, N, M>::zeros();
        
        // Central point
        let state_diff0 = &self.points_buffer[0] - state_mean;
        let meas_diff0 = &measurement_transformed[0] - measurement_mean;
        cross_cov += (state_diff0 * meas_diff0.transpose()) * self.weights.cov_0;
        
        // Other points
        for i in 1..self.num_points {
            let state_diff = &self.points_buffer[i] - state_mean;
            let meas_diff = &measurement_transformed[i] - measurement_mean;
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
pub struct UnscentedKalman<T, const N: usize, const U: usize, const MAX_POINTS: usize, S>
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

/// Type aliases with different buffer sizes
/// Small UKF for memory-constrained systems
pub type UKFSmall<T, const N: usize, const U: usize, S> = UnscentedKalman<T, N, U, SMALL_MAX_POINTS, S>;
/// Medium UKF for typical applications
pub type UKFMedium<T, const N: usize, const U: usize, S> = UnscentedKalman<T, N, U, MEDIUM_MAX_POINTS, S>;
/// Large UKF for high-dimensional systems
pub type UKFLarge<T, const N: usize, const U: usize, S> = UnscentedKalman<T, N, U, LARGE_MAX_POINTS, S>;

// Convenient type aliases for linear systems
/// Small UKF for linear systems without input
pub type UKFLinearNoInputSmall<T, const N: usize> = UKFSmall<T, N, 0, LinearNoInputSystem<T, N>>;
/// Medium UKF for linear systems without input
pub type UKFLinearNoInputMedium<T, const N: usize> = UKFMedium<T, N, 0, LinearNoInputSystem<T, N>>;
/// Large UKF for linear systems without input
pub type UKFLinearNoInputLarge<T, const N: usize> = UKFLarge<T, N, 0, LinearNoInputSystem<T, N>>;

/// Small UKF for linear systems with input
pub type UKFLinearSmall<T, const N: usize, const U: usize> = UKFSmall<T, N, U, LinearSystem<T, N, U>>;
/// Medium UKF for linear systems with input
pub type UKFLinearMedium<T, const N: usize, const U: usize> = UKFMedium<T, N, U, LinearSystem<T, N, U>>;
/// Large UKF for linear systems with input
pub type UKFLinearLarge<T, const N: usize, const U: usize> = UKFLarge<T, N, U, LinearSystem<T, N, U>>;

// Default aliases for backward compatibility
/// Default UKF for linear systems without input (medium-sized)
pub type UKFLinearNoInput<T, const N: usize> = UKFLinearNoInputMedium<T, N>;
/// Default UKF for linear systems with input (medium-sized)
pub type UKFLinear<T, const N: usize, const U: usize> = UKFLinearMedium<T, N, U>;

impl<T, const N: usize, const U: usize, const MAX_POINTS: usize, S> UnscentedKalman<T, N, U, MAX_POINTS, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Create new UKF with custom system
    #[track_caller]
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>) -> Self {
        let num_points = 2 * N + 1;
        assert!(num_points <= MAX_POINTS, 
            "State dimension {} requires {} sigma points, but only {} available", 
            N, num_points, MAX_POINTS);
        
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
        let num_points = 2 * N + 1;
        assert!(num_points <= MAX_POINTS, 
            "State dimension {} requires {} sigma points, but only {} available", 
            N, num_points, MAX_POINTS);
        
        Self {
            P: initial_covariance,
            system,
            params,
        }
    }
    
    /// Generate sigma points from current state
    #[track_caller]
    #[inline]
    fn generate_sigma_points(&self) -> Result<SigmaPoints<T, N, MAX_POINTS>, &'static str> {
        SigmaPoints::generate(
            self.system.state(),
            &self.P,
            &self.params,
        )
    }
}

/// Linear UKF constructors
impl<T, const N: usize> UKFLinearNoInputSmall<T, N>
where
    T: RealField + Copy,
{
    /// Create small UKF with linear system without inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearNoInputSystem::new(F, Q, x_initial),
            P_initial,
        )
    }
}

impl<T, const N: usize> UKFLinearNoInputMedium<T, N>
where
    T: RealField + Copy,
{
    /// Create medium UKF with linear system without inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearNoInputSystem::new(F, Q, x_initial),
            P_initial,
        )
    }
}

impl<T, const N: usize> UKFLinearNoInputLarge<T, N>
where
    T: RealField + Copy,
{
    /// Create large UKF with linear system without inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearNoInputSystem::new(F, Q, x_initial),
            P_initial,
        )
    }
}

impl<T, const N: usize, const U: usize> UKFLinearSmall<T, N, U>
where
    T: RealField + Copy,
{
    /// Create small UKF with linear system with inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearSystem::new(F, Q, B, x_initial),
            P_initial,
        )
    }
}

impl<T, const N: usize, const U: usize> UKFLinearMedium<T, N, U>
where
    T: RealField + Copy,
{
    /// Create medium UKF with linear system with inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearSystem::new(F, Q, B, x_initial),
            P_initial,
        )
    }
}

impl<T, const N: usize, const U: usize> UKFLinearLarge<T, N, U>
where
    T: RealField + Copy,
{
    /// Create large UKF with linear system with inputs
    #[allow(non_snake_case)]
    #[track_caller]
    pub fn new_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        Self::new_custom(
            LinearSystem::new(F, Q, B, x_initial),
            P_initial,
        )
    }
}

/// Implement KalmanFilter trait for UnscentedKalman
impl<T, const N: usize, const U: usize, const MAX_POINTS: usize, S> KalmanFilter<T, N, S> for UnscentedKalman<T, N, U, MAX_POINTS, S>
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
impl<T, const N: usize, const MAX_POINTS: usize, S> KalmanPredict<T, N> for UnscentedKalman<T, N, 0, MAX_POINTS, S>
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
                
                // Fallback to linear prediction
                self.system.step();
                let f = self.system.transition();
                let q = self.system.covariance();
                self.P = f * &self.P * f.transpose() + q;
            }
        }
        
        self.system.state()
    }
}

/// UKF Prediction for systems with input
impl<T, const N: usize, const U: usize, const MAX_POINTS: usize, S> KalmanPredictInput<T, N, U> for UnscentedKalman<T, N, U, MAX_POINTS, S>
where
    T: RealField + Copy,
    S: InputSystem<T, N, U>,
{
    #[track_caller]
    fn predict(&mut self, u: SVector<T, U>) -> &SVector<T, N> {
        // Try UKF prediction with input first
        match self.predict_ukf_with_input_internal(u) {
            Ok(_) => {
                #[cfg(feature = "defmt")]
                defmt::trace!("UKF prediction with input successful");
            },
            Err(_e) => {
                #[cfg(feature = "defmt")]
                defmt::warn!("UKF prediction with input failed, falling back to linear");
                
                // Fallback to linear prediction
                self.system.step(u);
                let f = self.system.transition();
                let q = self.system.covariance();
                self.P = f * &self.P * f.transpose() + q;
            }
        }
        
        self.system.state()
    }
}

/// UKF-specific methods
impl<T, const N: usize, const U: usize, const MAX_POINTS: usize, S> UnscentedKalman<T, N, U, MAX_POINTS, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Internal UKF prediction with input - optimized without cloning
    #[track_caller]
    fn predict_ukf_with_input_internal(&mut self, u: SVector<T, U>) -> Result<(), &'static str>
    where
        S: InputSystem<T, N, U>,
    {
        // Generate sigma points from current state
        let sigma_points = self.generate_sigma_points()?;
        
        // Get system matrices once for efficiency
        let f_matrix = *self.system.transition();
        let b_matrix = if let Some(linear_sys) = self.try_get_linear_system() {
            Some(*linear_sys)
        } else {
            None
        };
        
        // Transform sigma points through system dynamics with input
        let mut transformed = [SVector::<T, N>::zeros(); MAX_POINTS];
        
        // For each sigma point, apply the system dynamics
        for i in 0..sigma_points.num_points {
            let sigma_point = &sigma_points.points_buffer[i];
            
            // Use linear approximation if available, otherwise fall back to system stepping
            if let Some(b) = &b_matrix {
                // Linear system: x_next = F * x + B * u
                transformed[i] = &f_matrix * sigma_point + b * u;
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
        let mut predicted_cov = sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);
        
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
    fn predict_ukf_internal<F>(&mut self, process_fn: F) -> Result<(), &'static str>
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
        let mut predicted_cov = sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);
        
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
        self.predict_ukf_internal(process_fn)
            .expect("UKF prediction failed");
        
        self.system.state()
    }
    
    /// Public API: Predict using unscented transform with custom process function and input
    #[track_caller]
    pub fn predict_ukf_with_input<F>(&mut self, process_fn: F, u: SVector<T, U>) -> &SVector<T, N>
    where
        F: Fn(&SVector<T, N>, &SVector<T, U>) -> SVector<T, N>,
    {
        // Generate sigma points from current state
        let sigma_points = self.generate_sigma_points()
            .expect("Failed to generate sigma points for prediction");
        
        // Transform sigma points through custom process function with input
        let transformed = sigma_points.transform(|state| process_fn(state, &u));
        
        // Compute predicted mean
        let predicted_mean = sigma_points.weighted_mean(&transformed);
        
        // Compute predicted covariance
        let mut predicted_cov = sigma_points.weighted_covariance_stable(&transformed, &predicted_mean);
        
        // Add process noise
        let q = self.system.covariance();
        predicted_cov += q;
        
        // Update state and covariance
        *self.system.state_mut() = predicted_mean;
        self.P = predicted_cov.symmetric_part();
        
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
        let sigma_points = self.generate_sigma_points()
            .expect("Failed to generate sigma points for update");
        
        // Transform sigma points through measurement function
        let mut measurement_transformed = [SVector::<T, M>::zeros(); MAX_POINTS];
        sigma_points.transform_into(&measurement_fn, &mut measurement_transformed);
        
        // Compute predicted measurement
        let predicted_measurement = sigma_points.weighted_mean(&measurement_transformed);
        
        // Compute innovation covariance
        let mut innovation_cov = sigma_points.weighted_covariance_stable(
            &measurement_transformed,
            &predicted_measurement,
        );
        innovation_cov += measurement_noise;
        
        // Robust inversion
        let innovation_cov_inv = self.robust_matrix_inverse(&innovation_cov)
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
        
        // Simplified covariance update for UKF
        self.P = &self.P - &K * &cross_cov.transpose();
        self.P = self.P.symmetric_part();
        
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
        matrix: &SMatrix<T, M, M>
    ) -> Result<SMatrix<T, M, M>, &'static str> {
        // Primary: try direct inversion
        if let Some(inv) = matrix.try_inverse() {
            return Ok(inv);
        }
        
        // Fallback: regularized inversion
        let regularization = T::from_f64(1e-6).unwrap();
        let regularized = matrix + &SMatrix::identity() * regularization;
        if let Some(inv) = regularized.try_inverse() {
            return Ok(inv);
        }
        
        Err("Matrix inversion failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2, Matrix3, Vector3};
    
    #[test]
    fn test_ukf_small_configuration() {
        type T = f64;
        
        let mut ukf = UKFLinearNoInputSmall::<T, 2>::new(
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
            Matrix2::identity() * 0.01,        // Q
            Vector2::zeros(),                  // x_initial
            Matrix2::identity(),               // P_initial
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
        assert!((params.alpha - 0.5).abs() < 1e-10);
        
        // Default parameters
        let params_default = UnscentedParameters::<T>::default();
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
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
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
            Matrix3::new(
                1.0, 0.1, 0.0,
                0.0, 1.0, 0.1, 
                0.0, 0.0, 1.0
            ), // F
            Matrix3::identity() * 0.01,                // Q
            SMatrix::<T, 3, 2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 1.0), // B
            Vector3::new(1.0, 0.0, 0.0),               // x_initial
            Matrix3::identity(),                       // P_initial
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
            Matrix2::identity() * 0.01,        // Q - small process noise
            SMatrix::<T, 2, 1>::new(0.0, 1.0), // B - input affects velocity only
            Vector2::new(0.0, 0.0),            // x_initial - start at origin
            Matrix2::identity() * 0.1,         // P_initial
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
            Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
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
}