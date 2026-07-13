//! Unscented Kalman Filter implementation

use nalgebra::{RealField, SMatrix, SVector};

use crate::{
    kalman::{KalmanFilter, KalmanPredict},
    measurement::Measurement,
    system::{InputSystem, LinearNoInputSystem, LinearSystem, NoInputSystem, System},
    KalmanPredictInput, KalmanUpdate,
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

/// Result type for UKF operations
pub type UKFResult<T> = Result<T, UKFError>;

/// Parameters for the Unscented Transform with validation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct UKFParameters<T: RealField + Copy> {
    /// Alpha parameter (0 < alpha <= 1), controls spread of sigma points
    pub alpha: T,
    /// Beta parameter (beta >= 0), incorporates prior knowledge of distribution
    pub beta: T,
    /// Kappa parameter, secondary scaling parameter (usually 0 or 3-n)
    pub kappa: T,
}

impl<T: RealField + Copy> UKFParameters<T> {
    /// Create new UKF parameters with validation
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
    pub fn lambda(&self, n: usize) -> T {
        let n_f = T::from_usize(n).unwrap();
        self.alpha * self.alpha * (n_f + self.kappa) - n_f
    }
}

impl<T: RealField + Copy> Default for UKFParameters<T> {
    fn default() -> Self {
        Self::new_default()
    }
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
    pub points: [SVector<T, N>; X],
    /// Mean weight
    pub mean_weight: T,
    /// Mean weight for the covariance
    pub mean_cov_weight: T,
    /// Weight for the rest of the points
    pub weight: T,
}

impl<T: RealField + Copy, const N: usize, const X: usize> SigmaPoints<T, N, X> {
    /// Generate sigma points based on the common use of the parameters (e.g Wan and van der Merwe)
    pub fn new(
        mean: &SVector<T, N>,
        covariance: &SMatrix<T, N, N>,
        params: &UKFParameters<T>,
    ) -> UKFResult<Self> {
        // Use const assert to check that S = 2N + 1
        #[allow(clippy::let_unit_value)]
        let _ = <Self as ValidSigma>::VALID;

        let n_f = T::from_usize(N).unwrap();
        let mut points = [SVector::<T, N>::zeros(); X];

        // First point is the mean
        points[0] = *mean;
        // Assign weights
        let lambda = params.lambda(N);
        // n_f + lambda is a divisor for every weight below and is square-rooted
        // for the sigma point spread; zero (e.g. kappa == -N) or negative values
        // would silently produce NaN/Inf rather than a usable filter.
        if n_f + lambda <= T::zero() {
            return Err(UKFError::InvalidParameters);
        }
        let mean_weight = lambda / (n_f + lambda);
        let mean_cov_weight = mean_weight + (T::one() - params.alpha.powi(2) + params.beta);
        let weight = T::one() / (T::from_f32(2.0).unwrap() * (n_f + lambda));

        // Covariance square root
        let sqrt =
            (covariance).cholesky().ok_or(UKFError::SingularMatrix)?.l() * (n_f + lambda).sqrt();

        // Fill the rest of the points
        // Split into positive and negative halfs, ignore the first point
        let (pos, neg) = points[1..].split_at_mut(N);
        // Iteratve over the pos/neg pairs and each colum of sqrt
        for ((p, r), c) in pos.iter_mut().zip(neg.iter_mut()).zip(sqrt.column_iter()) {
            *p = mean + c;
            *r = mean - c;
        }

        Ok(Self {
            points,
            mean_weight,
            mean_cov_weight,
            weight,
        })
    }

    /// Take a vector of either system or measurement predictions and calculate the mean and covariance
    fn mean_and_covariance<const L: usize>(
        &self,
        predictions: &[SVector<T, L>; X],
        covariance: &SMatrix<T, L, L>,
    ) -> (SVector<T, L>, SMatrix<T, L, L>) {
        // Calculate weighted mean
        let mut mean = predictions[0] * self.mean_weight;
        for pred in predictions.iter().skip(1) {
            mean += pred * self.weight;
        }

        // Calculate weighted covariance
        let mut cov =
            ((predictions[0] - mean) * (predictions[0] - mean).transpose()) * self.mean_cov_weight;
        for pred in predictions.iter().skip(1) {
            let diff = pred - mean;
            cov += (diff * diff.transpose()) * self.weight;
        }
        cov += covariance;
        (mean, cov)
    }

    // Cross covariance between the sigma points and the observations
    fn cross_covariance<const L: usize>(
        &self,
        predicted_observations: &[SVector<T, L>; X],
    ) -> SMatrix<T, N, L> {
        // The mean of the sigma points is the first entry
        // Get mean of the predicted observations
        // let obs_mean = predicted_observations
        //     .iter()
        //     .map(|obs| obs * self.weight)
        //     .sum();
        self.points
            .iter()
            .zip(predicted_observations.iter())
            .skip(1) // Skip the first point (mean)
            .map(|(state, obs)| {
                let state_diff = state - self.points[0];
                let obs_diff = obs - predicted_observations[0];
                state_diff * obs_diff.transpose() * self.weight
            })
            .sum()
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
    P: SMatrix<T, N, N>,
    /// System model
    pub system: S,
    /// UKF parameters
    params: UKFParameters<T>,
}

impl<T: RealField + Copy, const N: usize, const U: usize, const X: usize, S> ValidSigma
    for UnscentedKalman<T, N, U, X, S>
where
    S: System<T, N, U>,
{
    const VALID: () = assert!(X == 2 * N + 1);
}

impl<T, const N: usize, const U: usize, const X: usize, S> UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
{
    /// Create new UKF with custom system
    pub fn new_custom(system: S, initial_covariance: SMatrix<T, N, N>) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = <Self as ValidSigma>::VALID;
        Self {
            P: initial_covariance,
            system,
            params: UKFParameters::default(),
        }
    }

    /// Create new UKF with custom parameters
    pub fn new_custom_with_params(
        system: S,
        initial_covariance: SMatrix<T, N, N>,
        params: UKFParameters<T>,
    ) -> Self {
        Self {
            P: initial_covariance,
            system,
            params,
        }
    }

    /// Generate sigma points from current state
    fn generate_sigma_points(&self) -> UKFResult<SigmaPoints<T, N, X>> {
        SigmaPoints::new(self.system.state(), &self.P, &self.params)
    }
}

/// Implement KalmanFilter trait for UnscentedKalman
impl<T, const N: usize, const U: usize, const X: usize, S> KalmanFilter<T, N, S>
    for UnscentedKalman<T, N, U, X, S>
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
    fn predict(&mut self) -> Result<&SVector<T, N>, Self::Error> {
        // Generate sigma points and weights, pass through system predict
        // and calculate predicted mean and covariance

        // Generate sigma points
        let sigma_points = self.generate_sigma_points()?;
        // map each point to a predicted state
        let predicted_states = sigma_points.points.map(|point| self.system.predict(&point));
        let (predicted_mean, cov) =
            sigma_points.mean_and_covariance(&predicted_states, self.system.covariance());
        // Assign the new mean and covariance to the system
        *self.system.state_mut() = predicted_mean;
        self.P = cov;
        // Return reference to new state
        Ok(self.system.state())
    }
}

/// UKF Prediction for system with input
impl<T, const N: usize, const U: usize, const X: usize, S> KalmanPredictInput<T, N, U>
    for UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: InputSystem<T, N, U>,
{
    type Error = UKFError;
    fn predict(&mut self, u: SVector<T, U>) -> Result<&SVector<T, N>, Self::Error> {
        // Generate sigma points and weights, pass through system predict
        // and calculate predicted mean and covariance

        // Generate sigma points
        let sigma_points = self.generate_sigma_points()?;
        // map each point to a predicted state
        let predicted_states = sigma_points
            .points
            .map(|point| self.system.predict(&point, &u));
        let (predicted_mean, cov) =
            sigma_points.mean_and_covariance(&predicted_states, self.system.covariance());
        // Assign the new mean and covariance to the system
        *self.system.state_mut() = predicted_mean;
        self.P = cov;
        // Return reference to new state
        Ok(self.system.state())
    }
}

/// UKF Update
impl<T, const N: usize, const U: usize, const X: usize, S, const M: usize, ME>
    KalmanUpdate<T, N, M, ME> for UnscentedKalman<T, N, U, X, S>
where
    T: RealField + Copy,
    S: System<T, N, U>,
    ME: Measurement<T, N, M>,
{
    type Error = UKFError;

    fn update(&mut self, measurement: &ME) -> Result<&SVector<T, N>, Self::Error> {
        // Generate sigma points from current state(mean) and covariance
        let sigma_points = self.generate_sigma_points()?;
        // map each point to a predicted observation based on the measurement function
        let predicted_observations = sigma_points.points.map(|point| measurement.predict(&point));
        // Calculate the mean and covariance of the predictions
        let (predicted_mean, innovation_cov) =
            sigma_points.mean_and_covariance(&predicted_observations, measurement.covariance());
        // calculate cross covariance between sigma points and predicted observations
        let cross_cov = sigma_points.cross_covariance(&predicted_observations);
        // Kalman gain is product of cross_covariance and inverse of innovation_cov
        let kalman_gain = cross_cov
            * innovation_cov
                .try_inverse()
                .ok_or(UKFError::SingularMatrix)?;
        // Update state based on current state and measurement error
        *self.system.state_mut() += kalman_gain * (measurement.measurement() - predicted_mean);
        // Update covariance
        self.P -= kalman_gain * innovation_cov * kalman_gain.transpose();
        // Ensure covariance remains symmetric
        self.P = self.P.symmetric_part();
        Ok(self.system.state())
    }
}

/// Type alias for a UKF with linear system and measurement. Used to verify with KF
pub type UKFLinear<T, const N: usize, const U: usize, const X: usize> =
    UnscentedKalman<T, N, U, X, LinearSystem<T, N, U>>;

/// Type alias for a UKF with no input and linear measurement. Used to verify with KF
pub type UKFLinearNoInput<T, const N: usize, const X: usize> =
    UnscentedKalman<T, N, 0, X, LinearNoInputSystem<T, N>>;

impl<T: RealField + Copy, const N: usize, const U: usize, const X: usize> UKFLinear<T, N, U, X> {
    #[allow(non_snake_case)]
    /// Create a UKF with linear system and measurement
    pub fn new_linear_with_input(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        B: SMatrix<T, N, U>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        let system = LinearSystem::new(F, Q, B, x_initial);
        Self::new_custom(system, P_initial)
    }
}

impl<T: RealField + Copy, const N: usize, const X: usize> UKFLinearNoInput<T, N, X> {
    #[allow(non_snake_case)]
    /// Create a UKF with linear system and measurement
    pub fn new_linear(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        x_initial: SVector<T, N>,
        P_initial: SMatrix<T, N, N>,
    ) -> Self {
        let system = LinearNoInputSystem::new(F, Q, x_initial);
        Self::new_custom(system, P_initial)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use crate::Kalman;

    use super::*;
    use log::debug;
    use nalgebra::{Matrix2, Vector2};

    use rand::Rng;
    use test_log::test;

    fn rand_covariance<const N: usize>() -> SMatrix<f64, N, N> {
        let mut rng = rand::thread_rng();
        SMatrix::from_diagonal(&SVector::from_fn(|_, _| rng.gen_range(0.1..1.0)))
    }

    fn rand_state<const N: usize>() -> SVector<f64, N> {
        let mut rng = rand::thread_rng();
        SVector::from_fn(|_, _| rng.gen_range(-1.0..1.0))
    }

    fn check_sigma_points<const N: usize, const U: usize, const X: usize>() {
        for _ in 0..100 {
            let alpha = rand::thread_rng().gen_range(0.0001..1.0);
            let beta = rand::thread_rng().gen_range(0.0..5.0);
            let kappa = rand::thread_rng().gen_range(0.0..3.0);
            let params = UKFParameters::new(alpha, beta, kappa).unwrap();
            // Print the params
            debug!("UKF Parameters: {params:?}");
            debug!("Lambda: {}", params.lambda(N));
            // create mean and covariance
            let mean = rand_state::<N>();
            let cov = rand_covariance::<N>();
            debug!("Mean: {mean:?}");
            debug!("Covariance: {cov:?}");
            let sigma_points = SigmaPoints::<f64, N, X>::new(&mean, &cov, &params).unwrap();
            debug!("Sigma Points: {sigma_points:?}");
            // Check that the sum of weights is 1
            let weight = 2.0 * (N as f64) * sigma_points.weight + sigma_points.mean_weight;
            debug!("Sum of weights: {weight}");
            assert!((weight - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn sigma_points() {
        check_sigma_points::<2, 0, 5>();
        check_sigma_points::<3, 0, 7>();
        check_sigma_points::<4, 0, 9>();
    }

    fn compare_to_kf_predict_identity<const N: usize, const X: usize>() {
        // Create linear no input kf and check ukf produces same result
        let x_initial = SVector::<f64, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let mut kf = Kalman::<f64, N, 0, _>::new(
            SMatrix::identity(),
            SMatrix::identity(),
            x_initial,
            SMatrix::identity(),
        );

        let mut ukf: UnscentedKalman<f64, N, 0, X, _> = UnscentedKalman::new_linear(
            SMatrix::identity(),
            SMatrix::identity(),
            x_initial,
            SMatrix::identity(),
        );

        for _ in 0..10 {
            let kf_state = kf.predict().unwrap();
            let ukf_state = ukf.predict().unwrap();
            debug!("KF State: {kf_state:?}");
            debug!("UKF State: {ukf_state:?}");
            // Check that predicted state is the same
            assert!(kf_state
                .iter()
                .zip(ukf_state.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
            debug!("KF Covariance: {:?}", kf.covariance());
            debug!("UKF Covariance: {:?}", ukf.covariance());
            // Check that covariance is the same
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    fn compare_to_kf_predict_random<const N: usize, const X: usize>() {
        // Create linear no input kf and check ukf produces same result
        let x_initial =
            SVector::<f64, N>::from_fn(|_, _| rand::prelude::thread_rng().gen_range(-1.0..1.0));
        // Create a random F matrix
        let F = SMatrix::<f64, N, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let Q = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.01..0.1)
        }));
        let P = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));
        let mut kf = Kalman::<f64, N, 0, _>::new(F, Q, x_initial, P);

        let mut ukf: UnscentedKalman<f64, N, 0, X, _> =
            UnscentedKalman::new_linear(F, Q, x_initial, P);

        for _ in 0..10 {
            kf.predict().unwrap();
            ukf.predict().unwrap();
            debug!(target: "test", "KF Covariance: {:?}", kf.covariance());
            debug!(target: "test", "UKF Covariance: {:?}", ukf.covariance());
            // Check that each covariance value is with tolerance
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    fn compare_to_kf_predict_input<const N: usize, const U: usize, const X: usize>() {
        // Create linear no input kf and check ukf produces same result
        let x_initial = SVector::<f64, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let F = SMatrix::<f64, N, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let B = SMatrix::<f64, N, U>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let Q = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.01..0.1)
        }));
        let P = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));

        let mut kf = Kalman::<f64, N, U, _>::new_with_input(F, Q, B, x_initial, P);

        let mut ukf: UnscentedKalman<f64, N, U, X, _> =
            UnscentedKalman::new_linear_with_input(F, Q, B, x_initial, P);

        let u = SVector::<f64, U>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let kf_state = kf.predict(u).unwrap();
        let ukf_state = ukf.predict(u).unwrap();
        debug!("KF State: {kf_state:?}");
        debug!("UKF State: {ukf_state:?}");
        // Check that predicted state is the same
        assert!(kf_state
            .iter()
            .zip(ukf_state.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5));
        debug!("KF Covariance: {:?}", kf.covariance());
        debug!("UKF Covariance: {:?}", ukf.covariance());
        // Check that covariance is the same
        assert!(kf
            .covariance()
            .iter()
            .zip(ukf.covariance().iter())
            .all(|(a, b)| (a - b).abs() < 1e-5));
    }
    #[test]
    fn compare_to_kf_predict() {
        // Compare different sizes to linear kf with no input
        compare_to_kf_predict_identity::<1, 3>();
        compare_to_kf_predict_identity::<2, 5>();
        compare_to_kf_predict_identity::<3, 7>();
        compare_to_kf_predict_identity::<5, 11>();
        compare_to_kf_predict_identity::<12, 25>();

        // Do the same with random F matrices
        compare_to_kf_predict_random::<1, 3>();
        compare_to_kf_predict_random::<2, 5>();
        compare_to_kf_predict_random::<3, 7>();
        compare_to_kf_predict_random::<5, 11>();
        compare_to_kf_predict_random::<12, 25>();

        // System with inputs
        compare_to_kf_predict_input::<1, 1, 3>();
        compare_to_kf_predict_input::<2, 1, 5>();
        compare_to_kf_predict_input::<2, 2, 5>();
        compare_to_kf_predict_input::<2, 3, 5>();
        compare_to_kf_predict_input::<6, 6, 13>();
    }

    fn compare_to_kf<const N: usize, const X: usize, const M: usize>() {
        // Create linear no input kf and check ukf produces same result
        let x_initial = SVector::<f64, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        // Create a random F matrix
        let F = SMatrix::<f64, N, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let Q = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.01..0.1)
        }));
        let H = SMatrix::<f64, M, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let R = SMatrix::from_diagonal(&SVector::<f64, M>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));
        let P = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));
        let mut kf = Kalman::<f64, N, 0, _>::new(F, Q, x_initial, P);
        let measurement =
            SVector::<f64, M>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let meas_model = crate::measurement::LinearMeasurement::new(H, R, measurement);

        let mut ukf: UnscentedKalman<f64, N, 0, X, _> =
            UnscentedKalman::new_linear(F, Q, x_initial, P);

        for _ in 0..10 {
            kf.predict().unwrap();
            ukf.predict().unwrap();
            debug!(target: "test", "KF Covariance: {:?}", kf.covariance());
            debug!(target: "test", "UKF Covariance: {:?}", ukf.covariance());
            // Check that each covariance value is with tolerance
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));

            kf.update(&meas_model).unwrap();
            ukf.update(&meas_model).unwrap();
            debug!(target: "test", "KF State: {:?}", kf.state());
            debug!(target: "test", "UKF State: {:?}", ukf.state());
            // Check that each state value is with tolerance
            assert!(kf
                .state()
                .iter()
                .zip(ukf.state().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
            debug!(target: "test", "KF Covariance after update: {:?}", kf.covariance());
            debug!(target: "test", "UKF Covariance after update: {:?}", ukf.covariance());
            // Check that each covariance value is with tolerance
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    fn compare_to_kf_input<const N: usize, const U: usize, const X: usize, const M: usize>() {
        // Create linear no input kf and check ukf produces same result
        let x_initial = SVector::<f64, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        // Create a random F matrix
        let F = SMatrix::<f64, N, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let B = SMatrix::<f64, N, U>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let Q = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.01..0.1)
        }));
        let H = SMatrix::<f64, M, N>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let R = SMatrix::from_diagonal(&SVector::<f64, M>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));
        let P = SMatrix::from_diagonal(&SVector::<f64, N>::from_fn(|_, _| {
            rand::thread_rng().gen_range(0.1..1.0)
        }));
        let mut kf = Kalman::<f64, N, U, _>::new_with_input(F, Q, B, x_initial, P);
        let measurement =
            SVector::<f64, M>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let meas_model = crate::measurement::LinearMeasurement::new(H, R, measurement);

        let mut ukf: UnscentedKalman<f64, N, U, X, _> =
            UnscentedKalman::new_linear_with_input(F, Q, B, x_initial, P);

        for _ in 0..10 {
            let u = SVector::<f64, U>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
            kf.predict(u).unwrap();
            ukf.predict(u).unwrap();
            debug!(target: "test", "KF Covariance: {:?}", kf.covariance());
            debug!(target: "test", "UKF Covariance: {:?}", ukf.covariance());
            // Check that each covariance value is with tolerance
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));

            kf.update(&meas_model).unwrap();
            ukf.update(&meas_model).unwrap();
            debug!(target: "test", "KF State: {:?}", kf.state());
            debug!(target: "test", "UKF State: {:?}", ukf.state());
            // Check that each state value is with tolerance
            assert!(kf
                .state()
                .iter()
                .zip(ukf.state().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
            debug!(target: "test", "KF Covariance after update: {:?}", kf.covariance());
            debug!(target: "test", "UKF Covariance after update: {:?}", ukf.covariance());
            // Check that each covariance value is with tolerance
            assert!(kf
                .covariance()
                .iter()
                .zip(ukf.covariance().iter())
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }

    #[test]
    fn compare_to_kf_update() {
        compare_to_kf::<1, 3, 1>();
        compare_to_kf::<2, 5, 1>();
        compare_to_kf::<3, 7, 3>();
        compare_to_kf::<6, 13, 6>();

        compare_to_kf_input::<1, 1, 3, 1>();
        compare_to_kf_input::<2, 2, 5, 1>();
        compare_to_kf_input::<3, 3, 7, 3>();
        compare_to_kf_input::<6, 6, 13, 6>();
    }

    #[test]
    fn test_ukf_parameters() {
        type T = f64;

        // Valid parameters
        let params = UKFParameters::<T>::new(0.5, 2.0, 0.0);
        assert!(params.is_ok());

        // Invalid parameters
        let invalid_params = UKFParameters::<T>::new(0.0, 2.0, 0.0);
        assert!(invalid_params.is_err());
    }

    #[test]
    fn test_sigma_points_generation() {
        type T = f64;

        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();
        let params = UKFParameters::<T>::default();

        let sigma_points = SigmaPoints::<T, 2, 5>::new(&mean, &cov, &params).unwrap();
        println!("Sigma Points: {sigma_points:?}");

        assert!((sigma_points.points[0] - mean).norm() < 1e-10);
    }

    #[test]
    fn sigma_points_reject_degenerate_kappa() {
        type T = f64;

        // N = 2, kappa = -N makes n + lambda == 0, which previously divided by
        // zero and produced NaN sigma point weights instead of an error.
        let params = UKFParameters::<T>::new(1.0, 0.0, -2.0).unwrap();
        let mean = Vector2::new(1.0, 2.0);
        let cov = Matrix2::identity();

        let result = SigmaPoints::<T, 2, 5>::new(&mean, &cov, &params);
        assert!(matches!(result, Err(UKFError::InvalidParameters)));
    }

    // #[test]
    // fn test_ukf_nonlinear_measurement() {
    //     type T = f64;

    //     let mut ukf = UKFLinearNoInputMedium::<T, 2>::new(
    //         Matrix2::new(1.0, 0.1, 0.0, 1.0),
    //         Matrix2::identity() * 0.01,
    //         Vector2::new(1.0, 0.0),
    //         Matrix2::identity(),
    //     );

    //     let measurement = Vector2::new(1.0, 0.5);
    //     let measurement_noise = Matrix2::identity() * 0.1;

    //     // Test with nonlinear measurement function
    //     let _result = ukf.update_ukf(
    //         |state| Vector2::new(state[0] * state[0], state[1].abs()),
    //         &measurement,
    //         &measurement_noise,
    //     );
    // }

    // #[test]
    // fn test_ukf_with_input() {
    //     type T = f64;

    //     let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
    //         Matrix2::new(1.0, 0.1, 0.0, 1.0),  // F
    //         Matrix2::identity() * 0.01,        // Q
    //         SMatrix::<T, 2, 1>::new(0.0, 1.0), // B (input affects velocity)
    //         Vector2::zeros(),                  // x_initial
    //         Matrix2::identity(),               // P_initial
    //     );

    //     // Test prediction with input
    //     let input = SVector::<T, 1>::new(0.5);
    //     ukf.predict(input);

    //     // Test custom UKF prediction with input
    //     let input2 = SVector::<T, 1>::new(0.3);
    //     let _result = ukf.predict_ukf_with_input(
    //         |state, input| {
    //             // Custom nonlinear dynamics
    //             let mut new_state = *state;
    //             new_state[0] += state[1] * 0.1 + input[0] * 0.05;
    //             new_state[1] += input[0] * 0.8;
    //             new_state
    //         },
    //         input2,
    //     );

    //     // Verify state was updated
    //     assert!(ukf.state().norm() > 0.0);
    // }

    // #[test]
    // fn test_ukf_input_system_robustness() {
    //     type T = f64;

    //     let mut ukf = UKFLinearMedium::<T, 3, 2>::new_with_input(
    //         Matrix3::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1.0), // F
    //         Matrix3::identity() * 0.01,                                // Q
    //         SMatrix::<T, 3, 2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),     // B
    //         Vector3::new(1.0, 0.0, 0.0),                               // x_initial
    //         Matrix3::identity(),                                       // P_initial
    //     );

    //     // Multiple prediction steps with different inputs
    //     for i in 0..10 {
    //         let input = SVector::<T, 2>::new(i as f64 * 0.1, (i as f64 * 0.1).sin());
    //         ukf.predict(input);

    //         // Verify state remains finite
    //         assert!(ukf.state().iter().all(|x| x.is_finite()));
    //         assert!(ukf.covariance().iter().all(|x| x.is_finite()));
    //     }
    // }

    // #[test]
    // fn test_ukf_with_input_prediction_comparison() {
    //     type T = f64;

    //     // Test that UKF with input produces reasonable results
    //     let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
    //         Matrix2::new(1.0, 0.1, 0.0, 1.0), // F - simple position/velocity model
    //         Matrix2::identity() * 0.01,       // Q - small process noise
    //         SMatrix::<T, 2, 1>::new(0.0, 1.0), // B - input affects velocity only
    //         Vector2::new(0.0, 0.0),           // x_initial - start at origin
    //         Matrix2::identity() * 0.1,        // P_initial
    //     );

    //     let initial_state = ukf.state().clone();

    //     // Apply constant acceleration input
    //     let input = SVector::<T, 1>::new(1.0);
    //     ukf.predict(input);

    //     let final_state = ukf.state();

    //     // With constant acceleration, velocity should increase
    //     assert!(final_state[1] > initial_state[1]);

    //     // Position should also change due to initial velocity and acceleration
    //     // Even if initial velocity is 0, position changes due to dt*v term
    //     // where v is updated by input
    //     assert!((final_state[0] - initial_state[0]).abs() >= 0.0); // Should be >= due to UKF effects
    // }

    // #[test]
    // fn test_ukf_predict_with_custom_nonlinear_function() {
    //     type T = f64;

    //     let mut ukf = UKFLinearMedium::<T, 2, 1>::new_with_input(
    //         Matrix2::new(1.0, 0.1, 0.0, 1.0),  // F
    //         Matrix2::identity() * 0.01,        // Q
    //         SMatrix::<T, 2, 1>::new(0.0, 1.0), // B
    //         Vector2::new(1.0, 0.5),            // x_initial
    //         Matrix2::identity() * 0.1,         // P_initial
    //     );

    //     let input = SVector::<T, 1>::new(0.2);

    //     // Test custom nonlinear prediction function
    //     let result = ukf.predict_ukf_with_input(
    //         |state, input| {
    //             // Nonlinear dynamics with quadratic terms
    //             let pos = state[0];
    //             let vel = state[1];
    //             let acc = input[0];

    //             Vector2::new(
    //                 pos + vel * 0.1 + 0.005 * vel * vel.abs(), // nonlinear position update
    //                 vel + acc + 0.01 * pos.signum() * pos * pos, // nonlinear velocity update
    //             )
    //         },
    //         input,
    //     );

    //     // Verify result is reasonable
    //     assert!(result.iter().all(|x| x.is_finite()));
    //     assert!(result.norm() > 0.0);
    // }

    // #[test]
    // fn test_ukf_builder_pattern() {
    //     type T = f64;

    //     let params = UnscentedParameters::<T>::new(0.1, 2.0, 0.0).unwrap();
    //     let ukf = UKFBuilder::<T>::new()
    //         .with_params(params)
    //         .build_linear_no_input(
    //             Matrix2::new(1.0, 0.1, 0.0, 1.0), // F
    //             Matrix2::identity() * 0.01,       // Q
    //             Vector2::zeros(),                 // x_initial
    //             Matrix2::identity(),              // P_initial
    //         );

    //     // Verify UKF was created successfully
    //     assert_eq!(ukf.state(), &Vector2::zeros());
    // }

    // #[test]
    // fn test_ukf_error_types() {
    //     type T = f64;

    //     // Test error handling
    //     let mean = Vector2::new(1.0, 2.0);
    //     let cov = Matrix2::identity();
    //     let params = UnscentedParameters::<T>::default();

    //     // This should work for medium-sized storage
    //     let result = SigmaPointsStorage::<T, 2>::new_optimal(&mean, &cov, &params);
    //     assert!(result.is_ok());

    //     let storage = result.unwrap();
    //     assert_eq!(storage.len(), 5); // 2*2 + 1 sigma points
    // }
}
