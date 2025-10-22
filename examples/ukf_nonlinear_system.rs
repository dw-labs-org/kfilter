//! Nonlinear Measurement UKF Example
//!
//! This example demonstrates the Unscented Kalman Filter with nonlinear
//! measurement functions, showcasing UKF's ability to handle nonlinear systems.

use kfilter::measurement::{LinearMeasurement, Measurement};
use kfilter::ukf::UKFLinearNoInput;
use kfilter::{KalmanFilter, KalmanPredict, KalmanUpdate};
use nalgebra::{Matrix2, SMatrix, SVector, Vector2};

fn main() {
    println!("=== UKF Nonlinear Measurement Examples ===\n");

    example_linear_measurement();
    println!();

    example_range_measurement();
    println!();

    example_range_bearing_measurement();
}

/// Example 1: Linear Measurement with UKF
/// Demonstrates that UKF works correctly with linear measurements
fn example_linear_measurement() {
    println!("--- Example 1: Linear Measurement ---");

    // For N=2, we need X=2*2+1=5 sigma points
    let mut ukf: UKFLinearNoInput<f64, 2, 5> = UKFLinearNoInput::new_linear(
        Matrix2::new(1.0, 0.1, 0.0, 1.0), // Simple kinematic model
        Matrix2::identity() * 0.01,       // Process noise
        Vector2::new(1.0, 0.5),           // Initial state: pos=1, vel=0.5
        Matrix2::identity() * 0.5,        // Initial uncertainty
    );

    println!("Initial state: pos={:.3}, vel={:.3}", ukf.state()[0], ukf.state()[1]);

    // Measurement matrix H observes position only
    let h = Matrix2::new(
        1.0, 0.0,  // Measure position
        0.0, 0.0,  // Don't measure velocity
    );
    let r = Matrix2::new(
        0.1, 0.0,  // Position measurement noise
        0.0, 1e6,  // Large noise for unobserved dimension
    );

    // Simulate 5 steps with prediction and update
    for step in 1..=5 {
        // Predict
        ukf.predict().unwrap();

        // Create measurement (position + noise)
        let true_position = ukf.state()[0];
        let noisy_position = true_position + 0.05 * (step as f64 % 2.0 - 0.5);
        let measurement = Vector2::new(noisy_position, 0.0);

        // Update with linear measurement
        let meas_model = LinearMeasurement::new(h, r, measurement);
        ukf.update(&meas_model).unwrap();

        println!(
            "Step {}: pos={:.3}, vel={:.3}, meas={:.3}",
            step,
            ukf.state()[0],
            ukf.state()[1],
            noisy_position
        );
    }
}

/// Custom nonlinear measurement: Range from origin
/// Measures distance: h(x) = sqrt(x^2 + y^2)
struct RangeMeasurement {
    measurement: Vector2<f64>,
    covariance: Matrix2<f64>,
}

impl RangeMeasurement {
    fn new(range: f64, noise_variance: f64) -> Self {
        Self {
            measurement: Vector2::new(range, 0.0),
            covariance: Matrix2::new(noise_variance, 0.0, 0.0, 1e6),
        }
    }
}

impl Measurement<f64, 2, 2> for RangeMeasurement {
    fn predict(&self, state: &SVector<f64, 2>) -> SVector<f64, 2> {
        // Nonlinear measurement function: range from origin
        let range = (state[0].powi(2) + state[1].powi(2)).sqrt();
        Vector2::new(range, 0.0)
    }

    fn measurement(&self) -> &SVector<f64, 2> {
        &self.measurement
    }

    fn covariance(&self) -> &SMatrix<f64, 2, 2> {
        &self.covariance
    }

    fn set_measurement(&mut self, z: SVector<f64, 2>) {
        self.measurement = z;
    }
}

/// Example 2: Range Measurement
/// Demonstrates nonlinear measurement function
fn example_range_measurement() {
    println!("--- Example 2: Nonlinear Range Measurement ---");

    // For N=2, we need X=2*2+1=5 sigma points
    let mut ukf: UKFLinearNoInput<f64, 2, 5> = UKFLinearNoInput::new_linear(
        Matrix2::identity(),           // Position stays constant
        Matrix2::identity() * 0.01,    // Small process noise
        Vector2::new(3.0, 4.0),        // Initial state: x=3, y=4 (range=5)
        Matrix2::identity() * 0.5,     // Initial uncertainty
    );

    let initial_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
    println!(
        "Initial state: x={:.3}, y={:.3}, range={:.3}",
        ukf.state()[0],
        ukf.state()[1],
        initial_range
    );

    // Simulate 5 steps with range measurements
    for step in 1..=5 {
        // Predict
        ukf.predict().unwrap();

        // Calculate true range
        let true_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();

        // Add measurement noise
        let noisy_range = true_range + 0.1 * ((step as f64).sin());

        // Update with range measurement
        let meas_model = RangeMeasurement::new(noisy_range, 0.05);
        ukf.update(&meas_model).unwrap();

        let estimated_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();

        println!(
            "Step {}: x={:.3}, y={:.3}, true_range={:.3}, meas={:.3}, est_range={:.3}",
            step,
            ukf.state()[0],
            ukf.state()[1],
            true_range,
            noisy_range,
            estimated_range
        );
    }
}

/// Custom nonlinear measurement: Range and bearing
/// Measures: h(x) = [sqrt(x^2 + y^2), atan2(y, x)]
struct RangeBearingMeasurement {
    measurement: Vector2<f64>,
    covariance: Matrix2<f64>,
}

impl RangeBearingMeasurement {
    fn new(range: f64, bearing: f64, range_noise: f64, bearing_noise: f64) -> Self {
        Self {
            measurement: Vector2::new(range, bearing),
            covariance: Matrix2::new(range_noise, 0.0, 0.0, bearing_noise),
        }
    }
}

impl Measurement<f64, 2, 2> for RangeBearingMeasurement {
    fn predict(&self, state: &SVector<f64, 2>) -> SVector<f64, 2> {
        // Nonlinear measurement: range and bearing
        let range = (state[0].powi(2) + state[1].powi(2)).sqrt();
        let bearing = state[1].atan2(state[0]);
        Vector2::new(range, bearing)
    }

    fn measurement(&self) -> &SVector<f64, 2> {
        &self.measurement
    }

    fn covariance(&self) -> &SMatrix<f64, 2, 2> {
        &self.covariance
    }

    fn set_measurement(&mut self, z: SVector<f64, 2>) {
        self.measurement = z;
    }
}

/// Example 3: Range and Bearing Measurement
/// Demonstrates multiple nonlinear measurements
fn example_range_bearing_measurement() {
    println!("--- Example 3: Range and Bearing Measurement ---");

    // For N=2, we need X=2*2+1=5 sigma points
    let mut ukf: UKFLinearNoInput<f64, 2, 5> = UKFLinearNoInput::new_linear(
        Matrix2::new(1.0, 0.0, 0.0, 1.0), // Static position
        Matrix2::identity() * 0.01,        // Process noise
        Vector2::new(3.0, 4.0),            // Initial: x=3, y=4
        Matrix2::identity() * 1.0,         // Initial uncertainty
    );

    let initial_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
    let initial_bearing = ukf.state()[1].atan2(ukf.state()[0]);

    println!(
        "Initial state: x={:.3}, y={:.3}, range={:.3}, bearing={:.3} rad ({:.1}°)",
        ukf.state()[0],
        ukf.state()[1],
        initial_range,
        initial_bearing,
        initial_bearing.to_degrees()
    );

    // Simulate 5 steps
    for step in 1..=5 {
        // Predict
        ukf.predict().unwrap();

        // Calculate true range and bearing
        let true_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
        let true_bearing = ukf.state()[1].atan2(ukf.state()[0]);

        // Add measurement noise
        let noisy_range = true_range + 0.1 * ((step as f64 * 0.5).sin());
        let noisy_bearing = true_bearing + 0.05 * ((step as f64).cos());

        // Update with range-bearing measurement
        let meas_model = RangeBearingMeasurement::new(
            noisy_range,
            noisy_bearing,
            0.05, // range noise
            0.03, // bearing noise (radians)
        );
        ukf.update(&meas_model).unwrap();

        let est_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
        let est_bearing = ukf.state()[1].atan2(ukf.state()[0]);

        println!(
            "Step {}: x={:.3}, y={:.3}, range={:.3}, bearing={:.2} rad ({:.0}°)",
            step,
            ukf.state()[0],
            ukf.state()[1],
            est_range,
            est_bearing,
            est_bearing.to_degrees()
        );
    }
}
