// examples/ukf_basic_usage.rs
//! Basic UKF usage example demonstrating core functionality

use kfilter::system::LinearNoInputSystem;
use kfilter::ukf::*;
use kfilter::{KalmanFilter, KalmanPredict, KalmanPredictInput};
use nalgebra::{Matrix2, SMatrix, SVector, Vector2};

fn main() {
    println!("=== UKF Basic Usage Example ===\n");

    basic_2d_tracking();
    println!();

    nonlinear_measurement_example();
    println!();

    input_system_example();
    println!();

    parameter_comparison_example();
}

fn basic_2d_tracking() {
    println!("--- Basic 2D Position/Velocity Tracking ---");

    // Create a simple 2D position-velocity system
    let mut ukf = UKFLinearNoInputMedium::<f64, 2>::new(
        Matrix2::new(1.0, 0.1, 0.0, 1.0), // F: position = position + velocity*dt
        Matrix2::identity() * 0.01,       // Q: small process noise
        Vector2::new(0.0, 1.0),           // Initial state: [position=0, velocity=1]
        Matrix2::identity() * 0.5,        // P: initial uncertainty
    );

    println!("Initial state: {:?}", ukf.state());

    // Simulate 10 time steps
    for step in 0..10 {
        // Predict
        ukf.predict();

        // Simulate a position measurement with noise
        let true_position = ukf.state()[0];
        let measurement_noise = 0.1 * (rand::random::<f64>() - 0.5);
        let measurement = Vector2::new(true_position + measurement_noise, 0.0);

        // Update with position measurement
        let measurement_noise_cov = Matrix2::new(0.01, 0.0, 0.0, 1e6); // Only measure position
        ukf.update_ukf(
            |state| Vector2::new(state[0], 0.0), // Observe position only
            &measurement,
            &measurement_noise_cov,
        );

        println!(
            "Step {}: pos={:.3}, vel={:.3}, uncertainty={:.3}",
            step,
            ukf.state()[0],
            ukf.state()[1],
            ukf.covariance().diagonal().norm()
        );
    }
}

fn nonlinear_measurement_example() {
    println!("--- Nonlinear Measurement Function ---");

    let mut ukf = UKFLinearNoInputMedium::<f64, 2>::new(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),
        Matrix2::identity() * 0.01,
        Vector2::new(5.0, 0.0), // Start at position 5
        Matrix2::identity() * 0.1,
    );

    println!("Initial state: {:?}", ukf.state());

    // Use range measurement (distance from origin)
    for step in 0..5 {
        ukf.predict();

        // True range from origin
        let true_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
        let measurement_noise = 0.05 * (rand::random::<f64>() - 0.5);
        let range_measurement = Vector2::new(true_range + measurement_noise, 0.0);

        // Nonlinear measurement function: h(x) = sqrt(x^2 + y^2)
        ukf.update_ukf(
            |state| {
                let range = (state[0].powi(2) + state[1].powi(2)).sqrt();
                Vector2::new(range, 0.0)
            },
            &range_measurement,
            &Matrix2::new(0.01, 0.0, 0.0, 1e6),
        );

        let current_range = (ukf.state()[0].powi(2) + ukf.state()[1].powi(2)).sqrt();
        println!(
            "Step {}: pos=({:.3}, {:.3}), estimated_range={:.3}",
            step,
            ukf.state()[0],
            ukf.state()[1],
            current_range
        );
    }
}

fn input_system_example() {
    println!("--- System with Control Input ---");

    // 2D system with acceleration control
    let mut ukf = UKFLinearMedium::<f64, 2, 1>::new_with_input(
        Matrix2::new(1.0, 0.1, 0.0, 1.0),      // F
        Matrix2::identity() * 0.01,            // Q
        SMatrix::<f64, 2, 1>::new(0.005, 0.1), // B: input affects velocity
        Vector2::new(0.0, 0.0),                // Initial state
        Matrix2::identity() * 0.1,             // P
    );

    println!("Controlling system to reach target position 5.0");

    let target_position = 5.0;

    for step in 0..20 {
        // Simple proportional controller
        let error = target_position - ukf.state()[0];
        let control_input = SVector::<f64, 1>::new(error * 0.5);

        // Predict with control input
        ukf.predict(control_input);

        // Simulate position measurement
        let measurement = Vector2::new(ukf.state()[0] + 0.01 * (rand::random::<f64>() - 0.5), 0.0);
        ukf.update_ukf(
            |state| Vector2::new(state[0], 0.0),
            &measurement,
            &Matrix2::new(0.01, 0.0, 0.0, 1e6),
        );

        if step % 5 == 0 {
            println!(
                "Step {}: pos={:.3}, vel={:.3}, control={:.3}",
                step,
                ukf.state()[0],
                ukf.state()[1],
                control_input[0]
            );
        }
    }

    println!(
        "Final position: {:.3} (target: {:.3})",
        ukf.state()[0],
        target_position
    );
}

fn parameter_comparison_example() {
    println!("--- UKF Parameter Effects ---");

    let initial_state = Vector2::new(1.0, 0.5);
    let initial_cov = Matrix2::identity() * 0.1;

    let alphas = [1e-3, 1e-2, 1e-1, 0.5];

    for &alpha in &alphas {
        let params = UnscentedParameters::new(alpha, 2.0, 0.0).unwrap();
        let mut ukf = UKFLinearNoInputMedium::<f64, 2>::new_custom_with_params(
            LinearNoInputSystem::new(
                Matrix2::new(1.0, 0.1, 0.0, 1.0),
                Matrix2::identity() * 0.01,
                initial_state,
            ),
            initial_cov,
            params,
        );

        // Single prediction step
        ukf.predict();

        println!(
            "Alpha {:.3}: final_state=[{:.6}, {:.6}]",
            alpha,
            ukf.state()[0],
            ukf.state()[1]
        );
    }
}
