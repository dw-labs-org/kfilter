//! Linear System UKF Example
//!
//! This example demonstrates the Unscented Kalman Filter with linear systems,
//! including both systems without input and systems with control input.

use kfilter::ukf::{UKFLinear, UKFLinearNoInput};
use kfilter::{KalmanFilter, KalmanPredict, KalmanPredictInput};
use nalgebra::{Matrix2, SMatrix, SVector, Vector2};

fn main() {
    println!("=== UKF Linear System Examples ===\n");

    example_1d_position_tracking();
    println!();

    example_2d_position_velocity();
    println!();

    example_with_control_input();
}

/// Example 1: Simple 1D position tracking
/// State: [position]
/// Demonstrates basic UKF usage with a 1-dimensional system
fn example_1d_position_tracking() {
    println!("--- Example 1: 1D Position Tracking ---");

    // For N=1, we need X=2*1+1=3 sigma points
    let mut ukf: UKFLinearNoInput<f64, 1, 3> = UKFLinearNoInput::new_linear(
        SMatrix::<f64, 1, 1>::identity(), // F: position stays constant
        SMatrix::<f64, 1, 1>::identity() * 0.01, // Q: small process noise
        SVector::<f64, 1>::new(0.0),      // Initial position: 0
        SMatrix::<f64, 1, 1>::identity() * 0.5, // P: initial uncertainty
    );

    println!("Initial state: position = {:.3}", ukf.state()[0]);
    println!("Initial covariance: {:.3}", ukf.covariance()[(0, 0)]);

    // Run 5 prediction steps
    for step in 1..=5 {
        ukf.predict().unwrap();
        println!(
            "Step {}: position = {:.3}, uncertainty = {:.3}",
            step,
            ukf.state()[0],
            ukf.covariance()[(0, 0)]
        );
    }
}

/// Example 2: 2D Position-Velocity System
/// State: [position, velocity]
/// Demonstrates tracking with a simple kinematic model
fn example_2d_position_velocity() {
    println!("--- Example 2: 2D Position-Velocity Tracking ---");

    let dt = 0.1; // time step

    // For N=2, we need X=2*2+1=5 sigma points
    let mut ukf: UKFLinearNoInput<f64, 2, 5> = UKFLinearNoInput::new_linear(
        Matrix2::new(
            1.0, dt,   // position = position + velocity * dt
            0.0, 1.0,  // velocity stays constant
        ),
        Matrix2::identity() * 0.01, // Q: process noise
        Vector2::new(0.0, 1.0),     // Initial state: pos=0, vel=1
        Matrix2::identity() * 0.5,  // P: initial uncertainty
    );

    println!("Initial state: pos={:.3}, vel={:.3}", ukf.state()[0], ukf.state()[1]);

    // Simulate 10 time steps
    for step in 1..=10 {
        ukf.predict().unwrap();

        if step % 2 == 0 {
            println!(
                "Step {}: pos={:.3}, vel={:.3}",
                step,
                ukf.state()[0],
                ukf.state()[1]
            );
        }
    }

    println!(
        "Final position: {:.3} (expected ~1.0 after 10 steps with dt=0.1)",
        ukf.state()[0]
    );
}

/// Example 3: System with Control Input
/// State: [position, velocity], Input: [acceleration]
/// Demonstrates UKF with control input
fn example_with_control_input() {
    println!("--- Example 3: System with Control Input ---");

    let dt = 0.1;

    // For N=2, U=1, we need X=2*2+1=5 sigma points
    let mut ukf: UKFLinear<f64, 2, 1, 5> = UKFLinear::new_linear_with_input(
        Matrix2::new(
            1.0, dt,   // position dynamics
            0.0, 1.0,  // velocity dynamics
        ),
        Matrix2::identity() * 0.01,         // Q: process noise
        SMatrix::<f64, 2, 1>::new(
            0.5 * dt * dt, // position affected by acceleration
            dt,            // velocity affected by acceleration
        ),
        Vector2::new(0.0, 0.0),            // Initial state: at rest
        Matrix2::identity() * 0.1,         // P: initial uncertainty
    );

    println!("Controlling system to reach target position 5.0");
    println!("Initial: pos={:.3}, vel={:.3}", ukf.state()[0], ukf.state()[1]);

    let target_position = 5.0;

    // Simple proportional controller
    for step in 1..=20 {
        // Calculate control input (proportional controller)
        let error = target_position - ukf.state()[0];
        let control_acceleration = SVector::<f64, 1>::new(error * 2.0);

        // Predict with control input
        ukf.predict(control_acceleration).unwrap();

        if step % 5 == 0 {
            println!(
                "Step {}: pos={:.3}, vel={:.3}, control={:.3}",
                step,
                ukf.state()[0],
                ukf.state()[1],
                control_acceleration[0]
            );
        }
    }

    println!(
        "Final: pos={:.3} (target: {:.3}), vel={:.3}",
        ukf.state()[0],
        target_position,
        ukf.state()[1]
    );
}
