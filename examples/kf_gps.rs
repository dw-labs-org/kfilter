//! Simple example of position/velocity tracking with GPS and velocity measurements

use kfilter::{
    measurement::{LinearMeasurement, Measurement},
    system::LinearNoInputSystem,
    KalmanLinearNoInput, KalmanPredict, KalmanUpdate,
};
use nalgebra::{Matrix1, Matrix1x2, SMatrix, Vector2};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea},
    series::LineSeries,
    style::{full_palette::BLUE, BLACK, GREEN, RED, WHITE},
};
use rand_distr::Distribution;
#[allow(non_snake_case)]
fn main() {
    // Regular velocity measurements and lower rate GPS for 1D motion
    // Assume constant velocity
    let dt = 0.1;
    let F = SMatrix::<f64, 2, 2>::new(1.0, dt, 0.0, 1.0);
    let Q = SMatrix::<f64, 2, 2>::identity() * 0.1;
    let x_initial = Vector2::new(0.0, 0.0);
    let sys = LinearNoInputSystem::new(F, Q, x_initial);

    let mut kf = KalmanLinearNoInput::new_custom(sys, SMatrix::zeros());

    // GPS measurement
    let mut gps = LinearMeasurement::new(
        Matrix1x2::new(1.0, 0.0),
        Matrix1::new(0.5),
        Matrix1::new(0.0),
    );

    // Velocity measurement
    let mut vel = LinearMeasurement::new(
        Matrix1x2::new(0.0, 1.0),
        Matrix1::new(0.5),
        Matrix1::new(0.0),
    );

    // Track real system values
    let mut x_real = x_initial;
    let acc = 1.0; // m/s^2
    let vel_noise = rand_distr::Normal::new(0.0, 0.2).unwrap();
    let vel_bias = 0.2;
    let gps_noise = rand_distr::Normal::new(0.0, 0.2).unwrap();
    let mut rng = rand::thread_rng();

    let mut state = Vec::new();
    let mut measured = Vec::new();
    let mut predicted = Vec::new();
    let mut updated = Vec::new();
    let n_samples = 100;

    for t in 0..n_samples {
        // update state
        x_real.y += (acc - 0.2 * x_real.y) * dt;
        x_real.x += x_real.y * dt;
        state.push(x_real);
        let kf_pred = kf.predict().unwrap();
        predicted.push(*kf_pred);
        // Make a noisy velocity measurement
        let vel_meas = x_real.y + vel_noise.sample(&mut rng) + vel_bias;
        vel.set_measurement(Matrix1::new(vel_meas));
        let kf_update = kf.update(&vel).unwrap();
        // let kf_update = kf_pred;
        // Make a noisy GPS measurement every 10 timesteps
        if t % 10 == 0 {
            let gps_meas = x_real.x + gps_noise.sample(&mut rng);
            gps.set_measurement(Matrix1::new(gps_meas));
            let kf_update = kf.update(&gps).unwrap();
            updated.push(*kf_update);
            measured.push(Vector2::new(gps_meas, vel_meas));
        } else {
            updated.push(*kf_update);
            measured.push(Vector2::new(0.0, vel_meas));
        }
    }
    for i in 0..state.len() {
        println!(
            "{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}",
            state[i].x,
            state[i].y,
            measured[i].x,
            measured[i].y,
            predicted[i].x,
            predicted[i].y,
            updated[i].x,
            updated[i].y
        );
    }

    let t = (0..n_samples).map(|x| x as f64 * dt).collect::<Vec<f64>>();
    let x_actual: Vec<f64> = state.iter().map(|s| s.x).collect();
    let v_actual: Vec<f64> = state.iter().map(|s| s.y).collect();
    let x_measured: Vec<f64> = measured.iter().map(|m| m.x).collect();
    let v_measured: Vec<f64> = measured.iter().map(|m| m.y).collect();
    let x_predicted: Vec<f64> = predicted.iter().map(|p| p.x).collect();
    let v_predicted: Vec<f64> = predicted.iter().map(|p| p.y).collect();
    let x_updated: Vec<f64> = updated.iter().map(|u| u.x).collect();
    let v_updated: Vec<f64> = updated.iter().map(|u| u.y).collect();

    // Plot positions
    let root = BitMapBackend::new("kf_gps_pos.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            t.first().copied().unwrap()..t.last().copied().unwrap(),
            x_actual.first().copied().unwrap()..x_actual.last().copied().unwrap(),
        )
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(x_actual), &RED))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(x_measured), &BLUE))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(x_predicted), &GREEN))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(x_updated), &BLACK))
        .unwrap();
    root.present().unwrap();

    // Plot velocities
    let root = BitMapBackend::new("kf_gps_vel.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            t.first().copied().unwrap()..t.last().copied().unwrap(),
            v_actual.first().copied().unwrap()..v_actual.last().copied().unwrap(),
        )
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(v_actual), &RED))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(v_measured), &BLUE))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(v_predicted), &GREEN))
        .unwrap();
    chart
        .draw_series(LineSeries::new(t.iter().cloned().zip(v_updated), &BLACK))
        .unwrap();
    root.present().unwrap();
}
