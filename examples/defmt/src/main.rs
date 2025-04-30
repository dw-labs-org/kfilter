#![no_std]
#![no_main]

use defmt_rtt as _;
use kfilter::{Kalman1M, KalmanFilter, KalmanPredictInput};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Matrix2x1, SMatrix};
use panic_probe as _;

use cortex_m_rt::entry;

#[entry]
unsafe fn main() -> ! {
    defmt::info!("Hello, world!");
    let m = kfilter::measurement::LinearMeasurement::new(
        Matrix1::new(1.0),
        Matrix1::new(1.0),
        Matrix1::new(1.0),
    );
    defmt::info!("Measurement created: {:?}", m);
    let q = Matrix2::new(1.0, 0.1, 0.0, 1.0);
    let mut k = Kalman1M::new_with_input(
        q,
        SMatrix::identity(),
        Matrix2x1::new(0.0, 1.0),
        Matrix1x2::new(1.0, 0.0),
        SMatrix::identity(),
        SMatrix::zeros(),
    );

    defmt::info!("Kalman filter created: {:?}", k);
    defmt::flush();
    // Fairly useless loop but does run
    for i in 0.. {
        let x_predicted = k.predict(Matrix1::new(1.0)).x;
        let x_measured = i as f32; // Replace with actual measurement
        k.update(Matrix1::new(x_measured));
        let x_updated = k.state().x;
        defmt::info!(
            "Predicted: {:?}, Measured: {:?}, Updated: {:?}",
            x_predicted,
            x_measured,
            x_updated
        );
        defmt::info!("Kalman filter state: {:?}", k);
    }

    loop {}
}
