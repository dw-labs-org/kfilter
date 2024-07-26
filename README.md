
# Kfilter
A no-std  implementation of the [Kalman](https://en.wikipedia.org/wiki/Kalman_filter)
and [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter).

See the [documentation](https://docs.rs/kfilter) and [examples](https://github.com/dw-labs-org/kfilter/tree/main/examples) for usage.

See this [blog post](https://domwil.co.uk/posts/kalman/) to understand why the library looks the way it does.

## Features
- Systems for modelling state transition / prediction
  - Linear or Non-linear
  - With and without inputs
- Measurements for modelling sensors / observations
  - Linear or Non-linear
- Single measurement Kalman filter (Kalman1M) or multi-measurment / multi-rate (Kalman)

## Quickstart
The example below is a position and velocity 2 state Kalman filter with position measurements.

```rust
use kfilter::{Kalman1M, KalmanPredict};
use nalgebra::{Matrix1, Matrix1x2, Matrix2, SMatrix};

// Create a new 2 state kalman filter
let mut k = Kalman1M::new(
    Matrix2::new(1.0, 0.1, 0.0, 1.0),   // F
    SMatrix::identity(),                // Q
    Matrix1x2::new(1.0, 0.0),           // H
    SMatrix::identity(),                // R
    SMatrix::zeros(),                   // x
);
// Run 100 timesteps
for i in 0..100 {
    // predict based on system model
    k.predict();
    // update based on new measurement
    k.update(Matrix1::new(i as f64));
}
```