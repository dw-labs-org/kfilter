#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
//! A no-std  implementation of the [Kalman](https://en.wikipedia.org/wiki/Kalman_filter)
//!  and [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
//! based on [nalgebra::SMatrix] statically sized matrices.
//!
//! The base [Kalman] type is the most configurable, allowing multi-rate sensor measurements.
//! For simpler systems with single measurements, [Kalman1M] is easier to use.
//!
//! See this [blog post](https://domwil.co.uk/posts/kalman/) to understand why the library looks the way it does.
//!
//! # Getting started
//! See [Kalman1M] for an example of linear Kalman filter for a 2-state system model
//! with no inputs.
//!
//! See [Kalman] for an example of a linear Kalman filter for a 2-state system model
//! with no inputs and multiple sensors.
//!
//! See the [examples](https://github.com/dw-labs-org/kfilter/tree/main/examples) for more realistic simulations including systems with inputs, non-linear
//! systems (Extended Kalman Filter)
//!
//!
pub mod kalman;
pub mod measurement;
pub mod system;

pub use kalman::*;
