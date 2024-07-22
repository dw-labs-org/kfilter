use nalgebra::{RealField, SMatrix, SVector};

/// Representation of Kalman filter
#[derive(Debug)]
struct Kalman<T, const N: usize, const M: usize> {
    /// Estimated state
    pub x: SVector<T, N>,
    /// Discrete state transition
    F: SMatrix<T, N, N>,
    /// Covariance
    pub P: SMatrix<T, N, N>,
    /// Process Covariance
    Q: SMatrix<T, N, N>,
    /// Observation Matrix
    H: SMatrix<T, M, N>,
    /// Measurment covariance
    R: SMatrix<T, M, M>,
}

impl<T: RealField + Copy, const N: usize, const M: usize> Kalman<T, N, M> {
    pub fn predict(&mut self) {
        self.x = self.F * self.x;
        self.P = self.F * self.P * self.F.transpose();
    }

    pub fn update(&mut self, z: &SVector<T, M>) {
        // innovation
        let y = z - (self.H * self.x);
        // innovation covariance
        let S = self.H * self.P * self.H.transpose() + self.R;
        // Optimal gain
        let K = self.P * self.H.transpose() * S.try_inverse().unwrap();
        // state update
        self.x += K * y;
        // covariance update
        self.P = (SMatrix::identity() - K * self.H) * self.P;
    }

    pub fn new(
        F: SMatrix<T, N, N>,
        Q: SMatrix<T, N, N>,
        H: SMatrix<T, M, N>,
        R: SMatrix<T, M, M>,
    ) -> Self {
        Kalman {
            x: SMatrix::zeros(),
            F,
            P: SMatrix::zeros(),
            Q,
            H,
            R,
        }
    }
}
