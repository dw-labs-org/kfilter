use nalgebra::{SMatrix, SVector};

pub struct Measurement<T, const N: usize, const M: usize> {
    pub z: SVector<T, M>,
    pub H: SMatrix<T, M, N>,
    pub R: SMatrix<T, M, M>,
}
