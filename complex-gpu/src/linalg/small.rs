//! Решение малой спектральной задачи

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::linalg::magma::{
    SchurError, SchurOutput, select_right_eigenvectors, zgeev_right_eigenpairs,
};

#[derive(Debug, Clone)]
pub struct RitzValue {
    pub value: Complex64,
    pub residual_estimate: f64,
    pub ritz_vector: Array1<Complex64>,
}

pub fn compute_ritz_values(hessenberg: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    zgeev_right_eigenpairs(hessenberg)
}

pub fn retrieve_ritz_vectors(
    decomposition: &SchurOutput,
    ritz_indices: &[usize],
    dim: usize,
) -> Result<Array2<Complex64>, SchurError> {
    select_right_eigenvectors(decomposition, ritz_indices, dim)
}

/// Backward-compatible spelling alias. Prefer `retrieve_ritz_vectors`.
pub fn retrive_ritz_vectors(
    decomposition: &SchurOutput,
    ritz_indices: &[usize],
    dim: usize,
) -> Result<Array2<Complex64>, SchurError> {
    retrieve_ritz_vectors(decomposition, ritz_indices, dim)
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use num_complex::Complex64;

    use super::compute_ritz_values;

    #[test]
    fn schur_unpack_recovers_complex_triangular_ritz_values() {
        let hessenberg = arr2(&[
            [Complex64::new(2.0, 1.0), Complex64::new(1.0, -1.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-3.0, 0.5)],
        ]);

        let values = compute_ritz_values(&hessenberg).unwrap();

        assert_eq!(values.w.len(), 2);
        assert!(
            values
                .w
                .iter()
                .any(|value| (value - Complex64::new(2.0, 1.0)).norm() <= 1.0e-10)
        );
        assert!(
            values
                .w
                .iter()
                .any(|value| (value - Complex64::new(-3.0, 0.5)).norm() <= 1.0e-10)
        );
    }
}
