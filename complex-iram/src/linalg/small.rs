//! Решение малой спектральной задачи

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::linalg::lapack::*;

#[derive(Debug, Clone)]
pub struct RitzValue {
    pub value: Complex64,
    pub residual_estimate: f64,
    pub ritz_vector: Array1<Complex64>,
}

pub fn compute_ritz_values(hessenberg: &Array2<Complex64>) -> SchurOutput {
    zhseqr_schur(hessenberg).unwrap()
}

pub fn retrive_ritz_vectors(
    decomposition: &mut SchurOutput,
    ritz_indices: &[usize],
    dim: usize,
) -> Array2<Complex64> {
    ztrevc_right_selected(decomposition, ritz_indices, dim).unwrap()
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

        let values = compute_ritz_values(&hessenberg);

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
