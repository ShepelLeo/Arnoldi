//! Решение малой спектральной задачи

use nalgebra::{DMatrix, linalg::Schur};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::normalize_complex;

const SCHUR_PIVOT_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub struct RitzValue {
    pub value: Complex64,
    pub residual_estimate: f64,
}

pub fn compute_ritz_values(
    hessenberg: &Array2<Complex64>,
    trailing_subdiagonal: f64,
) -> Result<Vec<RitzValue>, IramError> {
    let order = hessenberg.nrows();

    if order == 0 {
        return Ok(Vec::new());
    }

    let storage = hessenberg.iter().copied().collect::<Vec<_>>();
    let matrix = DMatrix::from_row_slice(order, order, &storage);
    let schur = Schur::new(matrix);
    let (q, t) = schur.unpack();

    (0..order)
        .map(|index| {
            let eigenvalue = t[(index, index)];
            let schur_vector = schur_vector_for_index(&t, index, eigenvalue)?;
            Ok(build_ritz_value(
                &q,
                &schur_vector,
                eigenvalue,
                trailing_subdiagonal,
            ))
        })
        .collect()
}

fn build_ritz_value(
    schur_q: &DMatrix<Complex64>,
    schur_vector: &Array1<Complex64>,
    eigenvalue: Complex64,
    trailing_subdiagonal: f64,
) -> RitzValue {
    let last_row = schur_q.nrows() - 1;
    let last_component_abs = schur_vector
        .iter()
        .enumerate()
        .map(|(column, &entry)| schur_q[(last_row, column)] * entry)
        .sum::<Complex64>()
        .norm();

    RitzValue {
        value: eigenvalue,
        residual_estimate: trailing_subdiagonal * last_component_abs,
    }
}

fn schur_vector_for_index(
    t: &DMatrix<Complex64>,
    eigen_index: usize,
    eigenvalue: Complex64,
) -> Result<Array1<Complex64>, IramError> {
    let order = t.nrows();
    let mut vector = Array1::from_elem(order, Complex64::new(0.0, 0.0));
    vector[eigen_index] = Complex64::new(1.0, 0.0);

    let mut cursor = eigen_index;

    while cursor > 0 {
        let row = cursor - 1;
        let rhs = -tail_coupling(t, &vector, row, row + 1, eigen_index);
        let scale = 1.0
            + eigenvalue.norm()
            + (row..=eigen_index)
                .map(|column| t[(row, column)].norm())
                .sum::<f64>();
        let pivot = stabilize_pivot(t[(row, row)] - eigenvalue, scale);
        vector[row] = rhs / pivot;
        cursor -= 1;
    }

    normalize_complex(&mut vector)?;
    Ok(vector)
}

fn tail_coupling(
    t: &DMatrix<Complex64>,
    vector: &Array1<Complex64>,
    row: usize,
    start_column: usize,
    end_column: usize,
) -> Complex64 {
    if start_column > end_column {
        return Complex64::new(0.0, 0.0);
    }

    (start_column..=end_column)
        .map(|column| t[(row, column)] * vector[column])
        .sum::<Complex64>()
}

fn stabilize_pivot(pivot: Complex64, scale: f64) -> Complex64 {
    if pivot.norm() <= SCHUR_PIVOT_TOL * scale {
        pivot + Complex64::new(SCHUR_PIVOT_TOL * scale, 0.0)
    } else {
        pivot
    }
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

        let values = compute_ritz_values(&hessenberg, 0.25)
            .expect("Schur-based Ritz extraction should succeed");

        assert_eq!(values.len(), 2);
        assert!(
            values
                .iter()
                .any(|value| (value.value - Complex64::new(2.0, 1.0)).norm() <= 1.0e-10)
        );
        assert!(
            values
                .iter()
                .any(|value| (value.value - Complex64::new(-3.0, 0.5)).norm() <= 1.0e-10)
        );
    }
}
