//! Решение малой спектральной задачи
//! Белиберда, надо подумать как лучше сделать
//! 
use nalgebra::{DMatrix, linalg::Schur};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::normalize_complex;

const SCHUR_BLOCK_TOL: f64 = 1.0e-12;
const SCHUR_PIVOT_TOL: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub struct RitzValue {
    pub value: Complex64,
    pub residual_estimate: f64,
}

#[derive(Debug, Clone, Copy)]
enum DiagonalBlock {
    Real(usize),
    ComplexPair(usize),
}

pub fn compute_ritz_values(
    hessenberg: &Array2<f64>,
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
    let mut values = Vec::with_capacity(order);

    for block in schur_blocks(&t) {
        match block {
            DiagonalBlock::Real(index) => {
                let eigenvalue = Complex64::new(t[(index, index)], 0.0);
                let schur_vector = schur_vector_for_block(&t, index, 1, eigenvalue)?;
                values.push(build_ritz_value(
                    &q,
                    &schur_vector,
                    eigenvalue,
                    trailing_subdiagonal,
                ));
            }
            DiagonalBlock::ComplexPair(start) => {
                for eigenvalue in eigenvalues_of_two_by_two_block(&t, start) {
                    let schur_vector = schur_vector_for_block(&t, start, 2, eigenvalue)?;
                    values.push(build_ritz_value(
                        &q,
                        &schur_vector,
                        eigenvalue,
                        trailing_subdiagonal,
                    ));
                }
            }
        }
    }

    Ok(values)
}

fn build_ritz_value(
    schur_q: &DMatrix<f64>,
    schur_vector: &Array1<Complex64>,
    eigenvalue: Complex64,
    trailing_subdiagonal: f64,
) -> RitzValue {
    let last_row = schur_q.nrows() - 1;
    let last_component_abs = schur_vector
        .iter()
        .enumerate()
        .map(|(column, &entry)| Complex64::new(schur_q[(last_row, column)], 0.0) * entry)
        .sum::<Complex64>()
        .norm();

    RitzValue {
        value: eigenvalue,
        residual_estimate: trailing_subdiagonal.abs() * last_component_abs,
    }
}

fn schur_blocks(t: &DMatrix<f64>) -> Vec<DiagonalBlock> {
    let mut blocks = Vec::new();
    let mut index = 0usize;

    while index < t.nrows() {
        if is_two_by_two_block_start(t, index) {
            blocks.push(DiagonalBlock::ComplexPair(index));
            index += 2;
        } else {
            blocks.push(DiagonalBlock::Real(index));
            index += 1;
        }
    }

    blocks
}

fn is_two_by_two_block_start(t: &DMatrix<f64>, start: usize) -> bool {
    if start + 1 >= t.nrows() {
        return false;
    }

    let subdiagonal = t[(start + 1, start)].abs();
    let scale = 1.0
        + t[(start, start)].abs()
        + t[(start + 1, start + 1)].abs()
        + t[(start, start + 1)].abs();

    subdiagonal > SCHUR_BLOCK_TOL * scale
}

fn schur_vector_for_block(
    t: &DMatrix<f64>,
    block_start: usize,
    block_size: usize,
    eigenvalue: Complex64,
) -> Result<Array1<Complex64>, IramError> {
    let order = t.nrows();
    let block_end = block_start + block_size - 1;
    let mut vector = Array1::from_elem(order, Complex64::new(0.0, 0.0));

    if block_size == 1 {
        vector[block_start] = Complex64::new(1.0, 0.0);
    } else {
        let [first, second] = eigenvector_of_two_by_two_block(t, block_start, eigenvalue)?;
        vector[block_start] = first;
        vector[block_start + 1] = second;
    }

    let mut cursor = block_start;

    while cursor > 0 {
        let previous_end = cursor - 1;

        if previous_end > 0 && is_two_by_two_block_start(t, previous_end - 1) {
            let row0 = previous_end - 1;
            let row1 = previous_end;
            let rhs0 = -tail_coupling(t, &vector, row0, row1 + 1, block_end);
            let rhs1 = -tail_coupling(t, &vector, row1, row1 + 1, block_end);
            let [first, second] = solve_two_by_two_block(t, row0, eigenvalue, rhs0, rhs1)?;

            vector[row0] = first;
            vector[row1] = second;
            cursor -= 2;
        } else {
            let row = previous_end;
            let rhs = -tail_coupling(t, &vector, row, row + 1, block_end);
            let pivot = stabilize_pivot(
                Complex64::new(t[(row, row)], 0.0) - eigenvalue,
                1.0 + eigenvalue.norm() + t[(row, row)].abs(),
            );
            vector[row] = rhs / pivot;
            cursor -= 1;
        }
    }

    normalize_complex(&mut vector)?;
    Ok(vector)
}

fn tail_coupling(
    t: &DMatrix<f64>,
    vector: &Array1<Complex64>,
    row: usize,
    start_column: usize,
    end_column: usize,
) -> Complex64 {
    if start_column > end_column {
        return Complex64::new(0.0, 0.0);
    }

    (start_column..=end_column)
        .map(|column| Complex64::new(t[(row, column)], 0.0) * vector[column])
        .sum::<Complex64>()
}

fn solve_two_by_two_block(
    t: &DMatrix<f64>,
    start: usize,
    eigenvalue: Complex64,
    rhs0: Complex64,
    rhs1: Complex64,
) -> Result<[Complex64; 2], IramError> {
    let mut a11 = Complex64::new(t[(start, start)], 0.0) - eigenvalue;
    let a12 = Complex64::new(t[(start, start + 1)], 0.0);
    let a21 = Complex64::new(t[(start + 1, start)], 0.0);
    let mut a22 = Complex64::new(t[(start + 1, start + 1)], 0.0) - eigenvalue;
    let scale = 1.0 + a11.norm() + a12.norm() + a21.norm() + a22.norm() + eigenvalue.norm();

    let mut determinant = a11 * a22 - a12 * a21;

    if determinant.norm() <= SCHUR_PIVOT_TOL * scale {
        let regularization = Complex64::new(SCHUR_PIVOT_TOL * scale, 0.0);
        a11 += regularization;
        a22 += regularization;
        determinant = a11 * a22 - a12 * a21;
    }

    if determinant.norm() <= f64::EPSILON {
        return Err(IramError::Spectral(
            "cannot recover a Schur-space Ritz vector from a singular 2x2 block".to_string(),
        ));
    }

    Ok([
        (rhs0 * a22 - a12 * rhs1) / determinant,
        (a11 * rhs1 - rhs0 * a21) / determinant,
    ])
}

fn stabilize_pivot(pivot: Complex64, scale: f64) -> Complex64 {
    if pivot.norm() <= SCHUR_PIVOT_TOL * scale {
        pivot + Complex64::new(SCHUR_PIVOT_TOL * scale, 0.0)
    } else {
        pivot
    }
}

fn eigenvalues_of_two_by_two_block(t: &DMatrix<f64>, start: usize) -> [Complex64; 2] {
    let a = t[(start, start)];
    let b = t[(start, start + 1)];
    let c = t[(start + 1, start)];
    let d = t[(start + 1, start + 1)];
    let trace = a + d;
    let determinant = a * d - b * c;
    let discriminant = Complex64::new(trace * trace - 4.0 * determinant, 0.0).sqrt();

    [
        Complex64::new(0.5 * trace, 0.0) + 0.5 * discriminant,
        Complex64::new(0.5 * trace, 0.0) - 0.5 * discriminant,
    ]
}

fn eigenvector_of_two_by_two_block(
    t: &DMatrix<f64>,
    start: usize,
    eigenvalue: Complex64,
) -> Result<[Complex64; 2], IramError> {
    let a = Complex64::new(t[(start, start)], 0.0);
    let b = Complex64::new(t[(start, start + 1)], 0.0);
    let c = Complex64::new(t[(start + 1, start)], 0.0);
    let d = Complex64::new(t[(start + 1, start + 1)], 0.0);
    let candidate1 = [b, eigenvalue - a];
    let candidate2 = [eigenvalue - d, c];
    let norm1 = candidate1
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let norm2 = candidate2
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt();

    if norm1.max(norm2) <= f64::EPSILON {
        return Err(IramError::Spectral(
            "cannot recover a nonzero eigenvector for a 2x2 Schur block".to_string(),
        ));
    }

    Ok(if norm1 >= norm2 {
        candidate1
    } else {
        candidate2
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::compute_ritz_values;

    #[test]
    fn schur_unpack_recovers_complex_ritz_pair_from_rotation_block() {
        let hessenberg = Array2::from_shape_vec((2, 2), vec![0.0, -1.0, 1.0, 0.0])
            .expect("2x2 block should reshape");
        let values = compute_ritz_values(&hessenberg, 0.0)
            .expect("Schur-based Ritz extraction should succeed");

        assert_eq!(values.len(), 2);
        assert!(
            values
                .iter()
                .any(|value| (value.value - Complex64::new(0.0, 1.0)).norm() <= 1.0e-10)
        );
        assert!(
            values
                .iter()
                .any(|value| (value.value - Complex64::new(0.0, -1.0)).norm() <= 1.0e-10)
        );
        assert!(
            values
                .iter()
                .all(|value| value.residual_estimate.abs() <= 1.0e-12)
        );
    }
}
