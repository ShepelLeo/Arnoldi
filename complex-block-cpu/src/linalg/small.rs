//! Решение малой спектральной задачи

use ndarray::Array2;
use num_complex::Complex64;

use crate::linalg::lapack::{
    DenseSchurWorkspace, SchurOutput, TrevcWorkspace, zgees_schur_with_workspace,
    ztrevc_right_selected_with_workspace,
};

pub fn compute_dense_ritz_values_with_workspace(
    matrix: &Array2<Complex64>,
    workspace: &mut DenseSchurWorkspace,
) -> SchurOutput {
    zgees_schur_with_workspace(matrix, workspace).unwrap()
}

pub fn retrive_ritz_vectors_with_workspace(
    decomposition: &mut SchurOutput,
    ritz_indices: &[usize],
    dim: usize,
    workspace: &mut TrevcWorkspace,
) -> Array2<Complex64> {
    ztrevc_right_selected_with_workspace(decomposition, ritz_indices, dim, workspace).unwrap()
}
