//! Малая спектральная задача — generic-обёртка над LAPACK ZHSEQR/ZTREVC.
//!
//! Используется только бэкендом LAPACK (см. `backend/lapack.rs::small_eig`).
//! Здесь оставлены тонкие шимы, чтобы прежние внешние пользователи могли
//! продолжать звать `compute_ritz_values` / `retrive_ritz_vectors`.

use num_complex::Complex64;

use crate::linalg::lapack::{SchurOutput, zhseqr_schur_slice, ztrevc_all_right_slice};

#[derive(Debug, Clone)]
pub struct RitzValue {
    pub value: Complex64,
    pub residual_estimate: f64,
    pub ritz_vector: Vec<Complex64>,
}

/// Шифма для обратной совместимости: H — column-major, размером `n×n`.
pub fn compute_ritz_values(hessenberg: &[Complex64], n: usize) -> SchurOutput {
    zhseqr_schur_slice(hessenberg, n).expect("zhseqr failed")
}

/// Возвращает все правые собственные векторы (column-major, `dim × dim`).
pub fn retrieve_ritz_vectors(
    decomposition: &mut SchurOutput,
    dim: usize,
) -> Vec<Complex64> {
    ztrevc_all_right_slice(decomposition, dim).expect("ztrevc failed")
}
