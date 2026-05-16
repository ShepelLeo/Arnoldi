//! Backend abstraction for the shared IRAM algorithm.
//!
//! The algorithm calls this trait in readable places: Arnoldi workspace creation,
//! basis orthogonalization, small Ritz solve, Ritz vector extraction and dense
//! matrix products. Concrete backends keep their LAPACK/MAGMA details here.

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::OrthogonalizedVector;

pub mod lapack;
#[cfg(feature = "magma")]
pub mod magma;

pub use lapack::LapackBackend;
#[cfg(feature = "magma")]
pub use magma::MagmaBackend;

pub trait Backend {
    type ArnoldiWorkspace;
    type RitzDecomposition;

    fn name(&self) -> &'static str;

    fn create_arnoldi_workspace(
        &mut self,
        basis: &Array2<Complex64>,
        dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError>;

    fn orthogonalize_arnoldi_candidate(
        &mut self,
        workspace: &mut Self::ArnoldiWorkspace,
        basis: &Array2<Complex64>,
        basis_columns: usize,
        candidate: &mut Array1<Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector;

    fn append_arnoldi_basis_column(
        &mut self,
        workspace: &mut Self::ArnoldiWorkspace,
        column: usize,
        values: &Array1<Complex64>,
    );

    fn orthogonalize_restart_residual(
        &mut self,
        residual: &mut Array1<Complex64>,
        basis: &ArrayView2<'_, Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector;

    fn compute_ritz_values(
        &mut self,
        hessenberg: &Array2<Complex64>,
    ) -> Result<Self::RitzDecomposition, IramError>;

    fn ritz_values<'a>(&self, decomposition: &'a Self::RitzDecomposition) -> &'a [Complex64];

    fn retrieve_ritz_vectors(
        &mut self,
        decomposition: &mut Self::RitzDecomposition,
        ritz_indices: &[usize],
        dim: usize,
    ) -> Result<Array2<Complex64>, IramError>;

    fn zgemm_nn(
        &mut self,
        a: ArrayView2<'_, Complex64>,
        b: ArrayView2<'_, Complex64>,
    ) -> Array2<Complex64>;
}
