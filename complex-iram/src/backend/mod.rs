//! Backend abstraction for the shared IRAM algorithm.
//!
//! The algorithm calls this trait in readable places: operator preparation,
//! Arnoldi workspace creation, backend matvec, basis orthogonalization, shifted
//! QR restart filtering, small Ritz solve, Ritz vector extraction and dense
//! matrix products. Concrete backends keep their LAPACK/MAGMA/cuSPARSE details here.

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{self, OrthogonalizedVector};
use crate::operator::LinearOperator;

pub mod lapack;
#[cfg(feature = "magma")]
pub mod magma;

pub use lapack::LapackBackend;
#[cfg(feature = "magma")]
pub use magma::MagmaBackend;

pub trait Backend {
    type OperatorWorkspace;
    type ArnoldiWorkspace;
    type RitzDecomposition;

    fn name(&self) -> &'static str;

    fn prepare_operator(
        &mut self,
        operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorWorkspace, IramError>;

    fn create_arnoldi_workspace(
        &mut self,
        basis: &Array2<Complex64>,
        dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError>;

    /// Backend-owned matrix-vector product used by Arnoldi.
    ///
    /// LAPACK applies the host-side `LinearOperator`. MAGMA can keep a prepared
    /// CSR descriptor on the device and call cuSPARSE against a basis column.
    fn apply_operator_to_basis_vector(
        &mut self,
        operator_workspace: &mut Self::OperatorWorkspace,
        arnoldi_workspace: &mut Self::ArnoldiWorkspace,
        operator: &dyn LinearOperator,
        basis: &Array2<Complex64>,
        column: usize,
        output: &mut Array1<Complex64>,
    ) -> Result<(), IramError>;

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

    fn shifted_qr_filter(
        &mut self,
        hessenberg: &Array2<Complex64>,
        shifts: &[Complex64],
    ) -> Result<(Array2<Complex64>, Array2<Complex64>), IramError>;

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

    fn vector_norm2(&mut self, vector: &Array1<Complex64>) -> f64 {
        ops::norm2(vector)
    }

    fn normalize_vector(
        &mut self,
        vector: &mut Array1<Complex64>,
        context: &'static str,
    ) -> Result<f64, IramError> {
        ops::normalize(vector, context)
    }

    fn scale_vector_in_place(&mut self, vector: &mut Array1<Complex64>, alpha: Complex64) {
        ops::scale_in_place(vector, alpha);
    }

    fn add_scaled_vector_in_place(
        &mut self,
        target: &mut Array1<Complex64>,
        alpha: Complex64,
        source: &Array1<Complex64>,
    ) {
        ops::axpy_in_place(target, alpha, source);
    }
}
