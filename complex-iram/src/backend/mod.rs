//! Backend abstraction for the shared IRAM algorithm.
//!
//! The IRAM orchestration owns no numerical kernel directly. A selected backend
//! must prepare the operator, perform MatVec, orthogonalize Arnoldi vectors,
//! solve the small Ritz problem, run the implicit shifted-QR restart filter, and
//! multiply dense basis blocks.

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::OrthogonalizedVector;
use crate::operator::LinearOperator;

pub mod lapack;
#[cfg(feature = "magma")]
pub mod magma;

pub use lapack::LapackBackend;
#[cfg(feature = "magma")]
pub use magma::MagmaBackend;

pub trait Backend {
    type PreparedOperator<'operator>
    where
        Self: 'operator;
    type ArnoldiWorkspace;
    type RitzDecomposition;

    fn name(&self) -> &'static str;

    fn prepare_operator<'operator>(
        &mut self,
        operator: &'operator dyn LinearOperator,
    ) -> Result<Self::PreparedOperator<'operator>, IramError>;

    fn prepared_operator_dimension(&self, operator: &Self::PreparedOperator<'_>) -> usize;

    fn prepared_operator_description(&self, operator: &Self::PreparedOperator<'_>) -> String;

    fn create_arnoldi_workspace(
        &mut self,
        basis: &Array2<Complex64>,
        dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError>;

    /// Computes `candidate = A * basis[:, column]` using the selected backend.
    ///
    /// The return value is the norm of the unorthogonalized candidate and is used
    /// as Arnoldi's reference norm for numerical-breakdown detection.
    fn apply_operator_to_arnoldi_column(
        &mut self,
        operator: &Self::PreparedOperator<'_>,
        workspace: &mut Self::ArnoldiWorkspace,
        basis: &Array2<Complex64>,
        column: usize,
        candidate: &mut Array1<Complex64>,
    ) -> Result<f64, IramError>;

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
}
