use ndarray::{Array1, Array2, ArrayView2, s};
use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::lapack::{self, SchurOutput, ZgemmTranspose};
use crate::linalg::shifted_qr::shifted_qr_filter;
use crate::operator::LinearOperator;
use crate::linalg::ops::{OrthogonalizedVector, orthogonalize_with_reorthogonalization};

#[derive(Debug, Default)]
pub struct LapackBackend;

impl LapackBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for LapackBackend {
    type OperatorWorkspace = ();
    type ArnoldiWorkspace = ();
    type RitzDecomposition = SchurOutput;

    fn name(&self) -> &'static str {
        "lapack"
    }

    fn prepare_operator(
        &mut self,
        _operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorWorkspace, IramError> {
        Ok(())
    }

    fn create_arnoldi_workspace(
        &mut self,
        _basis: &Array2<Complex64>,
        _dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError> {
        Ok(())
    }


    fn apply_operator_to_basis_vector(
        &mut self,
        _operator_workspace: &mut Self::OperatorWorkspace,
        _arnoldi_workspace: &mut Self::ArnoldiWorkspace,
        operator: &dyn LinearOperator,
        basis: &Array2<Complex64>,
        column: usize,
        output: &mut Array1<Complex64>,
    ) -> Result<(), IramError> {
        operator.apply_into(basis.column(column), output.view_mut())
    }

    fn orthogonalize_arnoldi_candidate(
        &mut self,
        _workspace: &mut Self::ArnoldiWorkspace,
        basis: &Array2<Complex64>,
        basis_columns: usize,
        candidate: &mut Array1<Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        orthogonalize_with_reorthogonalization(
            candidate,
            &basis.slice(s![.., 0..basis_columns]),
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }

    fn append_arnoldi_basis_column(
        &mut self,
        _workspace: &mut Self::ArnoldiWorkspace,
        _column: usize,
        _values: &Array1<Complex64>,
    ) {
    }

    fn orthogonalize_restart_residual(
        &mut self,
        residual: &mut Array1<Complex64>,
        basis: &ArrayView2<'_, Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        orthogonalize_with_reorthogonalization(
            residual,
            basis,
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }


    fn shifted_qr_filter(
        &mut self,
        hessenberg: &Array2<Complex64>,
        shifts: &[Complex64],
    ) -> Result<(Array2<Complex64>, Array2<Complex64>), IramError> {
        shifted_qr_filter(hessenberg, shifts).map_err(IramError::Spectral)
    }

    fn compute_ritz_values(
        &mut self,
        hessenberg: &Array2<Complex64>,
    ) -> Result<Self::RitzDecomposition, IramError> {
        lapack::zhseqr_schur(hessenberg)
            .map_err(|error| IramError::Spectral(format!("small Ritz problem failed: {error:?}")))
    }

    fn ritz_values<'a>(&self, decomposition: &'a Self::RitzDecomposition) -> &'a [Complex64] {
        &decomposition.w
    }

    fn retrieve_ritz_vectors(
        &mut self,
        decomposition: &mut Self::RitzDecomposition,
        ritz_indices: &[usize],
        dim: usize,
    ) -> Result<Array2<Complex64>, IramError> {
        lapack::ztrevc_right_selected(decomposition, ritz_indices, dim).map_err(|error| {
            IramError::Spectral(format!("Ritz vector extraction failed: {error:?}"))
        })
    }

    fn zgemm_nn(
        &mut self,
        a: ArrayView2<'_, Complex64>,
        b: ArrayView2<'_, Complex64>,
    ) -> Array2<Complex64> {
        lapack::zgemm(ZgemmTranspose::None, ZgemmTranspose::None, a, b)
    }
}
