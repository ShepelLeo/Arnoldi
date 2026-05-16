use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::magma::{
    self, DeviceMatrix, DeviceVector, MagmaSession, SchurOutput, ZgemmTranspose,
};
use crate::linalg::ops::{
    OrthogonalizedVector, orthogonalize_with_device_basis_reorthogonalization,
    orthogonalize_with_magma_reorthogonalization,
};

pub struct MagmaArnoldiWorkspace {
    d_basis: DeviceMatrix,
    d_candidate: DeviceVector,
}

pub struct MagmaBackend {
    session: MagmaSession,
}

impl MagmaBackend {
    pub fn new() -> Self {
        Self {
            session: MagmaSession::new(),
        }
    }
}

impl Default for MagmaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MagmaBackend {
    type ArnoldiWorkspace = MagmaArnoldiWorkspace;
    type RitzDecomposition = SchurOutput;

    fn name(&self) -> &'static str {
        "magma"
    }

    fn create_arnoldi_workspace(
        &mut self,
        basis: &Array2<Complex64>,
        dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError> {
        Ok(MagmaArnoldiWorkspace {
            d_basis: DeviceMatrix::from_column_major(&self.session, basis.view()),
            d_candidate: DeviceVector::new(dimension),
        })
    }

    fn orthogonalize_arnoldi_candidate(
        &mut self,
        workspace: &mut Self::ArnoldiWorkspace,
        _basis: &Array2<Complex64>,
        basis_columns: usize,
        candidate: &mut Array1<Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        let candidate_slice = candidate
            .as_slice()
            .expect("Arnoldi candidate must be contiguous");
        workspace
            .d_candidate
            .copy_from_slice(&self.session, candidate_slice);

        orthogonalize_with_device_basis_reorthogonalization(
            &self.session,
            &workspace.d_basis,
            basis_columns,
            candidate,
            &mut workspace.d_candidate,
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }

    fn append_arnoldi_basis_column(
        &mut self,
        workspace: &mut Self::ArnoldiWorkspace,
        column: usize,
        values: &Array1<Complex64>,
    ) {
        let values = values
            .as_slice()
            .expect("Arnoldi basis column must be contiguous");
        workspace
            .d_basis
            .copy_column_from_slice(&self.session, column, values);
    }

    fn orthogonalize_restart_residual(
        &mut self,
        residual: &mut Array1<Complex64>,
        basis: &ArrayView2<'_, Complex64>,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        orthogonalize_with_magma_reorthogonalization(
            &self.session,
            residual,
            basis,
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }

    fn compute_ritz_values(
        &mut self,
        hessenberg: &Array2<Complex64>,
    ) -> Result<Self::RitzDecomposition, IramError> {
        magma::zgeev_right_eigenpairs(hessenberg)
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
        magma::select_right_eigenvectors(decomposition, ritz_indices, dim).map_err(|error| {
            IramError::Spectral(format!("Ritz vector extraction failed: {error:?}"))
        })
    }

    fn zgemm_nn(
        &mut self,
        a: ArrayView2<'_, Complex64>,
        b: ArrayView2<'_, Complex64>,
    ) -> Array2<Complex64> {
        magma::zgemm(ZgemmTranspose::None, ZgemmTranspose::None, a, b)
    }
}
