use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::magma::{
    self, DeviceCsrMatrix, DeviceMatrix, DeviceVector, MagmaSession, SchurOutput, ZgemmTranspose,
};
use crate::linalg::ops::{
    norm2, OrthogonalizedVector, orthogonalize_with_device_basis_reorthogonalization,
    orthogonalize_with_magma_reorthogonalization,
};
use crate::operator::{CsrMatrix, LinearOperator};

pub struct MagmaArnoldiWorkspace {
    d_basis: DeviceMatrix,
    d_candidate: DeviceVector,
    d_projection: DeviceVector,
    projection: Vec<Complex64>,
    candidate_is_device_current: bool,
}

pub struct MagmaPreparedOperator {
    dimension: usize,
    description: String,
    _host_csr: CsrMatrix,
    d_csr: DeviceCsrMatrix,
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
    type PreparedOperator<'operator> = MagmaPreparedOperator where Self: 'operator;
    type ArnoldiWorkspace = MagmaArnoldiWorkspace;
    type RitzDecomposition = SchurOutput;

    fn name(&self) -> &'static str {
        "magma"
    }

    fn prepare_operator<'operator>(
        &mut self,
        operator: &'operator dyn LinearOperator,
    ) -> Result<Self::PreparedOperator<'operator>, IramError> {
        let csr = operator.as_csr().cloned().or_else(|| operator.to_csr()).ok_or_else(|| {
            IramError::InvalidConfig(format!(
                "operator '{}' cannot be prepared for MAGMA: it does not expose CSR storage",
                operator.description(),
            ))
        })?;

        let d_csr = DeviceCsrMatrix::from_csr(
            &self.session,
            csr.rows(),
            csr.columns(),
            csr.row_offsets(),
            csr.column_indices(),
            csr.values(),
        )
        .map_err(|error| IramError::Spectral(format!("MAGMA CSR upload failed: {error:?}")))?;

        Ok(MagmaPreparedOperator {
            dimension: csr.rows(),
            description: operator.description(),
            _host_csr: csr,
            d_csr,
        })
    }

    fn prepared_operator_dimension(&self, operator: &Self::PreparedOperator<'_>) -> usize {
        operator.dimension
    }

    fn prepared_operator_description(&self, operator: &Self::PreparedOperator<'_>) -> String {
        operator.description.clone()
    }

    fn create_arnoldi_workspace(
        &mut self,
        basis: &Array2<Complex64>,
        dimension: usize,
    ) -> Result<Self::ArnoldiWorkspace, IramError> {
        Ok(MagmaArnoldiWorkspace {
            d_basis: DeviceMatrix::from_column_major(&self.session, basis.view()),
            d_candidate: DeviceVector::new(dimension),
            d_projection: DeviceVector::new(basis.ncols()),
            projection: vec![Complex64::ZERO; basis.ncols()],
            candidate_is_device_current: false,
        })
    }

    fn apply_operator_to_arnoldi_column(
        &mut self,
        operator: &Self::PreparedOperator<'_>,
        workspace: &mut Self::ArnoldiWorkspace,
        _basis: &Array2<Complex64>,
        column: usize,
        candidate: &mut Array1<Complex64>,
    ) -> Result<f64, IramError> {
        operator
            .d_csr
            .spmv_from_matrix_column(
                &self.session,
                &workspace.d_basis,
                column,
                &mut workspace.d_candidate,
            )
            .map_err(|error| IramError::Spectral(format!("MAGMA CSR SpMV failed: {error:?}")))?;

        let candidate_slice = candidate
            .as_slice_mut()
            .expect("Arnoldi candidate must be contiguous");
        workspace
            .d_candidate
            .copy_to_slice(&self.session, candidate_slice);
        workspace.candidate_is_device_current = true;
        Ok(norm2(candidate))
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
        if !workspace.candidate_is_device_current {
            let candidate_slice = candidate
                .as_slice()
                .expect("Arnoldi candidate must be contiguous");
            workspace
                .d_candidate
                .copy_from_slice(&self.session, candidate_slice);
        }

        let result = orthogonalize_with_device_basis_reorthogonalization(
            &self.session,
            &workspace.d_basis,
            basis_columns,
            candidate,
            &mut workspace.d_candidate,
            &mut workspace.d_projection,
            &mut workspace.projection[..basis_columns],
            h_column,
            reference_norm,
            breakdown_tol,
        );
        workspace.candidate_is_device_current = true;
        result
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
        workspace.candidate_is_device_current = false;
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

    fn shifted_qr_filter(
        &mut self,
        hessenberg: &Array2<Complex64>,
        shifts: &[Complex64],
    ) -> Result<(Array2<Complex64>, Array2<Complex64>), IramError> {
        magma::shifted_qr_filter_with_session(&self.session, hessenberg, shifts)
            .map_err(IramError::Spectral)
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
        magma::zgemm_with_session(&self.session, ZgemmTranspose::None, ZgemmTranspose::None, a, b)
    }
}
