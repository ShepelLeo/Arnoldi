use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::magma::{
    self, DeviceCsrMatrix, DeviceMatrix, DeviceVector, MagmaSession, SchurOutput, ZgemmTranspose,
};
use crate::linalg::ops::{
    OrthogonalizedVector, orthogonalize_with_device_basis_reorthogonalization,
    orthogonalize_with_magma_reorthogonalization,
};
use crate::operator::LinearOperator;

pub struct MagmaOperatorWorkspace {
    d_operator: DeviceCsrMatrix,
}

pub struct MagmaArnoldiWorkspace {
    d_basis: DeviceMatrix,
    d_candidate: DeviceVector,
    d_projection: DeviceVector,
    projection: Vec<Complex64>,
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
    type OperatorWorkspace = MagmaOperatorWorkspace;
    type ArnoldiWorkspace = MagmaArnoldiWorkspace;
    type RitzDecomposition = SchurOutput;

    fn name(&self) -> &'static str {
        "magma"
    }

    fn prepare_operator(
        &mut self,
        operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorWorkspace, IramError> {
        let csr = operator.csr_matrix().ok_or_else(|| {
            IramError::InvalidConfig(format!(
                "MAGMA backend requires a CSR-materializable operator for cuSPARSE matvec; '{}' does not provide one",
                operator.description(),
            ))
        })?;
        csr.validate()?;

        let d_operator = DeviceCsrMatrix::from_csr(
            &self.session,
            csr.dimension,
            &csr.row_offsets,
            &csr.columns,
            &csr.values,
        )
        .map_err(|error| IramError::Spectral(format!("cuSPARSE CSR upload failed: {error:?}")))?;

        Ok(MagmaOperatorWorkspace { d_operator })
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
        })
    }

    fn apply_operator_to_basis_vector(
        &mut self,
        operator_workspace: &mut Self::OperatorWorkspace,
        arnoldi_workspace: &mut Self::ArnoldiWorkspace,
        _operator: &dyn LinearOperator,
        _basis: &Array2<Complex64>,
        column: usize,
        output: &mut Array1<Complex64>,
    ) -> Result<(), IramError> {
        let x = arnoldi_workspace.d_basis.column_ptr(column);
        let y = arnoldi_workspace.d_candidate.mut_ptr();
        operator_workspace
            .d_operator
            .spmv_raw(&self.session, x, y)
            .map_err(|error| IramError::Spectral(format!("cuSPARSE SpMV failed: {error:?}")))?;

        let output_slice = output
            .as_slice_mut()
            .expect("Arnoldi candidate must be contiguous");
        arnoldi_workspace
            .d_candidate
            .copy_to_slice(&self.session, output_slice);
        Ok(())
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
        // `apply_operator_to_basis_vector` has already written the candidate to
        // `workspace.d_candidate`. Re-uploading `candidate` here creates a full
        // host -> device copy on every Arnoldi step and cancels much of the
        // benefit of doing matvec on the GPU.
        //
        // The host `candidate` remains a mirror used for norm/breakdown logic;
        // the device vector is the source of truth for the BLAS projections.
        orthogonalize_with_device_basis_reorthogonalization(
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

    fn shifted_qr_filter(
        &mut self,
        hessenberg: &Array2<Complex64>,
        shifts: &[Complex64],
    ) -> Result<(Array2<Complex64>, Array2<Complex64>), IramError> {
        // The shifted QR problem here is tiny (ncv ~= 200). The CUDA prototype
        // moves H/Q to the device, launches a mostly serial kernel, and copies
        // both matrices back on every restart. That is slower than the CPU path
        // and increases device allocations/transfers, so keep QR on host until
        // restart rotation is fused into a fully device-resident pipeline.
        crate::linalg::shifted_qr::shifted_qr_filter(hessenberg, shifts)
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
