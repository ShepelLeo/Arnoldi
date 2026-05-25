//! MAGMA / cuSPARSE-бэкенд для нового generic-API.
//!
//! Базис Крылова, текущий кандидат и проекция владеются устройством через
//! `DeviceMatrix` / `DeviceVector`. Ядро работает только со срезами на хосте;
//! бэкенд сам синхронизирует device-зеркало.

use num_complex::Complex64;

use crate::backend::{Backend, DenseColMajor, SmallEig};
use crate::error::IramError;
use crate::linalg::magma::{
    DeviceCsrMatrix, DeviceMatrix, DeviceVector, MagmaSession, ZgemmTranspose,
    ZgemvTranspose as MagmaZgemvTranspose, zgemm_with_session_slice,
    zgeev_right_eigenpairs_slice,
};
use crate::linalg::ops::{
    OrthogonalizedVector, REORTHOGONALIZATION_THRESHOLD, Trans, is_numerical_breakdown, nrm2,
    scal,
};
use crate::linalg::shifted_qr::shifted_qr_filter_slice;
use crate::operator::LinearOperator;

pub struct MagmaOperatorHandle {
    d_operator: DeviceCsrMatrix,
}

pub struct MagmaBasisHandle {
    d_basis: DeviceMatrix,
    rows: usize,
    capacity: usize,
}

pub struct MagmaVectorHandle {
    d_vector: DeviceVector,
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

fn trans_to_zgemm(trans: Trans) -> ZgemmTranspose {
    match trans {
        Trans::None => ZgemmTranspose::None,
        Trans::ConjugateTranspose => ZgemmTranspose::ConjugateTranspose,
    }
}

/// CGS + однократная реортогонализация. Базис — устройство, кандидат — пара
/// (host, device).
fn orthogonalize_against_device_basis(
    session: &MagmaSession,
    d_basis: &DeviceMatrix,
    basis_columns: usize,
    candidate_host: &mut [Complex64],
    d_candidate: &mut DeviceVector,
    d_projection: &mut DeviceVector,
    projection_host: &mut [Complex64],
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    let m = d_basis.rows();
    let n = basis_columns;
    debug_assert!(n <= d_basis.columns());
    debug_assert_eq!(candidate_host.len(), m);
    debug_assert_eq!(d_candidate.len(), m);
    debug_assert!(d_projection.len() >= n);
    debug_assert!(projection_host.len() >= n);
    debug_assert!(h_column.len() >= n);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::ZERO;
    let minus_one = Complex64::new(-1.0, 0.0);
    let projection_host = &mut projection_host[..n];

    for _ in 0..2 {
        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::ConjugateTranspose,
            one,
            d_candidate,
            zero,
            d_projection,
        );
        d_projection.copy_prefix_to_slice(session, projection_host);

        for (h, &p) in h_column.iter_mut().zip(projection_host.iter()) {
            *h += p;
        }

        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::None,
            minus_one,
            d_projection,
            one,
            d_candidate,
        );
    }

    d_candidate.copy_to_slice(session, candidate_host);

    let mut residual_norm = nrm2(candidate_host);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scal(candidate_host, Complex64::new(1.0 / residual_norm, 0.0));
    d_candidate.copy_from_slice(session, candidate_host);

    d_basis.zgemv_leading_columns(
        session,
        n,
        MagmaZgemvTranspose::ConjugateTranspose,
        one,
        d_candidate,
        zero,
        d_projection,
    );
    d_projection.copy_prefix_to_slice(session, projection_host);

    let correction_norm = nrm2(projection_host);

    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection_host.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::None,
            minus_one,
            d_projection,
            one,
            d_candidate,
        );
        d_candidate.copy_to_slice(session, candidate_host);

        let reorthogonalized_norm = nrm2(candidate_host);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }
        scal(candidate_host, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
        d_candidate.copy_from_slice(session, candidate_host);
    }

    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// CGS + реортогонализация для host-базиса (используется на рестарте). Сначала
/// host -> device загрузка, потом то же ядро.
fn orthogonalize_against_host_basis_via_magma(
    session: &MagmaSession,
    candidate_host: &mut [Complex64],
    basis_host: &[Complex64],
    basis_rows: usize,
    basis_columns: usize,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    debug_assert_eq!(candidate_host.len(), basis_rows);
    debug_assert!(basis_host.len() >= basis_rows * basis_columns);
    debug_assert!(h_column.len() >= basis_columns);

    let mut d_basis = DeviceMatrix::new(basis_rows, basis_columns);
    d_basis.copy_from_host_slice(session, basis_host, basis_rows);

    let mut d_candidate = DeviceVector::from_slice(session, candidate_host);
    let mut d_projection = DeviceVector::new(basis_columns);
    let mut projection_host = vec![Complex64::ZERO; basis_columns];

    orthogonalize_against_device_basis(
        session,
        &d_basis,
        basis_columns,
        candidate_host,
        &mut d_candidate,
        &mut d_projection,
        &mut projection_host,
        h_column,
        reference_norm,
        breakdown_tol,
    )
}


impl Backend for MagmaBackend {
    type OperatorHandle = MagmaOperatorHandle;
    type BasisHandle = MagmaBasisHandle;
    type VectorHandle = MagmaVectorHandle;

    fn name(&self) -> &'static str {
        "magma"
    }

    fn prepare_operator(
        &mut self,
        operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorHandle, IramError> {
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

        Ok(MagmaOperatorHandle { d_operator })
    }

    fn alloc_basis(
        &mut self,
        dimension: usize,
        capacity: usize,
    ) -> Result<Self::BasisHandle, IramError> {
        Ok(MagmaBasisHandle {
            d_basis: DeviceMatrix::new(dimension, capacity),
            rows: dimension,
            capacity,
        })
    }

    fn alloc_vector(&mut self, dimension: usize) -> Result<Self::VectorHandle, IramError> {
        Ok(MagmaVectorHandle {
            d_vector: DeviceVector::new(dimension),
        })
    }

    fn write_basis_column(
        &mut self,
        basis: &mut Self::BasisHandle,
        column: usize,
        values: &[Complex64],
    ) {
        basis
            .d_basis
            .copy_column_from_slice(&self.session, column, values);
    }

    fn read_basis_column(
        &mut self,
        basis: &Self::BasisHandle,
        column: usize,
        out: &mut [Complex64],
    ) {
        debug_assert!(column < basis.capacity);
        debug_assert_eq!(out.len(), basis.rows);
        basis.d_basis.copy_column_to_slice(&self.session, column, out);
    }

    fn write_vector(&mut self, vector: &mut Self::VectorHandle, values: &[Complex64]) {
        vector.d_vector.copy_from_slice(&self.session, values);
    }

    fn read_vector(&mut self, vector: &Self::VectorHandle, out: &mut [Complex64]) {
        vector.d_vector.copy_to_slice(&self.session, out);
    }

    fn spmv_basis_column(
        &mut self,
        operator: &mut Self::OperatorHandle,
        _operator_obj: &dyn LinearOperator,
        basis: &Self::BasisHandle,
        column: usize,
        out_vector: &mut Self::VectorHandle,
        out_host_mirror: &mut [Complex64],
    ) -> Result<(), IramError> {
        let x = basis.d_basis.column_ptr(column);
        let y = out_vector.d_vector.mut_ptr();
        operator
            .d_operator
            .spmv_raw(&self.session, x, y)
            .map_err(|error| IramError::Spectral(format!("cuSPARSE SpMV failed: {error:?}")))?;
        out_vector
            .d_vector
            .copy_to_slice(&self.session, out_host_mirror);
        Ok(())
    }

    fn orthogonalize_against_basis(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        candidate_host: &mut [Complex64],
        candidate_vec: &mut Self::VectorHandle,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        // Прокидываем рабочие буферы basis (d_projection, projection_host)
        // через unsafe-окно: мы не можем мутировать `basis: &Self::BasisHandle`,
        // поэтому держим отдельные временные буферы.
        let mut d_projection = DeviceVector::new(basis_columns);
        let mut projection_host = vec![Complex64::ZERO; basis_columns];

        orthogonalize_against_device_basis(
            &self.session,
            &basis.d_basis,
            basis_columns,
            candidate_host,
            &mut candidate_vec.d_vector,
            &mut d_projection,
            &mut projection_host,
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }

    fn orthogonalize_against_host_basis(
        &mut self,
        residual: &mut [Complex64],
        basis: &[Complex64],
        basis_rows: usize,
        basis_columns: usize,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector {
        orthogonalize_against_host_basis_via_magma(
            &self.session,
            residual,
            basis,
            basis_rows,
            basis_columns,
            h_column,
            reference_norm,
            breakdown_tol,
        )
    }

    fn gemm(
        &mut self,
        trans_a: Trans,
        trans_b: Trans,
        m: usize,
        n: usize,
        k: usize,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        c: &mut [Complex64],
        ldc: usize,
    ) {
        zgemm_with_session_slice(
            &self.session,
            trans_to_zgemm(trans_a),
            trans_to_zgemm(trans_b),
            m,
            n,
            k,
            a,
            lda,
            b,
            ldb,
            c,
            ldc,
        );
    }

    fn multishift_qr_filter(
        &mut self,
        hessenberg: &DenseColMajor,
        shifts: &[Complex64],
    ) -> Result<(DenseColMajor, DenseColMajor), IramError> {
        // Маленький Хессенберг (ncv ~ 200). CUDA-прототип медленнее CPU-пути
        // и плодит D2H/H2D-трафик. Держим QR на хосте, пока рестартная
        // ротация не интегрирована в device-resident pipeline.
        if hessenberg.rows != hessenberg.cols {
            return Err(IramError::Spectral(
                "multishift QR expects a square Hessenberg matrix".to_string(),
            ));
        }
        let n = hessenberg.rows;
        let (q, h) = shifted_qr_filter_slice(&hessenberg.data, n, shifts)
            .map_err(IramError::Spectral)?;
        Ok((
            DenseColMajor {
                data: q,
                rows: n,
                cols: n,
            },
            DenseColMajor {
                data: h,
                rows: n,
                cols: n,
            },
        ))
    }

    fn small_eig(&mut self, matrix: &DenseColMajor) -> Result<SmallEig, IramError> {
        if matrix.rows != matrix.cols {
            return Err(IramError::Spectral(
                "small_eig expects a square matrix".to_string(),
            ));
        }
        let n = matrix.rows;
        let schur = zgeev_right_eigenpairs_slice(&matrix.data, n).map_err(|error| {
            IramError::Spectral(format!("magma_zgeev failed: {error:?}"))
        })?;
        // У zgeev собственные векторы уже лежат в `schur.z`, column-major dim×dim.
        Ok(SmallEig {
            values: schur.w,
            vectors: schur.z,
            dim: n,
        })
    }
}

