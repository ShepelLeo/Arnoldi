//! MAGMA / cuSPARSE-бэкенд: владение device-резидентным базисом + generic
//! primitives. Никакой алгоритмической логики (CGS, Арнольди, рестарт) здесь
//! нет — её предоставляет ядро.

use num_complex::Complex64;

use crate::backend::{Backend, DenseColMajor, SmallEig};
use crate::error::IramError;
use crate::linalg::magma::{
    DeviceCsrMatrix, DeviceMatrix, DeviceVector, MagmaSession, ZgemmTranspose,
    ZgemvTranspose as MagmaZgemvTranspose, zgemm_with_session_slice,
    zgeev_right_eigenpairs_slice,
};
use crate::linalg::ops::{Trans, nrm2, scal};
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

    fn vector_nrm2(&mut self, vector: &Self::VectorHandle) -> f64 {
        // MAGMA не экспортирует `magma_dznrm2` в текущей биндинг-сборке.
        // Делаем D2H в локальный буфер и считаем норму на host.
        let len = vector.d_vector.len();
        let mut mirror = vec![Complex64::ZERO; len];
        vector.d_vector.copy_to_slice(&self.session, &mut mirror);
        nrm2(&mirror)
    }

    fn vector_scale(&mut self, vector: &mut Self::VectorHandle, alpha: Complex64) {
        // Аналогично: D2H → host-scal → H2D.
        let len = vector.d_vector.len();
        let mut mirror = vec![Complex64::ZERO; len];
        vector.d_vector.copy_to_slice(&self.session, &mut mirror);
        scal(&mut mirror, alpha);
        vector.d_vector.copy_from_slice(&self.session, &mirror);
    }

    fn basis_prefix_conj_dot_vector(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        vector: &Self::VectorHandle,
        out_projection: &mut [Complex64],
    ) {
        debug_assert!(basis_columns <= basis.capacity);
        debug_assert!(out_projection.len() >= basis_columns);
        if basis_columns == 0 {
            return;
        }
        let mut d_projection = DeviceVector::new(basis_columns);
        basis.d_basis.zgemv_leading_columns(
            &self.session,
            basis_columns,
            MagmaZgemvTranspose::ConjugateTranspose,
            Complex64::new(1.0, 0.0),
            &vector.d_vector,
            Complex64::ZERO,
            &mut d_projection,
        );
        d_projection.copy_prefix_to_slice(&self.session, &mut out_projection[..basis_columns]);
    }

    fn basis_prefix_sub_mul(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        projection: &[Complex64],
        vector: &mut Self::VectorHandle,
    ) {
        debug_assert!(basis_columns <= basis.capacity);
        debug_assert!(projection.len() >= basis_columns);
        if basis_columns == 0 {
            return;
        }
        let mut d_projection = DeviceVector::new(basis_columns);
        d_projection.copy_prefix_from_slice(&self.session, &projection[..basis_columns]);
        basis.d_basis.zgemv_leading_columns(
            &self.session,
            basis_columns,
            MagmaZgemvTranspose::None,
            Complex64::new(-1.0, 0.0),
            &d_projection,
            Complex64::new(1.0, 0.0),
            &mut vector.d_vector,
        );
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
        Ok(SmallEig {
            values: schur.w,
            vectors: schur.z,
            dim: n,
        })
    }
}
