//! LAPACK/OpenBLAS-бэкенд для нового generic-API.
//!
//! Реализация прокидывает примитивы Backend в существующие FFI-обёртки в
//! `linalg/lapack.rs` и `linalg/shifted_qr.rs`. Базис Крылова хранится здесь
//! как обычный `Vec<Complex64>` column-major с шагом `ld = dimension`. Это и
//! есть "владение крупным буфером на стороне бэкенда" для CPU-варианта.

use num_complex::Complex64;

use crate::backend::{Backend, DenseColMajor, SmallEig};
use crate::error::IramError;
use crate::linalg::lapack::{
    ZgemmTranspose, zgemm_slice, zhseqr_schur_slice, ztrevc_all_right_slice,
};
use crate::linalg::ops::{
    OrthogonalizedVector, Trans, orthogonalize_against_host_basis_slice,
};
use crate::linalg::shifted_qr::shifted_qr_filter_slice;
use crate::operator::LinearOperator;

/// CPU-вариант хранилища базиса. Поле `data` имеет длину `rows * capacity`
/// и хранится column-major; `ld == rows`.
pub struct HostBasis {
    data: Vec<Complex64>,
    rows: usize,
    #[allow(dead_code)]
    capacity: usize,
}

impl HostBasis {
    fn new(rows: usize, capacity: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; rows * capacity],
            rows,
            capacity,
        }
    }

    #[inline]
    fn column(&self, col: usize) -> &[Complex64] {
        debug_assert!(col < self.capacity);
        let start = col * self.rows;
        &self.data[start..start + self.rows]
    }

    #[inline]
    fn column_mut(&mut self, col: usize) -> &mut [Complex64] {
        debug_assert!(col < self.capacity);
        let start = col * self.rows;
        &mut self.data[start..start + self.rows]
    }
}

pub struct HostVector {
    data: Vec<Complex64>,
}

impl HostVector {
    fn new(len: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; len],
        }
    }
}

#[derive(Debug, Default)]
pub struct LapackBackend;

impl LapackBackend {
    pub fn new() -> Self {
        Self
    }
}

fn trans_to_zgemm(trans: Trans) -> ZgemmTranspose {
    match trans {
        Trans::None => ZgemmTranspose::None,
        Trans::ConjugateTranspose => ZgemmTranspose::ConjugateTranspose,
    }
}

impl Backend for LapackBackend {
    type OperatorHandle = ();
    type BasisHandle = HostBasis;
    type VectorHandle = HostVector;

    fn name(&self) -> &'static str {
        "lapack"
    }

    fn prepare_operator(
        &mut self,
        _operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorHandle, IramError> {
        Ok(())
    }

    fn alloc_basis(
        &mut self,
        dimension: usize,
        capacity: usize,
    ) -> Result<Self::BasisHandle, IramError> {
        Ok(HostBasis::new(dimension, capacity))
    }

    fn alloc_vector(&mut self, dimension: usize) -> Result<Self::VectorHandle, IramError> {
        Ok(HostVector::new(dimension))
    }

    fn write_basis_column(
        &mut self,
        basis: &mut Self::BasisHandle,
        column: usize,
        values: &[Complex64],
    ) {
        basis.column_mut(column).copy_from_slice(values);
    }

    fn read_basis_column(
        &mut self,
        basis: &Self::BasisHandle,
        column: usize,
        out: &mut [Complex64],
    ) {
        out.copy_from_slice(basis.column(column));
    }

    fn write_vector(&mut self, vector: &mut Self::VectorHandle, values: &[Complex64]) {
        vector.data.copy_from_slice(values);
    }

    fn read_vector(&mut self, vector: &Self::VectorHandle, out: &mut [Complex64]) {
        out.copy_from_slice(&vector.data);
    }

    fn spmv_basis_column(
        &mut self,
        _operator: &mut Self::OperatorHandle,
        operator_obj: &dyn LinearOperator,
        basis: &Self::BasisHandle,
        column: usize,
        out_vector: &mut Self::VectorHandle,
        out_host_mirror: &mut [Complex64],
    ) -> Result<(), IramError> {
        let source = basis.column(column);
        operator_obj.apply(source, out_host_mirror)?;
        // Поддерживаем host-mirror и device-handle в синхронном состоянии
        // (для CPU это одна и та же память, но трейту нужно явное обновление).
        out_vector.data.copy_from_slice(out_host_mirror);
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
        let result = orthogonalize_against_host_basis_slice(
            candidate_host,
            &basis.data,
            basis.rows,
            basis_columns,
            h_column,
            reference_norm,
            breakdown_tol,
        );
        candidate_vec.data.copy_from_slice(candidate_host);
        result
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
        orthogonalize_against_host_basis_slice(
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
        zgemm_slice(
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
        let mut schur = zhseqr_schur_slice(&matrix.data, n)
            .map_err(|error| IramError::Spectral(format!("zhseqr_schur failed: {error:?}")))?;
        let vectors = ztrevc_all_right_slice(&mut schur, n)
            .map_err(|error| IramError::Spectral(format!("ztrevc failed: {error:?}")))?;
        Ok(SmallEig {
            values: schur.w,
            vectors,
            dim: n,
        })
    }
}

