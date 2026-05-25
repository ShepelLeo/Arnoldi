//! Процесс Арнольди — ядро без зависимости от конкретного бэкенда.
//!
//! Ядро работает только с примитивами `Backend`. Базис V (размер `m × (ncv+1)`)
//! хранится в дескрипторе, который владеет бэкенд (CPU-память или GPU-память).
//! Хессенберг H живёт в простом `Vec<Complex64>` column-major с явным `ld`,
//! здесь же владение остаётся за ядром, поскольку матрица маленькая
//! (`(ncv+1) × ncv`).

use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::ops::{normalize, nrm2, orthogonalize_against_backend_basis};
use crate::operator::LinearOperator;

/// Маленькая верхнехессенбергова матрица в column-major раскладке.
#[derive(Debug, Clone)]
pub struct HessenbergMatrix {
    pub data: Vec<Complex64>,
    pub rows: usize,
    pub cols: usize,
}

impl HessenbergMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; rows * cols],
            rows,
            cols,
        }
    }

    #[inline]
    pub fn ld(&self) -> usize {
        self.rows.max(1)
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row + col * self.rows] = value;
    }
}

/// Результат процесса Арнольди.
#[derive(Debug, Clone)]
pub struct ArnoldiFactorization<H> {
    /// Дескриптор базиса (живёт на стороне бэкенда).
    pub basis: H,
    /// Хессенберг размером `(target_steps+1) × target_steps`.
    pub hessenberg: HessenbergMatrix,
    pub performed_steps: usize,
    pub happy_breakdown: bool,
}

impl<H> ArnoldiFactorization<H> {
    /// Возвращает квадратный обрезок H ([0..k, 0..k]) копией column-major.
    pub fn square_hessenberg(&self) -> Vec<Complex64> {
        let k = self.performed_steps;
        let mut out = vec![Complex64::ZERO; k * k];
        for j in 0..k {
            for i in 0..k {
                out[i + j * k] = self.hessenberg.get(i, j);
            }
        }
        out
    }

    /// Норма последнего поддиагонального элемента
    /// `h_{performed_steps, performed_steps-1}`.
    pub fn trailing_subdiagonal(&self) -> f64 {
        if self.happy_breakdown || self.performed_steps == 0 {
            0.0
        } else {
            let k = self.performed_steps;
            self.hessenberg.get(k, k - 1).norm()
        }
    }
}

/// Запускает процесс Арнольди с нуля.
pub fn run_arnoldi<B: Backend>(
    backend: &mut B,
    operator_handle: &mut B::OperatorHandle,
    operator: &dyn LinearOperator,
    start_vector: &[Complex64],
    steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization<B::BasisHandle>, IramError> {
    let dim = operator.dimension();
    if start_vector.len() != dim {
        return Err(IramError::DimensionMismatch {
            expected: dim,
            got: start_vector.len(),
        });
    }

    let mut normalized_start = start_vector.to_vec();
    normalize(&mut normalized_start, "Arnoldi start vector")?;

    let mut basis = backend.alloc_basis(dim, steps + 1)?;
    backend.write_basis_column(&mut basis, 0, &normalized_start);
    let hessenberg = HessenbergMatrix::zeros(steps + 1, steps);

    continue_arnoldi(
        backend,
        operator_handle,
        operator,
        basis,
        hessenberg,
        0,
        steps,
        breakdown_tol,
        matvec_count,
    )
}

/// Продолжает (или начинает с шага `start_step`) процесс Арнольди.
pub fn continue_arnoldi<B: Backend>(
    backend: &mut B,
    operator_handle: &mut B::OperatorHandle,
    operator: &dyn LinearOperator,
    mut basis: B::BasisHandle,
    mut hessenberg: HessenbergMatrix,
    start_step: usize,
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization<B::BasisHandle>, IramError> {
    if hessenberg.rows != target_steps + 1 || hessenberg.cols != target_steps {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation expected Hessenberg shape {}x{}, got {}x{}",
            target_steps + 1,
            target_steps,
            hessenberg.rows,
            hessenberg.cols,
        )));
    }

    let dim = operator.dimension();
    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    let mut candidate_vec = backend.alloc_vector(dim)?;
    let mut candidate_host = vec![Complex64::ZERO; dim];
    let mut h_column = vec![Complex64::ZERO; target_steps];

    for step in start_step..target_steps {
        backend.spmv_basis_column(
            operator_handle,
            operator,
            &basis,
            step,
            &mut candidate_vec,
            &mut candidate_host,
        )?;
        let candidate_old = nrm2(&candidate_host);
        *matvec_count += 1;

        for value in h_column[..=step].iter_mut() {
            *value = Complex64::ZERO;
        }

        let orthogonalized = orthogonalize_against_backend_basis(
            backend,
            &basis,
            step + 1,
            &mut candidate_host,
            &mut candidate_vec,
            &mut h_column[..=step],
            candidate_old,
            breakdown_tol,
        );

        for row in 0..=step {
            hessenberg.set(row, step, h_column[row]);
        }
        performed_steps = step + 1;

        if orthogonalized.happy_breakdown {
            happy_breakdown = true;
            hessenberg.set(step + 1, step, Complex64::ZERO);
            break;
        }

        hessenberg.set(
            step + 1,
            step,
            Complex64::new(orthogonalized.residual_norm, 0.0),
        );
        backend.write_basis_column(&mut basis, step + 1, &candidate_host);
    }

    Ok(ArnoldiFactorization {
        basis,
        hessenberg,
        performed_steps,
        happy_breakdown,
    })
}
