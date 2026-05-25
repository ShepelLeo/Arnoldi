//! Базовые операции над комплексными векторами/матрицами в формате column-major.
//!
//! Слой `ops` — это набор generic-примитивов, на которых живёт ядро алгоритма
//! (Arnoldi/IRAM). Здесь нет ни одного упоминания шага Арнольди, рестарта,
//! Ритц-пары — только линейная алгебра по непрерывным `&[Complex64]`.

use num_complex::Complex64;
use rand::Rng;

use crate::error::IramError;

pub const REORTHOGONALIZATION_THRESHOLD: f64 = f64::EPSILON * 1000.0;

/// Опции переноса для GEMV/GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    None,
    ConjugateTranspose,
}

/// Результат ортогонализации очередного вектора-кандидата.
#[derive(Debug, Clone, Copy)]
pub struct OrthogonalizedVector {
    pub residual_norm: f64,
    pub happy_breakdown: bool,
}

/// 2-норма для непрерывного среза.
#[inline]
pub fn nrm2(vector: &[Complex64]) -> f64 {
    vector
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// Скалярное произведение <x, y> = sum conj(x_i) * y_i.
#[inline]
pub fn dotc(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&l, &r)| l.conj() * r)
        .sum::<Complex64>()
}

/// x *= alpha
#[inline]
pub fn scal(vector: &mut [Complex64], alpha: Complex64) {
    vector.iter_mut().for_each(|entry| *entry *= alpha);
}

/// y += alpha * x
#[inline]
pub fn axpy(target: &mut [Complex64], alpha: Complex64, source: &[Complex64]) {
    debug_assert_eq!(target.len(), source.len());
    for (t, &s) in target.iter_mut().zip(source.iter()) {
        *t += alpha * s;
    }
}

/// Нормализация. Возвращает исходную норму.
pub fn normalize(vector: &mut [Complex64], context: &'static str) -> Result<f64, IramError> {
    let norm = nrm2(vector);
    if norm <= f64::EPSILON {
        return Err(IramError::ZeroVector(context));
    }
    scal(vector, Complex64::new(1.0 / norm, 0.0));
    Ok(norm)
}

/// Проверка численного breakdown.
#[inline]
pub fn is_numerical_breakdown(residual_norm: f64, reference_norm: f64, tolerance: f64) -> bool {
    residual_norm <= tolerance * reference_norm
}

/// Reorthogonalized classical Gram-Schmidt против host-матрицы (column-major).
///
/// Использует BLAS-уровневые примитивы `zgemv_slice` через два прохода CGS
/// + однократную реортогонализацию по необходимости. Это generic-операция,
/// нужная не только Arnoldi, поэтому живёт в `ops`.
pub fn orthogonalize_against_host_basis_slice(
    candidate: &mut [Complex64],
    basis: &[Complex64],
    basis_rows: usize,
    basis_columns: usize,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    use crate::linalg::lapack::{ZgemvTranspose, zgemv_slice};

    debug_assert_eq!(candidate.len(), basis_rows);
    debug_assert!(basis.len() >= basis_rows * basis_columns);
    debug_assert!(h_column.len() >= basis_columns);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::ZERO;
    let minus_one = Complex64::new(-1.0, 0.0);
    let mut projection = vec![Complex64::ZERO; basis_columns];

    for _ in 0..2 {
        projection.fill(Complex64::ZERO);

        zgemv_slice(
            ZgemvTranspose::ConjugateTranspose,
            basis_rows,
            basis_columns,
            one,
            basis,
            basis_rows.max(1),
            candidate,
            zero,
            &mut projection,
        );

        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p;
        }

        zgemv_slice(
            ZgemvTranspose::None,
            basis_rows,
            basis_columns,
            minus_one,
            basis,
            basis_rows.max(1),
            &projection,
            one,
            candidate,
        );
    }

    let mut residual_norm = nrm2(candidate);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scal(candidate, Complex64::new(1.0 / residual_norm, 0.0));

    projection.fill(Complex64::ZERO);
    zgemv_slice(
        ZgemvTranspose::ConjugateTranspose,
        basis_rows,
        basis_columns,
        one,
        basis,
        basis_rows.max(1),
        candidate,
        zero,
        &mut projection,
    );

    let correction_norm = nrm2(&projection);
    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        zgemv_slice(
            ZgemvTranspose::None,
            basis_rows,
            basis_columns,
            minus_one,
            basis,
            basis_rows.max(1),
            &projection,
            one,
            candidate,
        );

        let reorthogonalized_norm = nrm2(candidate);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }
        scal(candidate, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
    }

    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// Генерация нормализованного случайного вектора (host).
pub fn normalized_random_vector<R>(
    dimension: usize,
    rng: &mut R,
) -> Result<Vec<Complex64>, IramError>
where
    R: Rng + ?Sized,
{
    let mut vector: Vec<Complex64> = (0..dimension)
        .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)))
        .collect();
    normalize(&mut vector, "random start vector generation")?;
    Ok(vector)
}
