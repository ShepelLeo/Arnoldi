//! Базовые операции над комплексными векторами/матрицами в формате column-major
//! плюс реализация классического Gram-Schmidt с реортогонализацией.
//!
//! Этот модуль — часть ядра алгоритма: он знает про CGS/reorthogonalization,
//! breakdown-criterion и т.д. Бэкенд видит только vector/basis primitives.

use num_complex::Complex64;
use rand::Rng;

use crate::backend::Backend;
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

// ---------- host-side vector ops над срезами ----------

/// 2-норма для непрерывного среза.
#[inline]
pub fn nrm2(vector: &[Complex64]) -> f64 {
    vector.iter().map(|entry| entry.norm_sqr()).sum::<f64>().sqrt()
}

/// Скалярное произведение `<x, y> = sum conj(x_i) * y_i`.
#[inline]
pub fn dotc(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&l, &r)| l.conj() * r)
        .sum::<Complex64>()
}

/// `x *= alpha`.
#[inline]
pub fn scal(vector: &mut [Complex64], alpha: Complex64) {
    vector.iter_mut().for_each(|entry| *entry *= alpha);
}

/// `y += alpha * x`.
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

// ---------- Classical Gram-Schmidt с реортогонализацией ----------

/// CGS+reortho против первых `basis_columns` колонок backend-owned базиса.
///
/// Алгоритм:
/// 1. Дважды выполняется `proj := V[:, 0..n]^H * x`,  `x -= V[:, 0..n] * proj`,
///    с накоплением `h_column += proj`.
/// 2. Если получившаяся норма меньше `breakdown_tol * reference_norm`, ставим
///    `happy_breakdown` и возвращаем результат.
/// 3. Иначе нормализуем кандидата, делаем однократную реортогонализацию
///    (третий проход проекции) и при превышении
///    `REORTHOGONALIZATION_THRESHOLD` повторно вычитаем проекцию и
///    нормализуем.
///
/// Все «крупные» операции — `V^H x` и `V * proj` — делегируются бэкенду
/// через `basis_prefix_conj_dot_vector` и `basis_prefix_sub_mul`. Сам CGS
/// живёт в ядре.
pub fn orthogonalize_against_backend_basis<B: Backend>(
    backend: &mut B,
    basis: &B::BasisHandle,
    basis_columns: usize,
    candidate_host: &mut [Complex64],
    candidate_vec: &mut B::VectorHandle,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    debug_assert!(h_column.len() >= basis_columns);

    let mut projection = vec![Complex64::ZERO; basis_columns];

    // Два прохода CGS.
    for _ in 0..2 {
        backend.basis_prefix_conj_dot_vector(basis, basis_columns, candidate_vec, &mut projection);

        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p;
        }

        backend.basis_prefix_sub_mul(basis, basis_columns, &projection, candidate_vec);
    }

    let mut residual_norm = backend.vector_nrm2(candidate_vec);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        backend.read_vector(candidate_vec, candidate_host);
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    backend.vector_scale(candidate_vec, Complex64::new(1.0 / residual_norm, 0.0));

    // Один проход проверочной реортогонализации.
    backend.basis_prefix_conj_dot_vector(basis, basis_columns, candidate_vec, &mut projection);

    let correction_norm = nrm2(&projection);
    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        backend.basis_prefix_sub_mul(basis, basis_columns, &projection, candidate_vec);

        let reorthogonalized_norm = backend.vector_nrm2(candidate_vec);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            backend.read_vector(candidate_vec, candidate_host);
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }

        backend.vector_scale(candidate_vec, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
    }

    backend.read_vector(candidate_vec, candidate_host);
    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// CGS+reortho против host-матрицы `basis` (column-major, `rows × cols`,
/// ld = `rows`). Используется в рестарт-пути, когда «повернутый» базис уже
/// лежит на host после `gemm` и не требует backend-owned контейнера.
///
/// Реализация — наивный двойной CGS на чистых срезах. Сложность `O(m·k)`,
/// что соответствует `zgemv`-варианту; BLAS здесь не нужен — это редкий путь.
pub fn orthogonalize_against_host_basis_slice(
    candidate: &mut [Complex64],
    basis: &[Complex64],
    basis_rows: usize,
    basis_columns: usize,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    debug_assert_eq!(candidate.len(), basis_rows);
    debug_assert!(basis.len() >= basis_rows * basis_columns);
    debug_assert!(h_column.len() >= basis_columns);

    let mut projection = vec![Complex64::ZERO; basis_columns];

    for _ in 0..2 {
        compute_basis_conj_dot_host(basis, basis_rows, basis_columns, candidate, &mut projection);

        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p;
        }

        apply_basis_sub_mul_host(basis, basis_rows, basis_columns, &projection, candidate);
    }

    let mut residual_norm = nrm2(candidate);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scal(candidate, Complex64::new(1.0 / residual_norm, 0.0));

    compute_basis_conj_dot_host(basis, basis_rows, basis_columns, candidate, &mut projection);

    let correction_norm = nrm2(&projection);
    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        apply_basis_sub_mul_host(basis, basis_rows, basis_columns, &projection, candidate);

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

/// `projection[j] = sum_i conj(basis[i, j]) * candidate[i]`.
fn compute_basis_conj_dot_host(
    basis: &[Complex64],
    basis_rows: usize,
    basis_columns: usize,
    candidate: &[Complex64],
    projection: &mut [Complex64],
) {
    for j in 0..basis_columns {
        let column = &basis[j * basis_rows..(j + 1) * basis_rows];
        projection[j] = dotc(column, candidate);
    }
}

/// `candidate -= sum_j basis[:, j] * projection[j]`.
fn apply_basis_sub_mul_host(
    basis: &[Complex64],
    basis_rows: usize,
    basis_columns: usize,
    projection: &[Complex64],
    candidate: &mut [Complex64],
) {
    for j in 0..basis_columns {
        let column = &basis[j * basis_rows..(j + 1) * basis_rows];
        axpy(candidate, -projection[j], column);
    }
}
