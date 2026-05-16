//! Basic vector operations and backend-specific orthogonalization kernels.

use ndarray::{Array1, ArrayView2, Zip};
use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::error::IramError;
use crate::linalg::lapack::{ZgemvTranspose, zgemv};

#[cfg(feature = "magma")]
use crate::linalg::magma::{DeviceMatrix, DeviceVector, MagmaSession, ZgemvTranspose as MagmaZgemvTranspose};

const REORTHOGONALIZATION_THRESHOLD: f64 = f64::EPSILON * 1000.0;

#[derive(Debug, Clone, Copy)]
pub struct OrthogonalizedVector {
    pub residual_norm: f64,
    pub happy_breakdown: bool,
}

/// Нормализация вектора
pub fn normalize(vector: &mut Array1<Complex64>, context: &'static str) -> Result<f64, IramError> {
    let norm = norm2(vector);

    if norm <= f64::EPSILON {
        return Err(IramError::ZeroVector(context));
    }

    scale_in_place(vector, Complex64::new(1.0 / norm, 0.0));
    Ok(norm)
}

/// 2-норма
pub fn norm2(vector: &Array1<Complex64>) -> f64 {
    norm2_slice(vector.as_slice().expect("norm2 expects contiguous vector"))
}

#[inline]
pub(crate) fn norm2_slice(vector: &[Complex64]) -> f64 {
    vector.iter().map(|entry| entry.norm_sqr()).sum::<f64>().sqrt()
}

/// Скалярное произведение в комплексном пространстве
pub fn inner_product(left: &Array1<Complex64>, right: &Array1<Complex64>) -> Complex64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left_entry, &right_entry)| left_entry.conj() * right_entry)
        .sum::<Complex64>()
}

/// Проверка невязки
pub fn is_numerical_breakdown(residual_norm: f64, reference_norm: f64, tolerance: f64) -> bool {
    residual_norm <= tolerance * reference_norm
}

/// x *= a
pub fn scale_in_place(vector: &mut Array1<Complex64>, alpha: Complex64) {
    scale_slice_in_place(
        vector
            .as_slice_mut()
            .expect("scale_in_place expects contiguous vector"),
        alpha,
    );
}

#[inline]
pub(crate) fn scale_slice_in_place(vector: &mut [Complex64], alpha: Complex64) {
    vector.iter_mut().for_each(|entry| *entry *= alpha);
}

/// y += a * x
pub fn axpy_in_place(target: &mut Array1<Complex64>, alpha: Complex64, source: &Array1<Complex64>) {
    Zip::from(target)
        .and(source)
        .for_each(|target_entry, &source_entry| {
            *target_entry += alpha * source_entry;
        });
}

/// Reorthogonalized classical Gram-Schmidt using the LAPACK/OpenBLAS backend.
pub fn orthogonalize_with_reorthogonalization(
    candidate: &mut Array1<Complex64>,
    basis: &ArrayView2<Complex64>,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    let (m, n) = basis.dim();
    assert_eq!(candidate.len(), m);
    assert!(h_column.len() >= n);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::ZERO;
    let minus_one = Complex64::new(-1.0, 0.0);
    let mut projection = vec![Complex64::ZERO; n];

    {
        let x = candidate
            .as_slice_mut()
            .expect("candidate must be contiguous");

        for _ in 0..2 {
            projection.fill(Complex64::ZERO);

            zgemv(
                ZgemvTranspose::ConjugateTranspose,
                *basis,
                one,
                x,
                zero,
                &mut projection,
            );

            for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
                *h += p;
            }

            zgemv(ZgemvTranspose::None, *basis, minus_one, &projection, one, x);
        }
    }

    let mut residual_norm = norm2(candidate);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scale_in_place(candidate, Complex64::new(1.0 / residual_norm, 0.0));

    projection.fill(Complex64::ZERO);
    {
        let x = candidate
            .as_slice_mut()
            .expect("candidate must be contiguous");
        zgemv(
            ZgemvTranspose::ConjugateTranspose,
            *basis,
            one,
            x,
            zero,
            &mut projection,
        );
    }

    let correction_norm = norm2_slice(&projection);

    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        {
            let x = candidate
                .as_slice_mut()
                .expect("candidate must be contiguous");
            zgemv(ZgemvTranspose::None, *basis, minus_one, &projection, one, x);
        }

        let reorthogonalized_norm = norm2(candidate);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }

        scale_in_place(candidate, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
    }

    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// Reorthogonalized classical Gram-Schmidt against a basis that is already
/// resident on the GPU.
#[cfg(feature = "magma")]
pub fn orthogonalize_with_device_basis_reorthogonalization(
    session: &MagmaSession,
    d_basis: &DeviceMatrix,
    basis_columns: usize,
    candidate: &mut Array1<Complex64>,
    d_candidate: &mut DeviceVector,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    let m = d_basis.rows();
    let n = basis_columns;
    assert!(n <= d_basis.columns());
    assert_eq!(candidate.len(), m);
    assert_eq!(d_candidate.len(), m);
    assert!(h_column.len() >= n);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::ZERO;
    let minus_one = Complex64::new(-1.0, 0.0);

    let x = candidate
        .as_slice_mut()
        .expect("candidate must be contiguous");
    let mut d_projection = DeviceVector::new(n);
    let mut projection = vec![Complex64::ZERO; n];

    for _ in 0..2 {
        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::ConjugateTranspose,
            one,
            d_candidate,
            zero,
            &mut d_projection,
        );
        d_projection.copy_to_slice(session, &mut projection);

        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p;
        }

        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::None,
            minus_one,
            &d_projection,
            one,
            d_candidate,
        );
    }

    d_candidate.copy_to_slice(session, x);

    let mut residual_norm = norm2_slice(x);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scale_slice_in_place(x, Complex64::new(1.0 / residual_norm, 0.0));
    d_candidate.copy_from_slice(session, x);

    d_basis.zgemv_leading_columns(
        session,
        n,
        MagmaZgemvTranspose::ConjugateTranspose,
        one,
        d_candidate,
        zero,
        &mut d_projection,
    );
    d_projection.copy_to_slice(session, &mut projection);

    let correction_norm = norm2_slice(&projection);

    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        d_basis.zgemv_leading_columns(
            session,
            n,
            MagmaZgemvTranspose::None,
            minus_one,
            &d_projection,
            one,
            d_candidate,
        );
        d_candidate.copy_to_slice(session, x);

        let reorthogonalized_norm = norm2_slice(x);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }

        scale_slice_in_place(x, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
        d_candidate.copy_from_slice(session, x);
    }

    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// One-shot MAGMA reorthogonalization for restart residuals.
#[cfg(feature = "magma")]
pub fn orthogonalize_with_magma_reorthogonalization(
    session: &MagmaSession,
    candidate: &mut Array1<Complex64>,
    basis: &ArrayView2<Complex64>,
    h_column: &mut [Complex64],
    reference_norm: f64,
    breakdown_tol: f64,
) -> OrthogonalizedVector {
    let (m, n) = basis.dim();
    assert_eq!(candidate.len(), m);
    assert!(h_column.len() >= n);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::ZERO;
    let minus_one = Complex64::new(-1.0, 0.0);

    let d_basis = DeviceMatrix::from_column_major(session, *basis);
    let x = candidate
        .as_slice_mut()
        .expect("candidate must be contiguous");
    let mut d_x = DeviceVector::from_slice(session, x);
    let mut d_projection = DeviceVector::new(n);
    let mut projection = vec![Complex64::ZERO; n];

    for _ in 0..2 {
        d_basis.zgemv(
            session,
            MagmaZgemvTranspose::ConjugateTranspose,
            one,
            &d_x,
            zero,
            &mut d_projection,
        );
        d_projection.copy_to_slice(session, &mut projection);

        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p;
        }

        d_basis.zgemv(
            session,
            MagmaZgemvTranspose::None,
            minus_one,
            &d_projection,
            one,
            &mut d_x,
        );
    }

    d_x.copy_to_slice(session, x);

    let mut residual_norm = norm2_slice(x);
    if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
        return OrthogonalizedVector {
            residual_norm,
            happy_breakdown: true,
        };
    }

    scale_slice_in_place(x, Complex64::new(1.0 / residual_norm, 0.0));
    d_x.copy_from_slice(session, x);

    d_basis.zgemv(
        session,
        MagmaZgemvTranspose::ConjugateTranspose,
        one,
        &d_x,
        zero,
        &mut d_projection,
    );
    d_projection.copy_to_slice(session, &mut projection);

    let correction_norm = norm2_slice(&projection);

    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        for (h, &p) in h_column.iter_mut().zip(projection.iter()) {
            *h += p * Complex64::new(residual_norm, 0.0);
        }

        d_basis.zgemv(
            session,
            MagmaZgemvTranspose::None,
            minus_one,
            &d_projection,
            one,
            &mut d_x,
        );
        d_x.copy_to_slice(session, x);

        let reorthogonalized_norm = norm2_slice(x);
        residual_norm *= reorthogonalized_norm;
        if is_numerical_breakdown(residual_norm, reference_norm, breakdown_tol) {
            return OrthogonalizedVector {
                residual_norm,
                happy_breakdown: true,
            };
        }

        scale_slice_in_place(x, Complex64::new(1.0 / reorthogonalized_norm, 0.0));
        d_x.copy_from_slice(session, x);
    }

    OrthogonalizedVector {
        residual_norm,
        happy_breakdown: false,
    }
}

/// Генерация нормализованного случайного вектора
pub fn normalized_random_vector<R>(
    dimension: usize,
    rng: &mut R,
) -> Result<Array1<Complex64>, IramError>
where
    R: Rng + ?Sized,
{
    let mut vector = Array1::from_iter(
        (0..dimension)
            .map(|_| Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))),
    );
    normalize(&mut vector, "random start vector generation")?;
    Ok(vector)
}

/// Нормализация комплекснозначного вектора
pub fn normalize_complex(vector: &mut Array1<Complex64>) -> Result<f64, IramError> {
    let norm = norm2(vector);

    if norm <= f64::EPSILON {
        return Err(IramError::Spectral(
            "complex eigenvector estimate collapsed to zero".to_string(),
        ));
    }

    scale_in_place(vector, Complex64::new(1.0 / norm, 0.0));
    Ok(norm)
}
