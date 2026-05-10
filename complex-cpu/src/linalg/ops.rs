//! Операции

use ndarray::{Array1, ArrayView2, Zip};
use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::error::IramError;
use crate::linalg::lapack::{ZgemvTranspose, zgemv};

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
    vector
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt()
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

/// a * x + b * y + c * z
pub fn linear_combination3(
    first: &Array1<Complex64>,
    first_alpha: Complex64,
    second: &Array1<Complex64>,
    second_alpha: Complex64,
    third: &Array1<Complex64>,
    third_alpha: Complex64,
) -> Array1<Complex64> {
    first_alpha * first + second_alpha * second + third_alpha * third
    // Array1::from_iter(first.iter().zip(second.iter()).zip(third.iter()).map(
    //     |((&first_entry, &second_entry), &third_entry)| {
    //         first_alpha * first_entry + second_alpha * second_entry + third_alpha * third_entry
    //     },
    // ))
}

/// Ортогонализация вектора по базису
// pub fn orthogonalize_twice(
//     candidate: &mut Array1<Complex64>,
//     basis: &[Array1<Complex64>],
//     h_column: &mut [Complex64],
// ) {
//     (0..2).for_each(|_| {
//         basis.iter().enumerate().for_each(|(index, basis_vector)| {
//             let projection = inner_product(basis_vector, candidate);
//             h_column[index] += projection;
//             axpy_in_place(candidate, -projection, basis_vector);
//         });
//     });
// }

// pub fn orthogonalize2_twice(
//     candidate: &mut Array1<Complex64>,
//     basis: &ArrayView2<Complex64>,
//     h_column: &mut [Complex64],
// ) {
//     for _ in 0..2 {
//         let projection = basis
//             .t()
//             .mapv(|z| z.conj())
//             .dot(candidate);

//         for (h, p) in h_column.iter_mut().zip(projection.iter()) {
//                 *h += *p;
//             }

//         *candidate -= &basis.dot(&projection);
//     }
// }

// pub fn orthogonalize2_twice(
//     candidate: &mut Array1<Complex64>,
//     basis: &ArrayView2<Complex64>,
//     h_column: &mut [Complex64],
// ) {
//     for _ in 0..2 {
//         for (i, basis_col) in basis.axis_iter(Axis(1)).enumerate() {
//             let proj = Zip::from(&basis_col)
//                 .and(&*candidate)
//                 .fold(Complex64::ZERO, |acc, &b, &c| acc + b.conj() * c);

//             h_column[i] += proj;
//             Zip::from(&mut *candidate)
//                 .and(&basis_col)
//                 .for_each(|c, &b| *c -= proj * b);
//         }
//     }
// }

pub fn orthogonalize2_twice(
    candidate: &mut Array1<Complex64>,
    basis: &ArrayView2<Complex64>,
    h_column: &mut [Complex64],
) {
    let (m, n) = basis.dim();
    assert_eq!(candidate.len(), m);
    assert!(h_column.len() >= n);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let minus_one = Complex64::new(-1.0, 0.0);

    let x = candidate
        .as_slice_mut()
        .expect("candidate must be contiguous");

    let mut projection = vec![Complex64::ZERO; n];

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

    let correction_norm = projection
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt();

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
