//! Операции

use ndarray::{Array1, Zip};
use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::error::IramError;

/// Нормализация вектора
pub fn normalize(vector: &mut Array1<f64>, context: &'static str) -> Result<f64, IramError> {
    let norm = norm2(vector);

    if norm <= f64::EPSILON {
        return Err(IramError::ZeroVector(context));
    }

    scale_in_place(vector, 1.0 / norm);
    Ok(norm)
}

/// 2-Норма
pub fn norm2(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}

/// Проверка невязки
pub fn is_numerical_breakdown(residual_norm: f64, reference_norm: f64, tolerance: f64) -> bool {
    residual_norm <= tolerance * reference_norm
}

/// x *= a
pub fn scale_in_place(vector: &mut Array1<f64>, alpha: f64) {
    vector.iter_mut().for_each(|entry| *entry *= alpha);
}

/// a * x + y
pub fn axpy_in_place(target: &mut Array1<f64>, alpha: f64, source: &Array1<f64>) {
    Zip::from(target)
        .and(source)
        .for_each(|target_entry, &source_entry| {
            *target_entry += alpha * source_entry;
        });
}

/// a * x + b * y + c * z
pub fn linear_combination3(
    first: &Array1<f64>,
    first_alpha: f64,
    second: &Array1<f64>,
    second_alpha: f64,
    third: &Array1<f64>,
    third_alpha: f64,
) -> Array1<f64> {
    Array1::from_iter(first.iter().zip(second.iter()).zip(third.iter()).map(
        |((&first_entry, &second_entry), &third_entry)| {
            first_alpha * first_entry + second_alpha * second_entry + third_alpha * third_entry
        },
    ))
}

/// Ортогонализация вектора по базису
pub fn orthogonalize_twice(
    candidate: &mut Array1<f64>,
    basis: &[Array1<f64>],
    h_column: &mut [f64],
) {
    (0..2).for_each(|_| {
        basis.iter().enumerate().for_each(|(index, basis_vector)| {
            let projection = basis_vector.dot(candidate);
            h_column[index] += projection;
            axpy_in_place(candidate, -projection, basis_vector);
        });
    });
}

/// Генерация нормализованного случайного вектора
pub fn normalized_random_vector<R>(dimension: usize, rng: &mut R) -> Result<Array1<f64>, IramError>
where
    R: Rng + ?Sized,
{
    let mut vector = Array1::from_iter((0..dimension).map(|_| rng.random_range(-1.0..=1.0)));
    normalize(&mut vector, "random start vector generation")?;
    Ok(vector)
}


/// Нормализация коплекснозначного вектора
pub fn normalize_complex(vector: &mut Array1<Complex64>) -> Result<f64, IramError> {
    let norm = vector
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt();

    if norm <= f64::EPSILON {
        return Err(IramError::Spectral(
            "complex eigenvector estimate collapsed to zero".to_string(),
        ));
    }

    vector.iter_mut().for_each(|entry| *entry /= norm);
    Ok(norm)
}
