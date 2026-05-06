//! Операции

use ndarray::{Array1, Array2, ShapeBuilder};
use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::error::IramError;
use crate::linalg::lapack::zgeqp3_qr_rank;

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

/// x *= a
pub fn scale_in_place(vector: &mut Array1<Complex64>, alpha: Complex64) {
    vector.iter_mut().for_each(|entry| *entry *= alpha);
}

/// Генерация случайной матрицы с ортонормальными столбцами.
pub fn normalized_random_unitary_matrix<R>(
    dimension: usize,
    block_size: usize,
    rng: &mut R,
) -> Result<Array2<Complex64>, IramError>
where
    R: Rng + ?Sized,
{
    if block_size == 0 {
        return Err(IramError::InvalidConfig(
            "block_size must be strictly positive".to_string(),
        ));
    }

    if block_size > dimension {
        return Err(IramError::InvalidConfig(format!(
            "block_size ({block_size}) cannot exceed the operator dimension ({dimension})",
        )));
    }

    let matrix = Array2::from_shape_fn((dimension, block_size).f(), |_| {
        Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
    });
    let qr = zgeqp3_qr_rank(&matrix, f64::EPSILON).map_err(IramError::Spectral)?;

    if qr.rank != block_size {
        return Err(IramError::Spectral(format!(
            "random start block has numerical rank {}, expected {}",
            qr.rank, block_size,
        )));
    }

    Ok(qr.q)
}
