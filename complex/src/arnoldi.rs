//! Процесс Арнольди
use ndarray::{Array1, Array2, s};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{is_numerical_breakdown, norm2, normalize, orthogonalize_twice};
use crate::operator::LinearOperator;

/// Ответ процесса
#[derive(Debug, Clone)]
pub struct ArnoldiFactorization {
    pub basis: Vec<Array1<Complex64>>,
    pub hessenberg: Array2<Complex64>,
    pub performed_steps: usize,
    pub happy_breakdown: bool,
}

impl ArnoldiFactorization {
    pub fn square_hessenberg(&self) -> Array2<Complex64> {
        self.hessenberg
            .slice(s![0..self.performed_steps, 0..self.performed_steps])
            .to_owned()
    }

    pub fn trailing_subdiagonal(&self) -> f64 {
        if self.happy_breakdown || self.performed_steps == 0 {
            0.0
        } else {
            self.hessenberg[[self.performed_steps, self.performed_steps - 1]].norm()
        }
    }
}

/// Вход в процесс Арнольди, первый прогон при инициализации пространства Крылова
pub fn run_arnoldi(
    operator: &dyn LinearOperator,
    start_vector: &Array1<Complex64>,
    steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let mut normalized_start = start_vector.clone();
    normalize(&mut normalized_start, "Arnoldi start vector")?;

    let basis = vec![normalized_start];
    let hessenberg = Array2::<Complex64>::from_elem((steps + 1, steps), Complex64::new(0.0, 0.0));

    continue_arnoldi(
        operator,
        basis,
        hessenberg,
        0,
        steps,
        breakdown_tol,
        matvec_count,
    )
}

/// Вход в процесс Арнольди, второй и последующие прогоны, пополняем пространство Крыллова
pub fn continue_arnoldi(
    operator: &dyn LinearOperator,
    mut basis: Vec<Array1<Complex64>>,
    mut hessenberg: Array2<Complex64>,
    start_step: usize,
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    if hessenberg.nrows() != target_steps + 1 || hessenberg.ncols() != target_steps {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation expected Hessenberg shape {}x{}, got {}x{}",
            target_steps + 1,
            target_steps,
            hessenberg.nrows(),
            hessenberg.ncols(),
        )));
    }

    if basis.len() < start_step + 1 {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation needs at least {} basis vectors, got {}",
            start_step + 1,
            basis.len(),
        )));
    }

    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    for step in start_step..target_steps {
        let current_vector = basis[step].clone();
        let mut candidate = operator.apply(&current_vector)?;
        let candidate_old = norm2(&candidate);
        *matvec_count += 1;

        let mut h_column = vec![Complex64::new(0.0, 0.0); step + 1];
        orthogonalize_twice(&mut candidate, &basis[..=step], &mut h_column);

        h_column
            .iter()
            .enumerate()
            .for_each(|(row, &value)| hessenberg[[row, step]] = value);

        let candidate_norm = norm2(&candidate);
        performed_steps = step + 1;

        if is_numerical_breakdown(candidate_norm, candidate_old, breakdown_tol) {
            happy_breakdown = true;
            hessenberg[[step + 1, step]] = Complex64::new(0.0, 0.0);
            break;
        }

        hessenberg[[step + 1, step]] = Complex64::new(candidate_norm, 0.0);
        candidate
            .iter_mut()
            .for_each(|entry| *entry /= candidate_norm);
        basis.push(candidate);
    }

    Ok(ArnoldiFactorization {
        basis,
        hessenberg,
        performed_steps,
        happy_breakdown,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use num_complex::Complex64;

    use crate::operator::IdentityOperator;

    use super::run_arnoldi;

    #[test]
    fn identity_operator_breaks_down_after_one_step() {
        let operator = IdentityOperator::new(4);
        let start = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let mut matvec_count = 0;
        let factorization = run_arnoldi(&operator, &start, 3, 1.0e-14, &mut matvec_count)
            .expect("Arnoldi factorization should succeed");

        assert_eq!(factorization.performed_steps, 1);
        assert!(factorization.happy_breakdown);
        assert_eq!(matvec_count, 1);
    }
}
