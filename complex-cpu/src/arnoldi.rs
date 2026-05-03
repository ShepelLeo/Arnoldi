//! Процесс Арнольди
use ndarray::{Array1, Array2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{is_numerical_breakdown, norm2, normalize, orthogonalize2_twice};
use crate::operator::LinearOperator;

/// Ответ процесса
#[derive(Debug, Clone)]
pub struct ArnoldiFactorization {
    pub basis: Array2<Complex64>,
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

    let mut basis = Array2::zeros((operator.dimension(), steps + 1).f());
    basis.column_mut(0).assign(&normalized_start);
    let hessenberg = Array2::zeros((steps + 1, steps));

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
    mut basis: Array2<Complex64>,
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

    if basis.ncols() < target_steps + 1 {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation needs at least {} basis vectors, got {}",
            target_steps + 1,
            basis.ncols(),
        )));
    }

    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    let mut h_column = vec![Complex64::default(); target_steps];
    let mut candidate = Array1::zeros(operator.dimension());

    for step in start_step..target_steps {
        operator.apply_into(basis.column(step), candidate.view_mut())?;
        let candidate_old = norm2(&candidate);
        *matvec_count += 1;

        h_column[..=step].fill(Complex64::ZERO);
        orthogonalize2_twice(
            &mut candidate,
            &basis.slice(s![.., 0..=step]),
            &mut h_column[..=step],
        );

        for row in 0..=step {
            hessenberg[[row, step]] = h_column[row];
        }
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
        basis.column_mut(step + 1).assign(&candidate);
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
