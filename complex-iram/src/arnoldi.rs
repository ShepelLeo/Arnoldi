//! Процесс Арнольди
use ndarray::{Array1, Array2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::backend::Backend;
use crate::error::IramError;
use crate::linalg::ops::normalize;

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

/// Вход в процесс Арнольди, первый прогон при инициализации пространства Крылова.
pub fn run_arnoldi<B: Backend>(
    backend: &mut B,
    operator: &B::PreparedOperator<'_>,
    start_vector: &Array1<Complex64>,
    steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let mut normalized_start = start_vector.clone();
    normalize(&mut normalized_start, "Arnoldi start vector")?;

    let dimension = backend.prepared_operator_dimension(operator);
    let mut basis = Array2::zeros((dimension, steps + 1).f());
    basis.column_mut(0).assign(&normalized_start);
    let hessenberg = Array2::zeros((steps + 1, steps));

    continue_arnoldi(
        backend,
        operator,
        basis,
        hessenberg,
        0,
        steps,
        breakdown_tol,
        matvec_count,
    )
}

/// Вход в процесс Арнольди, второй и последующие прогоны, пополняем пространство Крылова.
///
/// Алгоритм не знает, где выполняется ортогонализация. Он только сообщает
/// бэкенду: создать рабочее состояние, ортогонализовать текущий кандидат и
/// синхронизировать добавленный базисный вектор.
pub fn continue_arnoldi<B: Backend>(
    backend: &mut B,
    operator: &B::PreparedOperator<'_>,
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

    let dimension = backend.prepared_operator_dimension(operator);
    if basis.nrows() != dimension {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi basis row count must equal operator dimension {}, got {}",
            dimension,
            basis.nrows(),
        )));
    }

    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    let mut workspace = backend.create_arnoldi_workspace(&basis, dimension)?;
    let mut h_column = vec![Complex64::default(); target_steps];
    let mut candidate = Array1::zeros(dimension);

    for step in start_step..target_steps {
        let candidate_old = backend.apply_operator_to_arnoldi_column(
            operator,
            &mut workspace,
            &basis,
            step,
            &mut candidate,
        )?;
        *matvec_count += 1;

        h_column[..=step].fill(Complex64::ZERO);
        let orthogonalized = backend.orthogonalize_arnoldi_candidate(
            &mut workspace,
            &basis,
            step + 1,
            &mut candidate,
            &mut h_column[..=step],
            candidate_old,
            breakdown_tol,
        );

        for row in 0..=step {
            hessenberg[[row, step]] = h_column[row];
        }
        performed_steps = step + 1;

        if orthogonalized.happy_breakdown {
            happy_breakdown = true;
            hessenberg[[step + 1, step]] = Complex64::new(0.0, 0.0);
            break;
        }

        hessenberg[[step + 1, step]] = Complex64::new(orthogonalized.residual_norm, 0.0);
        basis.column_mut(step + 1).assign(&candidate);
        backend.append_arnoldi_basis_column(&mut workspace, step + 1, &candidate);
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
    use ndarray::{Array1, Array2, ShapeBuilder, s};
    use num_complex::Complex64;

    use crate::backend::{Backend, LapackBackend};
    use crate::operator::{ConvectionDiffusionOperator, IdentityOperator, LinearOperator};

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
        let mut backend = LapackBackend::new();

        let prepared = backend.prepare_operator(&operator).unwrap();
        let result = run_arnoldi(&mut backend, &prepared, &start, 3, 1.0e-12, &mut matvec_count)
            .expect("identity Arnoldi should not fail");

        assert_eq!(result.performed_steps, 1);
        assert!(result.happy_breakdown);
        assert_eq!(matvec_count, 1);
    }

    #[test]
    fn arnoldi_relation_holds_for_convection_diffusion() {
        let operator = ConvectionDiffusionOperator::new(5, 0.0);
        let start = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ]);
        let mut matvec_count = 0;
        let mut backend = LapackBackend::new();

        let prepared = backend.prepare_operator(&operator).unwrap();
        let factorization = run_arnoldi(
            &mut backend,
            &prepared,
            &start,
            3,
            1.0e-12,
            &mut matvec_count,
        )
        .expect("Arnoldi should run");

        let k = factorization.performed_steps;
        let v_k = factorization.basis.slice(s![.., 0..k]);
        let v_k_plus_1 = factorization.basis.slice(s![.., 0..=k]);
        let h = factorization.hessenberg.slice(s![0..=k, 0..k]);

        let mut av = Array2::<Complex64>::zeros((operator.dimension(), k).f());
        for column in 0..k {
            let image = operator
                .apply(&v_k.column(column).to_owned())
                .expect("operator application should succeed");
            av.column_mut(column).assign(&image);
        }

        let vh = v_k_plus_1.dot(&h);
        let error = (&av - &vh)
            .iter()
            .map(|entry| entry.norm_sqr())
            .sum::<f64>()
            .sqrt();

        assert!(error < 1.0e-10, "Arnoldi relation error = {error}");
    }
}
