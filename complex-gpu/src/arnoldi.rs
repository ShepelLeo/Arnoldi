//! Процесс Арнольди
use ndarray::{Array1, Array2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::magma::{DeviceMatrix, DeviceVector, MagmaSession};
use crate::linalg::ops::{
    norm2, normalize, orthogonalize_with_device_basis_reorthogonalization,
};
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

/// Вход в процесс Арнольди, второй и последующие прогоны, пополняем пространство Крыллова.
///
/// MAGMA-optimized variant:
/// - creates one MAGMA queue/session for the whole continuation loop;
/// - uploads the current basis matrix once at the start;
/// - appends each newly normalized Arnoldi vector to the device basis;
/// - reuses one device candidate buffer across all Arnoldi steps.
///
/// The linear operator is still CPU-side because `LinearOperator::apply_into`
/// accepts/returns ndarray views. The optimization target here is the dense
/// Gram-Schmidt part, which previously re-uploaded the whole basis matrix in
/// every `zgemv` call.
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

    if basis.nrows() != operator.dimension() {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi basis row count must equal operator dimension {}, got {}",
            operator.dimension(),
            basis.nrows(),
        )));
    }

    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    let session = MagmaSession::new();
    let mut d_basis = DeviceMatrix::from_column_major(&session, basis.view());
    let mut d_candidate = DeviceVector::new(operator.dimension());

    let mut h_column = vec![Complex64::default(); target_steps];
    let mut candidate = Array1::zeros(operator.dimension());

    for step in start_step..target_steps {
        // Operator application remains host-side with the current LinearOperator API.
        operator.apply_into(basis.column(step), candidate.view_mut())?;
        let candidate_old = norm2(&candidate);
        *matvec_count += 1;

        let candidate_slice = candidate
            .as_slice()
            .expect("Arnoldi candidate must be contiguous");
        d_candidate.copy_from_slice(&session, candidate_slice);

        h_column[..=step].fill(Complex64::ZERO);
        let orthogonalized = orthogonalize_with_device_basis_reorthogonalization(
            &session,
            &d_basis,
            step + 1,
            &mut candidate,
            &mut d_candidate,
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

        // Keep both host and device basis in sync. The host copy is still needed
        // by LinearOperator::apply_into and by downstream IRAM restart logic.
        basis.column_mut(step + 1).assign(&candidate);
        let candidate_slice = candidate
            .as_slice()
            .expect("Arnoldi candidate must be contiguous");
        d_basis.copy_column_from_slice(&session, step + 1, candidate_slice);
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
        let factorization = run_arnoldi(&operator, &start, 3, 1.0e-14, &mut matvec_count)
            .expect("Arnoldi factorization should succeed");

        assert_eq!(factorization.performed_steps, 1);
        assert!(factorization.happy_breakdown);
        assert_eq!(matvec_count, 1);
    }

    #[test]
    fn arnoldi_factorization_preserves_relation_and_orthogonality() {
        let operator = ConvectionDiffusionOperator::new(4, 0.0);
        let start = Array1::from_iter(
            (0..operator.dimension())
                .map(|index| Complex64::new(index as f64 + 1.0, -(index as f64 + 0.5))),
        );
        let target_steps = 5;
        let mut matvec_count = 0;
        let factorization =
            run_arnoldi(&operator, &start, target_steps, 1.0e-12, &mut matvec_count)
                .expect("Arnoldi factorization should succeed");

        assert_eq!(factorization.performed_steps, target_steps);
        assert!(!factorization.happy_breakdown);

        let steps = factorization.performed_steps;
        let basis = factorization.basis.slice(s![.., 0..steps]).to_owned();
        let extended_basis = factorization.basis.slice(s![.., 0..=steps]).to_owned();
        let hessenberg = factorization
            .hessenberg
            .slice(s![0..=steps, 0..steps])
            .to_owned();

        let mut applied = Array2::zeros((operator.dimension(), steps).f());
        for column in 0..steps {
            operator
                .apply_into(basis.column(column), applied.column_mut(column))
                .unwrap();
        }

        let relation_error = frobenius_norm(&(applied - extended_basis.dot(&hessenberg)));
        assert!(relation_error < 1.0e-8, "relation_error={relation_error}");

        let gram = extended_basis
            .t()
            .mapv(|entry| entry.conj())
            .dot(&extended_basis);
        let mut identity = Array2::zeros((steps + 1, steps + 1).f());
        for diagonal in 0..=steps {
            identity[[diagonal, diagonal]] = Complex64::new(1.0, 0.0);
        }
        let orthogonality_error = frobenius_norm(&(gram - identity));
        assert!(
            orthogonality_error < 1.0e-8,
            "orthogonality_error={orthogonality_error}"
        );
    }

    fn frobenius_norm(matrix: &Array2<Complex64>) -> f64 {
        matrix
            .iter()
            .map(|entry| entry.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}
