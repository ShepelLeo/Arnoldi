//! Ход IRAM
//! Процесс Арнольди, рестарты
use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::arnoldi::{ArnoldiFactorization, continue_arnoldi, run_arnoldi};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::linalg::lapack::*;
use crate::linalg::ops::{is_numerical_breakdown, norm2, normalize, orthogonalize2_twice};
use crate::linalg::small::{compute_ritz_values, retrive_ritz_vectors};
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, RitzEstimate, SolveReport};
use crate::selection::*;

/// Вход в алгоритм
pub fn solve(
    operator: &dyn LinearOperator,
    start_vector: Array1<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    config.validate(operator.dimension())?;

    // Инициализируем стартовый вектор
    let mut current_start = start_vector;
    normalize(&mut current_start, "solver start vector")?;

    let mut total_matvecs = 0usize;
    // Запуск процесса Арнольди
    let mut factorization = run_arnoldi(
        operator,
        &current_start,
        config.ncv,
        config.breakdown_tol,
        &mut total_matvecs,
    )?;
    let mut history = Vec::new();
    let mut final_values = Vec::new();
    let mut note = None;
    let mut fully_converged = false;
    let mut happy_breakdown = false;
    let mut converged = 0usize;

    // Рестарты
    for restart in 0..=config.max_restarts {
        let krylov_dim = factorization.performed_steps;
        let square_hessenberg = factorization.square_hessenberg();
        // Крайний поддиагональный элемент
        let trailing_subdiagonal = factorization.trailing_subdiagonal();

        // Малая спектральная задача
        let mut hessenberg_schur = compute_ritz_values(&square_hessenberg);

        // Выбираем желаемые СЗН матрицы Хессенберга
        let selection =
            select_ritz_values(&hessenberg_schur.w, config.target, config.nev, krylov_dim)?;

        let ritz_vectors =
            retrive_ritz_vectors(&mut hessenberg_schur, &selection.wanted, krylov_dim);

        let result: Vec<RitzEstimate> = selection
            .wanted
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let value = hessenberg_schur.w[idx];

                let residual_estimate =
                    (trailing_subdiagonal * ritz_vectors[[krylov_dim - 1, i]]).norm();

                RitzEstimate {
                    value,
                    residual_estimate,
                }
            })
            .collect();

        // Сколько сошлось
        converged = result
            .iter()
            .filter(|estimate| estimate.residual_estimate <= config.tol)
            .count();
        happy_breakdown = factorization.happy_breakdown;

        final_values = result;

        history.push(IterationLog {
            restart,
            krylov_dimension: krylov_dim,
            converged,
            total_matvecs,
            peak_memory_bytes: memory::peak_bytes_since_reset(),
            happy_breakdown,
            wanted: final_values.clone(),
            shifts: selection.shifts.clone(),
        });

        if converged >= config.nev {
            fully_converged = true;
            break; // Ура, мы сошлись
        }

        if factorization.happy_breakdown {
            note = Some(if krylov_dim < config.nev {
                "happy breakdown occurred before the Krylov space became large enough; with a single starting vector the chosen operator cannot expose that many independent eigen-directions"
                    .to_string()
            } else {
                "happy breakdown detected; the current Krylov subspace is already invariant"
                    .to_string()
            });
            break; // Беда
        }

        if restart == config.max_restarts {
            note = Some("maximum number of restarts reached before full convergence".to_string());
            break; // Недобили
        }

        if selection.shifts.is_empty() {
            note = Some("no unwanted Ritz values remain for the restart filter".to_string());
            break; // Беда
        }

        // Запускаем рестарты
        factorization = implicit_restart_and_extend(
            operator,
            &factorization,
            &selection.shifts,
            config.ncv + converged,
            config.breakdown_tol,
            &mut total_matvecs,
            selection.wanted.len(),
        )?;
    }

    Ok(SolveReport {
        operator_description: operator.description(),
        start_description: start_description.into(),
        dimension: operator.dimension(),
        config,
        elapsed_seconds: 0.0,
        total_restarts: history.len(),
        total_matvecs,
        peak_memory_bytes: memory::peak_bytes_since_reset(),
        converged,
        fully_converged,
        happy_breakdown,
        final_values,
        history,
        note,
    })
}

/// Вход в рестарты
fn implicit_restart_and_extend(
    operator: &dyn LinearOperator,
    factorization: &ArnoldiFactorization,
    shifts: &[Complex64],
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
    k: usize,
) -> Result<ArnoldiFactorization, IramError> {
    let m = factorization.performed_steps;

    if k == 0 || k >= m {
        return Err(IramError::InvalidConfig(format!(
            "implicit restart requires 0 < retained_dimension < krylov_dimension, got retained_dimension={k}, krylov_dimension={m}",
        )));
    }

    if factorization.basis.ncols() < m + 1 {
        return Err(IramError::InvalidConfig(
            "implicit restart requires the trailing Arnoldi residual vector".to_string(),
        ));
    }

    // A V_m = V_m H_m + beta v_{m+1} e_m^T
    let beta = factorization.trailing_subdiagonal();
    let (rotation, h) = zlaqr52(&mut factorization.square_hessenberg(), shifts).unwrap();

    let rotated_basis = rotate_basis(&factorization.basis.slice(s![.., ..m]), &rotation, k);
    let mut restarted_hessenberg =
        Array2::<Complex64>::from_elem((target_steps + 1, target_steps), Complex64::ZERO);

    (0..k).for_each(|column| {
        (0..k).for_each(|row| {
            restarted_hessenberg[[row, column]] = h[[row, column]];
        });
    });

    // r_new = h_{k+1,k} * (V_m q_{k+1}) + beta * q_{m,k} * v_{m+1}
    let h_coupling = h[(k, k - 1)];
    let residual_coupling = Complex64::new(beta, 0.0) * rotation[[m - 1, k - 1]];

    let mut residual = factorization
        .basis
        .slice(s![.., ..m])
        .dot(&rotation.column(k));

    residual *= h_coupling;
    residual.scaled_add(residual_coupling, &factorization.basis.column(m));

    let mut h_column_correction = vec![Complex64::default(); k];
    orthogonalize2_twice(
        &mut residual,
        &rotated_basis.view(),
        &mut h_column_correction,
    );
    h_column_correction
        .iter()
        .enumerate()
        .for_each(|(row, &value)| restarted_hessenberg[[row, k - 1]] += value);

    let residual_norm = norm2(&residual);
    let residual_reference_norm = (h_coupling.norm_sqr() + residual_coupling.norm_sqr()).sqrt();
    if is_numerical_breakdown(residual_norm, residual_reference_norm, breakdown_tol) {
        restarted_hessenberg[[k, k - 1]] = Complex64::ZERO;
        return Ok(ArnoldiFactorization {
            basis: rotated_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: k,
            happy_breakdown: true,
        });
    }

    restarted_hessenberg[[k, k - 1]] = Complex64::new(residual_norm, 0.0);
    residual
        .iter_mut()
        .for_each(|entry| *entry /= residual_norm);

    let mut continued_basis =
        Array2::<Complex64>::zeros((factorization.basis.nrows(), target_steps + 1).f());
    continued_basis
        .slice_mut(s![.., 0..k])
        .assign(&rotated_basis);
    continued_basis.column_mut(k).assign(&residual);

    continue_arnoldi(
        operator,
        continued_basis,
        restarted_hessenberg,
        k,
        target_steps,
        breakdown_tol,
        matvec_count,
    )
}

fn rotate_basis(
    basis: &ArrayView2<Complex64>,
    q_total: &Array2<Complex64>,
    column: usize,
) -> Array2<Complex64> {
    let product = basis.dot(&q_total.slice(s![.., 0..column]));
    let mut result = Array2::zeros((product.nrows(), product.ncols()).f());
    result.assign(&product);
    result
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use num_complex::Complex64;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::config::{SolverConfig, SpectrumTarget, recommended_ncv};
    use crate::linalg::ops::normalized_random_vector;
    use crate::operator::{ConvectionDiffusionOperator, IdentityOperator, LinearOperator};

    use super::solve;

    #[test]
    fn identity_operator_converges_for_one_eigenvalue() {
        let operator = IdentityOperator::new(8);
        let start = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
        ]);
        let config = SolverConfig {
            nev: 1,
            ncv: recommended_ncv(1, 8),
            max_restarts: 5,
            tol: 1.0e-10,
            breakdown_tol: 1.0e-12,
            target: SpectrumTarget::LargestMagnitude,
        };
        let report = solve(&operator, start, config, "unit vector")
            .expect("the identity problem should be solvable");

        assert_eq!(report.converged, 1);
        assert!(report.fully_converged);
        assert!((report.final_values[0].value - Complex64::new(1.0, 0.0)).norm() < 1.0e-10);
    }

    #[test]
    fn implicit_restart_preserves_convection_diffusion_wanted_space() {
        let operator = ConvectionDiffusionOperator::new(10, 0.0);
        let mut rng = StdRng::seed_from_u64(0);
        let start = normalized_random_vector(operator.dimension(), &mut rng)
            .expect("the deterministic random start vector should be nonzero");
        let config = SolverConfig {
            nev: 4,
            ncv: 20,
            max_restarts: 20,
            tol: 1.0e-10,
            breakdown_tol: 1.0e-12,
            target: SpectrumTarget::SmallestMagnitude,
        };

        let report = solve(&operator, start, config, "seeded random vector")
            .expect("the convection-diffusion problem should be solvable");

        assert!(
            report.fully_converged,
            "expected full convergence, got {} converged values after {} restarts; note={:?}",
            report.converged, report.total_restarts, report.note,
        );
        assert_eq!(report.converged, 4);
        assert!(!report.happy_breakdown);
    }
}
