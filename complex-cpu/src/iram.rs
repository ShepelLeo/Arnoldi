//! Ход IRAM
//! Процесс Арнольди, рестарты
use ndarray::{Array1, Array2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::arnoldi::{ArnoldiFactorization, continue_arnoldi, run_arnoldi};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::linalg::lapack::*;
use crate::linalg::ops::{norm2, normalize};
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

        // Выбираем желаемые СЗН матрицы Хессенберга. Для thick restart
        // оставляем хотя бы один свободный шаг под продолжение Арнольди.
        let selection = select_ritz_values(
            &hessenberg_schur.w,
            config.target,
            config.nev,
            config.ncv.saturating_sub(1).min(krylov_dim),
            config.ritz_inflation,
        )?;

        let ritz_vectors = retrive_ritz_vectors(&mut hessenberg_schur, &selection.retained, krylov_dim);

        let result: Vec<RitzEstimate> = selection.wanted
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
            shifts: [Complex64::ZERO].to_vec(),
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

        let target_steps = config
            .ncv
            .saturating_add(converged)
            .min(operator.dimension());

        if selection.retained.is_empty() || selection.retained.len() >= target_steps {
             note = Some("no room remains to extend the thick-restarted Krylov space".to_string());
             break; // Беда
        }

        let ritz_vectors =
            retrive_ritz_vectors(&mut hessenberg_schur, &selection.retained, krylov_dim);

        // Запускаем рестарты
        factorization = thick_restart_and_extend(
            operator,
            &factorization,
            &square_hessenberg,
            &ritz_vectors,
            target_steps,
            config.breakdown_tol,
            &mut total_matvecs,
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

/// Вход в thick restart через Ritz-векторы малой задачи:
/// Z = U R через QR-отражения, V_+ = V_m U.
fn thick_restart_and_extend(
    operator: &dyn LinearOperator,
    factorization: &ArnoldiFactorization,
    square_hessenberg: &Array2<Complex64>,
    ritz_vectors: &Array2<Complex64>,
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let m = factorization.performed_steps;
    let retained = zgeqrf_qr_rank(ritz_vectors, breakdown_tol).map_err(IramError::Spectral)?;
    let k = retained.rank;

    if k == 0 || k >= m {
        return Err(IramError::InvalidConfig(format!(
            "thick restart requires 0 < retained_dimension < krylov_dimension, got retained_dimension={k}, krylov_dimension={m}",
        )));
    }

    if k >= target_steps {
        return Err(IramError::InvalidConfig(format!(
            "thick restart retained {k} vectors, but target_steps is {target_steps}",
        )));
    }

    if factorization.basis.ncols() < m + 1 {
        return Err(IramError::InvalidConfig(
            "thick restart requires the trailing Arnoldi residual vector".to_string(),
        ));
    }

    // A V_m = V_m H_m + beta v_{m+1} e_m^T
    let beta = factorization.trailing_subdiagonal();
    let u = retained.q;
    let restarted_basis = factorization.basis.slice(s![.., ..m]).dot(&u);
    let h_u = square_hessenberg.dot(&u);
    let u_star = u.t().mapv(|entry| entry.conj());
    let restarted_square = u_star.dot(&h_u);

    let mut restarted_hessenberg = Array2::zeros((target_steps + 1, target_steps).f());
    restarted_hessenberg
        .slice_mut(s![0..k, 0..k])
        .assign(&restarted_square);

    let trailing_row = u.row(m - 1).to_owned() * Complex64::new(beta, 0.0);
    for column in 0..k {
        restarted_hessenberg[[k, column]] = trailing_row[column];
    }

    let residual_reference_norm = norm2(&trailing_row);
    if residual_reference_norm <= breakdown_tol * beta.max(1.0) {
        return Ok(ArnoldiFactorization {
            basis: restarted_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: k,
            happy_breakdown: true,
        });
    }

    let mut continued_basis =
        Array2::<Complex64>::zeros((operator.dimension(), target_steps + 1).f());
    continued_basis
        .slice_mut(s![.., 0..k])
        .assign(&restarted_basis);
    continued_basis
        .column_mut(k)
        .assign(&factorization.basis.column(m));

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
            ritz_inflation: 1.0,
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
            ritz_inflation: 1.0,
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
