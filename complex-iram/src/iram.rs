//! Ход IRAM
//! Процесс Арнольди, рестарты
use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::arnoldi::{ArnoldiFactorization, continue_arnoldi, run_arnoldi};
use crate::backend::{Backend, LapackBackend};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, RitzEstimate, SolveReport};
use crate::selection::*;

/// Backward-compatible LAPACK entry point.
pub fn solve(
    operator: &dyn LinearOperator,
    start_vector: Array1<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    let mut backend = LapackBackend::new();
    solve_with_backend(&mut backend, operator, start_vector, config, start_description)
}

/// Вход в алгоритм с явным бэкендом.
pub fn solve_with_backend<B: Backend>(
    backend: &mut B,
    operator: &dyn LinearOperator,
    start_vector: Array1<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    config.validate(operator.dimension())?;

    // Инициализируем стартовый вектор
    let mut current_start = start_vector;
    backend.normalize_vector(&mut current_start, "solver start vector")?;

    let mut operator_workspace = backend.prepare_operator(operator)?;
    let mut total_matvecs = 0usize;
    // Запуск процесса Арнольди
    let mut factorization = run_arnoldi(
        backend,
        &mut operator_workspace,
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
        let mut hessenberg_decomposition = backend.compute_ritz_values(&square_hessenberg)?;

        // Выбираем желаемые СЗН матрицы Хессенберга
        let selection = select_ritz_values(
            backend.ritz_values(&hessenberg_decomposition),
            config.target,
            config.nev,
            krylov_dim,
            config.ritz_inflation,
        )?;

        let ritz_vectors = backend.retrieve_ritz_vectors(
            &mut hessenberg_decomposition,
            &selection.wanted,
            krylov_dim,
        )?;

        let result: Vec<RitzEstimate> = selection.wanted
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let value = backend.ritz_values(&hessenberg_decomposition)[idx];

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
            peak_device_memory_bytes: memory::peak_device_bytes_since_reset(),
            device_allocations: memory::device_allocation_count_since_reset(),
            host_to_device_bytes: memory::host_to_device_bytes_since_reset(),
            device_to_host_bytes: memory::device_to_host_bytes_since_reset(),
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
            backend,
            &mut operator_workspace,
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
        operator_description: format!("{} [{} backend]", operator.description(), backend.name()),
        start_description: start_description.into(),
        dimension: operator.dimension(),
        config,
        elapsed_seconds: 0.0,
        total_restarts: history.len(),
        total_matvecs,
        peak_memory_bytes: memory::peak_bytes_since_reset(),
        peak_device_memory_bytes: memory::peak_device_bytes_since_reset(),
        device_allocations: memory::device_allocation_count_since_reset(),
        host_to_device_bytes: memory::host_to_device_bytes_since_reset(),
        device_to_host_bytes: memory::device_to_host_bytes_since_reset(),
        converged,
        fully_converged,
        happy_breakdown,
        final_values,
        history,
        note,
    })
}

/// Вход в рестарты
fn implicit_restart_and_extend<B: Backend>(
    backend: &mut B,
    operator_workspace: &mut B::OperatorWorkspace,
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
    let (rotation, h) = backend.shifted_qr_filter(&factorization.square_hessenberg(), shifts)?;

    // Compute [V_m Q(:,0:k-1), V_m Q(:,k)] in one backend GEMM.
    let rotated_block = rotate_basis_block(
        backend,
        &factorization.basis.slice(s![.., ..m]),
        &rotation,
        k + 1,
    );
    let mut rotated_basis = Array2::<Complex64>::zeros((factorization.basis.nrows(), k).f());
    rotated_basis
        .slice_mut(s![.., ..k])
        .assign(&rotated_block.slice(s![.., ..k]));

    let mut restarted_hessenberg =
        Array2::<Complex64>::from_elem((target_steps + 1, target_steps), Complex64::ZERO);

    restarted_hessenberg
        .slice_mut(s![0..=k, 0..k])
        .assign(&h.slice(s![0..=k, 0..k]));

    // r_new = h_{k+1,k} * (V_m q_{k+1}) + beta * q_{m,k} * v_{m+1}
    let h_coupling = h[(k, k - 1)];
    let residual_coupling = Complex64::new(beta, 0.0) * rotation[[m - 1, k - 1]];

    let mut residual = rotated_block.column(k).to_owned();
    backend.scale_vector_in_place(&mut residual, h_coupling);
    backend.add_scaled_vector_in_place(
        &mut residual,
        residual_coupling,
        &factorization.basis.column(m).to_owned(),
    );

    let residual_reference_norm = (h_coupling.norm_sqr() + residual_coupling.norm_sqr()).sqrt();
    let mut h_column_correction = vec![Complex64::default(); k];
    let orthogonalized = backend.orthogonalize_restart_residual(
        &mut residual,
        &rotated_basis.view(),
        &mut h_column_correction,
        residual_reference_norm,
        breakdown_tol,
    );
    h_column_correction
        .iter()
        .enumerate()
        .for_each(|(row, &value)| restarted_hessenberg[[row, k - 1]] += value);

    if orthogonalized.happy_breakdown {
        restarted_hessenberg[[k, k - 1]] = Complex64::ZERO;
        return Ok(ArnoldiFactorization {
            basis: rotated_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: k,
            happy_breakdown: true,
        });
    }

    restarted_hessenberg[[k, k - 1]] = Complex64::new(orthogonalized.residual_norm, 0.0);

    let mut continued_basis =
        Array2::<Complex64>::zeros((factorization.basis.nrows(), target_steps + 1).f());
    continued_basis
        .slice_mut(s![.., 0..k])
        .assign(&rotated_basis);
    continued_basis.column_mut(k).assign(&residual);

    continue_arnoldi(
        backend,
        operator_workspace,
        operator,
        continued_basis,
        restarted_hessenberg,
        k,
        target_steps,
        breakdown_tol,
        matvec_count,
    )
}

fn rotate_basis_block<B: Backend>(
    backend: &mut B,
    basis: &ArrayView2<Complex64>,
    q_total: &Array2<Complex64>,
    columns: usize,
) -> Array2<Complex64> {
    backend.zgemm_nn(*basis, q_total.slice(s![.., 0..columns]))
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
            ritz_inflation: Some(1.0),
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
            ritz_inflation: Some(1.0),
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
