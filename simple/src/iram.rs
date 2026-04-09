//! Ход IRAM
//! Процесс Арнольди, рестарты
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::arnoldi::{ArnoldiFactorization, continue_arnoldi, run_arnoldi};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::linalg::ops::{
    axpy_in_place, is_numerical_breakdown, norm2, normalize, orthogonalize_twice,
};
use crate::linalg::small::compute_ritz_values;
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, SolveReport};
use crate::selection::select_ritz_values;

/// Вход в алгоритм
pub fn solve(
    operator: &dyn LinearOperator,
    start_vector: Array1<f64>,
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
        let krylov_dimension = factorization.performed_steps;
        let square_hessenberg = factorization.square_hessenberg();
        // Крайний поддиагональный элемент
        let trailing_subdiagonal = factorization.trailing_subdiagonal();

        // Получаем числа Ритца с оценкой
        // Малая спектральная задача
        let ritz_values = compute_ritz_values(&square_hessenberg, trailing_subdiagonal)?;
        let keep_limit = krylov_dimension;

        // Выбираем желаемые СЗН матрицы Хессенберга
        // # Их может быть больше, чем искомых nev, т.к у выбранных восполняем их комплексно-сопряенные
        let selection = select_ritz_values(&ritz_values, config.target, config.nev, keep_limit)?;

        // Сколько сошлось
        converged = selection
            .wanted
            .iter()
            .filter(|estimate| estimate.residual_estimate <= config.tol)
            .count();
        final_values = selection.wanted.clone();
        happy_breakdown = factorization.happy_breakdown;

        
        history.push(IterationLog {
            restart,
            krylov_dimension,
            retained_dimension: selection.retained_dimension,
            converged,
            total_matvecs,
            peak_memory_bytes: memory::peak_bytes_since_reset(),
            happy_breakdown,
            wanted: selection.wanted.clone(),
            shifts: selection.shifts.clone(),
        });

        if converged >= config.nev {
            fully_converged = true;
            break; // Ура, мы сошлись
        }

        if factorization.happy_breakdown {
            note = Some(if krylov_dimension < config.nev {
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
            selection.retained_dimension,
            &selection.shifts,
            config.ncv,
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


/// Вход в рестарты 
fn implicit_restart_and_extend(
    operator: &dyn LinearOperator,
    factorization: &ArnoldiFactorization,
    retained_dimension: usize,
    shifts: &[Complex64],
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let m = factorization.performed_steps;
    let k = retained_dimension;

    if k == 0 || k >= m {
        return Err(IramError::InvalidConfig(format!(
            "implicit restart requires 0 < retained_dimension < krylov_dimension, got retained_dimension={k}, krylov_dimension={m}",
        ))); // Беда
    }

    if factorization.basis.len() < m + 1 {
        return Err(IramError::InvalidConfig(
            "implicit restart requires the trailing Arnoldi residual vector".to_string(),
        )); // Беда
    }

    // A V_m = V_m H_m + beta v_{m+1} e_m^T
    let beta = factorization.trailing_subdiagonal();
    let old_hessenberg = factorization.square_hessenberg();
    let mut h = array_to_dmatrix(&old_hessenberg);
    let mut q_total = DMatrix::<f64>::identity(m, m);

    // Делаем QR со сдвигами
    apply_shifted_qr_steps(&mut h, &mut q_total, shifts)?;

    // Поворачиваем базис пространства крылова
    let rotated_basis = (0..k)
        .map(|column| rotate_basis_column(&factorization.basis[..m], &q_total, column))
        .collect::<Vec<_>>();
    let mut restarted_hessenberg = Array2::<f64>::zeros((target_steps + 1, target_steps));

    (0..k).for_each(|column| {
        (0..k).for_each(|row| {
            restarted_hessenberg[[row, column]] = h[(row, column)];
        });
    });


    // r_new = h_{k+1,k} * (V_m q_k) + beta * q_{m,k} * v_{m+1}
    let h_coupling = h[(k, k - 1)];
    let residual_coupling = beta * q_total[(m - 1, k - 1)];
    let mut residual = rotate_basis_column(&factorization.basis[..m], &q_total, k);
    residual.iter_mut().for_each(|entry| *entry *= h_coupling);
    axpy_in_place(&mut residual, residual_coupling, &factorization.basis[m]);

    // Доортогонализуем r к новому базису
    let mut h_column_correction = vec![0.0; k];
    orthogonalize_twice(&mut residual, &rotated_basis, &mut h_column_correction);
    h_column_correction
        .iter()
        .enumerate()
        .for_each(|(row, &value)| restarted_hessenberg[[row, k - 1]] += value);

    // Проверяем ЛЗ пополнения в новом базисе
    let residual_norm = norm2(&residual);
    let residual_reference_norm = h_coupling.hypot(residual_coupling);
    if is_numerical_breakdown(residual_norm, residual_reference_norm, breakdown_tol) {
        restarted_hessenberg[[k, k - 1]] = 0.0;
        return Ok(ArnoldiFactorization {
            basis: rotated_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: k,
            happy_breakdown: true,
        });
    }

    // A V_k^+ = V_k^+ H_k^+ + r_new e_k^T
    restarted_hessenberg[[k, k - 1]] = residual_norm;
    residual
        .iter_mut()
        .for_each(|entry| *entry /= residual_norm);

    let mut basis = rotated_basis;
    basis.push(residual);

    // Рестартуем процесс Арнольди с нового базиса
    continue_arnoldi(
        operator,
        basis,
        restarted_hessenberg,
        k,
        target_steps,
        breakdown_tol,
        matvec_count,
    )
}


/// Вход в QR-shifted 
fn apply_shifted_qr_steps(
    h: &mut DMatrix<f64>,
    q_total: &mut DMatrix<f64>,
    shifts: &[Complex64],
) -> Result<(), IramError> {
    let pair_map = build_shift_pair_map(shifts);
    let mut used = vec![false; shifts.len()];

    for index in 0..shifts.len() {
        if used[index] {
            continue;
        }

        let shift = shifts[index];

        if shift.im.abs() <= 1.0e-12 {
            apply_real_shifted_qr_step(h, q_total, shift.re);
            used[index] = true;
            continue;
        }

        let pair_index = pair_map[index].ok_or_else(|| {
            IramError::Spectral(
                "complex restart shifts must appear in conjugate pairs for the real IRAM filter"
                    .to_string(),
            )
        })?;
        apply_complex_pair_shifted_qr_step(h, q_total, shift);
        used[index] = true;
        used[pair_index] = true;
    }

    Ok(())
}

/// Вещественный сдвиг
fn apply_real_shifted_qr_step(h: &mut DMatrix<f64>, q_total: &mut DMatrix<f64>, shift: f64) {
    let order = h.nrows();
    let identity = DMatrix::<f64>::identity(order, order);
    let shifted = h.clone() - identity.clone() * shift;
    let (q, r) = shifted.qr().unpack();

    *h = r * q.clone() + identity * shift;
    *q_total = q_total.clone() * q;
    enforce_upper_hessenberg(h);
}

/// Комплексноспряженный сдвиг
fn apply_complex_pair_shifted_qr_step(
    h: &mut DMatrix<f64>,
    q_total: &mut DMatrix<f64>,
    shift: Complex64,
) {
    let order = h.nrows();
    let identity = DMatrix::<f64>::identity(order, order);
    let polynomial =
        h.clone() * h.clone() - h.clone() * (2.0 * shift.re) + identity * shift.norm_sqr();
    let (q, _) = polynomial.qr().unpack();

    *h = q.transpose() * h.clone() * q.clone();
    *q_total = q_total.clone() * q;
    enforce_upper_hessenberg(h);
}

/// Грубо сохраняем Хессенберговость
fn enforce_upper_hessenberg(h: &mut DMatrix<f64>) {
    let order = h.nrows();

    (0..order).for_each(|row| {
        (0..row.saturating_sub(1)).for_each(|column| {
            if h[(row, column)].abs() <= 1.0e-10 {
                h[(row, column)] = 0.0;
            }
        });
    });
}

/// Поворот базиса 
fn rotate_basis_column(
    basis: &[Array1<f64>],
    q_total: &DMatrix<f64>,
    column: usize,
) -> Array1<f64> {
    let mut result = Array1::<f64>::zeros(basis[0].len());

    basis.iter().enumerate().for_each(|(row, basis_vector)| {
        let coefficient = q_total[(row, column)];
        if coefficient != 0.0 {
            axpy_in_place(&mut result, coefficient, basis_vector);
        }
    });

    result
}

/// Переводим ndarray матрицу в nalgebra, нужно чтобы достать QR
fn array_to_dmatrix(array: &Array2<f64>) -> DMatrix<f64> {
    let storage = array.iter().copied().collect::<Vec<_>>();
    DMatrix::from_row_slice(array.nrows(), array.ncols(), &storage)
}

/// Собираем комплексносопряженные пары чисел Ритца в индекс -> индекс сопряженного
fn build_shift_pair_map(shifts: &[Complex64]) -> Vec<Option<usize>> {
    let tolerance = 1.0e-8;
    let mut pair_map = vec![None; shifts.len()];
    let mut used = vec![false; shifts.len()];

    shifts.iter().enumerate().for_each(|(index, shift)| {
        if used[index] || shift.im <= tolerance {
            return;
        }

        let conjugate = shift.conj();
        let candidate = shifts
            .iter()
            .enumerate()
            .filter(|(other_index, other_shift)| {
                *other_index != index && !used[*other_index] && other_shift.im < -tolerance
            })
            .min_by(|(_, left_shift), (_, right_shift)| {
                (*left_shift - conjugate)
                    .norm()
                    .partial_cmp(&(*right_shift - conjugate).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((pair_index, pair_shift)) = candidate {
            let scale = 1.0 + shift.norm().max(pair_shift.norm());
            if (*pair_shift - conjugate).norm() <= tolerance * scale {
                pair_map[index] = Some(pair_index);
                pair_map[pair_index] = Some(index);
                used[index] = true;
                used[pair_index] = true;
            }
        }
    });

    pair_map
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::config::{SolverConfig, SpectrumTarget, recommended_ncv};
    use crate::linalg::ops::normalized_random_vector;
    use crate::operator::{ConvectionDiffusionOperator, IdentityOperator, LinearOperator};

    use super::solve;

    #[test]
    fn identity_operator_converges_for_one_eigenvalue() {
        let operator = IdentityOperator::new(8);
        let start = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
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
        assert!((report.final_values[0].value.re - 1.0).abs() < 1.0e-10);
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
