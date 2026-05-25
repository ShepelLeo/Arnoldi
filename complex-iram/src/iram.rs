//! Ядро IRAM: процесс Арнольди + неявные шифтованные рестарты.
//!
//! Алгоритм работает поверх примитивов `Backend`. Базис V хранится бэкендом,
//! H — обычным `Vec<Complex64>` column-major в ядре. Никакого `ndarray`.

use num_complex::Complex64;

use crate::arnoldi::{ArnoldiFactorization, HessenbergMatrix, continue_arnoldi, run_arnoldi};
use crate::backend::{Backend, DenseColMajor, LapackBackend};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::linalg::ops::Trans;
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, RitzEstimate, SolveReport};
use crate::selection::*;

/// Backward-compatible LAPACK entry point.
pub fn solve(
    operator: &dyn LinearOperator,
    start_vector: Vec<Complex64>,
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
    start_vector: Vec<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    config.validate(operator.dimension())?;

    let mut current_start = start_vector;
    backend.normalize(&mut current_start, "solver start vector")?;

    let mut operator_handle = backend.prepare_operator(operator)?;
    let mut total_matvecs = 0usize;

    let mut factorization = run_arnoldi(
        backend,
        &mut operator_handle,
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

    for restart in 0..=config.max_restarts {
        let krylov_dim = factorization.performed_steps;
        let square_hessenberg = factorization.square_hessenberg();
        let trailing_subdiagonal = factorization.trailing_subdiagonal();

        // Малая спектральная задача
        let small_h = DenseColMajor {
            data: square_hessenberg,
            rows: krylov_dim,
            cols: krylov_dim,
        };
        let small_eig = backend.small_eig(&small_h)?;

        let selection = select_ritz_values(
            &small_eig.values,
            config.target,
            config.nev,
            krylov_dim,
            config.ritz_inflation,
        )?;

        // Из small_eig.vectors выбираем wanted-столбцы.
        let result: Vec<RitzEstimate> = selection
            .wanted
            .iter()
            .map(|&idx| {
                let value = small_eig.values[idx];
                let last_entry = small_eig.vectors[(krylov_dim - 1) + idx * krylov_dim];
                let residual_estimate = (trailing_subdiagonal * last_entry).norm();

                RitzEstimate {
                    value,
                    residual_estimate,
                }
            })
            .collect();

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
            break;
        }

        if factorization.happy_breakdown {
            note = Some(if krylov_dim < config.nev {
                "happy breakdown occurred before the Krylov space became large enough; with a single starting vector the chosen operator cannot expose that many independent eigen-directions"
                    .to_string()
            } else {
                "happy breakdown detected; the current Krylov subspace is already invariant"
                    .to_string()
            });
            break;
        }

        if restart == config.max_restarts {
            note = Some("maximum number of restarts reached before full convergence".to_string());
            break;
        }

        if selection.shifts.is_empty() {
            note = Some("no unwanted Ritz values remain for the restart filter".to_string());
            break;
        }

        factorization = implicit_restart_and_extend(
            backend,
            &mut operator_handle,
            operator,
            factorization,
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

/// Запуск шифтованного QR-рестарта + продолжение Арнольди.
fn implicit_restart_and_extend<B: Backend>(
    backend: &mut B,
    operator_handle: &mut B::OperatorHandle,
    operator: &dyn LinearOperator,
    factorization: ArnoldiFactorization<B::BasisHandle>,
    shifts: &[Complex64],
    target_steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
    k: usize,
) -> Result<ArnoldiFactorization<B::BasisHandle>, IramError> {
    let m = factorization.performed_steps;
    let dim = operator.dimension();

    if k == 0 || k >= m {
        return Err(IramError::InvalidConfig(format!(
            "implicit restart requires 0 < retained_dimension < krylov_dimension, got retained_dimension={k}, krylov_dimension={m}",
        )));
    }

    // A V_m = V_m H_m + beta v_{m+1} e_m^T
    let beta = factorization.trailing_subdiagonal();

    let square_h = DenseColMajor {
        data: factorization.square_hessenberg(),
        rows: m,
        cols: m,
    };
    let (rotation, h_after) = backend.multishift_qr_filter(&square_h, shifts)?;

    // rotated_basis: V_m * Q[:, 0..=k] — это (k+1) столбцов, размером dim×(k+1).
    // Соберём временно host-копию первых m столбцов базиса.
    let mut basis_host = vec![Complex64::ZERO; dim * m];
    {
        let mut column_buffer = vec![Complex64::ZERO; dim];
        for j in 0..m {
            backend.read_basis_column(&factorization.basis, j, &mut column_buffer);
            let dst = &mut basis_host[j * dim..(j + 1) * dim];
            dst.copy_from_slice(&column_buffer);
        }
    }

    // rotated_block = V_m * Q[:, 0..=k] (host gemm на (dim) × (k+1)).
    let mut rotated_block = vec![Complex64::ZERO; dim * (k + 1)];
    backend.gemm(
        Trans::None,
        Trans::None,
        dim,
        k + 1,
        m,
        &basis_host,
        dim.max(1),
        &rotation.data,
        rotation.ld(),
        &mut rotated_block,
        dim.max(1),
    );

    // Подготовим целевой Хессенберг (target_steps+1) × target_steps.
    let mut restarted_hessenberg = HessenbergMatrix::zeros(target_steps + 1, target_steps);
    for j in 0..k {
        for i in 0..=k {
            restarted_hessenberg.set(i, j, h_after.get(i, j));
        }
    }

    // Невязка для рестарта.
    let h_coupling = h_after.get(k, k - 1);
    let residual_coupling = Complex64::new(beta, 0.0) * rotation.get(m - 1, k - 1);

    // residual = h_coupling * rotated_block[:, k] + residual_coupling * V[:, m]
    let mut residual = vec![Complex64::ZERO; dim];
    {
        let rotated_kth = &rotated_block[k * dim..(k + 1) * dim];
        for (r, &v) in residual.iter_mut().zip(rotated_kth.iter()) {
            *r = h_coupling * v;
        }
    }
    let mut v_m = vec![Complex64::ZERO; dim];
    backend.read_basis_column(&factorization.basis, m, &mut v_m);
    backend.axpy(&mut residual, residual_coupling, &v_m);

    let residual_reference_norm = (h_coupling.norm_sqr() + residual_coupling.norm_sqr()).sqrt();
    let mut h_column_correction = vec![Complex64::ZERO; k];

    // Ортогонализация против rotated_basis (k столбцов, dim×k) — host-сторона.
    let orthogonalized = backend.orthogonalize_against_host_basis(
        &mut residual,
        &rotated_block[..dim * k],
        dim,
        k,
        &mut h_column_correction,
        residual_reference_norm,
        breakdown_tol,
    );

    for (row, &value) in h_column_correction.iter().enumerate() {
        let existing = restarted_hessenberg.get(row, k - 1);
        restarted_hessenberg.set(row, k - 1, existing + value);
    }

    // Освобождаем старый базис; аллоцируем новый.
    drop(factorization);
    let mut new_basis = backend.alloc_basis(dim, target_steps + 1)?;
    for j in 0..k {
        let column = &rotated_block[j * dim..(j + 1) * dim];
        backend.write_basis_column(&mut new_basis, j, column);
    }

    if orthogonalized.happy_breakdown {
        restarted_hessenberg.set(k, k - 1, Complex64::ZERO);
        return Ok(ArnoldiFactorization {
            basis: new_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: k,
            happy_breakdown: true,
        });
    }

    restarted_hessenberg.set(
        k,
        k - 1,
        Complex64::new(orthogonalized.residual_norm, 0.0),
    );
    backend.write_basis_column(&mut new_basis, k, &residual);

    continue_arnoldi(
        backend,
        operator_handle,
        operator,
        new_basis,
        restarted_hessenberg,
        k,
        target_steps,
        breakdown_tol,
        matvec_count,
    )
}
