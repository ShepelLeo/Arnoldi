//! Ход блочного IRAM
//! Блочный процесс Арнольди и толстые рестарты.
use ndarray::{Array2, ArrayView2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::block_arnoldi::{
    ArnoldiContinuation, ArnoldiFactorization, BlockViewExt, continue_arnoldi, run_arnoldi,
};
use crate::config::{SolverConfig, SpectrumTarget};
use crate::error::IramError;
use crate::linalg::ops::{
    DenseSchurWorkspace, HouseholderQrWorkspace, TrevcWorkspace, dense_schur_with_workspace,
    householder_qr_owned_fortran_with_workspace, matmul, matmul_conj_left, pivoted_qr_rank,
    selected_ritz_vectors_with_workspace,
};
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, RitzEstimate, SolveReport};
use crate::selection::select_ritz_values;

/// Вход в блочный алгоритм.
pub fn solve_block(
    operator: &dyn LinearOperator,
    start_block: Array2<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    config.validate(operator.dimension())?;
    let block_size = config.block_size;
    let current_start = orthonormal_start_block(&start_block, operator.dimension(), block_size)?;

    let mut total_matvecs = 0usize;
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
    let mut schur_workspace = DenseSchurWorkspace::default();
    let mut trevc_workspace = TrevcWorkspace::default();
    let mut restart_qr_workspace = HouseholderQrWorkspace::default();

    for restart in 0..=config.max_restarts {
        let krylov_blocks = factorization.performed_steps;
        let krylov_dimension = krylov_blocks * block_size;
        let square_hessenberg = factorization.square_hessenberg_view(block_size);
        let trailing_subdiagonal = factorization.trailing_subdiagonal(block_size);

        let mut hessenberg_schur =
            dense_schur_with_workspace(square_hessenberg.view(), &mut schur_workspace)
                .map_err(|error| IramError::Spectral(error.to_string()))?;

        let max_keep = config
            .ncv
            .saturating_sub(1)
            .min(krylov_blocks)
            .saturating_mul(block_size);
        let selection = select_ritz_values(
            &hessenberg_schur.w,
            config.target,
            config.nev,
            max_keep,
            config.ritz_inflation,
        )?;

        let retained_indices = retained_indices_with_wanted(
            &selection.retained,
            &selection.wanted,
            &hessenberg_schur.w,
            config.target,
            max_keep,
            block_size,
        )?;

        let retained_ritz_vectors = selected_ritz_vectors_with_workspace(
            &mut hessenberg_schur,
            &retained_indices,
            krylov_dimension,
            &mut trevc_workspace,
        )
        .map_err(|error| IramError::Spectral(error.to_string()))?;

        let retained_residual_estimates = retained_residual_estimates(
            &trailing_subdiagonal,
            &retained_ritz_vectors,
            krylov_blocks,
            block_size,
        );
        let wanted_positions = retained_positions(&retained_indices, &selection.wanted)?;

        let result = selection
            .wanted
            .iter()
            .enumerate()
            .map(|(position, &index)| RitzEstimate {
                value: hessenberg_schur.w[index],
                residual_estimate: retained_residual_estimates[wanted_positions[position]],
            })
            .collect::<Vec<_>>();

        converged = result
            .iter()
            .filter(|estimate| estimate.residual_estimate <= config.tol)
            .count();
        happy_breakdown = factorization.happy_breakdown;
        final_values = result;
        schur_workspace.recycle_schur_output(hessenberg_schur);

        history.push(IterationLog {
            restart,
            krylov_dimension,
            converged,
            total_matvecs,
            peak_memory_bytes: memory::peak_bytes_since_reset(),
            happy_breakdown,
            wanted: final_values.clone(),
        });

        if converged >= config.nev {
            fully_converged = true;
            break;
        }

        if factorization.happy_breakdown {
            note = Some(if krylov_dimension < config.nev {
                "happy breakdown occurred before the block Krylov space became large enough; the chosen operator cannot expose that many independent eigen-directions"
                    .to_string()
            } else {
                "happy breakdown detected; the current block Krylov subspace is already invariant"
                    .to_string()
            });
            break;
        }

        if restart == config.max_restarts {
            note = Some("maximum number of restarts reached before full convergence".to_string());
            break;
        }

        let target_blocks = config.ncv;
        if retained_indices.is_empty()
            || retained_indices.len() >= target_blocks.saturating_mul(block_size)
        {
            note = Some(
                "no room remains to extend the thick-restarted block Krylov space".to_string(),
            );
            break;
        }

        factorization = thick_restart_and_extend(
            operator,
            ThickRestartInput {
                factorization: &factorization,
                square_hessenberg,
                ritz_vectors: retained_ritz_vectors,
                target_blocks,
                block_size,
                breakdown_tol: config.breakdown_tol,
            },
            &mut total_matvecs,
            &mut restart_qr_workspace,
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

fn orthonormal_start_block(
    start_block: &Array2<Complex64>,
    dimension: usize,
    block_size: usize,
) -> Result<Array2<Complex64>, IramError> {
    if start_block.nrows() != dimension {
        return Err(IramError::DimensionMismatch {
            expected: dimension,
            got: start_block.nrows(),
        });
    }
    if start_block.ncols() != block_size {
        return Err(IramError::DimensionMismatch {
            expected: block_size,
            got: start_block.ncols(),
        });
    }

    let start_qr = pivoted_qr_rank(start_block, f64::EPSILON).map_err(IramError::Spectral)?;
    if start_qr.rank != block_size {
        return Err(IramError::Spectral(format!(
            "start block has numerical rank {}, expected {}",
            start_qr.rank, block_size,
        )));
    }

    Ok(start_qr.q)
}

fn retained_indices_with_wanted(
    retained: &[usize],
    wanted: &[usize],
    values: &[Complex64],
    target: SpectrumTarget,
    max_keep: usize,
    block_size: usize,
) -> Result<Vec<usize>, IramError> {
    let mut indices = retained.to_vec();
    indices.extend(wanted.iter().copied());
    indices.sort_unstable();
    indices.dedup();

    if indices.len() > max_keep {
        return Err(IramError::InvalidConfig(format!(
            "retained Ritz set has {} values, but max_keep is {}",
            indices.len(),
            max_keep,
        )));
    }

    let padded_len = indices.len().div_ceil(block_size) * block_size;
    if padded_len > max_keep {
        return Err(IramError::InvalidConfig(format!(
            "retained Ritz set of {} values cannot be padded to a full block without exceeding max_keep={}",
            indices.len(),
            max_keep,
        )));
    }

    if indices.len() < padded_len {
        let mut selected = vec![false; values.len()];
        for &index in &indices {
            if index < selected.len() {
                selected[index] = true;
            }
        }

        for index in ranked_ritz_indices(values, target) {
            if indices.len() == padded_len {
                break;
            }
            if !selected[index] {
                selected[index] = true;
                indices.push(index);
            }
        }
    }

    if indices.len() != padded_len {
        return Err(IramError::InvalidConfig(
            "not enough Ritz values to pad the retained set to a full block".to_string(),
        ));
    }

    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn retained_positions(retained: &[usize], wanted: &[usize]) -> Result<Vec<usize>, IramError> {
    wanted
        .iter()
        .map(|wanted_index| {
            retained.binary_search(wanted_index).map_err(|_| {
                IramError::Spectral("wanted Ritz value is absent from retained set".to_string())
            })
        })
        .collect()
}

fn retained_residual_estimates(
    trailing_subdiagonal: &Array2<Complex64>,
    ritz_vectors: &Array2<Complex64>,
    krylov_blocks: usize,
    block_size: usize,
) -> Vec<f64> {
    let last_block_start = (krylov_blocks - 1) * block_size;

    (0..ritz_vectors.ncols())
        .map(|column| {
            let last_block =
                ritz_vectors.slice(s![last_block_start..last_block_start + block_size, column]);
            let mut norm_sqr = 0.0;

            for row in 0..block_size {
                let mut entry = Complex64::ZERO;
                for inner in 0..block_size {
                    entry += trailing_subdiagonal[[row, inner]] * last_block[inner];
                }
                norm_sqr += entry.norm_sqr();
            }

            norm_sqr.sqrt()
        })
        .collect()
}

/// Блочный thick restart через Ritz-векторы малой задачи.
struct ThickRestartInput<'a> {
    factorization: &'a ArnoldiFactorization,
    square_hessenberg: ArrayView2<'a, Complex64>,
    ritz_vectors: Array2<Complex64>,
    target_blocks: usize,
    block_size: usize,
    breakdown_tol: f64,
}

fn thick_restart_and_extend(
    operator: &dyn LinearOperator,
    restart: ThickRestartInput<'_>,
    matvec_count: &mut usize,
    qr_workspace: &mut HouseholderQrWorkspace,
) -> Result<ArnoldiFactorization, IramError> {
    let ThickRestartInput {
        factorization,
        square_hessenberg,
        ritz_vectors,
        target_blocks,
        block_size,
        breakdown_tol,
    } = restart;

    let krylov_blocks = factorization.performed_steps;
    let krylov_dimension = krylov_blocks * block_size;
    let ritz_rows = ritz_vectors.nrows();
    let ritz_columns = ritz_vectors.ncols();
    let retained = householder_qr_owned_fortran_with_workspace(ritz_vectors, qr_workspace)
        .map_err(IramError::Spectral)?;
    let rank = qr_rank(&retained.r, ritz_rows, ritz_columns, breakdown_tol);

    if rank == 0 || rank >= krylov_dimension {
        return Err(IramError::InvalidConfig(format!(
            "thick restart requires 0 < retained_dimension < krylov_dimension, got retained_dimension={rank}, krylov_dimension={krylov_dimension}",
        )));
    }

    if !rank.is_multiple_of(block_size) {
        return Err(IramError::InvalidConfig(format!(
            "block thick restart requires retained_dimension to be divisible by block_size, got retained_dimension={rank}, block_size={block_size}",
        )));
    }

    let retained_blocks = rank / block_size;
    if retained_blocks >= target_blocks {
        return Err(IramError::InvalidConfig(format!(
            "thick restart retained {retained_blocks} blocks, but target_blocks is {target_blocks}",
        )));
    }

    if factorization.basis.ncols() < (krylov_blocks + 1) * block_size {
        return Err(IramError::InvalidConfig(
            "thick restart requires the trailing Arnoldi residual block".to_string(),
        ));
    }

    let u = if rank == retained.q.ncols() {
        retained.q
    } else {
        retained.q.slice(s![.., 0..rank]).to_owned()
    };

    let restarted_basis = matmul(
        factorization.basis.slice(s![.., 0..krylov_dimension]),
        u.view(),
    );

    let h_u = matmul(square_hessenberg, u.view());
    let restarted_square = matmul_conj_left(u.view(), h_u.view());

    let target_dimension = target_blocks * block_size;
    let mut restarted_hessenberg =
        Array2::zeros(((target_blocks + 1) * block_size, target_dimension).f());
    restarted_hessenberg
        .slice_mut(s![0..rank, 0..rank])
        .assign(&restarted_square);

    let trailing_subdiagonal = factorization.trailing_subdiagonal(block_size);
    let last_block = u.slice(s![
        (krylov_blocks - 1) * block_size..krylov_dimension,
        0..rank
    ]);
    let residual_coefficients = matmul(trailing_subdiagonal.view(), last_block);
    restarted_hessenberg
        .slice_mut(s![rank..rank + block_size, 0..rank])
        .assign(&residual_coefficients);

    let residual_reference_norm = residual_coefficients.norm_f();
    let trailing_norm = trailing_subdiagonal.norm_f();
    if residual_reference_norm <= breakdown_tol * trailing_norm.max(1.0) {
        return Ok(ArnoldiFactorization {
            basis: restarted_basis,
            hessenberg: restarted_hessenberg,
            performed_steps: retained_blocks,
            happy_breakdown: true,
        });
    }

    let mut continued_basis =
        Array2::<Complex64>::zeros((operator.dimension(), (target_blocks + 1) * block_size).f());
    continued_basis
        .slice_mut(s![.., 0..rank])
        .assign(&restarted_basis);
    continued_basis
        .slice_mut(s![.., rank..rank + block_size])
        .assign(
            &factorization
                .basis
                .slice(s![.., krylov_dimension..krylov_dimension + block_size]),
        );

    continue_arnoldi(
        operator,
        ArnoldiContinuation {
            basis: continued_basis,
            hessenberg: restarted_hessenberg,
            start_step: retained_blocks,
            block_size,
            target_steps: target_blocks,
            breakdown_tol,
        },
        matvec_count,
    )
}

fn qr_rank(r: &Array2<Complex64>, rows: usize, columns: usize, relative_tolerance: f64) -> usize {
    let diagonal = (0..rows.min(columns))
        .map(|index| r[[index, index]].norm())
        .collect::<Vec<_>>();
    let scale = diagonal.first().copied().unwrap_or(0.0);
    let cutoff = relative_tolerance.max(0.0) * rows.max(columns) as f64 * scale;

    if scale <= f64::EPSILON {
        0
    } else {
        diagonal.iter().take_while(|&&value| value > cutoff).count()
    }
}

fn ranked_ritz_indices(values: &[Complex64], target: SpectrumTarget) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();

    match target {
        SpectrumTarget::LargestMagnitude => indices.sort_unstable_by(|&left, &right| {
            values[right]
                .norm_sqr()
                .total_cmp(&values[left].norm_sqr())
                .then_with(|| left.cmp(&right))
        }),
        SpectrumTarget::SmallestMagnitude => indices.sort_unstable_by(|&left, &right| {
            values[left]
                .norm_sqr()
                .total_cmp(&values[right].norm_sqr())
                .then_with(|| left.cmp(&right))
        }),
        SpectrumTarget::LargestReal => indices.sort_unstable_by(|&left, &right| {
            values[right]
                .re
                .total_cmp(&values[left].re)
                .then_with(|| values[right].im.abs().total_cmp(&values[left].im.abs()))
                .then_with(|| left.cmp(&right))
        }),
        SpectrumTarget::SmallestReal => indices.sort_unstable_by(|&left, &right| {
            values[left]
                .re
                .total_cmp(&values[right].re)
                .then_with(|| values[left].im.abs().total_cmp(&values[right].im.abs()))
                .then_with(|| left.cmp(&right))
        }),
        SpectrumTarget::BothEndsReal => {
            let min_real = values
                .iter()
                .map(|entry| entry.re)
                .fold(f64::INFINITY, f64::min);
            let max_real = values
                .iter()
                .map(|entry| entry.re)
                .fold(f64::NEG_INFINITY, f64::max);
            let center = 0.5 * (min_real + max_real);

            indices.sort_unstable_by(|&left, &right| {
                (values[right].re - center)
                    .abs()
                    .total_cmp(&(values[left].re - center).abs())
                    .then_with(|| left.cmp(&right))
            });
        }
    }

    indices
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ShapeBuilder};
    use num_complex::Complex64;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::config::{SolverConfig, SpectrumTarget};
    use crate::linalg::ops::normalized_random_unitary_matrix;
    use crate::operator::{FnOperator, IdentityOperator};

    use super::solve_block;

    #[test]
    fn identity_operator_converges_for_full_start_block() {
        let operator = IdentityOperator::new(8);
        let mut start = Array2::<Complex64>::zeros((8, 2).f());
        start[[0, 0]] = Complex64::ONE;
        start[[1, 1]] = Complex64::ONE;
        let config = SolverConfig {
            nev: 2,
            block_size: 2,
            ncv: 3,
            max_restarts: 5,
            tol: 1.0e-10,
            breakdown_tol: 1.0e-12,
            ritz_inflation: 1.0,
            target: SpectrumTarget::LargestMagnitude,
        };

        let report = solve_block(&operator, start, config, "coordinate block")
            .expect("the identity problem should be solvable");

        assert_eq!(report.converged, 2);
        assert!(report.fully_converged);
        assert!(
            report
                .final_values
                .iter()
                .all(|estimate| (estimate.value - Complex64::ONE).norm() < 1.0e-10)
        );
    }

    #[test]
    fn thick_restart_converges_on_diagonal_problem() {
        let dimension = 40;
        let operator = FnOperator::new(
            dimension,
            "diagonal test operator",
            move |vector, mut output| {
                for (index, (&entry, target)) in vector.iter().zip(output.iter_mut()).enumerate() {
                    *target = Complex64::new((index + 1) as f64, 0.0) * entry;
                }
            },
        );
        let mut rng = StdRng::seed_from_u64(7);
        let start = normalized_random_unitary_matrix(dimension, 2, &mut rng)
            .expect("random start block should have full rank");
        let config = SolverConfig {
            nev: 2,
            block_size: 2,
            ncv: 5,
            max_restarts: 40,
            tol: 1.0e-8,
            breakdown_tol: 1.0e-12,
            ritz_inflation: 1.0,
            target: SpectrumTarget::SmallestMagnitude,
        };

        let report = solve_block(&operator, start, config, "seeded random block")
            .expect("the diagonal problem should be solvable");

        assert!(
            report.fully_converged,
            "expected full convergence, got {} converged values after {} restarts; note={:?}",
            report.converged, report.total_restarts, report.note,
        );
        assert_eq!(report.converged, 2);
        assert!(report.total_restarts > 1);
    }

    #[test]
    fn solve_allows_block_size_smaller_than_requested_eigenvalues() {
        let dimension = 48;
        let operator = FnOperator::new(
            dimension,
            "diagonal test operator",
            move |vector, mut output| {
                for (index, (&entry, target)) in vector.iter().zip(output.iter_mut()).enumerate() {
                    *target = Complex64::new((index + 1) as f64, 0.0) * entry;
                }
            },
        );
        let mut rng = StdRng::seed_from_u64(11);
        let start = normalized_random_unitary_matrix(dimension, 2, &mut rng)
            .expect("random start block should have full rank");
        let config = SolverConfig {
            nev: 4,
            block_size: 2,
            ncv: 8,
            max_restarts: 60,
            tol: 1.0e-8,
            breakdown_tol: 1.0e-12,
            ritz_inflation: 1.0,
            target: SpectrumTarget::SmallestMagnitude,
        };

        let report = solve_block(&operator, start, config, "seeded random block")
            .expect("block_size < nev should be a valid solve configuration");

        assert!(
            report.fully_converged,
            "expected full convergence, got {} converged values after {} restarts; note={:?}",
            report.converged, report.total_restarts, report.note,
        );
        assert_eq!(report.converged, 4);
    }
}
