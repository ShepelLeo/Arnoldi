//! Блочный IRAM с толстым рестартом.
use ndarray::{Array2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::block_arnoldi::{
    BlockArnoldiFactorization, BlockArnoldiWorkspaces, append_columns, column_norms,
    continue_block_arnoldi_from_parts, orthogonalize_block_residual,
    run_block_arnoldi_with_workspaces,
};
use crate::config::SolverConfig;
use crate::error::IramError;
use crate::linalg::lapack::{
    DenseSchurWorkspace, TrevcWorkspace, ZgemmTranspose, zgemm, zgemm_into,
    zgeqp3_qr_rank_with_workspace,
};
use crate::linalg::small::{
    compute_dense_ritz_values_with_workspace, retrive_ritz_vectors_with_workspace,
};
use crate::memory;
use crate::operator::LinearOperator;
use crate::report::{IterationLog, RitzEstimate, SolveReport};
use crate::selection::select_ritz_values;

pub fn solve_block(
    operator: &dyn LinearOperator,
    start_block: Array2<Complex64>,
    config: SolverConfig,
    start_description: impl Into<String>,
) -> Result<SolveReport, IramError> {
    config.validate(operator.dimension())?;

    if start_block.nrows() != operator.dimension() {
        return Err(IramError::DimensionMismatch {
            expected: operator.dimension(),
            got: start_block.nrows(),
        });
    }

    if start_block.ncols() > config.block_size {
        return Err(IramError::InvalidConfig(format!(
            "start block has {} columns, but block_size is {}",
            start_block.ncols(),
            config.block_size,
        )));
    }

    let mut total_matvecs = 0usize;
    let mut arnoldi_workspaces = BlockArnoldiWorkspaces::default();
    let mut schur_workspace = DenseSchurWorkspace::default();
    let mut trevc_workspace = TrevcWorkspace::default();
    let mut factorization = run_block_arnoldi_with_workspaces(
        operator,
        &start_block,
        config.ncv,
        config.breakdown_tol,
        &mut total_matvecs,
        &mut arnoldi_workspaces,
    )?;

    let mut history = Vec::new();
    let mut final_values = Vec::new();
    let mut note = None;
    let mut fully_converged = false;
    let mut happy_breakdown = false;
    let mut converged = 0usize;

    for restart in 0..=config.max_restarts {
        let krylov_dim = factorization.krylov_dimension();
        let square_hessenberg = factorization.square_hessenberg();
        let trailing_coupling = factorization.trailing_coupling();
        let last_block_range = factorization.last_block_range();

        //   H_m Z = Z T,  theta_i = T_ii.
        let mut hessenberg_schur =
            compute_dense_ritz_values_with_workspace(&square_hessenberg, &mut schur_workspace);
        //   |theta_j - c| <= ritz_inflation * max_i |theta_i - c|.
        let selection = select_ritz_values(
            &hessenberg_schur.w,
            config.target,
            config.nev,
            krylov_dim,
            config.ritz_inflation,
        )?;
        
        let ritz_vectors = retrive_ritz_vectors_with_workspace(
            &mut hessenberg_schur,
            &selection.retained,
            krylov_dim,
            &mut trevc_workspace,
        );

        // Arnoldi relation:
        //   A Q_m = Q_m H_m + Q_{m+1} C_m E_m^*.
        // Ritz residual estimate:
        //   ||A Q_m y - theta Q_m y||_2 = ||C_m E_m^* y||_2.
        let residual_estimates = if trailing_coupling.nrows() == 0 {
            vec![0.0; ritz_vectors.ncols()]
        } else {
            let tails = ritz_vectors.slice(s![last_block_range.clone(), ..]);
            let residuals = zgemm(
                ZgemmTranspose::None,
                ZgemmTranspose::None,
                trailing_coupling.view(),
                tails,
            );
            column_norms(residuals.view())
        };

        let result = selection
            .wanted
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let value = hessenberg_schur.w[idx];
                let residual_estimate = residual_estimates[i];

                RitzEstimate {
                    value,
                    residual_estimate,
                }
            })
            .collect::<Vec<_>>();

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
        });

        if converged >= config.nev {
            fully_converged = true;
            break;
        }

        if factorization.happy_breakdown {
            note = Some(if krylov_dim < config.nev {
                "happy breakdown occurred before the block Krylov space became large enough to expose the requested number of eigen-directions"
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

        if ritz_vectors.ncols() == 0 {
            note = Some("no Ritz vectors were selected for the thick restart".to_string());
            break;
        }

        factorization = restart_and_extend(
            operator,
            &factorization,
            &ritz_vectors,
            config.ncv,
            config.breakdown_tol,
            &mut total_matvecs,
            &mut arnoldi_workspaces,
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

fn restart_and_extend(
    operator: &dyn LinearOperator,
    factorization: &BlockArnoldiFactorization,
    ritz_vectors: &Array2<Complex64>,
    target_blocks: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
    workspaces: &mut BlockArnoldiWorkspaces,
) -> Result<BlockArnoldiFactorization, IramError> {
    let krylov_dim = factorization.krylov_dimension();
    let h = factorization
        .hessenberg
        .slice(s![0..krylov_dim, 0..krylov_dim]);
    let q_basis = factorization.krylov_basis();

    //   V P = U R,  rank(U) = p.
    let retained = zgeqp3_qr_rank_with_workspace(
        ritz_vectors,
        breakdown_tol,
        &mut workspaces.basis_pivoted_qr,
    )
    .map_err(IramError::Spectral)?;
    if retained.rank == 0 {
        return Err(IramError::Spectral(
            "selected Ritz vector block has numerical rank zero".to_string(),
        ));
    }

    let u = retained.q;
    let retained_size = u.ncols();
    //   Q_+ = Q_m U,  H_+ = U^* H_m U.
    let q_restarted = zgemm(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        q_basis,
        u.view(),
    );
    let h_u = zgemm(ZgemmTranspose::None, ZgemmTranspose::None, h, u.view());
    let mut h_restarted = zgemm(
        ZgemmTranspose::ConjugateTranspose,
        ZgemmTranspose::None,
        u.view(),
        h_u.view(),
    );

    //   G = (I - U U^*) H_m U = H_m U - U H_+,
    //   RES <- Q_m G.
    let mut q_residual_coeff = h_u;
    zgemm_into(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        Complex64::new(-1.0, 0.0),
        u.view(),
        h_restarted.view(),
        Complex64::new(1.0, 0.0),
        q_residual_coeff.view_mut(),
    );
    let mut residual = zgemm(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        q_basis,
        q_residual_coeff.view(),
    );

    if factorization.next_block_size > 0 {
        //   RES <- RES + Q_{m+1} C_m E_m^* U.
        let trailing_coupling = factorization.trailing_coupling();
        let last_range = factorization.last_block_range();
        let u_last = u.slice(s![last_range, ..]);
        let tail_coeff = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            trailing_coupling.view(),
            u_last,
        );
        let q_next = factorization
            .next_basis_block()
            .expect("next block must be present when next_block_size > 0");
        zgemm_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::new(1.0, 0.0),
            q_next,
            tail_coeff.view(),
            Complex64::new(1.0, 0.0),
            residual.view_mut(),
        );
    }

    // A Q_+ = Q_+ H_+ + RES.
    let mut aq_restarted = residual.clone();
    zgemm_into(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        Complex64::new(1.0, 0.0),
        q_restarted.view(),
        h_restarted.view(),
        Complex64::new(1.0, 0.0),
        aq_restarted.view_mut(),
    );
    let reference_norms = column_norms(aq_restarted.view());

    //   RES = Q_next R_next,
    //   H_new = [H_+; R_next].
    let orthogonalized = orthogonalize_block_residual(
        q_restarted.view(),
        &mut h_restarted,
        residual,
        &reference_norms,
        breakdown_tol,
        workspaces,
    )?;

    if orthogonalized.rank == 0 {
        return Ok(BlockArnoldiFactorization {
            basis: q_restarted,
            hessenberg: h_restarted,
            block_sizes: vec![retained_size],
            next_block_size: 0,
            performed_blocks: 1,
            happy_breakdown: true,
        });
    }

    // Q_total = [Q_+, Q_next],
    // H_total = [ H_+ ; R_next ] before continuing block Arnoldi.
    let basis = append_columns(q_restarted.view(), orthogonalized.q_next.view());
    let mut hessenberg = Array2::zeros((retained_size + orthogonalized.rank, retained_size).f());
    hessenberg
        .slice_mut(s![0..retained_size, 0..retained_size])
        .assign(&h_restarted);
    hessenberg
        .slice_mut(s![retained_size.., 0..retained_size])
        .assign(&orthogonalized.subdiagonal);

    continue_block_arnoldi_from_parts(
        operator,
        basis,
        hessenberg,
        vec![retained_size, orthogonalized.rank],
        1,
        target_blocks,
        breakdown_tol,
        matvec_count,
        workspaces,
    )
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ShapeBuilder};
    use num_complex::Complex64;

    use crate::config::{SolverConfig, SpectrumTarget};
    use crate::operator::IdentityOperator;

    use super::solve_block;

    #[test]
    fn identity_operator_converges_for_two_eigenvalues_from_block_start() {
        let operator = IdentityOperator::new(6);
        let start = Array2::from_shape_vec(
            (6, 2).f(),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        let config = SolverConfig {
            nev: 2,
            block_size: 2,
            ncv: 2,
            max_restarts: 2,
            tol: 1.0e-10,
            breakdown_tol: 1.0e-12,
            ritz_inflation: 1.0,
            target: SpectrumTarget::LargestMagnitude,
        };

        let report = solve_block(&operator, start, config, "unit block")
            .expect("identity problem should converge from a block start");

        assert_eq!(report.converged, 2);
        assert!(report.fully_converged);
    }
}
