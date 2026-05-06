//! Блочный процесс Арнольди.
use ndarray::{Array2, ArrayView2, ShapeBuilder, s};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::lapack::{
    HouseholderQrWorkspace, PivotedQrWorkspace, ZgemmTranspose, zgemm, zgemm_into,
    zgeqp3_qr_rank_with_workspace, zgeqrf_qr_with_workspace,
};
use crate::operator::LinearOperator;

#[derive(Debug, Clone)]
pub struct BlockArnoldiFactorization {
    pub basis: Array2<Complex64>,
    pub hessenberg: Array2<Complex64>,
    pub block_sizes: Vec<usize>,
    pub next_block_size: usize,
    pub performed_blocks: usize,
    pub happy_breakdown: bool,
}

#[derive(Debug)]
pub struct BlockOrthogonalization {
    pub q_next: Array2<Complex64>,
    pub subdiagonal: Array2<Complex64>,
    pub rank: usize,
}

#[derive(Debug, Default)]
pub(crate) struct BlockArnoldiWorkspaces {
    pub basis_pivoted_qr: PivotedQrWorkspace,
    residual_qr: HouseholderQrWorkspace,
    reorthogonalized_qr: HouseholderQrWorkspace,
    rank_pivoted_qr: PivotedQrWorkspace,
}

impl BlockArnoldiFactorization {
    pub fn krylov_dimension(&self) -> usize {
        self.block_sizes.iter().sum()
    }

    pub fn total_basis_columns(&self) -> usize {
        self.krylov_dimension() + self.next_block_size
    }

    pub fn square_hessenberg(&self) -> Array2<Complex64> {
        let dim = self.krylov_dimension();
        self.hessenberg.slice(s![0..dim, 0..dim]).to_owned()
    }

    pub fn last_block_range(&self) -> std::ops::Range<usize> {
        let end = self.krylov_dimension();
        let start = end - self.block_sizes.last().copied().unwrap_or(0);
        start..end
    }

    pub fn trailing_coupling(&self) -> Array2<Complex64> {
        if self.happy_breakdown || self.next_block_size == 0 {
            return Array2::zeros((0, self.block_sizes.last().copied().unwrap_or(0)).f());
        }

        let dim = self.krylov_dimension();
        let last = self.last_block_range();
        self.hessenberg
            .slice(s![dim..dim + self.next_block_size, last])
            .to_owned()
    }

    pub fn krylov_basis(&self) -> ArrayView2<'_, Complex64> {
        let dim = self.krylov_dimension();
        self.basis.slice(s![.., 0..dim])
    }

    pub fn next_basis_block(&self) -> Option<ArrayView2<'_, Complex64>> {
        if self.next_block_size == 0 {
            return None;
        }

        let dim = self.krylov_dimension();
        Some(self.basis.slice(s![.., dim..dim + self.next_block_size]))
    }
}

pub fn run_block_arnoldi(
    operator: &dyn LinearOperator,
    start_block: &Array2<Complex64>,
    target_blocks: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<BlockArnoldiFactorization, IramError> {
    let mut workspaces = BlockArnoldiWorkspaces::default();
    run_block_arnoldi_with_workspaces(
        operator,
        start_block,
        target_blocks,
        breakdown_tol,
        matvec_count,
        &mut workspaces,
    )
}

pub(crate) fn run_block_arnoldi_with_workspaces(
    operator: &dyn LinearOperator,
    start_block: &Array2<Complex64>,
    target_blocks: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
    workspaces: &mut BlockArnoldiWorkspaces,
) -> Result<BlockArnoldiFactorization, IramError> {
    if target_blocks == 0 {
        return Err(IramError::InvalidConfig(
            "block Arnoldi needs at least one block iteration".to_string(),
        ));
    }

    if start_block.nrows() != operator.dimension() {
        return Err(IramError::DimensionMismatch {
            expected: operator.dimension(),
            got: start_block.nrows(),
        });
    }

    let start_qr =
        zgeqp3_qr_rank_with_workspace(start_block, breakdown_tol, &mut workspaces.basis_pivoted_qr)
            .map_err(IramError::Spectral)?;
    if start_qr.rank == 0 {
        return Err(IramError::ZeroVector("block Arnoldi start block"));
    }

    let hessenberg = Array2::zeros((start_qr.rank, 0).f());
    continue_block_arnoldi_from_parts(
        operator,
        start_qr.q,
        hessenberg,
        vec![start_qr.rank],
        0,
        target_blocks,
        breakdown_tol,
        matvec_count,
        workspaces,
    )
}

pub(crate) fn continue_block_arnoldi_from_parts(
    operator: &dyn LinearOperator,
    basis: Array2<Complex64>,
    hessenberg: Array2<Complex64>,
    mut block_sizes: Vec<usize>,
    mut completed_blocks: usize,
    target_blocks: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
    workspaces: &mut BlockArnoldiWorkspaces,
) -> Result<BlockArnoldiFactorization, IramError> {
    let mut happy_breakdown = false;
    let max_block_size = block_sizes.iter().copied().max().unwrap_or(0);
    let remaining_blocks = target_blocks.saturating_sub(completed_blocks);
    let capacity_cols = basis
        .ncols()
        .saturating_add(remaining_blocks.saturating_mul(max_block_size))
        .min(operator.dimension())
        .max(basis.ncols())
        .max(hessenberg.nrows())
        .max(hessenberg.ncols());
    let mut basis_store = Array2::zeros((operator.dimension(), capacity_cols).f());
    basis_store
        .slice_mut(s![.., 0..basis.ncols()])
        .assign(&basis);
    let mut hessenberg_store = Array2::zeros((capacity_cols, capacity_cols).f());
    if hessenberg.nrows() > 0 && hessenberg.ncols() > 0 {
        hessenberg_store
            .slice_mut(s![0..hessenberg.nrows(), 0..hessenberg.ncols()])
            .assign(&hessenberg);
    }
    let mut active_basis_cols = basis.ncols();
    let mut active_h_rows = hessenberg.nrows();
    let mut active_h_cols = hessenberg.ncols();

    while completed_blocks < target_blocks {
        if completed_blocks >= block_sizes.len() {
            return Err(IramError::InvalidConfig(format!(
                "block Arnoldi continuation has no block {} to extend",
                completed_blocks + 1,
            )));
        }

        let current_offset = block_sizes[..completed_blocks].iter().sum::<usize>();
        let current_size = block_sizes[completed_blocks];
        let q_total_cols = current_offset + current_size;

        if active_h_rows != q_total_cols || active_h_cols != current_offset {
            return Err(IramError::InvalidConfig(format!(
                "block Hessenberg shape is {}x{}, expected {}x{} before block {}",
                active_h_rows,
                active_h_cols,
                q_total_cols,
                current_offset,
                completed_blocks + 1,
            )));
        }

        // Q_{1:k} = [Q_1, ..., Q_k], W = A Q_k.
        let q_total = basis_store.slice(s![.., 0..q_total_cols]);
        let qk = basis_store.slice(s![.., current_offset..q_total_cols]);
        let mut aqk = Array2::zeros((operator.dimension(), current_size).f());
        operator.apply_block_into(qk, aqk.view_mut())?;
        *matvec_count += current_size;

        // C_k = Q_{1:k}^* W,  W <- W - Q_{1:k} C_k.
        let reference_norms = column_norms(aqk.view());
        let mut column = zgemm(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            q_total,
            aqk.view(),
        );
        let mut residual = aqk;
        zgemm_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::new(-1.0, 0.0),
            q_total,
            column.view(),
            Complex64::new(1.0, 0.0),
            residual.view_mut(),
        );

        // Reorthogonalize W against Q_{1:k}; QR + rank-revealing QR gives
        // Q_{k+1} and the subdiagonal block R_{k+1}.
        let orthogonalized = orthogonalize_block_residual(
            q_total,
            &mut column,
            residual,
            &reference_norms,
            breakdown_tol,
            workspaces,
        )?;

        let new_rows = q_total_cols + orthogonalized.rank;
        let new_cols = q_total_cols;
        if new_rows > capacity_cols {
            return Err(IramError::InvalidConfig(format!(
                "block Arnoldi capacity {capacity_cols} is smaller than the required {new_rows} basis columns",
            )));
        }
        // H(1:k,k) = C_k.
        hessenberg_store
            .slice_mut(s![0..q_total_cols, current_offset..q_total_cols])
            .assign(&column);

        if orthogonalized.rank > 0 {
            // H(k+1,k) = R_{k+1},  Q_total <- [Q_total, Q_{k+1}].
            hessenberg_store
                .slice_mut(s![q_total_cols..new_rows, current_offset..q_total_cols])
                .assign(&orthogonalized.subdiagonal);
            basis_store
                .slice_mut(s![.., q_total_cols..new_rows])
                .assign(&orthogonalized.q_next);
            block_sizes.push(orthogonalized.rank);
            active_basis_cols = new_rows;
        } else {
            active_basis_cols = q_total_cols;
            happy_breakdown = true;
        }

        active_h_rows = new_rows;
        active_h_cols = new_cols;
        completed_blocks += 1;

        if happy_breakdown {
            break;
        }
    }

    if happy_breakdown {
        let basis_cols = block_sizes.iter().sum::<usize>();
        return Ok(BlockArnoldiFactorization {
            basis: basis_store.slice(s![.., 0..basis_cols]).to_owned(),
            hessenberg: hessenberg_store
                .slice(s![0..active_h_rows, 0..active_h_cols])
                .to_owned(),
            block_sizes,
            next_block_size: 0,
            performed_blocks: completed_blocks,
            happy_breakdown: true,
        });
    }

    let performed_blocks = completed_blocks.min(target_blocks);
    let basis_block_sizes = block_sizes[0..performed_blocks].to_vec();
    let basis_cols = basis_block_sizes.iter().sum::<usize>();
    let next_block_size = block_sizes.get(performed_blocks).copied().unwrap_or(0);
    let total_cols = basis_cols + next_block_size;
    debug_assert!(total_cols <= active_basis_cols);

    Ok(BlockArnoldiFactorization {
        basis: basis_store.slice(s![.., 0..total_cols]).to_owned(),
        hessenberg: hessenberg_store
            .slice(s![0..total_cols, 0..basis_cols])
            .to_owned(),
        block_sizes: basis_block_sizes,
        next_block_size,
        performed_blocks,
        happy_breakdown: false,
    })
}

pub(crate) fn orthogonalize_block_residual(
    basis: ArrayView2<'_, Complex64>,
    top_block: &mut Array2<Complex64>,
    mut residual: Array2<Complex64>,
    reference_norms: &[f64],
    breakdown_tol: f64,
    workspaces: &mut BlockArnoldiWorkspaces,
) -> Result<BlockOrthogonalization, IramError> {
    if residual.ncols() == 0 {
        return Ok(BlockOrthogonalization {
            q_next: Array2::zeros((basis.nrows(), 0).f()),
            subdiagonal: Array2::zeros((0, 0).f()),
            rank: 0,
        });
    }

    // C_hat = Q^* W,  C <- C + C_hat,  W <- W - Q C_hat.
    let correction = zgemm(
        ZgemmTranspose::ConjugateTranspose,
        ZgemmTranspose::None,
        basis,
        residual.view(),
    );
    *top_block += &correction;
    zgemm_into(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        Complex64::new(-1.0, 0.0),
        basis,
        correction.view(),
        Complex64::new(1.0, 0.0),
        residual.view_mut(),
    );

    // W = U R.
    let first_qr = zgeqrf_qr_with_workspace(&residual, &mut workspaces.residual_qr)
        .map_err(IramError::Spectral)?;
    let mut u = first_qr.q;
    let mut r = first_qr.r;

    // If Q^* U is not negligible:
    //   C <- C + (Q^* U) R,
    //   U <- U - Q (Q^* U),
    //   U = U_hat R_hat,  R <- R_hat R.
    let c_hat = zgemm(
        ZgemmTranspose::ConjugateTranspose,
        ZgemmTranspose::None,
        basis,
        u.view(),
    );
    if max_column_norm(c_hat.view()) > 1000.0 * f64::EPSILON {
        zgemm_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::new(1.0, 0.0),
            c_hat.view(),
            r.view(),
            Complex64::new(1.0, 0.0),
            top_block.view_mut(),
        );
        let mut reorthogonalized = u;
        zgemm_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::new(-1.0, 0.0),
            basis,
            c_hat.view(),
            Complex64::new(1.0, 0.0),
            reorthogonalized.view_mut(),
        );

        let second_qr =
            zgeqrf_qr_with_workspace(&reorthogonalized, &mut workspaces.reorthogonalized_qr)
                .map_err(IramError::Spectral)?;
        u = second_qr.q;
        r = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            second_qr.r.view(),
            r.view(),
        );
    }

    // D = diag(1 / ||(A Q_k)_i||_2), factor R D by pivoted QR:
    //   R D P = V T,  rank(T) = r.
    let mut scaled_r = r.clone();
    for column in 0..scaled_r.ncols() {
        let scale = reference_norms
            .get(column)
            .copied()
            .filter(|norm| *norm > f64::EPSILON)
            .map(|norm| 1.0 / norm)
            .unwrap_or(1.0);
        for row in 0..scaled_r.nrows() {
            scaled_r[[row, column]] *= Complex64::new(scale, 0.0);
        }
    }

    let pivoted =
        zgeqp3_qr_rank_with_workspace(&scaled_r, breakdown_tol, &mut workspaces.rank_pivoted_qr)
            .map_err(IramError::Spectral)?;
    if pivoted.rank == 0 {
        return Ok(BlockOrthogonalization {
            q_next: Array2::zeros((basis.nrows(), 0).f()),
            subdiagonal: Array2::zeros((0, scaled_r.ncols()).f()),
            rank: 0,
        });
    }

    // Q_{k+1} = U V(:,1:r),
    // R_{k+1} = T(1:r,:) P^T D^{-1}.
    let q_next = zgemm(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        u.view(),
        pivoted.q.view(),
    );
    let subdiagonal = unpivot_and_unscale(&pivoted.r, &pivoted.pivots, reference_norms);

    Ok(BlockOrthogonalization {
        q_next,
        subdiagonal,
        rank: pivoted.rank,
    })
}

pub(crate) fn append_columns(
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    assert_eq!(left.nrows(), right.nrows());

    let mut result = Array2::zeros((left.nrows(), left.ncols() + right.ncols()).f());
    result.slice_mut(s![.., 0..left.ncols()]).assign(&left);
    result.slice_mut(s![.., left.ncols()..]).assign(&right);
    result
}

pub(crate) fn column_norms(matrix: ArrayView2<'_, Complex64>) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|column| {
            matrix
                .column(column)
                .iter()
                .map(|entry| entry.norm_sqr())
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn frobenius_norm(matrix: ArrayView2<'_, Complex64>) -> f64 {
    matrix
        .iter()
        .map(|entry| entry.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn max_column_norm(matrix: ArrayView2<'_, Complex64>) -> f64 {
    column_norms(matrix).into_iter().fold(0.0, f64::max)
}

fn unpivot_and_unscale(
    pivoted_r: &Array2<Complex64>,
    pivots: &[usize],
    reference_norms: &[f64],
) -> Array2<Complex64> {
    let rank = pivoted_r.nrows();
    let columns = pivoted_r.ncols();
    let mut result = Array2::zeros((rank, columns).f());

    for pivoted_column in 0..columns {
        let original_column = pivots
            .get(pivoted_column)
            .copied()
            .unwrap_or(pivoted_column);
        if original_column >= columns {
            continue;
        }
        let scale = reference_norms
            .get(original_column)
            .copied()
            .filter(|norm| *norm > f64::EPSILON)
            .unwrap_or(1.0);

        for row in 0..rank {
            result[[row, original_column]] =
                pivoted_r[[row, pivoted_column]] * Complex64::new(scale, 0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ShapeBuilder, s};
    use num_complex::Complex64;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::linalg::lapack::{ZgemmTranspose, zgemm};
    use crate::linalg::ops::normalized_random_unitary_matrix;
    use crate::operator::{ConvectionDiffusionOperator, IdentityOperator, LinearOperator};

    use super::{frobenius_norm, run_block_arnoldi};

    #[test]
    fn identity_operator_breaks_down_after_one_block() {
        let operator = IdentityOperator::new(4);
        let start = Array2::from_shape_vec(
            (4, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        let mut matvec_count = 0;
        let factorization = run_block_arnoldi(&operator, &start, 3, 1.0e-12, &mut matvec_count)
            .expect("block Arnoldi should handle an invariant start block");

        assert_eq!(factorization.performed_blocks, 1);
        assert_eq!(factorization.krylov_dimension(), 2);
        assert!(factorization.happy_breakdown);
        assert_eq!(matvec_count, 2);
    }

    #[test]
    fn block_factorization_satisfies_arnoldi_relation() {
        let operator = ConvectionDiffusionOperator::new(4, 0.0);
        let mut rng = StdRng::seed_from_u64(0);
        let start = normalized_random_unitary_matrix(operator.dimension(), 2, &mut rng).unwrap();
        let mut matvec_count = 0;
        let factorization =
            run_block_arnoldi(&operator, &start, 4, 1.0e-12, &mut matvec_count).unwrap();

        let k = factorization.krylov_dimension();
        let total = factorization.total_basis_columns();
        let q = factorization.basis.slice(s![.., 0..k]).to_owned();
        let q_bar = factorization.basis.slice(s![.., 0..total]).to_owned();
        let h_bar = factorization
            .hessenberg
            .slice(s![0..total, 0..k])
            .to_owned();

        let mut aq = Array2::zeros((operator.dimension(), k).f());
        operator.apply_block_into(q.view(), aq.view_mut()).unwrap();
        let qh = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            q_bar.view(),
            h_bar.view(),
        );
        let relation_error = frobenius_norm((aq - qh).view());
        assert!(relation_error < 1.0e-8, "relation_error={relation_error}");

        let gram = zgemm(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            q_bar.view(),
            q_bar.view(),
        );
        let mut identity = Array2::zeros((total, total).f());
        for i in 0..total {
            identity[[i, i]] = Complex64::new(1.0, 0.0);
        }
        let orthogonality_error = frobenius_norm((gram - identity).view());
        assert!(
            orthogonality_error < 1.0e-8,
            "orthogonality_error={orthogonality_error}"
        );
    }
}
