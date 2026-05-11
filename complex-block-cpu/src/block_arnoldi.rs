//! Процесс Арнольди
use ndarray::{Array2, ArrayViewMut2, DataMut, ShapeBuilder, s};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{OrthogonalizationWorkspaces, orthogonalize_with_reorthogonalization};
use crate::operator::LinearOperator;

/// Ответ процесса
#[derive(Debug, Clone)]
pub struct ArnoldiFactorization {
    pub basis: Array2<Complex64>,
    pub hessenberg: Array2<Complex64>,
    pub performed_steps: usize,
    pub happy_breakdown: bool,
}

pub(crate) struct ArnoldiContinuation {
    pub basis: Array2<Complex64>,
    pub hessenberg: Array2<Complex64>,
    pub start_step: usize,
    pub block_size: usize,
    pub target_steps: usize,
    pub breakdown_tol: f64,
}

use ndarray::{ArrayBase, ArrayView2, Data, Ix2};
pub(crate) trait BlockViewExt<A> {
    /// Количество блоков по строкам.
    fn brows(&self, block_size: usize) -> usize;

    /// Количество блоков по столбцам.
    fn bcols(&self, block_size: usize) -> usize;

    fn block_view(&self, block_size: usize, block_i: usize, block_j: usize) -> ArrayView2<'_, A>;

    fn bcolumn(&self, block_size: usize, block_j: usize) -> ArrayView2<'_, A>;

    fn norm_f(&self) -> f64;
    fn norm_c(&self) -> f64;
}

pub(crate) trait BlockViewMutExt<A>: BlockViewExt<A> {
    fn block_view_mut(
        &mut self,
        block_size: usize,
        block_i: usize,
        block_j: usize,
    ) -> ArrayViewMut2<'_, A>;

    fn bcolumn_mut(&mut self, block_size: usize, block_j: usize) -> ArrayViewMut2<'_, A>;
}

trait ScalarNorm2 {
    fn abs2(&self) -> f64;
}

trait ScalarAbs {
    fn abs_value(&self) -> f64;
}

impl ScalarAbs for f64 {
    fn abs_value(&self) -> f64 {
        self.abs()
    }
}

impl ScalarAbs for Complex64 {
    fn abs_value(&self) -> f64 {
        self.norm()
    }
}

impl ScalarNorm2 for f64 {
    fn abs2(&self) -> f64 {
        self * self
    }
}

impl ScalarNorm2 for Complex64 {
    fn abs2(&self) -> f64 {
        self.norm_sqr()
    }
}

impl<A, S> BlockViewExt<A> for ArrayBase<S, Ix2>
where
    A: ScalarNorm2 + ScalarAbs,
    S: Data<Elem = A>,
{
    fn brows(&self, block_size: usize) -> usize {
        assert!(block_size > 0, "block_size must be > 0");

        self.nrows().div_ceil(block_size)
    }

    fn bcols(&self, block_size: usize) -> usize {
        assert!(block_size > 0, "block_size must be > 0");

        self.ncols().div_ceil(block_size)
    }

    fn block_view(&self, block_size: usize, block_i: usize, block_j: usize) -> ArrayView2<'_, A> {
        assert!(block_size > 0, "block_size must be > 0");

        assert!(block_i < self.brows(block_size), "block_i is out of bounds");
        assert!(block_j < self.bcols(block_size), "block_j is out of bounds");

        let row_start = block_i * block_size;
        let col_start = block_j * block_size;

        let row_end = (row_start + block_size).min(self.nrows());
        let col_end = (col_start + block_size).min(self.ncols());

        self.slice(s![row_start..row_end, col_start..col_end])
    }

    fn bcolumn(&self, block_size: usize, block_j: usize) -> ArrayView2<'_, A> {
        assert!(block_size > 0, "block_size must be > 0");

        assert!(block_j < self.bcols(block_size), "block_j is out of bounds");

        let col_start = block_j * block_size;
        let col_end = (col_start + block_size).min(self.ncols());

        self.slice(s![.., col_start..col_end])
    }

    fn norm_f(&self) -> f64 {
        self.iter().map(|x| x.abs2()).sum::<f64>().sqrt()
    }

    fn norm_c(&self) -> f64 {
        self.rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs_value()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }
}

impl<A, S> BlockViewMutExt<A> for ArrayBase<S, Ix2>
where
    A: ScalarNorm2 + ScalarAbs,
    S: DataMut<Elem = A>,
{
    fn block_view_mut(
        &mut self,
        block_size: usize,
        block_i: usize,
        block_j: usize,
    ) -> ArrayViewMut2<'_, A> {
        assert!(block_size > 0, "block_size must be > 0");

        assert!(block_i < self.brows(block_size), "block_i is out of bounds");
        assert!(block_j < self.bcols(block_size), "block_j is out of bounds");

        let row_start = block_i * block_size;
        let col_start = block_j * block_size;

        let row_end = (row_start + block_size).min(self.nrows());
        let col_end = (col_start + block_size).min(self.ncols());

        self.slice_mut(s![row_start..row_end, col_start..col_end])
    }

    fn bcolumn_mut(&mut self, block_size: usize, block_j: usize) -> ArrayViewMut2<'_, A> {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(block_j < self.bcols(block_size), "block_j is out of bounds");

        let col_start = block_j * block_size;
        let col_end = (col_start + block_size).min(self.ncols());

        self.slice_mut(s![.., col_start..col_end])
    }
}

impl ArnoldiFactorization {
    pub fn square_hessenberg_view(&self, block_size: usize) -> ArrayView2<'_, Complex64> {
        self.hessenberg.slice(s![
            0..self.performed_steps * block_size,
            0..self.performed_steps * block_size
        ])
    }

    pub fn trailing_subdiagonal(&self, block_size: usize) -> Array2<Complex64> {
        if self.happy_breakdown || self.performed_steps == 0 {
            Array2::<Complex64>::zeros((block_size, block_size))
        } else {
            self.hessenberg
                .block_view(block_size, self.performed_steps, self.performed_steps - 1)
                .to_owned()
        }
    }
}

/// Вход в процесс Арнольди, первый прогон при инициализации пространства Крылова
pub fn run_arnoldi(
    operator: &dyn LinearOperator,
    start_block: &Array2<Complex64>,
    steps: usize,
    breakdown_tol: f64,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let block_size = start_block.ncols();

    let mut basis = Array2::zeros((operator.dimension(), (steps + 1) * block_size).f());
    basis.slice_mut(s![.., 0..block_size]).assign(start_block);
    let hessenberg = Array2::zeros(((steps + 1) * block_size, steps * block_size).f());
    continue_arnoldi(
        operator,
        ArnoldiContinuation {
            basis,
            hessenberg,
            start_step: 0,
            block_size,
            target_steps: steps,
            breakdown_tol,
        },
        matvec_count,
    )
}

/// Вход в процесс Арнольди, второй и последующие прогоны, пополняем пространство Крыллова
pub(crate) fn continue_arnoldi(
    operator: &dyn LinearOperator,
    continuation: ArnoldiContinuation,
    matvec_count: &mut usize,
) -> Result<ArnoldiFactorization, IramError> {
    let ArnoldiContinuation {
        mut basis,
        mut hessenberg,
        start_step,
        block_size,
        target_steps,
        breakdown_tol,
    } = continuation;

    if hessenberg.brows(block_size) != target_steps + 1
        || hessenberg.bcols(block_size) != target_steps
    {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation expected Hessenberg block shape {}x{}, got {}x{}",
            target_steps + 1,
            target_steps,
            hessenberg.brows(block_size),
            hessenberg.bcols(block_size),
        )));
    }

    if basis.bcols(block_size) < target_steps + 1 {
        return Err(IramError::InvalidConfig(format!(
            "Arnoldi continuation needs at least {} basis vectors, got {}",
            target_steps + 1,
            basis.ncols(),
        )));
    }

    let mut performed_steps = start_step;
    let mut happy_breakdown = false;

    let mut h_column = Array2::zeros((block_size * target_steps, block_size).f());

    let mut candidate = Array2::zeros((operator.dimension(), block_size).f());
    let mut orthogonalization_workspaces = OrthogonalizationWorkspaces::default();

    for step in start_step..target_steps {
        operator.apply_block_into(basis.bcolumn(block_size, step), candidate.view_mut())?;
        let candidate_old = candidate.norm_f();
        *matvec_count += block_size;

        h_column.fill(Complex64::ZERO);
        let orthogonalized = orthogonalize_with_reorthogonalization(
            candidate.view_mut(),
            basis.slice(s![.., 0..(step + 1) * block_size]),
            h_column.view_mut(),
            candidate_old,
            breakdown_tol,
            block_size,
            &mut orthogonalization_workspaces,
        );

        for row in 0..=step {
            hessenberg
                .block_view_mut(block_size, row, step)
                .assign(&h_column.block_view(block_size, row, 0));
        }

        performed_steps = step + 1;

        if orthogonalized.happy_breakdown {
            happy_breakdown = true;
            hessenberg
                .block_view_mut(block_size, step + 1, step)
                .fill(Complex64::ZERO);
            break;
        }
        hessenberg
            .block_view_mut(block_size, step + 1, step)
            .assign(&orthogonalized.residual);
        basis.bcolumn_mut(block_size, step + 1).assign(&candidate);
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
    use ndarray::{Array2, s};
    use num_complex::Complex64;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::{
        block_arnoldi::BlockViewExt,
        linalg::ops::normalized_random_unitary_matrix,
        operator::{ConvectionDiffusionOperator, IdentityOperator, LinearOperator},
    };

    use super::run_arnoldi;

    #[test]
    fn identity_operator_breaks_down_after_one_step() {
        let operator = IdentityOperator::new(4);
        let start = Array2::<Complex64>::eye(4);
        let mut matvec_count = 0;
        let factorization = run_arnoldi(&operator, &start, 3, 1.0e-14, &mut matvec_count)
            .expect("Arnoldi factorization should succeed");

        assert_eq!(factorization.performed_steps, 1);
        assert!(factorization.happy_breakdown);
        assert_eq!(matvec_count, 4);
    }

    #[test]
    fn arnoldi_factorization_preserves_relation_and_orthogonality() {
        let operator = ConvectionDiffusionOperator::new(10, 100.0);
        let mut rng = StdRng::seed_from_u64(0);
        let block_size = 5;
        let start =
            normalized_random_unitary_matrix(operator.dimension(), block_size, &mut rng).unwrap();
        let target_steps = 3;
        let mut matvec_count = 0;
        let factorization =
            run_arnoldi(&operator, &start, target_steps, 1.0e-15, &mut matvec_count)
                .expect("Arnoldi factorization should succeed");

        assert_eq!(factorization.performed_steps, target_steps);
        assert!(!factorization.happy_breakdown);

        let mut lhs = Array2::zeros((operator.dimension(), (target_steps) * block_size));
        operator
            .apply_block_into(
                factorization
                    .basis
                    .slice(s![.., ..(target_steps) * block_size]),
                lhs.view_mut(),
            )
            .unwrap();

        let rhs = factorization.basis.dot(&factorization.hessenberg);
        let lhs_norm = lhs.norm_f();
        let relation_error = (lhs - rhs).norm_f();
        let relative_relation_error = relation_error / lhs_norm.max(1.0);
        assert!(
            relative_relation_error < 1.0e-12,
            "relative_relation_error={relative_relation_error}, relation_error={relation_error}, lhs_norm={lhs_norm}",
        );

        let total_basis_columns = (target_steps + 1) * block_size;
        let gram = factorization
            .basis
            .t()
            .mapv(|entry| entry.conj())
            .dot(&factorization.basis);
        let identity = Array2::<Complex64>::eye(total_basis_columns);
        let identity_norm = identity.norm_f();
        let orthogonality_error = (gram - identity).norm_f();
        let relative_orthogonality_error = orthogonality_error / identity_norm.max(1.0);
        assert!(
            relative_orthogonality_error < 1.0e-12,
            "relative_orthogonality_error={relative_orthogonality_error}, orthogonality_error={orthogonality_error}",
        );
    }
}
