//! Операции

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder, s};
use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::block_arnoldi::BlockViewExt;
use crate::error::IramError;
use crate::linalg::magma::{self, PivotedQrOutput, QrOutput, SchurError, SchurOutput};

pub(crate) use crate::linalg::magma::{
    DenseSchurWorkspace, HouseholderQrWorkspace, TrevcWorkspace, ZgemmTranspose, ZgemvTranspose,
};

const REORTHOGONALIZATION_THRESHOLD: f64 = f64::EPSILON * 1000.0;
// const ORTHOGONALIZATION_LOSS_THRESHOLD: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone)]
pub(crate) struct OrthogonalizedBlock {
    pub(crate) residual: Array2<Complex64>,
    pub(crate) happy_breakdown: bool,
}

#[derive(Debug, Default)]
pub(crate) struct OrthogonalizationWorkspaces {
    residual_qr: HouseholderQrWorkspace,
    reorthogonalized_qr: HouseholderQrWorkspace,
}

pub(crate) fn matvec_into(
    trans: ZgemvTranspose,
    matrix: ArrayView2<'_, Complex64>,
    alpha: Complex64,
    x: ArrayView1<'_, Complex64>,
    beta: Complex64,
    y: ArrayViewMut1<'_, Complex64>,
) {
    magma::zgemv_into(trans, matrix, alpha, x, beta, y);
}

pub(crate) fn matmul(
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let mut output = Array2::zeros((left.nrows(), right.ncols()).f());
    matmul_into(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        Complex64::ONE,
        left,
        right,
        Complex64::ZERO,
        output.view_mut(),
    );
    output
}

pub(crate) fn matmul_conj_left(
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let mut output = Array2::zeros((left.ncols(), right.ncols()).f());
    matmul_into(
        ZgemmTranspose::ConjugateTranspose,
        ZgemmTranspose::None,
        Complex64::ONE,
        left,
        right,
        Complex64::ZERO,
        output.view_mut(),
    );
    output
}

pub(crate) fn matmul_into(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    alpha: Complex64,
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
    beta: Complex64,
    output: ArrayViewMut2<'_, Complex64>,
) {
    magma::zgemm_into(trans_a, trans_b, alpha, left, right, beta, output);
}

pub(crate) fn householder_qr_with_workspace(
    matrix: ArrayView2<'_, Complex64>,
    workspace: &mut HouseholderQrWorkspace,
) -> Result<QrOutput, String> {
    magma::zgeqrf_qr_with_workspace(matrix, workspace)
}

pub(crate) fn householder_qr_owned_fortran_with_workspace(
    matrix: Array2<Complex64>,
    workspace: &mut HouseholderQrWorkspace,
) -> Result<QrOutput, String> {
    magma::zgeqrf_qr_owned_fortran_with_workspace(matrix, workspace)
}

pub(crate) fn pivoted_qr_rank(
    matrix: &Array2<Complex64>,
    relative_tolerance: f64,
) -> Result<PivotedQrOutput, String> {
    magma::zgeqp3_qr_rank(matrix, relative_tolerance)
}

pub(crate) fn dense_schur_with_workspace(
    matrix: ArrayView2<'_, Complex64>,
    workspace: &mut DenseSchurWorkspace,
) -> Result<SchurOutput, SchurError> {
    magma::zgees_schur_with_workspace(matrix, workspace)
}

pub(crate) fn selected_ritz_vectors_with_workspace(
    decomposition: &mut SchurOutput,
    ritz_indices: &[usize],
    dim: usize,
    workspace: &mut TrevcWorkspace,
) -> Result<Array2<Complex64>, SchurError> {
    magma::ztrevc_right_selected_with_workspace(decomposition, ritz_indices, dim, workspace)
}

/// Генерация случайной матрицы с ортонормальными столбцами.
pub fn normalized_random_unitary_matrix<R>(
    dimension: usize,
    block_size: usize,
    rng: &mut R,
) -> Result<Array2<Complex64>, IramError>
where
    R: Rng + ?Sized,
{
    if block_size == 0 {
        return Err(IramError::InvalidConfig(
            "block_size must be strictly positive".to_string(),
        ));
    }

    if block_size > dimension {
        return Err(IramError::InvalidConfig(format!(
            "block_size ({block_size}) cannot exceed the operator dimension ({dimension})",
        )));
    }

    let matrix = Array2::from_shape_fn((dimension, block_size).f(), |_| {
        Complex64::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0))
    });
    let qr = pivoted_qr_rank(&matrix, f64::EPSILON).map_err(IramError::Spectral)?;

    if qr.rank != block_size {
        return Err(IramError::Spectral(format!(
            "random start block has numerical rank {}, expected {}",
            qr.rank, block_size,
        )));
    }

    Ok(qr.q)
}

pub(crate) fn orthogonalize_with_reorthogonalization<'a>(
    mut candidate: ArrayViewMut2<'a, Complex64>,
    basis: ArrayView2<'a, Complex64>,
    mut h_column: ArrayViewMut2<'a, Complex64>,
    reference_norm: f64,
    breakdown_tol: f64,
    block_size: usize,
    workspaces: &mut OrthogonalizationWorkspaces,
) -> OrthogonalizedBlock {
    let basis_columns = basis.ncols();
    assert_eq!(candidate.nrows(), basis.nrows());
    assert!(h_column.nrows() >= basis_columns);
    assert_eq!(h_column.ncols(), block_size);
    let mut h_column = h_column.slice_mut(s![0..basis_columns, ..]);

    /*
        candidate block: X ∈ C^{m × block_size}
        basis:          V ∈ C^{m × n}
        h_column:       H ∈ C^{n × block_size}

        Двойная ортогонализация:

            P = Vᴴ X
            H += P
            X -= V P
    */

    let mut projection = Array2::<Complex64>::zeros((basis_columns, block_size).f());

    project_candidate(
        basis,
        candidate.view_mut(),
        h_column.view_mut(),
        projection.view_mut(),
    );

    // let first_pass_norm = candidate.norm_f();
    // if first_pass_norm <= ORTHOGONALIZATION_LOSS_THRESHOLD * reference_norm {
    //     project_candidate(
    //         basis,
    //         candidate.view_mut(),
    //         h_column.view_mut(),
    //         projection.view_mut(),
    //     );
    // }

    let residual_qr =
        householder_qr_with_workspace(candidate.view(), &mut workspaces.residual_qr).unwrap();

    let mut residual = residual_qr.r.to_owned();

    if is_numerical_breakdown(&residual.view(), reference_norm, breakdown_tol) {
        return OrthogonalizedBlock {
            residual,
            happy_breakdown: true,
        };
    }
    candidate.assign(&residual_qr.q);

    /*
        Проверяем потерю ортогональности после нормировки:

            P = Vᴴ Q

        Если P достаточно велико:

            H += residual_norm * P
            Q -= V P
            Q /= ||Q||
    */

    projection.fill(Complex64::ZERO);

    {
        matmul_into(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            Complex64::ONE,
            basis,
            candidate.view(),
            Complex64::ZERO,
            projection.view_mut(),
        );
    }

    let correction_norm = projection.norm_c();

    if correction_norm > REORTHOGONALIZATION_THRESHOLD {
        let mut h_correction = Array2::zeros((basis_columns, block_size).f());
        matmul_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::ONE,
            projection.view(),
            residual.view(),
            Complex64::ZERO,
            h_correction.view_mut(),
        );

        for (h, &s) in h_column.iter_mut().zip(h_correction.iter()) {
            *h += s;
        }

        {
            matmul_into(
                ZgemmTranspose::None,
                ZgemmTranspose::None,
                -Complex64::ONE,
                basis,
                projection.view(),
                Complex64::ONE,
                candidate.view_mut(),
            );
        }

        let residual_reqr =
            householder_qr_with_workspace(candidate.view(), &mut workspaces.reorthogonalized_qr)
                .unwrap();

        residual = residual_reqr.r.dot(&residual);

        if is_numerical_breakdown(&residual.view(), reference_norm, breakdown_tol) {
            return OrthogonalizedBlock {
                residual,
                happy_breakdown: true,
            };
        }

        candidate.assign(&residual_reqr.q);
    }

    OrthogonalizedBlock {
        residual,
        happy_breakdown: false,
    }
}

fn project_candidate(
    basis: ArrayView2<'_, Complex64>,
    mut candidate: ArrayViewMut2<'_, Complex64>,
    mut h_column: ArrayViewMut2<'_, Complex64>,
    mut projection: ArrayViewMut2<'_, Complex64>,
) {
    projection.fill(Complex64::ZERO);

    matmul_into(
        ZgemmTranspose::ConjugateTranspose,
        ZgemmTranspose::None,
        Complex64::ONE,
        basis,
        candidate.view(),
        Complex64::ZERO,
        projection.view_mut(),
    );

    for (h, &s) in h_column.iter_mut().zip(projection.iter()) {
        *h += s;
    }

    matmul_into(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        -Complex64::ONE,
        basis,
        projection.view(),
        Complex64::ONE,
        candidate.view_mut(),
    );
}

fn is_numerical_breakdown(
    residual_norm: &ArrayView2<Complex64>,
    reference_norm: f64,
    tolerance: f64,
) -> bool {
    residual_norm.norm_f() <= tolerance * reference_norm
}
