use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};
use num_complex::Complex64;
use std::os::raw::{c_char, c_int};

#[derive(Debug, Clone, Copy)]
pub enum ZgemvTranspose {
    None,
    ConjugateTranspose,
}

#[derive(Debug, Clone, Copy)]
pub enum ZgemmTranspose {
    None,
    ConjugateTranspose,
}

#[derive(Debug)]
pub struct QrOutput {
    pub q: Array2<Complex64>,
    pub r: Array2<Complex64>,
}

#[derive(Debug)]
pub struct PivotedQrOutput {
    pub q: Array2<Complex64>,
    pub r: Array2<Complex64>,
    pub pivots: Vec<usize>,
    pub rank: usize,
}

#[derive(Debug)]
pub struct SchurOutput {
    /// Eigenvalues.
    pub w: Vec<Complex64>,
    /// Schur form T in LAPACK/Fortran column-major layout.
    pub t: Vec<Complex64>,
    /// Schur vectors Z in LAPACK/Fortran column-major layout.
    pub z: Vec<Complex64>,
}

#[derive(Debug)]
pub enum SchurError {
    NotSquare,
    LapackIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
    InvalidEigenIndex(usize),
}

#[derive(Debug, Default)]
pub struct HouseholderQrWorkspace {
    a: Vec<Complex64>,
    tau: Vec<Complex64>,
    work: Vec<Complex64>,
    zgeqrf_work: Option<CachedWork>,
    zungqr_work: Option<CachedUngqrWork>,
}

#[derive(Debug, Default)]
pub struct PivotedQrWorkspace {
    a: Vec<Complex64>,
    jpvt: Vec<i32>,
    tau: Vec<Complex64>,
    rwork: Vec<f64>,
    work: Vec<Complex64>,
    zgeqp3_work: Option<CachedWork>,
    zungqr_work: Option<CachedUngqrWork>,
}

#[derive(Debug, Default)]
pub struct DenseSchurWorkspace {
    a: Vec<Complex64>,
    w: Vec<Complex64>,
    z: Vec<Complex64>,
    rwork: Vec<f64>,
    bwork: Vec<i32>,
    work: Vec<Complex64>,
    zgees_work: Option<CachedSingleWork>,
}

#[derive(Debug, Default)]
pub struct TrevcWorkspace {
    select: Vec<i32>,
    vr_selected: Vec<Complex64>,
    work: Vec<Complex64>,
    rwork: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedWork {
    m: i32,
    n: i32,
    lwork: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedUngqrWork {
    m: i32,
    n: i32,
    k: i32,
    lwork: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedSingleWork {
    n: i32,
    lwork: i32,
}

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
}

fn view_to_fortran_vec(a: ArrayView2<'_, Complex64>) -> Vec<Complex64> {
    let (rows, cols) = a.dim();
    let mut out = Vec::with_capacity(rows * cols);

    for j in 0..cols {
        for i in 0..rows {
            out.push(a[(i, j)]);
        }
    }

    out
}

#[inline]
fn fortran_view<'a>(
    rows: usize,
    cols: usize,
    data: &'a [Complex64],
) -> Result<ArrayView2<'a, Complex64>, SchurError> {
    if data.len() != rows * cols {
        return Err(SchurError::DimensionMismatch);
    }

    ArrayView2::from_shape((rows, cols).f(), data).map_err(|_| SchurError::DimensionMismatch)
}

fn copy_fortran_slice_to_array(rows: usize, cols: usize, data: &[Complex64]) -> Array2<Complex64> {
    Array2::from_shape_fn((rows, cols).f(), |(row, column)| data[row + column * rows])
}

enum BlasInput {
    Borrowed { ptr: *const Complex64, lda: c_int },
    Owned { data: Vec<Complex64>, lda: c_int },
}

impl BlasInput {
    fn ptr(&self) -> *const Complex64 {
        match self {
            Self::Borrowed { ptr, .. } => *ptr,
            Self::Owned { data, .. } => data.as_ptr(),
        }
    }

    fn lda(&self) -> c_int {
        match self {
            Self::Borrowed { lda, .. } | Self::Owned { lda, .. } => *lda,
        }
    }
}

fn blas_input(matrix: ArrayView2<'_, Complex64>) -> BlasInput {
    let rows = matrix.nrows();
    let strides = matrix.strides();
    let column_major = (rows <= 1 || strides[0] == 1) && strides[1] > 0;

    if column_major {
        let stride = usize::try_from(strides[1]).expect("positive matrix column stride");
        BlasInput::Borrowed {
            ptr: matrix.as_ptr(),
            lda: stride.max(rows).max(1) as c_int,
        }
    } else {
        BlasInput::Owned {
            data: view_to_fortran_vec(matrix),
            lda: rows.max(1) as c_int,
        }
    }
}

fn output_is_column_major(output: &ArrayViewMut2<'_, Complex64>) -> bool {
    let rows = output.nrows();
    let strides = output.strides();

    (rows <= 1 || strides[0] == 1) && strides[1] > 0
}

unsafe extern "C" {
    fn zgemv_(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: *const c_int,
        x: *const Complex64,
        incx: *const c_int,
        beta: *const Complex64,
        y: *mut Complex64,
        incy: *const c_int,
    );

    fn zgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: *const c_int,
        b: *const Complex64,
        ldb: *const c_int,
        beta: *const Complex64,
        c: *mut Complex64,
        ldc: *const c_int,
    );
}

pub fn zgemv(
    trans: ZgemvTranspose,
    matrix: ArrayView2<'_, Complex64>,
    alpha: Complex64,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    let (rows, columns) = matrix.dim();
    let strides = matrix.strides();
    assert!(
        rows <= 1 || strides[0] == 1,
        "zgemv expects column-major matrix storage"
    );
    assert!(
        columns <= 1 || strides[1] == rows as isize,
        "zgemv expects column-major matrix storage"
    );
    let matrix_column_major = matrix
        .as_slice_memory_order()
        .expect("zgemv expects contiguous matrix storage");

    let (trans_char, x_len, y_len) = match trans {
        ZgemvTranspose::None => (b'N' as c_char, columns, rows),
        ZgemvTranspose::ConjugateTranspose => (b'C' as c_char, rows, columns),
    };
    assert_eq!(x.len(), x_len);
    assert_eq!(y.len(), y_len);

    let rows_i = rows as c_int;
    let columns_i = columns as c_int;
    let lda = rows_i;
    let incx = 1 as c_int;
    let incy = 1 as c_int;

    unsafe {
        zgemv_(
            &trans_char,
            &rows_i,
            &columns_i,
            &alpha,
            matrix_column_major.as_ptr(),
            &lda,
            x.as_ptr(),
            &incx,
            &beta,
            y.as_mut_ptr(),
            &incy,
        );
    }
}

pub fn zgemv_into(
    trans: ZgemvTranspose,
    matrix: ArrayView2<'_, Complex64>,
    alpha: Complex64,
    x: ArrayView1<'_, Complex64>,
    beta: Complex64,
    mut y: ArrayViewMut1<'_, Complex64>,
) {
    let (rows, columns) = matrix.dim();
    let (x_len, y_len) = match trans {
        ZgemvTranspose::None => (columns, rows),
        ZgemvTranspose::ConjugateTranspose => (rows, columns),
    };
    assert_eq!(x.len(), x_len);
    assert_eq!(y.len(), y_len);

    if let (Some(x_slice), Some(y_slice)) =
        (x.as_slice_memory_order(), y.as_slice_memory_order_mut())
    {
        zgemv(trans, matrix, alpha, x_slice, beta, y_slice);
        return;
    }

    let x_temp = x.iter().copied().collect::<Vec<_>>();
    let mut y_temp = y.iter().copied().collect::<Vec<_>>();
    zgemv(trans, matrix, alpha, &x_temp, beta, &mut y_temp);
    for (target, value) in y.iter_mut().zip(y_temp.into_iter()) {
        *target = value;
    }
}

/// BLAS ZGEMM: плотное матричное умножение комплексных матриц в column-major буферах.
pub fn zgemm(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let (left_rows, left_cols) = left.dim();
    let (right_rows, right_cols) = right.dim();

    let (m, k_left) = match trans_a {
        ZgemmTranspose::None => (left_rows, left_cols),
        ZgemmTranspose::ConjugateTranspose => (left_cols, left_rows),
    };
    let (k_right, n) = match trans_b {
        ZgemmTranspose::None => (right_rows, right_cols),
        ZgemmTranspose::ConjugateTranspose => (right_cols, right_rows),
    };

    assert_eq!(
        k_left, k_right,
        "zgemm dimension mismatch: left inner dimension {} != right inner dimension {}",
        k_left, k_right,
    );

    let mut output = Array2::zeros((m, n).f());
    zgemm_into(
        trans_a,
        trans_b,
        Complex64::new(1.0, 0.0),
        left,
        right,
        Complex64::ZERO,
        output.view_mut(),
    );
    output
}

/// BLAS ZGEMM: C := alpha * op(A) * op(B) + beta * C.
///
/// Column-major ndarray views are passed directly to BLAS, including full-row
/// slices with a larger leading dimension. Non-column-major inputs fall back to
/// one temporary Fortran copy; non-column-major outputs use one temporary result.
pub fn zgemm_into(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    alpha: Complex64,
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
    beta: Complex64,
    mut output: ArrayViewMut2<'_, Complex64>,
) {
    let (left_rows, left_cols) = left.dim();
    let (right_rows, right_cols) = right.dim();

    let (m, k_left) = match trans_a {
        ZgemmTranspose::None => (left_rows, left_cols),
        ZgemmTranspose::ConjugateTranspose => (left_cols, left_rows),
    };
    let (k_right, n) = match trans_b {
        ZgemmTranspose::None => (right_rows, right_cols),
        ZgemmTranspose::ConjugateTranspose => (right_cols, right_rows),
    };

    assert_eq!(
        k_left, k_right,
        "zgemm dimension mismatch: left inner dimension {} != right inner dimension {}",
        k_left, k_right,
    );
    assert_eq!(
        output.dim(),
        (m, n),
        "zgemm output shape mismatch: expected {}x{}, got {}x{}",
        m,
        n,
        output.nrows(),
        output.ncols(),
    );

    let trans_a_char = match trans_a {
        ZgemmTranspose::None => b'N' as c_char,
        ZgemmTranspose::ConjugateTranspose => b'C' as c_char,
    };
    let trans_b_char = match trans_b {
        ZgemmTranspose::None => b'N' as c_char,
        ZgemmTranspose::ConjugateTranspose => b'C' as c_char,
    };

    if m == 0 || n == 0 {
        return;
    }

    if !output_is_column_major(&output) {
        let mut temp = Array2::zeros((m, n).f());
        temp.assign(&output);
        zgemm_into(trans_a, trans_b, alpha, left, right, beta, temp.view_mut());
        output.assign(&temp);
        return;
    }

    let left_input = blas_input(left);
    let right_input = blas_input(right);
    let m_i = m as c_int;
    let n_i = n as c_int;
    let k_i = k_left as c_int;
    let output_strides = output.strides();
    let output_column_stride =
        usize::try_from(output_strides[1]).expect("positive output column stride");
    let ldc = output_column_stride.max(m).max(1) as c_int;

    unsafe {
        zgemm_(
            &trans_a_char,
            &trans_b_char,
            &m_i,
            &n_i,
            &k_i,
            &alpha,
            left_input.ptr(),
            &left_input.lda(),
            right_input.ptr(),
            &right_input.lda(),
            &beta,
            output.as_mut_ptr(),
            &ldc,
        );
    }
}

impl HouseholderQrWorkspace {
    fn zgeqrf_lwork(&mut self, m: i32, n: i32, lda: i32) -> Result<i32, String> {
        if let Some(cache) = self.zgeqrf_work {
            if cache.m == m && cache.n == n {
                return Ok(cache.lwork);
            }
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            lapack::zgeqrf(
                m,
                n,
                &mut self.a,
                lda,
                &mut self.tau,
                &mut work_query,
                -1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zgeqrf workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n).max(1);
        self.zgeqrf_work = Some(CachedWork { m, n, lwork });
        Ok(lwork)
    }

    fn zungqr_lwork(&mut self, m: i32, n: i32, k: i32, lda: i32) -> Result<i32, String> {
        if let Some(cache) = self.zungqr_work {
            if cache.m == m && cache.n == n && cache.k == k {
                return Ok(cache.lwork);
            }
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            lapack::zungqr(
                m,
                n,
                k,
                &mut self.a,
                lda,
                &self.tau,
                &mut work_query,
                -1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zungqr workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n).max(1);
        self.zungqr_work = Some(CachedUngqrWork { m, n, k, lwork });
        Ok(lwork)
    }

    fn compute_thin_qr(&mut self, matrix: &Array2<Complex64>) -> Result<QrOutput, String> {
        let (rows, columns) = matrix.dim();
        if rows < columns {
            return Err(format!(
                "thin QR expects rows >= columns, got {rows}x{columns}",
            ));
        }

        if columns == 0 {
            return Ok(QrOutput {
                q: Array2::zeros((rows, 0).f()),
                r: Array2::zeros((0, 0).f()),
            });
        }

        let m = rows as i32;
        let n = columns as i32;
        let lda = m.max(1);
        self.a.clear();
        self.a.extend(view_to_fortran_vec(matrix.view()));
        self.tau.resize(columns, zero());

        let lwork = self.zgeqrf_lwork(m, n, lda)?;
        self.work.resize(lwork as usize, zero());
        let mut info = 0;
        unsafe {
            lapack::zgeqrf(
                m,
                n,
                &mut self.a,
                lda,
                &mut self.tau,
                &mut self.work,
                lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zgeqrf failed, info = {info}"));
        }

        let mut r = Array2::zeros((columns, columns).f());
        for column in 0..columns {
            for row in 0..=column {
                r[[row, column]] = self.a[row + column * rows];
            }
        }

        let lwork = self.zungqr_lwork(m, n, n, lda)?;
        self.work.resize(lwork as usize, zero());
        unsafe {
            lapack::zungqr(
                m,
                n,
                n,
                &mut self.a,
                lda,
                &self.tau,
                &mut self.work,
                lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zungqr failed, info = {info}"));
        }

        Ok(QrOutput {
            q: copy_fortran_slice_to_array(rows, columns, &self.a),
            r,
        })
    }
}

/// LAPACK ZGEQRF + ZUNGQR: тонкое Householder QR-разложение A = Q R.
pub fn zgeqrf_qr(matrix: &Array2<Complex64>) -> Result<QrOutput, String> {
    let mut workspace = HouseholderQrWorkspace::default();
    zgeqrf_qr_with_workspace(matrix, &mut workspace)
}

pub fn zgeqrf_qr_with_workspace(
    matrix: &Array2<Complex64>,
    workspace: &mut HouseholderQrWorkspace,
) -> Result<QrOutput, String> {
    workspace.compute_thin_qr(matrix)
}

impl PivotedQrWorkspace {
    fn zgeqp3_lwork(&mut self, m: i32, n: i32, lda: i32) -> Result<i32, String> {
        if let Some(cache) = self.zgeqp3_work {
            if cache.m == m && cache.n == n {
                return Ok(cache.lwork);
            }
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            lapack::zgeqp3(
                m,
                n,
                &mut self.a,
                lda,
                &mut self.jpvt,
                &mut self.tau,
                &mut work_query,
                -1,
                &mut self.rwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zgeqp3 workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n + 1).max(1);
        self.zgeqp3_work = Some(CachedWork { m, n, lwork });
        Ok(lwork)
    }

    fn zungqr_lwork(&mut self, m: i32, n: i32, k: i32, lda: i32) -> Result<i32, String> {
        if let Some(cache) = self.zungqr_work {
            if cache.m == m && cache.n == n && cache.k == k {
                return Ok(cache.lwork);
            }
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            lapack::zungqr(
                m,
                n,
                k,
                &mut self.a,
                lda,
                &self.tau,
                &mut work_query,
                -1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zungqr workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n).max(1);
        self.zungqr_work = Some(CachedUngqrWork { m, n, k, lwork });
        Ok(lwork)
    }

    fn compute_rank_revealing_qr(
        &mut self,
        matrix: &Array2<Complex64>,
        relative_tolerance: f64,
    ) -> Result<PivotedQrOutput, String> {
        let (rows, columns) = matrix.dim();
        if rows < columns {
            return Err(format!(
                "pivoted thin QR expects rows >= columns, got {rows}x{columns}",
            ));
        }

        if columns == 0 {
            return Ok(PivotedQrOutput {
                q: Array2::zeros((rows, 0).f()),
                r: Array2::zeros((0, 0).f()),
                pivots: Vec::new(),
                rank: 0,
            });
        }

        let m = rows as i32;
        let n = columns as i32;
        let lda = m.max(1);
        let min_mn = rows.min(columns);
        self.a.clear();
        self.a.extend(view_to_fortran_vec(matrix.view()));
        self.jpvt.resize(columns, 0);
        self.jpvt.fill(0);
        self.tau.resize(min_mn, zero());
        self.rwork.resize(2 * columns, 0.0);

        let lwork = self.zgeqp3_lwork(m, n, lda)?;
        self.jpvt.fill(0);
        self.work.resize(lwork as usize, zero());
        let mut info = 0;
        unsafe {
            lapack::zgeqp3(
                m,
                n,
                &mut self.a,
                lda,
                &mut self.jpvt,
                &mut self.tau,
                &mut self.work,
                lwork,
                &mut self.rwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zgeqp3 failed, info = {info}"));
        }

        let diagonal = (0..min_mn)
            .map(|index| self.a[index + index * rows].norm())
            .collect::<Vec<_>>();
        let scale = diagonal.first().copied().unwrap_or(0.0);
        let cutoff = relative_tolerance.max(0.0) * rows.max(columns) as f64 * scale;
        let rank = if scale <= f64::EPSILON {
            0
        } else {
            diagonal.iter().take_while(|&&value| value > cutoff).count()
        };

        let mut r = Array2::zeros((rank, columns).f());
        for column in 0..columns {
            let row_limit = rank.min(column + 1);
            for row in 0..row_limit {
                r[[row, column]] = self.a[row + column * rows];
            }
        }

        let pivots = self
            .jpvt
            .iter()
            .map(|&pivot| {
                usize::try_from(pivot)
                    .ok()
                    .and_then(|value| value.checked_sub(1))
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();

        if rank == 0 {
            return Ok(PivotedQrOutput {
                q: Array2::zeros((rows, 0).f()),
                r,
                pivots,
                rank,
            });
        }

        let q_columns = min_mn as i32;
        let lwork = self.zungqr_lwork(m, q_columns, q_columns, lda)?;
        self.work.resize(lwork as usize, zero());
        unsafe {
            lapack::zungqr(
                m,
                q_columns,
                q_columns,
                &mut self.a,
                lda,
                &self.tau,
                &mut self.work,
                lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("zungqr failed, info = {info}"));
        }

        let q_full = copy_fortran_slice_to_array(rows, min_mn, &self.a);
        Ok(PivotedQrOutput {
            q: q_full.slice(ndarray::s![.., 0..rank]).to_owned(),
            r,
            pivots,
            rank,
        })
    }
}

/// LAPACK ZGEQP3 + ZUNGQR: rank-revealing QR с перестановкой столбцов.
pub fn zgeqp3_qr_rank(
    matrix: &Array2<Complex64>,
    relative_tolerance: f64,
) -> Result<PivotedQrOutput, String> {
    let mut workspace = PivotedQrWorkspace::default();
    zgeqp3_qr_rank_with_workspace(matrix, relative_tolerance, &mut workspace)
}

pub fn zgeqp3_qr_rank_with_workspace(
    matrix: &Array2<Complex64>,
    relative_tolerance: f64,
    workspace: &mut PivotedQrWorkspace,
) -> Result<PivotedQrOutput, String> {
    workspace.compute_rank_revealing_qr(matrix, relative_tolerance)
}

impl DenseSchurWorkspace {
    fn zgees_lwork(&mut self, n: i32) -> Result<i32, SchurError> {
        if let Some(cache) = self.zgees_work {
            if cache.n == n {
                return Ok(cache.lwork);
            }
        }

        let mut sdim = 0_i32;
        let mut work_query = [zero(); 1];
        let mut info = 0_i32;
        unsafe {
            lapack::zgees(
                b'V',
                b'N',
                None,
                n,
                &mut self.a,
                n,
                &mut sdim,
                &mut self.w,
                &mut self.z,
                n,
                &mut work_query,
                -1,
                &mut self.rwork,
                &mut self.bwork,
                &mut info,
            );
        }

        if info < 0 {
            return Err(SchurError::LapackIllegalArgument(-info));
        }

        let lwork = (work_query[0].re as i32).max(2 * n).max(1);
        self.zgees_work = Some(CachedSingleWork { n, lwork });
        Ok(lwork)
    }

    fn compute_schur(&mut self, matrix: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
        let (n, m) = matrix.dim();
        if n != m {
            return Err(SchurError::NotSquare);
        }

        if n == 0 {
            return Ok(SchurOutput {
                w: Vec::new(),
                t: Vec::new(),
                z: Vec::new(),
            });
        }

        let n_i = n as i32;
        self.a.clear();
        self.a.extend(view_to_fortran_vec(matrix.view()));
        self.w.resize(n, zero());
        self.z.resize(n * n, zero());
        self.rwork.resize(n, 0.0);
        self.bwork.resize(n, 0);

        let lwork = self.zgees_lwork(n_i)?;
        self.work.resize(lwork as usize, zero());
        let mut sdim = 0_i32;
        let mut info = 0_i32;

        unsafe {
            lapack::zgees(
                b'V',
                b'N',
                None,
                n_i,
                &mut self.a,
                n_i,
                &mut sdim,
                &mut self.w,
                &mut self.z,
                n_i,
                &mut self.work,
                lwork,
                &mut self.rwork,
                &mut self.bwork,
                &mut info,
            );
        }

        if info < 0 {
            return Err(SchurError::LapackIllegalArgument(-info));
        }
        if info > 0 {
            return Err(SchurError::NoConvergence(info));
        }

        Ok(SchurOutput {
            w: self.w.clone(),
            t: self.a.clone(),
            z: self.z.clone(),
        })
    }
}

pub fn zgees_schur(a: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    let mut workspace = DenseSchurWorkspace::default();
    zgees_schur_with_workspace(a, &mut workspace)
}

pub fn zgees_schur_with_workspace(
    a: &Array2<Complex64>,
    workspace: &mut DenseSchurWorkspace,
) -> Result<SchurOutput, SchurError> {
    workspace.compute_schur(a)
}

pub fn ztrevc_right_selected(
    decomposition: &mut SchurOutput,
    indices: &[usize],
    dim: usize,
) -> Result<Array2<Complex64>, SchurError> {
    let mut workspace = TrevcWorkspace::default();
    ztrevc_right_selected_with_workspace(decomposition, indices, dim, &mut workspace)
}

pub fn ztrevc_right_selected_with_workspace(
    decomposition: &mut SchurOutput,
    indices: &[usize],
    dim: usize,
    workspace: &mut TrevcWorkspace,
) -> Result<Array2<Complex64>, SchurError> {
    if decomposition.t.len() != dim * dim || decomposition.z.len() != dim * dim {
        return Err(SchurError::DimensionMismatch);
    }

    for &j in indices {
        if j >= dim {
            return Err(SchurError::InvalidEigenIndex(j));
        }
    }

    if dim == 0 || indices.is_empty() {
        return Ok(Array2::zeros((dim, 0).f()));
    }

    workspace.select.resize(dim, 0);
    workspace.select.fill(0);
    for &j in indices {
        workspace.select[j] = 1;
    }

    let mm = indices.len() as i32;
    let mut m_out = 0_i32;

    let mut vl_dummy = [zero(); 1];
    workspace.vr_selected.resize(dim * indices.len(), zero());
    workspace.work.resize(2 * dim, zero());
    workspace.rwork.resize(dim, 0.0);
    let mut info = 0_i32;

    unsafe {
        lapack::ztrevc(
            b'R',
            b'S',
            &workspace.select,
            dim as i32,
            &mut decomposition.t,
            dim as i32,
            &mut vl_dummy,
            1,
            &mut workspace.vr_selected,
            dim as i32,
            mm,
            &mut m_out,
            &mut workspace.work,
            &mut workspace.rwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
    }

    let x_sel = fortran_view(dim, m_out as usize, &workspace.vr_selected)?;
    let z = fortran_view(dim, dim, &decomposition.z)?;

    // Единственная новая матрица здесь — результат Z * X; считаем его через BLAS.
    Ok(zgemm(ZgemmTranspose::None, ZgemmTranspose::None, z, x_sel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use num_complex::Complex64;

    #[test]
    fn zgemv_wraps_column_major_blas() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::ZERO;
        let a = Array2::from_shape_vec(
            (2, 2).f(),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let x = vec![Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0)];
        let mut y = vec![Complex64::ZERO; 2];
        zgemv(ZgemvTranspose::None, a.view(), one, &x, zero, &mut y);
        assert_eq!(
            y,
            vec![Complex64::new(17.0, 0.0), Complex64::new(39.0, 0.0)]
        );

        let x = vec![Complex64::new(7.0, 0.0), Complex64::new(11.0, 0.0)];
        let mut y = vec![Complex64::ZERO; 2];
        zgemv(
            ZgemvTranspose::ConjugateTranspose,
            a.view(),
            one,
            &x,
            zero,
            &mut y,
        );
        assert_eq!(
            y,
            vec![Complex64::new(40.0, 0.0), Complex64::new(58.0, 0.0)]
        );
    }

    #[test]
    #[should_panic(expected = "zgemv expects column-major matrix storage")]
    fn zgemv_rejects_row_major_matrix() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut y = vec![Complex64::ZERO; 2];
        zgemv(
            ZgemvTranspose::None,
            a.view(),
            Complex64::new(1.0, 0.0),
            &x,
            Complex64::ZERO,
            &mut y,
        );
    }

    #[test]
    fn zgemm_wraps_column_major_blas() {
        let a = Array2::from_shape_vec(
            (2, 2).f(),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (2, 2).f(),
            vec![
                Complex64::new(5.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
        )
        .unwrap();

        let product = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            a.view(),
            b.view(),
        );
        assert_eq!(
            product,
            Array2::from_shape_vec(
                (2, 2).f(),
                vec![
                    Complex64::new(19.0, 0.0),
                    Complex64::new(43.0, 0.0),
                    Complex64::new(22.0, 0.0),
                    Complex64::new(50.0, 0.0),
                ],
            )
            .unwrap(),
        );

        let gram = zgemm(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            a.view(),
            a.view(),
        );
        assert_eq!(
            gram,
            Array2::from_shape_vec(
                (2, 2).f(),
                vec![
                    Complex64::new(10.0, 0.0),
                    Complex64::new(14.0, 0.0),
                    Complex64::new(14.0, 0.0),
                    Complex64::new(20.0, 0.0),
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn zgees_schur_handles_dense_non_hessenberg_matrix() {
        let a = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];

        let out = zgees_schur(&a).unwrap();

        assert_eq!(out.w.len(), 2);
        assert!(
            out.w
                .iter()
                .any(|value| (value - Complex64::new(-0.3722813232690143, 0.0)).norm() < 1.0e-12)
        );
        assert!(
            out.w
                .iter()
                .any(|value| (value - Complex64::new(5.372281323269014, 0.0)).norm() < 1.0e-12)
        );
    }

    #[test]
    fn ztrevc_selected_vectors_have_requested_count() {
        let a = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let mut schur = zgees_schur(&a).unwrap();
        let vectors = ztrevc_right_selected(&mut schur, &[0], 2).unwrap();

        assert_eq!(vectors.dim(), (2, 1));
    }
}
