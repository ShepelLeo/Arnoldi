use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};
use num_complex::Complex64;
use std::ffi::c_void;
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Once;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ZgemvTranspose {
    None,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ZgemmTranspose {
    None,
    ConjugateTranspose,
}

#[derive(Debug)]
pub(crate) struct QrOutput {
    pub q: Array2<Complex64>,
    pub r: Array2<Complex64>,
}

#[derive(Debug)]
pub(crate) struct PivotedQrOutput {
    pub q: Array2<Complex64>,
    pub rank: usize,
}

#[derive(Debug)]
pub(crate) struct SchurOutput {
    /// Eigenvalues.
    pub w: Vec<Complex64>,
    /// Work matrix in Fortran column-major layout.
    pub t: Vec<Complex64>,
    /// Right eigenvectors in Fortran column-major layout.
    pub z: Vec<Complex64>,
}

#[derive(Debug)]
pub(crate) enum SchurError {
    NotSquare,
    MagmaIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
    InvalidEigenIndex(usize),
}

impl fmt::Display for SchurError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare => write!(formatter, "Schur decomposition expects a square matrix"),
            Self::MagmaIllegalArgument(argument) => {
                write!(formatter, "MAGMA rejected argument {argument}")
            }
            Self::NoConvergence(info) => {
                write!(
                    formatter,
                    "Schur decomposition did not converge, info = {info}"
                )
            }
            Self::DimensionMismatch => write!(formatter, "matrix dimension mismatch"),
            Self::InvalidEigenIndex(index) => write!(formatter, "invalid eigenvalue index {index}"),
        }
    }
}

type MagmaQueue = *mut c_void;

const MAGMA_NO_TRANS: c_int = 111;
const MAGMA_CONJ_TRANS: c_int = 113;
const MAGMA_NO_VEC: c_int = 301;
const MAGMA_VEC: c_int = 302;
const MAGMA_SUCCESS: c_int = 0;
const MAGMA_FUNC: *const c_char = b"rust\0".as_ptr().cast();
const MAGMA_FILE: *const c_char = b"src/linalg/magma.rs\0".as_ptr().cast();

static MAGMA_INIT: Once = Once::new();

unsafe extern "C" {
    fn magma_init() -> c_int;
    fn magma_getdevice(device: *mut c_int);
    fn magma_queue_create_internal(
        device: c_int,
        queue_ptr: *mut MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );
    fn magma_queue_destroy_internal(
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );
    fn magma_malloc(ptr_ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn magma_free_internal(
        ptr: *mut c_void,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    ) -> c_int;
    fn magma_zsetmatrix(
        m: c_int,
        n: c_int,
        h_a: *const Complex64,
        lda: c_int,
        d_a: *mut Complex64,
        ldda: c_int,
        queue: MagmaQueue,
    );
    fn magma_zgetmatrix(
        m: c_int,
        n: c_int,
        d_a: *const Complex64,
        ldda: c_int,
        h_a: *mut Complex64,
        lda: c_int,
        queue: MagmaQueue,
    );
    fn magma_zsetvector(
        n: c_int,
        h_x: *const Complex64,
        incx: c_int,
        d_x: *mut Complex64,
        incx_dev: c_int,
        queue: MagmaQueue,
    );
    fn magma_zgetvector(
        n: c_int,
        d_x: *const Complex64,
        incx_dev: c_int,
        h_x: *mut Complex64,
        incx: c_int,
        queue: MagmaQueue,
    );
    fn magma_zgemm(
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: Complex64,
        d_a: *const Complex64,
        ldda: c_int,
        d_b: *const Complex64,
        lddb: c_int,
        beta: Complex64,
        d_c: *mut Complex64,
        lddc: c_int,
        queue: MagmaQueue,
    );
    fn magma_zgemv(
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: Complex64,
        d_a: *const Complex64,
        ldda: c_int,
        d_x: *const Complex64,
        incx: c_int,
        beta: Complex64,
        d_y: *mut Complex64,
        incy: c_int,
        queue: MagmaQueue,
    );
    fn magma_zgeqrf(
        m: c_int,
        n: c_int,
        a: *mut Complex64,
        lda: c_int,
        tau: *mut Complex64,
        work: *mut Complex64,
        lwork: c_int,
        info: *mut c_int,
    ) -> c_int;
    fn magma_zungqr2(
        m: c_int,
        n: c_int,
        k: c_int,
        a: *mut Complex64,
        lda: c_int,
        tau: *mut Complex64,
        info: *mut c_int,
    ) -> c_int;
    fn magma_zgeqp3(
        m: c_int,
        n: c_int,
        a: *mut Complex64,
        lda: c_int,
        jpvt: *mut c_int,
        tau: *mut Complex64,
        work: *mut Complex64,
        lwork: c_int,
        rwork: *mut f64,
        info: *mut c_int,
    ) -> c_int;
    fn magma_zgeev(
        jobvl: c_int,
        jobvr: c_int,
        n: c_int,
        a: *mut Complex64,
        lda: c_int,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: c_int,
        vr: *mut Complex64,
        ldvr: c_int,
        work: *mut Complex64,
        lwork: c_int,
        rwork: *mut f64,
        info: *mut c_int,
    ) -> c_int;
}

struct Queue {
    raw: MagmaQueue,
}

impl Queue {
    fn new() -> Self {
        ensure_magma_initialized();
        let mut device = 0;
        let mut raw = ptr::null_mut();
        unsafe {
            magma_getdevice(&mut device);
            magma_queue_create_internal(device, &mut raw, MAGMA_FUNC, MAGMA_FILE, line!() as c_int);
        }
        assert!(!raw.is_null(), "magma_queue_create returned a null queue");
        Self { raw }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            magma_queue_destroy_internal(self.raw, MAGMA_FUNC, MAGMA_FILE, line!() as c_int);
        }
    }
}

struct DeviceBuffer {
    ptr: *mut Complex64,
}

impl DeviceBuffer {
    fn new(len: usize) -> Self {
        ensure_magma_initialized();
        let mut ptr = ptr::null_mut();
        let bytes = len
            .checked_mul(std::mem::size_of::<Complex64>())
            .expect("MAGMA allocation size overflow");
        let status = unsafe { magma_malloc(&mut ptr, bytes) };
        assert_eq!(status, MAGMA_SUCCESS, "magma_malloc failed with status {status}");
        Self {
            ptr: ptr.cast::<Complex64>(),
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            magma_free_internal(self.ptr.cast::<c_void>(), MAGMA_FUNC, MAGMA_FILE, line!() as c_int);
        }
    }
}

fn ensure_magma_initialized() {
    MAGMA_INIT.call_once(|| {
        let status = unsafe { magma_init() };
        assert_eq!(status, MAGMA_SUCCESS, "magma_init failed with status {status}");
    });
}

#[derive(Debug, Default)]
pub(crate) struct HouseholderQrWorkspace {
    a: Vec<Complex64>,
    tau: Vec<Complex64>,
    work: Vec<Complex64>,
    zgeqrf_work: Option<CachedWork>,
}

#[derive(Debug, Default)]
pub(crate) struct PivotedQrWorkspace {
    a: Vec<Complex64>,
    jpvt: Vec<i32>,
    tau: Vec<Complex64>,
    rwork: Vec<f64>,
    work: Vec<Complex64>,
    zgeqp3_work: Option<CachedWork>,
}

#[derive(Debug, Default)]
pub(crate) struct DenseSchurWorkspace {
    a: Vec<Complex64>,
    w: Vec<Complex64>,
    z: Vec<Complex64>,
    rwork: Vec<f64>,
    bwork: Vec<i32>,
    work: Vec<Complex64>,
    zgees_work: Option<CachedSingleWork>,
}

#[derive(Debug, Default)]
pub(crate) struct TrevcWorkspace {
    select: Vec<i32>,
    vr_selected: Vec<Complex64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedWork {
    m: i32,
    n: i32,
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

fn copy_view_to_fortran_buffer(a: ArrayView2<'_, Complex64>, out: &mut Vec<Complex64>) {
    let (rows, cols) = a.dim();
    let len = rows * cols;
    out.clear();

    let strides = a.strides();
    let compact_fortran =
        (rows <= 1 || strides[0] == 1) && (cols <= 1 || strides[1] == rows as isize);
    if compact_fortran && let Some(slice) = a.as_slice_memory_order() {
        out.extend_from_slice(slice);
        return;
    }

    out.resize(len, zero());
    for column in 0..cols {
        let offset = column * rows;
        for row in 0..rows {
            out[offset + row] = a[(row, column)];
        }
    }
}

fn move_array_to_fortran_buffer(a: Array2<Complex64>, out: &mut Vec<Complex64>) {
    let (rows, cols) = a.dim();
    let len = rows * cols;
    let strides = a.strides();
    let compact_fortran =
        (rows <= 1 || strides[0] == 1) && (cols <= 1 || strides[1] == rows as isize);

    if compact_fortran && a.as_slice_memory_order().is_some() {
        let (data, offset) = a.into_raw_vec_and_offset();
        let offset = offset.unwrap_or(0);
        if offset == 0 && data.len() == len {
            *out = data;
        } else {
            out.clear();
            out.extend_from_slice(&data[offset..offset + len]);
        }
        return;
    }

    copy_view_to_fortran_buffer(a.view(), out);
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

fn copy_fortran_columns_to_array(
    rows: usize,
    cols: usize,
    data: &[Complex64],
) -> Array2<Complex64> {
    let len = rows * cols;
    assert!(
        data.len() >= len,
        "Fortran buffer has {} entries, expected at least {}",
        data.len(),
        len
    );

    let mut out = Array2::zeros((rows, cols).f());
    out.as_slice_memory_order_mut()
        .expect("Fortran-shaped Array2 must be contiguous")
        .copy_from_slice(&data[..len]);
    out
}

enum MatrixInput {
    Borrowed { ptr: *const Complex64, lda: c_int },
    Owned { data: Vec<Complex64>, lda: c_int },
}

impl MatrixInput {
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

fn magma_input(matrix: ArrayView2<'_, Complex64>) -> MatrixInput {
    let rows = matrix.nrows();
    let strides = matrix.strides();
    let column_major = (rows <= 1 || strides[0] == 1) && strides[1] > 0;

    if column_major {
        let stride = usize::try_from(strides[1]).expect("positive matrix column stride");
        MatrixInput::Borrowed {
            ptr: matrix.as_ptr(),
            lda: stride.max(rows).max(1) as c_int,
        }
    } else {
        MatrixInput::Owned {
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

fn copy_view_into_view(mut output: ArrayViewMut2<'_, Complex64>, input: ArrayView2<'_, Complex64>) {
    assert_eq!(output.dim(), input.dim());
    if let (Some(output_slice), Some(input_slice)) = (
        output.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        output_slice.copy_from_slice(input_slice);
        return;
    }

    output.zip_mut_with(&input, |target, value| *target = *value);
}

fn magma_trans(trans: ZgemmTranspose) -> c_int {
    match trans {
        ZgemmTranspose::None => MAGMA_NO_TRANS,
        ZgemmTranspose::ConjugateTranspose => MAGMA_CONJ_TRANS,
    }
}

fn magma_trans_from_zgemv(trans: ZgemvTranspose) -> c_int {
    match trans {
        ZgemvTranspose::None => MAGMA_NO_TRANS,
    }
}

fn scale_slice(values: &mut [Complex64], beta: Complex64) {
    if beta == Complex64::ZERO {
        values.fill(Complex64::ZERO);
    } else {
        values.iter_mut().for_each(|value| *value *= beta);
    }
}

pub(crate) fn zgemv(
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

    let (x_len, y_len) = match trans {
        ZgemvTranspose::None => (columns, rows),
    };
    assert_eq!(x.len(), x_len);
    assert_eq!(y.len(), y_len);

    if y_len == 0 {
        return;
    }
    if x_len == 0 {
        scale_slice(y, beta);
        return;
    }

    let rows_i = rows as c_int;
    let columns_i = columns as c_int;
    let lda = rows_i.max(1);
    let incx = 1 as c_int;
    let incy = 1 as c_int;
    let trans = magma_trans_from_zgemv(trans);

    let queue = Queue::new();
    let d_a = DeviceBuffer::new(matrix_column_major.len());
    let d_x = DeviceBuffer::new(x.len());
    let d_y = DeviceBuffer::new(y.len());

    unsafe {
        magma_zsetmatrix(
            rows_i,
            columns_i,
            matrix_column_major.as_ptr(),
            lda,
            d_a.ptr,
            lda,
            queue.raw,
        );
        magma_zsetvector(x_len as c_int, x.as_ptr(), incx, d_x.ptr, incx, queue.raw);
        magma_zsetvector(y_len as c_int, y.as_ptr(), incy, d_y.ptr, incy, queue.raw);
        magma_zgemv(
            trans,
            rows_i,
            columns_i,
            alpha,
            d_a.ptr,
            lda,
            d_x.ptr,
            incx,
            beta,
            d_y.ptr,
            incy,
            queue.raw,
        );
        magma_zgetvector(y_len as c_int, d_y.ptr, incy, y.as_mut_ptr(), incy, queue.raw);
    }
}

pub(crate) fn zgemv_into(
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

/// MAGMA ZGEMM: C := alpha * op(A) * op(B) + beta * C.
///
/// Column-major ndarray views are copied directly to the device, including full-row
/// slices with a larger leading dimension. Non-column-major inputs fall back to
/// one temporary Fortran copy; non-column-major outputs use one temporary result.
pub(crate) fn zgemm_into(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    alpha: Complex64,
    left: ArrayView2<'_, Complex64>,
    right: ArrayView2<'_, Complex64>,
    beta: Complex64,
    mut output: ArrayViewMut2<'_, Complex64>,
) {
    let trans_a_kind = trans_a;
    let trans_b_kind = trans_b;
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

    let trans_a = magma_trans(trans_a_kind);
    let trans_b = magma_trans(trans_b_kind);

    if m == 0 || n == 0 {
        return;
    }
    if k_left == 0 {
        output.map_inplace(|value| *value *= beta);
        return;
    }

    if !output_is_column_major(&output) {
        let mut temp = Array2::zeros((m, n).f());
        copy_view_into_view(temp.view_mut(), output.view());
        zgemm_into(
            trans_a_kind,
            trans_b_kind,
            alpha,
            left,
            right,
            beta,
            temp.view_mut(),
        );
        copy_view_into_view(output, temp.view());
        return;
    }

    let left_input = magma_input(left);
    let right_input = magma_input(right);
    let m_i = m as c_int;
    let n_i = n as c_int;
    let k_i = k_left as c_int;
    let left_rows_i = left_rows as c_int;
    let left_cols_i = left_cols as c_int;
    let right_rows_i = right_rows as c_int;
    let right_cols_i = right_cols as c_int;
    let left_ldd = left_rows_i.max(1);
    let right_ldd = right_rows_i.max(1);
    let output_ldd = m_i.max(1);
    let output_strides = output.strides();
    let output_column_stride =
        usize::try_from(output_strides[1]).expect("positive output column stride");
    let ldc = output_column_stride.max(m).max(1) as c_int;

    let queue = Queue::new();
    let d_left = DeviceBuffer::new(left_rows * left_cols);
    let d_right = DeviceBuffer::new(right_rows * right_cols);
    let d_output = DeviceBuffer::new(m * n);

    unsafe {
        magma_zsetmatrix(
            left_rows_i,
            left_cols_i,
            left_input.ptr(),
            left_input.lda(),
            d_left.ptr,
            left_ldd,
            queue.raw,
        );
        magma_zsetmatrix(
            right_rows_i,
            right_cols_i,
            right_input.ptr(),
            right_input.lda(),
            d_right.ptr,
            right_ldd,
            queue.raw,
        );
        magma_zsetmatrix(
            m_i,
            n_i,
            output.as_ptr(),
            ldc,
            d_output.ptr,
            output_ldd,
            queue.raw,
        );
        magma_zgemm(
            trans_a,
            trans_b,
            m_i,
            n_i,
            k_i,
            alpha,
            d_left.ptr,
            left_ldd,
            d_right.ptr,
            right_ldd,
            beta,
            d_output.ptr,
            output_ldd,
            queue.raw,
        );
        magma_zgetmatrix(
            m_i,
            n_i,
            d_output.ptr,
            output_ldd,
            output.as_mut_ptr(),
            ldc,
            queue.raw,
        );
    }
}

impl HouseholderQrWorkspace {
    fn zgeqrf_lwork(&mut self, m: i32, n: i32, lda: i32) -> Result<i32, String> {
        ensure_magma_initialized();
        if let Some(cache) = self.zgeqrf_work
            && cache.m == m
            && cache.n == n
        {
            return Ok(cache.lwork);
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            magma_zgeqrf(
                m,
                n,
                self.a.as_mut_ptr(),
                lda,
                self.tau.as_mut_ptr(),
                work_query.as_mut_ptr(),
                -1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zgeqrf workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n).max(1);
        self.zgeqrf_work = Some(CachedWork { m, n, lwork });
        Ok(lwork)
    }

    fn finish_thin_qr(&mut self, rows: usize, columns: usize) -> Result<QrOutput, String> {
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
        self.tau.resize(columns, zero());

        let lwork = self.zgeqrf_lwork(m, n, lda)?;
        self.work.resize(lwork as usize, zero());
        let mut info = 0;
        unsafe {
            magma_zgeqrf(
                m,
                n,
                self.a.as_mut_ptr(),
                lda,
                self.tau.as_mut_ptr(),
                self.work.as_mut_ptr(),
                lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zgeqrf failed, info = {info}"));
        }

        let mut r = Array2::zeros((columns, columns).f());
        for column in 0..columns {
            for row in 0..=column {
                r[[row, column]] = self.a[row + column * rows];
            }
        }

        unsafe {
            magma_zungqr2(
                m,
                n,
                n,
                self.a.as_mut_ptr(),
                lda,
                self.tau.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zungqr2 failed, info = {info}"));
        }

        Ok(QrOutput {
            q: copy_fortran_columns_to_array(rows, columns, &self.a),
            r,
        })
    }

    fn compute_thin_qr(&mut self, matrix: ArrayView2<'_, Complex64>) -> Result<QrOutput, String> {
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

        copy_view_to_fortran_buffer(matrix, &mut self.a);
        self.finish_thin_qr(rows, columns)
    }

    fn compute_thin_qr_owned_fortran(
        &mut self,
        matrix: Array2<Complex64>,
    ) -> Result<QrOutput, String> {
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

        move_array_to_fortran_buffer(matrix, &mut self.a);
        self.finish_thin_qr(rows, columns)
    }
}

pub(crate) fn zgeqrf_qr_with_workspace(
    matrix: ArrayView2<'_, Complex64>,
    workspace: &mut HouseholderQrWorkspace,
) -> Result<QrOutput, String> {
    workspace.compute_thin_qr(matrix)
}

pub(crate) fn zgeqrf_qr_owned_fortran_with_workspace(
    matrix: Array2<Complex64>,
    workspace: &mut HouseholderQrWorkspace,
) -> Result<QrOutput, String> {
    workspace.compute_thin_qr_owned_fortran(matrix)
}

impl PivotedQrWorkspace {
    fn zgeqp3_lwork(&mut self, m: i32, n: i32, lda: i32) -> Result<i32, String> {
        ensure_magma_initialized();
        if let Some(cache) = self.zgeqp3_work
            && cache.m == m
            && cache.n == n
        {
            return Ok(cache.lwork);
        }

        let mut work_query = [zero(); 1];
        let mut info = 0;
        unsafe {
            magma_zgeqp3(
                m,
                n,
                self.a.as_mut_ptr(),
                lda,
                self.jpvt.as_mut_ptr(),
                self.tau.as_mut_ptr(),
                work_query.as_mut_ptr(),
                -1,
                self.rwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zgeqp3 workspace query failed, info = {info}"));
        }

        let lwork = (work_query[0].re as i32).max(n + 1).max(1);
        self.zgeqp3_work = Some(CachedWork { m, n, lwork });
        Ok(lwork)
    }

    fn finish_rank_revealing_qr(
        &mut self,
        rows: usize,
        columns: usize,
        relative_tolerance: f64,
    ) -> Result<PivotedQrOutput, String> {
        if rows < columns {
            return Err(format!(
                "pivoted thin QR expects rows >= columns, got {rows}x{columns}",
            ));
        }

        if columns == 0 {
            return Ok(PivotedQrOutput {
                q: Array2::zeros((rows, 0).f()),
                rank: 0,
            });
        }

        let m = rows as i32;
        let n = columns as i32;
        let lda = m.max(1);
        let min_mn = rows.min(columns);
        self.jpvt.resize(columns, 0);
        self.jpvt.fill(0);
        self.tau.resize(min_mn, zero());
        self.rwork.resize(2 * columns, 0.0);

        let lwork = self.zgeqp3_lwork(m, n, lda)?;
        self.jpvt.fill(0);
        self.work.resize(lwork as usize, zero());
        let mut info = 0;
        unsafe {
            magma_zgeqp3(
                m,
                n,
                self.a.as_mut_ptr(),
                lda,
                self.jpvt.as_mut_ptr(),
                self.tau.as_mut_ptr(),
                self.work.as_mut_ptr(),
                lwork,
                self.rwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zgeqp3 failed, info = {info}"));
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

        if rank == 0 {
            return Ok(PivotedQrOutput {
                q: Array2::zeros((rows, 0).f()),
                rank,
            });
        }

        let q_columns = min_mn as i32;
        unsafe {
            magma_zungqr2(
                m,
                q_columns,
                q_columns,
                self.a.as_mut_ptr(),
                lda,
                self.tau.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("magma_zungqr2 failed, info = {info}"));
        }

        Ok(PivotedQrOutput {
            q: copy_fortran_columns_to_array(rows, rank, &self.a),
            rank,
        })
    }

    fn compute_rank_revealing_qr(
        &mut self,
        matrix: ArrayView2<'_, Complex64>,
        relative_tolerance: f64,
    ) -> Result<PivotedQrOutput, String> {
        let (rows, columns) = matrix.dim();
        copy_view_to_fortran_buffer(matrix, &mut self.a);
        self.finish_rank_revealing_qr(rows, columns, relative_tolerance)
    }
}

/// MAGMA ZGEQP3 + ZUNGQR: rank-revealing QR с перестановкой столбцов.
pub(crate) fn zgeqp3_qr_rank(
    matrix: &Array2<Complex64>,
    relative_tolerance: f64,
) -> Result<PivotedQrOutput, String> {
    let mut workspace = PivotedQrWorkspace::default();
    zgeqp3_qr_rank_with_workspace(matrix.view(), relative_tolerance, &mut workspace)
}

pub(crate) fn zgeqp3_qr_rank_with_workspace(
    matrix: ArrayView2<'_, Complex64>,
    relative_tolerance: f64,
    workspace: &mut PivotedQrWorkspace,
) -> Result<PivotedQrOutput, String> {
    workspace.compute_rank_revealing_qr(matrix, relative_tolerance)
}

impl DenseSchurWorkspace {
    pub(crate) fn recycle_schur_output(&mut self, mut output: SchurOutput) {
        self.w = std::mem::take(&mut output.w);
        self.a = std::mem::take(&mut output.t);
        self.z = std::mem::take(&mut output.z);
    }

    fn zgees_lwork(&mut self, n: i32) -> Result<i32, SchurError> {
        ensure_magma_initialized();
        if let Some(cache) = self.zgees_work
            && cache.n == n
        {
            return Ok(cache.lwork);
        }

        let mut vl_dummy = [zero(); 1];
        let mut work_query = [zero(); 1];
        let mut info = 0_i32;
        unsafe {
            magma_zgeev(
                MAGMA_NO_VEC,
                MAGMA_VEC,
                n,
                self.a.as_mut_ptr(),
                n,
                self.w.as_mut_ptr(),
                vl_dummy.as_mut_ptr(),
                1,
                self.z.as_mut_ptr(),
                n,
                work_query.as_mut_ptr(),
                -1,
                self.rwork.as_mut_ptr(),
                &mut info,
            );
        }

        if info < 0 {
            return Err(SchurError::MagmaIllegalArgument(-info));
        }
        if info > 0 {
            return Err(SchurError::NoConvergence(info));
        }

        let lwork = (work_query[0].re as i32).max(2 * n).max(1);
        self.zgees_work = Some(CachedSingleWork { n, lwork });
        Ok(lwork)
    }

    fn compute_schur(
        &mut self,
        matrix: ArrayView2<'_, Complex64>,
    ) -> Result<SchurOutput, SchurError> {
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
        copy_view_to_fortran_buffer(matrix, &mut self.a);
        self.w.resize(n, zero());
        self.z.resize(n * n, zero());
        self.rwork.resize(2 * n, 0.0);
        self.bwork.clear();

        let lwork = self.zgees_lwork(n_i)?;
        self.work.resize(lwork as usize, zero());
        let mut vl_dummy = [zero(); 1];
        let mut info = 0_i32;

        unsafe {
            magma_zgeev(
                MAGMA_NO_VEC,
                MAGMA_VEC,
                n_i,
                self.a.as_mut_ptr(),
                n_i,
                self.w.as_mut_ptr(),
                vl_dummy.as_mut_ptr(),
                1,
                self.z.as_mut_ptr(),
                n_i,
                self.work.as_mut_ptr(),
                lwork,
                self.rwork.as_mut_ptr(),
                &mut info,
            );
        }

        if info < 0 {
            return Err(SchurError::MagmaIllegalArgument(-info));
        }
        if info > 0 {
            return Err(SchurError::NoConvergence(info));
        }

        Ok(SchurOutput {
            w: std::mem::take(&mut self.w),
            t: std::mem::take(&mut self.a),
            z: std::mem::take(&mut self.z),
        })
    }
}

pub(crate) fn zgees_schur_with_workspace(
    a: ArrayView2<'_, Complex64>,
    workspace: &mut DenseSchurWorkspace,
) -> Result<SchurOutput, SchurError> {
    workspace.compute_schur(a)
}

pub(crate) fn ztrevc_right_selected_with_workspace(
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

    let selected_count = workspace.select.iter().filter(|&&flag| flag != 0).count();
    workspace
        .vr_selected
        .resize(dim * selected_count, Complex64::ZERO);
    let eigenvectors = fortran_view(dim, dim, &decomposition.z)?;
    let mut vectors = Array2::zeros((dim, selected_count).f());
    let mut target_column = 0;
    for source_column in 0..dim {
        if workspace.select[source_column] == 0 {
            continue;
        }
        vectors
            .column_mut(target_column)
            .assign(&eigenvectors.column(source_column));
        target_column += 1;
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use num_complex::Complex64;

    #[test]
    fn zgemv_wraps_column_major_magma() {
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
    fn zgemm_wraps_column_major_magma() {
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

        let mut product = Array2::zeros((2, 2).f());
        zgemm_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::ONE,
            a.view(),
            b.view(),
            Complex64::ZERO,
            product.view_mut(),
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

        let mut gram = Array2::zeros((2, 2).f());
        zgemm_into(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            Complex64::ONE,
            a.view(),
            a.view(),
            Complex64::ZERO,
            gram.view_mut(),
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

        let mut workspace = DenseSchurWorkspace::default();
        let out = zgees_schur_with_workspace(a.view(), &mut workspace).unwrap();

        assert_eq!(out.w.len(), 2);
        assert!(
            out.w
                .iter()
                .any(|value| (*value - Complex64::new(-0.3722813232690143, 0.0)).norm() < 1.0e-12)
        );
        assert!(
            out.w
                .iter()
                .any(|value| (*value - Complex64::new(5.372281323269014, 0.0)).norm() < 1.0e-12)
        );
    }

    #[test]
    fn ztrevc_selected_vectors_have_requested_count() {
        let a = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let mut schur_workspace = DenseSchurWorkspace::default();
        let mut schur = zgees_schur_with_workspace(a.view(), &mut schur_workspace).unwrap();
        let mut trevc_workspace = TrevcWorkspace::default();
        let vectors =
            ztrevc_right_selected_with_workspace(&mut schur, &[0], 2, &mut trevc_workspace)
                .unwrap();

        assert_eq!(vectors.dim(), (2, 1));
    }

    #[test]
    fn ztrevc_selected_vectors_are_returned_in_schur_index_order() {
        let dim = 4;
        let mut t = vec![Complex64::ZERO; dim * dim];
        let mut z = vec![Complex64::ZERO; dim * dim];
        let mut w = Vec::with_capacity(dim);

        for index in 0..dim {
            let value = Complex64::new((index + 1) as f64, 0.0);
            t[index + index * dim] = value;
            z[index + index * dim] = Complex64::new(1.0, 0.0);
            w.push(value);
        }

        let mut schur = SchurOutput { w, t, z };
        let mut workspace = TrevcWorkspace::default();
        let vectors =
            ztrevc_right_selected_with_workspace(&mut schur, &[3, 1], dim, &mut workspace).unwrap();
        let expected = Array2::from_shape_vec(
            (dim, 2).f(),
            vec![
                Complex64::ZERO,
                Complex64::new(1.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();

        assert_eq!(vectors, expected);
    }
}
