use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
use num_complex::Complex64;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Once;

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
pub struct SchurOutput {
    /// Eigenvalues.
    pub w: Vec<Complex64>,
    /// Work matrix in Fortran column-major layout.
    pub t: Vec<Complex64>,
    /// Right eigenvectors in Fortran column-major layout.
    pub z: Vec<Complex64>,
}

#[derive(Debug)]
pub struct HouseholderQrOutput {
    pub q: Array2<Complex64>,
    pub r: Array2<Complex64>,
    pub rank: usize,
}

#[derive(Debug)]
pub enum SchurError {
    NotSquare,
    BadIloIhi,
    MagmaIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
    InvalidEigenIndex(usize),
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

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
}

/// Copies an ndarray row-major/strided matrix into Fortran column-major storage.
///
/// This allocation is necessary unless the caller already owns a contiguous
/// Fortran-order buffer, because MAGMA mutates its input in-place.
fn to_fortran_vec(a: &Array2<Complex64>) -> Vec<Complex64> {
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

fn from_fortran_vec(rows: usize, cols: usize, data: Vec<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_vec((rows, cols).f(), data).expect("invalid Fortran buffer shape")
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

fn identity_fortran_vec(n: usize) -> Vec<Complex64> {
    let mut z = vec![zero(); n * n];
    for i in 0..n {
        z[i + i * n] = Complex64::new(1.0, 0.0);
    }
    z
}

fn magma_trans(trans: ZgemmTranspose) -> c_int {
    match trans {
        ZgemmTranspose::None => MAGMA_NO_TRANS,
        ZgemmTranspose::ConjugateTranspose => MAGMA_CONJ_TRANS,
    }
}

fn transposed_shape(rows: usize, columns: usize, trans: ZgemmTranspose) -> (usize, usize) {
    match trans {
        ZgemmTranspose::None => (rows, columns),
        ZgemmTranspose::ConjugateTranspose => (columns, rows),
    }
}

pub fn zgemm(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    a: ArrayView2<'_, Complex64>,
    b: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let (a_rows, a_columns) = a.dim();
    let (b_rows, b_columns) = b.dim();
    let (a_effective_rows, a_effective_columns) = transposed_shape(a_rows, a_columns, trans_a);
    let (b_effective_rows, b_effective_columns) = transposed_shape(b_rows, b_columns, trans_b);
    assert_eq!(a_effective_columns, b_effective_rows);

    let a_strides = a.strides();
    assert!(
        a_rows <= 1 || a_strides[0] == 1,
        "zgemm expects column-major left matrix storage"
    );
    assert!(
        a_columns <= 1 || a_strides[1] == a_rows as isize,
        "zgemm expects column-major left matrix storage"
    );
    let b_strides = b.strides();
    assert!(
        b_rows <= 1 || b_strides[0] == 1,
        "zgemm expects column-major right matrix storage"
    );
    assert!(
        b_columns <= 1 || b_strides[1] == b_rows as isize,
        "zgemm expects column-major right matrix storage"
    );

    let a_memory = a
        .as_slice_memory_order()
        .expect("zgemm expects contiguous left matrix storage");
    let b_memory = b
        .as_slice_memory_order()
        .expect("zgemm expects contiguous right matrix storage");
    let mut result = Array2::zeros((a_effective_rows, b_effective_columns).f());
    let result_memory = result
        .as_slice_memory_order_mut()
        .expect("zgemm result must be contiguous");

    let m = a_effective_rows as c_int;
    let n = b_effective_columns as c_int;
    let k = a_effective_columns as c_int;
    let lda = a_rows.max(1) as c_int;
    let ldb = b_rows.max(1) as c_int;
    let ldc = a_effective_rows.max(1) as c_int;
    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::ZERO;
    let transa = magma_trans(trans_a);
    let transb = magma_trans(trans_b);

    if m == 0 || n == 0 {
        return result;
    }
    if k == 0 {
        return result;
    }

    let queue = Queue::new();
    let d_a = DeviceBuffer::new(a_memory.len());
    let d_b = DeviceBuffer::new(b_memory.len());
    let d_c = DeviceBuffer::new(result_memory.len());

    unsafe {
        magma_zsetmatrix(
            a_rows as c_int,
            a_columns as c_int,
            a_memory.as_ptr(),
            lda,
            d_a.ptr,
            lda,
            queue.raw,
        );
        magma_zsetmatrix(
            b_rows as c_int,
            b_columns as c_int,
            b_memory.as_ptr(),
            ldb,
            d_b.ptr,
            ldb,
            queue.raw,
        );
        magma_zsetmatrix(m, n, result_memory.as_ptr(), ldc, d_c.ptr, ldc, queue.raw);
        magma_zgemm(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            d_a.ptr,
            lda,
            d_b.ptr,
            ldb,
            beta,
            d_c.ptr,
            ldc,
            queue.raw,
        );
        magma_zgetmatrix(m, n, d_c.ptr, ldc, result_memory.as_mut_ptr(), ldc, queue.raw);
    }

    result
}

fn scale_slice(values: &mut [Complex64], beta: Complex64) {
    if beta == Complex64::ZERO {
        values.fill(Complex64::ZERO);
    } else {
        values.iter_mut().for_each(|value| *value *= beta);
    }
}

fn magma_trans_from_zgemv(trans: ZgemvTranspose) -> c_int {
    match trans {
        ZgemvTranspose::None => MAGMA_NO_TRANS,
        ZgemvTranspose::ConjugateTranspose => MAGMA_CONJ_TRANS,
    }
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

    let (x_len, y_len) = match trans {
        ZgemvTranspose::None => (columns, rows),
        ZgemvTranspose::ConjugateTranspose => (rows, columns),
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

pub fn zhseqr_schur(h: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    magma_zgeev_right(h)
}

pub fn zgees_schur(a: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    magma_zgeev_right(a)
}

fn magma_zgeev_right(a: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    let (n, m) = a.dim();
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

    ensure_magma_initialized();
    let n_i = n as c_int;
    let mut a_col = to_fortran_vec(a);
    let mut w = vec![zero(); n];
    let mut vl_dummy = [zero(); 1];
    let mut z = vec![zero(); n * n];
    let mut rwork = vec![0.0; 2 * n];
    let mut work_query = [zero(); 1];
    let mut info = 0_i32;

    unsafe {
        magma_zgeev(
            MAGMA_NO_VEC,
            MAGMA_VEC,
            n_i,
            a_col.as_mut_ptr(),
            n_i,
            w.as_mut_ptr(),
            vl_dummy.as_mut_ptr(),
            1,
            z.as_mut_ptr(),
            n_i,
            work_query.as_mut_ptr(),
            -1,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::MagmaIllegalArgument(-info));
    }
    if info > 0 {
        return Err(SchurError::NoConvergence(info));
    }

    let lwork = (work_query[0].re as i32).max(2 * n_i).max(1);
    let mut work = vec![zero(); lwork as usize];
    let mut info = 0_i32;

    unsafe {
        magma_zgeev(
            MAGMA_NO_VEC,
            MAGMA_VEC,
            n_i,
            a_col.as_mut_ptr(),
            n_i,
            w.as_mut_ptr(),
            vl_dummy.as_mut_ptr(),
            1,
            z.as_mut_ptr(),
            n_i,
            work.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::MagmaIllegalArgument(-info));
    }
    if info > 0 {
        return Err(SchurError::NoConvergence(info));
    }

    Ok(SchurOutput { w, t: a_col, z })
}

pub fn ztrevc_right_selected(
    decomposition: &mut SchurOutput,
    indices: &[usize],
    dim: usize,
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

    let mut select = vec![false; dim];
    for &index in indices {
        select[index] = true;
    }

    let eigenvectors = fortran_view(dim, dim, &decomposition.z)?;
    let selected_count = select.iter().filter(|&&flag| flag).count();
    let mut selected = Array2::zeros((dim, selected_count).f());
    let mut target_column = 0;
    for (source_column, &is_selected) in select.iter().enumerate() {
        if !is_selected {
            continue;
        }
        selected
            .column_mut(target_column)
            .assign(&eigenvectors.column(source_column));
        target_column += 1;
    }

    Ok(selected)
}

fn magma_zgeqrf_lwork(
    m: c_int,
    n: c_int,
    a: &mut [Complex64],
    lda: c_int,
    tau: &mut [Complex64],
) -> Result<c_int, String> {
    ensure_magma_initialized();
    let mut work_query = [zero(); 1];
    let mut info = 0;
    unsafe {
        magma_zgeqrf(
            m,
            n,
            a.as_mut_ptr(),
            lda,
            tau.as_mut_ptr(),
            work_query.as_mut_ptr(),
            -1,
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("magma_zgeqrf workspace query failed, info = {info}"));
    }

    Ok((work_query[0].re as c_int).max(n).max(1))
}

fn magma_zgeqrf_factor(
    m: c_int,
    n: c_int,
    a: &mut [Complex64],
    lda: c_int,
    tau: &mut [Complex64],
    work: &mut [Complex64],
) -> Result<(), String> {
    let mut info = 0;
    unsafe {
        magma_zgeqrf(
            m,
            n,
            a.as_mut_ptr(),
            lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            work.len() as c_int,
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("magma_zgeqrf failed, info = {info}"));
    }
    Ok(())
}

fn magma_zungqr_generate(
    m: c_int,
    n: c_int,
    k: c_int,
    a: &mut [Complex64],
    lda: c_int,
    tau: &mut [Complex64],
) -> Result<(), String> {
    let mut info = 0;
    unsafe {
        magma_zungqr2(
            m,
            n,
            k,
            a.as_mut_ptr(),
            lda,
            tau.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("magma_zungqr2 failed, info = {info}"));
    }
    Ok(())
}

pub fn shifted_qr_filter(
    hessenberg: &Array2<Complex64>,
    shifts: &[Complex64],
) -> Result<(Array2<Complex64>, Array2<Complex64>), String> {
    let (h_rows, h_cols) = hessenberg.dim();
    if h_rows != h_cols {
        return Err("H must be square".into());
    }

    let n = h_rows;
    let mut h = from_fortran_vec(n, n, to_fortran_vec(hessenberg));
    let mut rotation = from_fortran_vec(n, n, identity_fortran_vec(n));

    for &shift in shifts {
        let mut shifted = h.clone();
        for index in 0..n {
            shifted[[index, index]] -= shift;
        }

        let q = zgeqrf_q(&shifted)?;
        let q_star_h = zgemm(
            ZgemmTranspose::ConjugateTranspose,
            ZgemmTranspose::None,
            q.view(),
            h.view(),
        );
        h = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            q_star_h.view(),
            q.view(),
        );
        cleanup_hessenberg_roundoff(&mut h);
        rotation = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            rotation.view(),
            q.view(),
        );
    }

    Ok((rotation, h))
}

fn zgeqrf_q(matrix: &Array2<Complex64>) -> Result<Array2<Complex64>, String> {
    let (rows, cols) = matrix.dim();
    let m = rows as i32;
    let n = cols as i32;
    let k = m.min(n);
    let lda = m.max(1);

    let mut q = to_fortran_vec(matrix);
    let mut tau = vec![zero(); k as usize];
    let lwork = magma_zgeqrf_lwork(m, n, &mut q, lda, &mut tau)?;
    let mut work = vec![zero(); lwork as usize];

    magma_zgeqrf_factor(m, n, &mut q, lda, &mut tau, &mut work)?;
    magma_zungqr_generate(m, n, k, &mut q, lda, &mut tau)?;

    Ok(from_fortran_vec(rows, cols, q))
}

pub fn zgeqrf_qr_rank(
    matrix: &Array2<Complex64>,
    relative_tolerance: f64,
) -> Result<HouseholderQrOutput, String> {
    let (rows, columns) = matrix.dim();
    if rows < columns {
        return Err(format!(
            "thin QR expects rows >= columns, got {rows}x{columns}",
        ));
    }

    if columns == 0 {
        return Ok(HouseholderQrOutput {
            q: Array2::zeros((rows, 0).f()),
            r: Array2::zeros((0, 0).f()),
            rank: 0,
        });
    }

    let m = rows as i32;
    let n = columns as i32;
    let lda = m.max(1);
    let min_mn = rows.min(columns);
    let mut a = to_fortran_vec(matrix);
    let mut tau = vec![zero(); min_mn];
    let lwork = magma_zgeqrf_lwork(m, n, &mut a, lda, &mut tau)?;
    let mut work = vec![zero(); lwork as usize];

    magma_zgeqrf_factor(m, n, &mut a, lda, &mut tau, &mut work)?;

    let diagonal = (0..min_mn)
        .map(|index| a[index + index * rows].norm())
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
            r[[row, column]] = a[row + column * rows];
        }
    }

    if rank == 0 {
        return Ok(HouseholderQrOutput {
            q: Array2::zeros((rows, 0).f()),
            r,
            rank,
        });
    }

    let q_columns = min_mn as i32;
    magma_zungqr_generate(m, q_columns, q_columns, &mut a, lda, &mut tau)?;

    Ok(HouseholderQrOutput {
        q: copy_fortran_columns_to_array(rows, rank, &a),
        r,
        rank,
    })
}

fn cleanup_hessenberg_roundoff(h: &mut Array2<Complex64>) {
    for row in 0..h.nrows() {
        for column in 0..row.saturating_sub(1) {
            h[[row, column]] = Complex64::ZERO;
        }
    }
}

pub fn last_r_col_without_diag_from_zgeqrf(
    a_cols: &[Array1<Complex64>],
    z: &Array1<Complex64>,
) -> Result<Vec<Complex64>, String> {
    let nrows = z.len();
    let k = a_cols.len();
    let ncols = k + 1;

    if nrows == 0 {
        return Err("empty vectors are not supported".into());
    }

    for (j, col) in a_cols.iter().enumerate() {
        if col.len() != nrows {
            return Err(format!(
                "column {} has length {}, expected {}",
                j,
                col.len(),
                nrows
            ));
        }
    }

    // Один буфер под A = [a_cols, z] в column-major layout.
    // Старый вариант делал clone всего Vec<Array1>, clone(z), push, flat_map/collect.
    let mut mat = Vec::with_capacity(nrows * ncols);
    for col in a_cols {
        mat.extend(col.iter().copied());
    }
    mat.extend(z.iter().copied());

    let m = nrows as i32;
    let n = ncols as i32;
    let lda = m.max(1);

    let min_mn = m.min(n);
    let mut tau = vec![zero(); min_mn as usize];

    let lwork = magma_zgeqrf_lwork(m, n, &mut mat, lda, &mut tau)?;
    if lwork <= 0 {
        return Err(format!("magma_zgeqrf returned invalid lwork = {}", lwork));
    }

    let mut work = vec![zero(); lwork as usize];
    magma_zgeqrf_factor(m, n, &mut mat, lda, &mut tau, &mut work)?;

    // R хранится в верхнем треугольнике mat. Последний столбец имеет offset k * lda.
    // Без диагонального элемента берём строки 0..rlen, где rlen = min(k, nrows).
    let rlen = k.min(nrows);
    let offset = k * lda as usize;

    Ok(mat[offset..offset + rlen].to_vec())
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

        let c = zgemm(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            a.view(),
            b.view(),
        );

        assert_eq!(
            c,
            Array2::from_shape_vec(
                (2, 2).f(),
                vec![
                    Complex64::new(19.0, 0.0),
                    Complex64::new(43.0, 0.0),
                    Complex64::new(22.0, 0.0),
                    Complex64::new(50.0, 0.0),
                ],
            )
            .unwrap()
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
    fn zhseqr_schur_smoke_test() {
        let h = array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, -1.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ];

        let mut out = zhseqr_schur(&h).unwrap();
        println!("{:?}", out.t);
        println!("{:?}\n\n", out.w);
        let vecs = ztrevc_right_selected(&mut out, &[0, 1], 3);
        println!("{:?}", vecs);
        assert_eq!(out.w.len(), h.nrows());
    }

    #[test]
    fn shifted_qr_filter_preserves_similarity() {
        let h = array![
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, -1.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
        ];

        let (q, filtered_h) = shifted_qr_filter(
            &h,
            &[
                Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -0.5),
            ],
        )
        .unwrap();

        let q_star = q.t().mapv(|x| x.conj());
        let reconstructed = q.dot(&filtered_h).dot(&q_star);
        let reconstruction_error = frobenius_norm(&(reconstructed - h));
        assert!(
            reconstruction_error < 1.0e-10,
            "reconstruction_error={reconstruction_error}"
        );

        let identity = q_star.dot(&q);
        let mut expected_identity = Array2::zeros((3, 3).f());
        for index in 0..3 {
            expected_identity[[index, index]] = Complex64::new(1.0, 0.0);
        }
        let unitary_error = frobenius_norm(&(identity - expected_identity));
        assert!(unitary_error < 1.0e-10, "unitary_error={unitary_error}");
    }

    fn frobenius_norm(matrix: &Array2<Complex64>) -> f64 {
        matrix
            .iter()
            .map(|entry| entry.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}
