use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
use num_complex::Complex64;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Once;

use crate::memory;

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
    /// Eigenvalues in the same order as the columns of `z`.
    pub w: Vec<Complex64>,
    /// MAGMA work matrix in Fortran column-major layout.
    ///
    /// This is the input matrix after `magma_zgeev` has overwritten it. It is
    /// kept for diagnostics/backward compatibility; it is not a Schur form.
    pub t: Vec<Complex64>,
    /// Right eigenvectors in Fortran column-major layout, one eigenvector per column.
    pub z: Vec<Complex64>,
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

//type MagmaQueue = *mut c_void;

const MAGMA_NO_TRANS: c_int = 111;
const MAGMA_CONJ_TRANS: c_int = 113;
const MAGMA_NO_VEC: c_int = 301;
const MAGMA_VEC: c_int = 302;
const MAGMA_RIGHT: c_int = 142;
const MAGMA_BACKTRANS_VEC: c_int = 307;
const MAGMA_SUCCESS: c_int = 0;
const MAGMA_FUNC: *const c_char = b"rust\0".as_ptr().cast();
const MAGMA_FILE: *const c_char = b"src/linalg/magma.rs\0".as_ptr().cast();

static MAGMA_INIT: Once = Once::new();

// MAGMA normally uses C int for magma_int_t unless built with ILP64.
type MagmaInt = c_int;
type MagmaDevice = c_int;

// Opaque MAGMA queue handle.
#[repr(C)]
pub struct MagmaQueueOpaque {
    _private: [u8; 0],
}

type MagmaQueue = *mut MagmaQueueOpaque;

unsafe extern "C" {
    fn magma_init() -> MagmaInt;
    fn magma_finalize() -> MagmaInt;

    fn magma_getdevice(device: *mut MagmaDevice);

    // magma_queue_create(...) is a C macro, so bind the real exported function.
    fn magma_queue_create_internal(
        device: MagmaDevice,
        queue_ptr: *mut MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    // magma_queue_destroy(...) is also a macro.
    fn magma_queue_destroy_internal(
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    fn magma_malloc(ptr_ptr: *mut *mut c_void, bytes: usize) -> MagmaInt;

    // magma_free(...) is a macro.
    fn magma_free_internal(
        ptr: *mut c_void,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    ) -> MagmaInt;

    // magma_zsetmatrix(...) / magma_zgetmatrix(...) are macros/static inline wrappers.
    // Bind the generic exported internal functions instead.
    fn magma_setmatrix_internal(
        m: MagmaInt,
        n: MagmaInt,
        elem_size: MagmaInt,
        h_a: *const c_void,
        lda: MagmaInt,
        d_a: *mut c_void,
        ldda: MagmaInt,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    fn magma_getmatrix_internal(
        m: MagmaInt,
        n: MagmaInt,
        elem_size: MagmaInt,
        d_a: *const c_void,
        ldda: MagmaInt,
        h_a: *mut c_void,
        lda: MagmaInt,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    fn magma_setvector_internal(
        n: MagmaInt,
        elem_size: MagmaInt,
        h_x: *const c_void,
        incx: MagmaInt,
        d_x: *mut c_void,
        incx_dev: MagmaInt,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    fn magma_getvector_internal(
        n: MagmaInt,
        elem_size: MagmaInt,
        d_x: *const c_void,
        incx_dev: MagmaInt,
        h_x: *mut c_void,
        incx: MagmaInt,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    );

    fn magma_zgemm(
        transa: MagmaInt,
        transb: MagmaInt,
        m: MagmaInt,
        n: MagmaInt,
        k: MagmaInt,
        alpha: Complex64,
        d_a: *const Complex64,
        ldda: MagmaInt,
        d_b: *const Complex64,
        lddb: MagmaInt,
        beta: Complex64,
        d_c: *mut Complex64,
        lddc: MagmaInt,
        queue: MagmaQueue,
    );

    fn magma_zgemv(
        trans: MagmaInt,
        m: MagmaInt,
        n: MagmaInt,
        alpha: Complex64,
        d_a: *const Complex64,
        ldda: MagmaInt,
        d_x: *const Complex64,
        incx: MagmaInt,
        beta: Complex64,
        d_y: *mut Complex64,
        incy: MagmaInt,
        queue: MagmaQueue,
    );

    fn magma_zgeev(
        jobvl: MagmaInt,
        jobvr: MagmaInt,
        n: MagmaInt,
        a: *mut Complex64,
        lda: MagmaInt,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: MagmaInt,
        vr: *mut Complex64,
        ldvr: MagmaInt,
        work: *mut Complex64,
        lwork: MagmaInt,
        rwork: *mut f64,
        info: *mut MagmaInt,
    ) -> MagmaInt;

    fn magma_zgeqrf(
        m: MagmaInt,
        n: MagmaInt,
        a: *mut Complex64,
        lda: MagmaInt,
        tau: *mut Complex64,
        work: *mut Complex64,
        lwork: MagmaInt,
        info: *mut MagmaInt,
    ) -> MagmaInt;

    fn magma_ztrevc3_mt(
        side: MagmaInt,
        howmany: MagmaInt,
        select: *mut MagmaInt,
        n: MagmaInt,
        t: *mut Complex64,
        ldt: MagmaInt,
        vl: *mut Complex64,
        ldvl: MagmaInt,
        vr: *mut Complex64,
        ldvr: MagmaInt,
        mm: MagmaInt,
        mout: *mut MagmaInt,
        work: *mut Complex64,
        lwork: MagmaInt,
        rwork: *mut f64,
        info: *mut MagmaInt,
    ) -> MagmaInt;
}

const MAGMA_COMPLEX64_SIZE: MagmaInt = std::mem::size_of::<Complex64>() as MagmaInt;

#[inline]
unsafe fn magma_zsetmatrix(
    m: MagmaInt,
    n: MagmaInt,
    h_a: *const Complex64,
    lda: MagmaInt,
    d_a: *mut Complex64,
    ldda: MagmaInt,
    queue: MagmaQueue,
) {
    unsafe {
        magma_setmatrix_internal(
            m,
            n,
            MAGMA_COMPLEX64_SIZE,
            h_a.cast::<c_void>(),
            lda,
            d_a.cast::<c_void>(),
            ldda,
            queue,
            MAGMA_FUNC,
            MAGMA_FILE,
            line!() as c_int,
        );
    }
}

#[inline]
unsafe fn magma_zgetmatrix(
    m: MagmaInt,
    n: MagmaInt,
    d_a: *const Complex64,
    ldda: MagmaInt,
    h_a: *mut Complex64,
    lda: MagmaInt,
    queue: MagmaQueue,
) {
    unsafe {
        magma_getmatrix_internal(
            m,
            n,
            MAGMA_COMPLEX64_SIZE,
            d_a.cast::<c_void>(),
            ldda,
            h_a.cast::<c_void>(),
            lda,
            queue,
            MAGMA_FUNC,
            MAGMA_FILE,
            line!() as c_int,
        );
    }
}

#[inline]
unsafe fn magma_zsetvector(
    n: MagmaInt,
    h_x: *const Complex64,
    incx: MagmaInt,
    d_x: *mut Complex64,
    incx_dev: MagmaInt,
    queue: MagmaQueue,
) {
    unsafe {
        magma_setvector_internal(
            n,
            MAGMA_COMPLEX64_SIZE,
            h_x.cast::<c_void>(),
            incx,
            d_x.cast::<c_void>(),
            incx_dev,
            queue,
            MAGMA_FUNC,
            MAGMA_FILE,
            line!() as c_int,
        );
    }
}

#[inline]
unsafe fn magma_zgetvector(
    n: MagmaInt,
    d_x: *const Complex64,
    incx_dev: MagmaInt,
    h_x: *mut Complex64,
    incx: MagmaInt,
    queue: MagmaQueue,
) {
    unsafe {
        magma_getvector_internal(
            n,
            MAGMA_COMPLEX64_SIZE,
            d_x.cast::<c_void>(),
            incx_dev,
            h_x.cast::<c_void>(),
            incx,
            queue,
            MAGMA_FUNC,
            MAGMA_FILE,
            line!() as c_int,
        );
    }
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
    bytes: usize,
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
        memory::record_device_allocation(bytes);
        Self {
            ptr: ptr.cast::<Complex64>(),
            bytes,
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            magma_free_internal(self.ptr.cast::<c_void>(), MAGMA_FUNC, MAGMA_FILE, line!() as c_int);
        }
        memory::record_device_deallocation(self.bytes);
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

fn magma_trans_from_zgemv(trans: ZgemvTranspose) -> c_int {
    match trans {
        ZgemvTranspose::None => MAGMA_NO_TRANS,
        ZgemvTranspose::ConjugateTranspose => MAGMA_CONJ_TRANS,
    }
}

fn transposed_shape(rows: usize, columns: usize, trans: ZgemmTranspose) -> (usize, usize) {
    match trans {
        ZgemmTranspose::None => (rows, columns),
        ZgemmTranspose::ConjugateTranspose => (columns, rows),
    }
}

/// Reusable MAGMA/CUDA execution context.
///
/// Keep one session across several BLAS calls to avoid recreating a MAGMA queue
/// for every small operation.
pub struct MagmaSession {
    queue: Queue,
}

impl MagmaSession {
    pub fn new() -> Self {
        Self { queue: Queue::new() }
    }
}

impl Default for MagmaSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Device-side vector buffer owned by Rust and allocated by MAGMA.
pub struct DeviceVector {
    buffer: DeviceBuffer,
    len: usize,
}

impl DeviceVector {
    pub fn new(len: usize) -> Self {
        Self {
            buffer: DeviceBuffer::new(len),
            len,
        }
    }

    pub fn from_slice(session: &MagmaSession, values: &[Complex64]) -> Self {
        let mut vector = Self::new(values.len());
        vector.copy_from_slice(session, values);
        vector
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_slice(&mut self, session: &MagmaSession, values: &[Complex64]) {
        assert_eq!(self.len, values.len());
        self.copy_prefix_from_slice(session, values);
    }

    pub fn copy_to_slice(&self, session: &MagmaSession, values: &mut [Complex64]) {
        assert_eq!(self.len, values.len());
        self.copy_prefix_to_slice(session, values);
    }

    /// Uploads a prefix of this vector without reallocating the device buffer.
    pub fn copy_prefix_from_slice(&mut self, session: &MagmaSession, values: &[Complex64]) {
        assert!(values.len() <= self.len);
        if values.is_empty() {
            return;
        }
        unsafe {
            magma_zsetvector(
                values.len() as c_int,
                values.as_ptr(),
                1,
                self.buffer.ptr,
                1,
                session.queue.raw,
            );
        }
        memory::record_host_to_device_copy(values.len() * std::mem::size_of::<Complex64>());
    }

    /// Downloads a prefix of this vector without reallocating host storage.
    pub fn copy_prefix_to_slice(&self, session: &MagmaSession, values: &mut [Complex64]) {
        assert!(values.len() <= self.len);
        if values.is_empty() {
            return;
        }
        unsafe {
            magma_zgetvector(
                values.len() as c_int,
                self.buffer.ptr,
                1,
                values.as_mut_ptr(),
                1,
                session.queue.raw,
            );
        }
        memory::record_device_to_host_copy(values.len() * std::mem::size_of::<Complex64>());
    }

    #[inline]
    fn ptr(&self) -> *const Complex64 {
        self.buffer.ptr
    }

    #[inline]
    fn mut_ptr(&mut self) -> *mut Complex64 {
        self.buffer.ptr
    }
}

/// Device-side column-major dense matrix owned by Rust and allocated by MAGMA.
pub struct DeviceMatrix {
    buffer: DeviceBuffer,
    rows: usize,
    columns: usize,
    lda: c_int,
}

impl DeviceMatrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            buffer: DeviceBuffer::new(rows.saturating_mul(columns)),
            rows,
            columns,
            lda: rows.max(1) as c_int,
        }
    }

    pub fn from_column_major(session: &MagmaSession, matrix: ArrayView2<'_, Complex64>) -> Self {
        let (rows, columns) = matrix.dim();
        assert_column_major_contiguous(matrix, "DeviceMatrix::from_column_major");
        let memory = matrix
            .as_slice_memory_order()
            .expect("column-major matrix must be contiguous");
        let mut device = Self::new(rows, columns);
        if rows != 0 && columns != 0 {
            unsafe {
                magma_zsetmatrix(
                    rows as c_int,
                    columns as c_int,
                    memory.as_ptr(),
                    rows.max(1) as c_int,
                    device.buffer.ptr,
                    device.lda,
                    session.queue.raw,
                );
                memory::record_host_to_device_copy(rows * columns * std::mem::size_of::<Complex64>());
            }
        }
        device
    }

    pub fn copy_to_column_major(
        &self,
        session: &MagmaSession,
        output: &mut [Complex64],
    ) {
        assert_eq!(output.len(), self.rows * self.columns);
        if self.rows == 0 || self.columns == 0 {
            return;
        }
        unsafe {
            magma_zgetmatrix(
                self.rows as c_int,
                self.columns as c_int,
                self.buffer.ptr,
                self.lda,
                output.as_mut_ptr(),
                self.rows.max(1) as c_int,
                session.queue.raw,
            );
        }
        memory::record_device_to_host_copy(self.rows * self.columns * std::mem::size_of::<Complex64>());
    }

    pub fn zgemv(
        &self,
        session: &MagmaSession,
        trans: ZgemvTranspose,
        alpha: Complex64,
        x: &DeviceVector,
        beta: Complex64,
        y: &mut DeviceVector,
    ) {
        let (x_len, y_len) = match trans {
            ZgemvTranspose::None => (self.columns, self.rows),
            ZgemvTranspose::ConjugateTranspose => (self.rows, self.columns),
        };
        assert!(x.len() >= x_len, "device zgemv input buffer is too small");
        assert!(y.len() >= y_len, "device zgemv output buffer is too small");

        if y_len == 0 {
            return;
        }
        if x_len == 0 {
            panic!("device zgemv with empty x is not supported here");
        }

        unsafe {
            magma_zgemv(
                magma_trans_from_zgemv(trans),
                self.rows as c_int,
                self.columns as c_int,
                alpha,
                self.buffer.ptr,
                self.lda,
                x.ptr(),
                1,
                beta,
                y.mut_ptr(),
                1,
                session.queue.raw,
            );
        }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Uploads one host vector into a column of this device matrix.
    ///
    /// The matrix is stored in column-major layout with leading dimension `lda`.
    /// This is the key primitive that lets Arnoldi keep the basis resident on
    /// the GPU and append one new vector per step instead of re-uploading the
    /// whole basis matrix for every orthogonalization call.
    pub fn copy_column_from_slice(
        &mut self,
        session: &MagmaSession,
        column: usize,
        values: &[Complex64],
    ) {
        assert!(column < self.columns);
        assert_eq!(values.len(), self.rows);
        if self.rows == 0 {
            return;
        }

        unsafe {
            let destination = self.buffer.ptr.add(column * self.lda as usize);
            magma_zsetvector(
                self.rows as c_int,
                values.as_ptr(),
                1,
                destination,
                1,
                session.queue.raw,
            );
            memory::record_host_to_device_copy(self.rows * std::mem::size_of::<Complex64>());
        }
    }

    /// Computes GEMV using only the leading `columns` columns of this device matrix.
    ///
    /// This avoids materializing a new device matrix view for `V[:, 0..j]` during
    /// Arnoldi. MAGMA sees the same base pointer and leading dimension, but the
    /// logical matrix width is restricted to `columns`.
    pub fn zgemv_leading_columns(
        &self,
        session: &MagmaSession,
        columns: usize,
        trans: ZgemvTranspose,
        alpha: Complex64,
        x: &DeviceVector,
        beta: Complex64,
        y: &mut DeviceVector,
    ) {
        assert!(columns <= self.columns);
        let (x_len, y_len) = match trans {
            ZgemvTranspose::None => (columns, self.rows),
            ZgemvTranspose::ConjugateTranspose => (self.rows, columns),
        };
        assert!(x.len() >= x_len, "device zgemv input buffer is too small");
        assert!(y.len() >= y_len, "device zgemv output buffer is too small");

        if y_len == 0 {
            return;
        }
        if x_len == 0 {
            panic!("device zgemv with empty x is not supported here");
        }

        unsafe {
            magma_zgemv(
                magma_trans_from_zgemv(trans),
                self.rows as c_int,
                columns as c_int,
                alpha,
                self.buffer.ptr,
                self.lda,
                x.ptr(),
                1,
                beta,
                y.mut_ptr(),
                1,
                session.queue.raw,
            );
        }
    }

}

#[inline]
fn assert_column_major_contiguous(matrix: ArrayView2<'_, Complex64>, context: &str) {
    let (rows, columns) = matrix.dim();
    let strides = matrix.strides();
    assert!(
        rows <= 1 || strides[0] == 1,
        "{context} expects column-major matrix storage"
    );
    assert!(
        columns <= 1 || strides[1] == rows as isize,
        "{context} expects column-major matrix storage"
    );
    matrix
        .as_slice_memory_order()
        .expect("column-major matrix must be contiguous");
}

pub fn zgemm_with_session(
    session: &MagmaSession,
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

    assert_column_major_contiguous(a, "zgemm left matrix");
    assert_column_major_contiguous(b, "zgemm right matrix");

    let mut result = Array2::zeros((a_effective_rows, b_effective_columns).f());
    if a_effective_rows == 0 || b_effective_columns == 0 || a_effective_columns == 0 {
        return result;
    }

    let d_a = DeviceMatrix::from_column_major(session, a);
    let d_b = DeviceMatrix::from_column_major(session, b);
    let d_c = DeviceMatrix::new(a_effective_rows, b_effective_columns);

    unsafe {
        magma_zgemm(
            magma_trans(trans_a),
            magma_trans(trans_b),
            a_effective_rows as c_int,
            b_effective_columns as c_int,
            a_effective_columns as c_int,
            Complex64::new(1.0, 0.0),
            d_a.buffer.ptr,
            d_a.lda,
            d_b.buffer.ptr,
            d_b.lda,
            Complex64::ZERO,
            d_c.buffer.ptr,
            d_c.lda,
            session.queue.raw,
        );
    }

    let result_memory = result
        .as_slice_memory_order_mut()
        .expect("zgemm result must be contiguous");
    d_c.copy_to_column_major(session, result_memory);
    result
}

pub fn zgemm(
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    a: ArrayView2<'_, Complex64>,
    b: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let session = MagmaSession::new();
    zgemm_with_session(&session, trans_a, trans_b, a, b)
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
    assert_column_major_contiguous(matrix, "zgemv matrix");

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

    let session = MagmaSession::new();
    let d_matrix = DeviceMatrix::from_column_major(&session, matrix);
    let d_x = DeviceVector::from_slice(&session, x);
    let mut d_y = DeviceVector::from_slice(&session, y);
    d_matrix.zgemv(&session, trans, alpha, &d_x, beta, &mut d_y);
    d_y.copy_to_slice(&session, y);
}

fn scale_slice(values: &mut [Complex64], beta: Complex64) {
    if beta == Complex64::ZERO {
        values.fill(Complex64::ZERO);
    } else {
        values.iter_mut().for_each(|value| *value *= beta);
    }
}

/// Computes eigenvalues and all right eigenvectors with MAGMA `zgeev`.
///
/// The input matrix is the small Arnoldi Hessenberg matrix. MAGMA in this build
/// exports `magma_zgeev` and `magma_ztrevc3_mt`, but not a public
/// `magma_zhseqr`; `zgeev` internally performs the Schur/eigenvector work.
/// Therefore this routine intentionally exposes the result as eigenpairs, not as
/// a true Schur decomposition.
pub fn zgeev_right_eigenpairs(h: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    magma_zgeev_right(h)
}

/// Backward-compatible alias for older call sites.
///
/// Despite the historical name, this does not call `zhseqr` and `SchurOutput.t`
/// is not a Schur form. Prefer `zgeev_right_eigenpairs`.
pub fn zhseqr_schur(h: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    zgeev_right_eigenpairs(h)
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
    select_right_eigenvectors(decomposition, indices, dim)
}

/// Selects right eigenvector columns from `SchurOutput.z` in exactly the order
/// requested by `indices`.
///
/// The previous implementation built a boolean mask and returned columns in
/// ascending eigenvalue index order. IRAM then paired those columns with
/// `selection.wanted` order, which could attach a residual estimate to the wrong
/// Ritz value.
pub fn select_right_eigenvectors(
    decomposition: &SchurOutput,
    indices: &[usize],
    dim: usize,
) -> Result<Array2<Complex64>, SchurError> {
    if decomposition.z.len() != dim * dim {
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

    let eigenvectors = fortran_view(dim, dim, &decomposition.z)?;
    let mut selected = Array2::zeros((dim, indices.len()).f());
    for (target_column, &source_column) in indices.iter().enumerate() {
        selected
            .column_mut(target_column)
            .assign(&eigenvectors.column(source_column));
    }

    Ok(selected)
}

/// Computes all right eigenvectors of a true Schur form `T`, optionally
/// back-transformed by Schur vectors `Q`, using `magma_ztrevc3_mt`.
///
/// This is not used by `zgeev_right_eigenpairs`, because that routine already
/// obtains eigenvectors from MAGMA's `zgeev` driver. It is provided for the
/// future path `CPU/LAPACK zhseqr -> MAGMA ztrevc3_mt`, where `t` is a true
/// Schur form and `q` contains the Schur vectors.
pub fn ztrevc3_right_backtrans_all(
    t: &[Complex64],
    q: &[Complex64],
    dim: usize,
) -> Result<Array2<Complex64>, SchurError> {
    if t.len() != dim * dim || q.len() != dim * dim {
        return Err(SchurError::DimensionMismatch);
    }
    if dim == 0 {
        return Ok(Array2::zeros((0, 0).f()));
    }

    ensure_magma_initialized();

    let n_i = dim as c_int;
    let mut t_work = t.to_vec();
    let mut right_vectors = q.to_vec();
    let mut left_dummy = vec![zero(); 1];
    let mut mout = 0_i32;

    // MAGMA requires at least 2*n workspace. A larger workspace enables the
    // blocked Level-3 path in ztrevc3_mt.
    let lwork = ((1 + 2 * 64) * dim).max(2 * dim).max(1) as c_int;
    let mut work = vec![zero(); lwork as usize];
    let mut rwork = vec![0.0; dim.max(1)];
    let mut info = 0_i32;

    unsafe {
        magma_ztrevc3_mt(
            MAGMA_RIGHT,
            MAGMA_BACKTRANS_VEC,
            ptr::null_mut(),
            n_i,
            t_work.as_mut_ptr(),
            n_i,
            left_dummy.as_mut_ptr(),
            1,
            right_vectors.as_mut_ptr(),
            n_i,
            n_i,
            &mut mout,
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
    if mout != n_i {
        return Err(SchurError::DimensionMismatch);
    }

    Ok(from_fortran_vec(dim, dim, right_vectors))
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

pub fn shifted_qr_filter(
    hessenberg: &Array2<Complex64>,
    shifts: &[Complex64],
) -> Result<(Array2<Complex64>, Array2<Complex64>), String> {
    crate::linalg::shifted_qr::shifted_qr_filter(hessenberg, shifts)
}

fn apply_implicit_shift(
    h: &mut Array2<Complex64>,
    rotation: &mut Array2<Complex64>,
    shift: Complex64,
    shift_index: usize,
) {
    let n = h.nrows();
    if n < 2 {
        return;
    }

    let mut istart = 0;
    while istart < n {
        let mut iend = n - 1;
        for i in istart..n - 1 {
            if should_deflate(h, i) {
                h[[i + 1, i]] = Complex64::ZERO;
                iend = i;
                break;
            }
        }

        if istart == iend {
            istart = iend + 1;
            continue;
        }

        let mut f = h[[istart, istart]] - shift;
        let mut g = h[[istart + 1, istart]];

        for i in istart..iend {
            let (c, s, r) = zlartg(f, g);
            if i > istart {
                h[[i, i - 1]] = r;
                h[[i + 1, i - 1]] = Complex64::ZERO;
            }

            apply_givens_from_left(h, i, c, s);
            apply_givens_from_right(h, i, iend, c, s);
            accumulate_givens(rotation, i, shift_index, c, s);

            if i < iend - 1 {
                f = h[[i + 1, i]];
                g = h[[i + 2, i]];
            }
        }

        istart = iend + 1;
    }
}

fn zlartg(f: Complex64, g: Complex64) -> (f64, Complex64, Complex64) {
    if g == Complex64::ZERO {
        return (1.0, Complex64::ZERO, f);
    }
    if f == Complex64::ZERO {
        return (0.0, g.conj() / g.norm(), Complex64::new(g.norm(), 0.0));
    }

    let f_abs = f.norm();
    let g_abs = g.norm();
    let scale = f_abs.max(g_abs);
    let fs = f_abs / scale;
    let gs = g_abs / scale;
    let r_abs = scale * (fs * fs + gs * gs).sqrt();
    let alpha = f / f_abs;
    let c = f_abs / r_abs;
    let s = alpha * g.conj() / r_abs;
    let r = alpha * r_abs;

    (c, s, r)
}

fn apply_givens_from_left(h: &mut Array2<Complex64>, i: usize, c: f64, s: Complex64) {
    let c = Complex64::new(c, 0.0);
    for column in i..h.ncols() {
        let upper = h[[i, column]];
        let lower = h[[i + 1, column]];
        h[[i, column]] = c * upper + s * lower;
        h[[i + 1, column]] = -s.conj() * upper + c * lower;
    }
}

fn apply_givens_from_right(h: &mut Array2<Complex64>, i: usize, iend: usize, c: f64, s: Complex64) {
    let c = Complex64::new(c, 0.0);
    for row in 0..=usize::min(i + 2, iend) {
        let left = h[[row, i]];
        let right = h[[row, i + 1]];
        h[[row, i]] = c * left + s.conj() * right;
        h[[row, i + 1]] = -s * left + c * right;
    }
}

fn accumulate_givens(
    rotation: &mut Array2<Complex64>,
    i: usize,
    shift_index: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let row_count = usize::min(i + shift_index + 2, rotation.nrows());

    for row in 0..row_count {
        let left = rotation[[row, i]];
        let right = rotation[[row, i + 1]];
        rotation[[row, i]] = c * left + s.conj() * right;
        rotation[[row, i + 1]] = -s * left + c * right;
    }
}

fn make_subdiagonal_real_nonnegative(h: &mut Array2<Complex64>, rotation: &mut Array2<Complex64>) {
    let n = h.nrows();
    for j in 0..n.saturating_sub(1) {
        let subdiagonal = h[[j + 1, j]];
        let magnitude = subdiagonal.norm();
        if magnitude == 0.0 || (subdiagonal.im == 0.0 && subdiagonal.re >= 0.0) {
            continue;
        }

        let phase = subdiagonal / magnitude;
        for column in j..n {
            h[[j + 1, column]] *= phase.conj();
        }
        for row in 0..=usize::min(j + 2, n - 1) {
            h[[row, j + 1]] *= phase;
        }
        for row in 0..rotation.nrows() {
            rotation[[row, j + 1]] *= phase;
        }
        h[[j + 1, j]] = Complex64::new(magnitude, 0.0);
    }
}

fn deflate_small_subdiagonals(h: &mut Array2<Complex64>) {
    for i in 0..h.nrows().saturating_sub(1) {
        if should_deflate(h, i) {
            h[[i + 1, i]] = Complex64::ZERO;
        }
    }
}

fn should_deflate(h: &Array2<Complex64>, i: usize) -> bool {
    let mut scale = zabs1(h[[i, i]]) + zabs1(h[[i + 1, i + 1]]);
    if scale == 0.0 {
        scale = hessenberg_one_norm(h);
    }

    h[[i + 1, i]].norm() <= (f64::EPSILON * scale).max(safe_minimum_threshold(h.nrows()))
}

fn safe_minimum_threshold(n: usize) -> f64 {
    f64::MIN_POSITIVE * (n.max(1) as f64 / f64::EPSILON)
}

fn hessenberg_one_norm(h: &Array2<Complex64>) -> f64 {
    (0..h.ncols())
        .map(|column| {
            let last_row = usize::min(column + 1, h.nrows().saturating_sub(1));
            (0..=last_row).map(|row| zabs1(h[[row, column]])).sum()
        })
        .fold(0.0, f64::max)
}

fn zabs1(value: Complex64) -> f64 {
    value.re.abs() + value.im.abs()
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
    fn zgeev_right_eigenpairs_smoke_test() {
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

        let mut out = zgeev_right_eigenpairs(&h).unwrap();
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
