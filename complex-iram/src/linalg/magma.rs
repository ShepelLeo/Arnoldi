//! MAGMA / cuSPARSE FFI слой. Только slice/column-major API.
//!
//! Все host-вход/выход — непрерывные `&[Complex64]` / `&mut [Complex64]` в
//! column-major раскладке с явным `ld`. На стороне устройства матрицы
//! хранятся как column-major блоки в `DeviceMatrix` с `lda = rows`.

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

/// Результат `magma_zgeev`: собственные значения + правые собственные
/// векторы в column-major раскладке.
#[derive(Debug)]
pub struct SchurOutput {
    /// Собственные значения, в том же порядке, что и колонки `z`.
    pub w: Vec<Complex64>,
    /// Рабочий буфер `a` после in-place `magma_zgeev`. Не Schur форма,
    /// сохранён для диагностики.
    pub t: Vec<Complex64>,
    /// Правые собственные векторы в column-major, одна колонка на вектор.
    pub z: Vec<Complex64>,
}

#[derive(Debug)]
pub enum SchurError {
    NotSquare,
    MagmaIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
}

#[derive(Debug)]
pub struct CusparseError(pub i32);

const MAGMA_NO_TRANS: c_int = 111;
const MAGMA_CONJ_TRANS: c_int = 113;
const MAGMA_NO_VEC: c_int = 301;
const MAGMA_VEC: c_int = 302;
const MAGMA_SUCCESS: c_int = 0;
const MAGMA_FUNC: *const c_char = b"rust\0".as_ptr().cast();
const MAGMA_FILE: *const c_char = b"src/linalg/magma.rs\0".as_ptr().cast();

static MAGMA_INIT: Once = Once::new();

type MagmaInt = c_int;
type MagmaDevice = c_int;
type CusparseStatus = c_int;
type CusparseHandle = *mut c_void;

const CUSPARSE_SUCCESS: CusparseStatus = 0;

#[repr(C)]
pub struct MagmaQueueOpaque {
    _private: [u8; 0],
}

type MagmaQueue = *mut MagmaQueueOpaque;

unsafe extern "C" {
    fn magma_init() -> MagmaInt;

    fn magma_getdevice(device: *mut MagmaDevice);

    fn magma_queue_create_internal(
        device: MagmaDevice,
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

    fn magma_malloc(ptr_ptr: *mut *mut c_void, bytes: usize) -> MagmaInt;

    fn magma_free_internal(
        ptr: *mut c_void,
        func: *const c_char,
        file: *const c_char,
        line: c_int,
    ) -> MagmaInt;

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

    fn cusparseCreate(handle: *mut CusparseHandle) -> CusparseStatus;
    fn cusparseDestroy(handle: CusparseHandle) -> CusparseStatus;

    // Минимальный набор CUDA Runtime API, нужный для форсирования primary
    // CUDA context на текущем CPU-thread до вызова `cusparseCreate`.
    // `cudart` уже линкуется через build.rs.
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaGetLastError() -> c_int;

    // CUDA 12.x больше не экспортирует устаревший `cusparseZcsrmv`;
    // C++/CUDA TU реализует эту обёртку поверх `cusparseSpMV`.
    fn complex_iram_cusparse_zcsrmv(
        handle: CusparseHandle,
        dimension: c_int,
        nnz: c_int,
        csr_row_offsets: *const c_int,
        csr_columns: *const c_int,
        csr_values: *const Complex64,
        x: *const Complex64,
        y: *mut Complex64,
    ) -> CusparseStatus;
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

struct DeviceI32Buffer {
    ptr: *mut c_int,
    #[allow(dead_code)]
    len: usize,
    bytes: usize,
}

impl DeviceI32Buffer {
    fn from_slice(values: &[c_int]) -> Self {
        ensure_magma_initialized();
        let mut ptr = ptr::null_mut();
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<c_int>())
            .expect("MAGMA i32 allocation size overflow");
        let status = unsafe { magma_malloc(&mut ptr, bytes) };
        assert_eq!(status, MAGMA_SUCCESS, "magma_malloc failed with status {status}");
        memory::record_device_allocation(bytes);

        if !values.is_empty() {
            unsafe {
                magma_setvector_internal(
                    values.len() as MagmaInt,
                    std::mem::size_of::<c_int>() as MagmaInt,
                    values.as_ptr().cast::<c_void>(),
                    1,
                    ptr,
                    1,
                    MagmaSession::new().queue.raw,
                    MAGMA_FUNC,
                    MAGMA_FILE,
                    line!() as c_int,
                );
            }
            memory::record_host_to_device_copy(bytes);
        }

        Self {
            ptr: ptr.cast::<c_int>(),
            len: values.len(),
            bytes,
        }
    }
}

impl Drop for DeviceI32Buffer {
    fn drop(&mut self) {
        unsafe {
            magma_free_internal(self.ptr.cast::<c_void>(), MAGMA_FUNC, MAGMA_FILE, line!() as c_int);
        }
        memory::record_device_deallocation(self.bytes);
    }
}

struct CusparseContext {
    raw: CusparseHandle,
}

impl CusparseContext {
    fn new() -> Self {
        // cuSPARSE требует валидный primary CUDA context, привязанный к
        // текущему CPU-thread. `ensure_magma_initialized` уже выполняет
        // `magma_init` + `cudaSetDevice` + `cudaFree(NULL)` warm-up на
        // первом входе в модуль, поэтому здесь достаточно повторно
        // проверить, что инициализация состоялась.
        ensure_magma_initialized();

        let mut raw = ptr::null_mut();
        let status = unsafe { cusparseCreate(&mut raw) };
        if status != CUSPARSE_SUCCESS || raw.is_null() {
            // Сообщение со списком переменных среды, которые чаще всего
            // приводят к этой ошибке на SLURM-узлах.
            panic!(
                "cusparseCreate failed (status {status}, handle null = {is_null}). \
                 Likely the CUDA primary context is not initialized for this \
                 thread/process. On SLURM make sure CUDA_VISIBLE_DEVICES is set, \
                 the node actually has a GPU allocated to this job, and that \
                 LD_LIBRARY_PATH points at the same CUDA runtime as the one \
                 used to build the cuSPARSE wrapper (libcusparse.so, \
                 libcudart.so).",
                is_null = raw.is_null(),
            );
        }
        Self { raw }
    }
}

impl Drop for CusparseContext {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        unsafe {
            cusparseDestroy(self.raw);
        }
    }
}

fn ensure_magma_initialized() {
    MAGMA_INIT.call_once(|| {
        let status = unsafe { magma_init() };
        if status != MAGMA_SUCCESS {
            panic!(
                "magma_init failed with status {status}. \
                 Verify MAGMA_DIR / MAGMA_LIB_DIR at build time and that the \
                 runtime CUDA driver/runtime matches the one MAGMA was built against."
            );
        }
        // Прогреваем CUDA Runtime primary context сразу после `magma_init`,
        // чтобы любой последующий вызов (cuSPARSE, MAGMA queue create,
        // magma_zsetvector в `DeviceI32Buffer::from_slice`, и т.д.) уже видел
        // привязанный context на этом thread.
        prime_cuda_runtime_context();
    });
}

/// Force-create a CUDA primary context on the current thread.
///
/// Called once from `ensure_magma_initialized`. The MAGMA driver-level context
/// is created by `magma_init`, but `cusparseCreate` and some MAGMA Runtime API
/// calls assume the CUDA Runtime API primary context is also bound to the
/// caller's thread. On a freshly-spawned Slurm task that's not guaranteed —
/// the Runtime primary context is materialized lazily on the first Runtime
/// call. We make that first call explicit:
///
/// 1. `magma_getdevice` — pick the device MAGMA is using.
/// 2. `cudaSetDevice(dev)` — bind the Runtime context to this thread.
/// 3. `cudaFree(NULL)` — canonical zero-cost warm-up that materializes the
///    Runtime primary context if it hasn't been created yet.
fn prime_cuda_runtime_context() {
    let mut device: c_int = 0;
    unsafe {
        magma_getdevice(&mut device);
    }

    let set_status = unsafe { cudaSetDevice(device) };
    if set_status != 0 {
        let _ = unsafe { cudaGetLastError() };
        panic!(
            "cudaSetDevice({device}) failed with CUDA error {set_status}. \
             Check CUDA_VISIBLE_DEVICES and that the Slurm job actually has a \
             GPU allocated."
        );
    }

    let free_status = unsafe { cudaFree(ptr::null_mut()) };
    if free_status != 0 {
        let _ = unsafe { cudaGetLastError() };
        panic!(
            "cudaFree(NULL) warm-up failed with CUDA error {free_status} on \
             device {device}. The CUDA Runtime primary context could not be \
             created — likely no GPU is visible to this process."
        );
    }

    let mut bound: c_int = -1;
    let get_status = unsafe { cudaGetDevice(&mut bound) };
    if get_status != 0 || bound < 0 {
        let _ = unsafe { cudaGetLastError() };
        panic!(
            "cudaGetDevice failed (status {get_status}, bound = {bound}) after \
             warm-up. CUDA context creation appears to have silently failed."
        );
    }
}

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
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

/// Reusable MAGMA/CUDA execution context.
pub struct MagmaSession {
    queue: Queue,
    cusparse: CusparseContext,
}

impl MagmaSession {
    pub fn new() -> Self {
        Self {
            queue: Queue::new(),
            cusparse: CusparseContext::new(),
        }
    }
}

impl Default for MagmaSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Device-side вектор. Владеет MAGMA-аллокацией.
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
    pub(crate) fn mut_ptr(&mut self) -> *mut Complex64 {
        self.buffer.ptr
    }
}

/// Device-side column-major плотная матрица. `lda == rows`.
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

    /// Загружает host column-major матрицу (`m × n`, ld = `host_ld`) в device.
    pub fn copy_from_host_slice(
        &mut self,
        session: &MagmaSession,
        host: &[Complex64],
        host_ld: usize,
    ) {
        let rows = self.rows;
        let cols = self.columns;
        if rows == 0 || cols == 0 {
            return;
        }
        assert!(host.len() >= host_ld * cols.saturating_sub(1) + rows);

        unsafe {
            magma_zsetmatrix(
                rows as c_int,
                cols as c_int,
                host.as_ptr(),
                host_ld.max(1) as c_int,
                self.buffer.ptr,
                self.lda,
                session.queue.raw,
            );
            memory::record_host_to_device_copy(rows * cols * std::mem::size_of::<Complex64>());
        }
    }

    /// Скачивает device матрицу в host column-major буфер с заданным ld.
    pub fn copy_to_host_slice(
        &self,
        session: &MagmaSession,
        host: &mut [Complex64],
        host_ld: usize,
    ) {
        let rows = self.rows;
        let cols = self.columns;
        if rows == 0 || cols == 0 {
            return;
        }
        assert!(host.len() >= host_ld * cols.saturating_sub(1) + rows);

        unsafe {
            magma_zgetmatrix(
                rows as c_int,
                cols as c_int,
                self.buffer.ptr,
                self.lda,
                host.as_mut_ptr(),
                host_ld.max(1) as c_int,
                session.queue.raw,
            );
            memory::record_device_to_host_copy(rows * cols * std::mem::size_of::<Complex64>());
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn column_ptr(&self, column: usize) -> *const Complex64 {
        assert!(column < self.columns);
        unsafe { self.buffer.ptr.add(column * self.lda as usize) }
    }

    /// Загружает host-вектор в указанный столбец device-матрицы.
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

    /// Скачивает один столбец device-матрицы в host-срез.
    pub fn copy_column_to_slice(
        &self,
        session: &MagmaSession,
        column: usize,
        values: &mut [Complex64],
    ) {
        assert!(column < self.columns);
        assert_eq!(values.len(), self.rows);
        if self.rows == 0 {
            return;
        }
        unsafe {
            let source = self.buffer.ptr.add(column * self.lda as usize);
            magma_zgetvector(
                self.rows as c_int,
                source,
                1,
                values.as_mut_ptr(),
                1,
                session.queue.raw,
            );
            memory::record_device_to_host_copy(self.rows * std::mem::size_of::<Complex64>());
        }
    }

    /// GEMV против первых `columns` колонок device-матрицы. Это primitive
    /// `Y_kry := alpha * V[:, 0..n] * x + beta * Y` без материализации
    /// subview-копии.
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
                x.buffer.ptr,
                1,
                beta,
                y.mut_ptr(),
                1,
                session.queue.raw,
            );
        }
    }
}

/// Device-side CSR матрица для cuSPARSE SpMV.
pub struct DeviceCsrMatrix {
    dimension: usize,
    nnz: usize,
    d_row_offsets: DeviceI32Buffer,
    d_columns: DeviceI32Buffer,
    d_values: DeviceBuffer,
}

impl DeviceCsrMatrix {
    pub fn from_csr(
        session: &MagmaSession,
        dimension: usize,
        row_offsets: &[usize],
        columns: &[usize],
        values: &[Complex64],
    ) -> Result<Self, CusparseError> {
        assert_eq!(row_offsets.len(), dimension + 1);
        assert_eq!(columns.len(), values.len());

        let row_offsets_i32 = row_offsets
            .iter()
            .map(|&value| c_int::try_from(value).expect("CSR row offset exceeds c_int"))
            .collect::<Vec<_>>();
        let columns_i32 = columns
            .iter()
            .map(|&value| c_int::try_from(value).expect("CSR column index exceeds c_int"))
            .collect::<Vec<_>>();

        let d_row_offsets = DeviceI32Buffer::from_slice(&row_offsets_i32);
        let d_columns = DeviceI32Buffer::from_slice(&columns_i32);
        let d_values = DeviceBuffer::new(values.len().max(1));
        if !values.is_empty() {
            unsafe {
                magma_zsetvector(
                    values.len() as c_int,
                    values.as_ptr(),
                    1,
                    d_values.ptr,
                    1,
                    session.queue.raw,
                );
            }
            memory::record_host_to_device_copy(values.len() * std::mem::size_of::<Complex64>());
        }

        Ok(Self {
            dimension,
            nnz: values.len(),
            d_row_offsets,
            d_columns,
            d_values,
        })
    }

    /// SpMV: `y = A * x`, где `x`, `y` — device-указатели длиной `dimension`.
    pub fn spmv_raw(
        &self,
        session: &MagmaSession,
        x: *const Complex64,
        y: *mut Complex64,
    ) -> Result<(), CusparseError> {
        let status = unsafe {
            complex_iram_cusparse_zcsrmv(
                session.cusparse.raw,
                self.dimension as c_int,
                self.nnz as c_int,
                self.d_row_offsets.ptr,
                self.d_columns.ptr,
                self.d_values.ptr,
                x,
                y,
            )
        };

        if status == CUSPARSE_SUCCESS {
            Ok(())
        } else {
            Err(CusparseError(status))
        }
    }
}

/// `C = op(A) * op(B)` через MAGMA `zgemm` с реиспользуемой сессией. Все буферы
/// — host column-major. Полезен для редких больших плотных GEMM в рестарте.
pub fn zgemm_with_session_slice(
    session: &MagmaSession,
    trans_a: ZgemmTranspose,
    trans_b: ZgemmTranspose,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    c: &mut [Complex64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let (a_rows, a_columns) = match trans_a {
        ZgemmTranspose::None => (m, k),
        ZgemmTranspose::ConjugateTranspose => (k, m),
    };
    let (b_rows, b_columns) = match trans_b {
        ZgemmTranspose::None => (k, n),
        ZgemmTranspose::ConjugateTranspose => (n, k),
    };

    let mut d_a = DeviceMatrix::new(a_rows, a_columns);
    let mut d_b = DeviceMatrix::new(b_rows, b_columns);
    let d_c = DeviceMatrix::new(m, n);

    d_a.copy_from_host_slice(session, a, lda);
    d_b.copy_from_host_slice(session, b, ldb);

    unsafe {
        magma_zgemm(
            magma_trans(trans_a),
            magma_trans(trans_b),
            m as c_int,
            n as c_int,
            k as c_int,
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

    d_c.copy_to_host_slice(session, c, ldc);
}

/// Eigenvalues + right eigenvectors малой плотной `n × n` матрицы через
/// MAGMA `zgeev`. Вход — column-major, не изменяется. Выход — column-major
/// `Z` (правые векторы), `w` (значения), `t` (in-place рабочий буфер).
pub fn zgeev_right_eigenpairs_slice(
    h_col_major: &[Complex64],
    n: usize,
) -> Result<SchurOutput, SchurError> {
    if h_col_major.len() != n * n {
        return Err(SchurError::DimensionMismatch);
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
    let mut a_col = h_col_major.to_vec();
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
