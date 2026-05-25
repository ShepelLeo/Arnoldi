//! LAPACK/OpenBLAS FFI слой. Только slice/column-major API.
//!
//! Все матрицы — непрерывные `&[Complex64]` / `&mut [Complex64]` в column-major
//! раскладке с явным `ld`. Никаких `ndarray`-сигнатур.

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

/// Результат малой спектральной задачи (Schur форма от `zhseqr`).
#[derive(Debug)]
pub struct SchurOutput {
    /// Собственные значения.
    pub w: Vec<Complex64>,
    /// Schur форма `T` в column-major раскладке (после `zhseqr`).
    pub t: Vec<Complex64>,
    /// Schur векторы `Z` в column-major раскладке.
    pub z: Vec<Complex64>,
}

#[derive(Debug)]
pub enum SchurError {
    NotSquare,
    LapackIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
}

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
}

unsafe extern "C" {
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
}

fn trans_char(trans: ZgemmTranspose) -> c_char {
    match trans {
        ZgemmTranspose::None => b'N' as c_char,
        ZgemmTranspose::ConjugateTranspose => b'C' as c_char,
    }
}

/// `C = op(A) * op(B)` (alpha=1, beta=0). Все буферы column-major, ld явные.
/// `m × n` — финальный размер `C`; `k` — общая размерность контракции.
pub fn zgemm_slice(
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
    if m == 0 || n == 0 {
        return;
    }

    let alpha = Complex64::new(1.0, 0.0);
    let beta = Complex64::ZERO;
    let m_i = m as c_int;
    let n_i = n as c_int;
    let k_i = k as c_int;
    let lda_i = lda.max(1) as c_int;
    let ldb_i = ldb.max(1) as c_int;
    let ldc_i = ldc.max(1) as c_int;
    let transa = trans_char(trans_a);
    let transb = trans_char(trans_b);

    unsafe {
        zgemm_(
            &transa,
            &transb,
            &m_i,
            &n_i,
            &k_i,
            &alpha,
            a.as_ptr(),
            &lda_i,
            b.as_ptr(),
            &ldb_i,
            &beta,
            c.as_mut_ptr(),
            &ldc_i,
        );
    }
}

/// `y = alpha * op(A) * x + beta * y`. Все буферы column-major, ld явное.
/// `rows × cols` — фактические размеры `A`; форма входа/выхода зависит от `trans`.
pub fn zgemv_slice(
    trans: ZgemvTranspose,
    rows: usize,
    cols: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
) {
    if rows == 0 || cols == 0 {
        return;
    }

    let trans_char = match trans {
        ZgemvTranspose::None => b'N' as c_char,
        ZgemvTranspose::ConjugateTranspose => b'C' as c_char,
    };
    let rows_i = rows as c_int;
    let cols_i = cols as c_int;
    let lda_i = lda.max(1) as c_int;
    let incx = 1 as c_int;
    let incy = 1 as c_int;

    unsafe {
        zgemv_(
            &trans_char,
            &rows_i,
            &cols_i,
            &alpha,
            a.as_ptr(),
            &lda_i,
            x.as_ptr(),
            &incx,
            &beta,
            y.as_mut_ptr(),
            &incy,
        );
    }
}

/// Schur-форма + собственные значения малой плотной `n×n` матрицы через
/// LAPACK `zhseqr`. Вход — column-major, не изменяется. Выход — column-major
/// `T`, `Z`, `w`.
pub fn zhseqr_schur_slice(
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

    let n_i = n as i32;
    let ilo = 1_i32;
    let ihi = n_i;

    let mut h_col = h_col_major.to_vec();
    let mut w = vec![zero(); n];
    let mut z = vec![zero(); n * n];

    let mut work_query = [zero(); 1];
    let mut info = 0_i32;

    unsafe {
        lapack::zhseqr(
            b'S', b'I', n_i, ilo, ihi, &mut h_col, n_i, &mut w, &mut z, n_i, &mut work_query, -1,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
    }

    let lwork = (work_query[0].re as i32).max(n_i).max(1);
    let mut work = vec![zero(); lwork as usize];

    unsafe {
        lapack::zhseqr(
            b'S', b'I', n_i, ilo, ihi, &mut h_col, n_i, &mut w, &mut z, n_i, &mut work, lwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
    }
    if info > 0 {
        return Err(SchurError::NoConvergence(info));
    }

    Ok(SchurOutput { w, t: h_col, z })
}

/// Все правые собственные векторы Schur-формы, бэк-трансформированные
/// Schur-векторами (т.е. `Z * X`). Выход — column-major `dim × dim`.
pub fn ztrevc_all_right_slice(
    decomposition: &mut SchurOutput,
    dim: usize,
) -> Result<Vec<Complex64>, SchurError> {
    if decomposition.t.len() != dim * dim || decomposition.z.len() != dim * dim {
        return Err(SchurError::DimensionMismatch);
    }
    if dim == 0 {
        return Ok(Vec::new());
    }

    let select = vec![1_i32; dim];
    let mm = dim as i32;
    let mut m_out = 0_i32;

    let mut vl_dummy = [zero(); 1];
    let mut vr_sel = vec![zero(); dim * dim];
    let mut work = vec![zero(); 2 * dim];
    let mut rwork = vec![0.0_f64; dim];
    let mut info = 0_i32;

    unsafe {
        lapack::ztrevc(
            b'R',
            b'S',
            &select,
            dim as i32,
            &mut decomposition.t,
            dim as i32,
            &mut vl_dummy,
            1,
            &mut vr_sel,
            dim as i32,
            mm,
            &mut m_out,
            &mut work,
            &mut rwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
    }

    let mut result = vec![zero(); dim * dim];
    zgemm_slice(
        ZgemmTranspose::None,
        ZgemmTranspose::None,
        dim,
        dim,
        dim,
        &decomposition.z,
        dim,
        &vr_sel,
        dim,
        &mut result,
        dim,
    );
    Ok(result)
}
