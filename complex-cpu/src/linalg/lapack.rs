use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
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
pub struct SchurOutput {
    /// Eigenvalues.
    pub w: Vec<Complex64>,
    /// Schur form T in LAPACK/Fortran column-major layout.
    pub t: Vec<Complex64>,
    /// Schur vectors Z in LAPACK/Fortran column-major layout.
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
    LapackIllegalArgument(i32),
    NoConvergence(i32),
    DimensionMismatch,
    InvalidEigenIndex(usize),
}

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
}

/// Copies an ndarray row-major/strided matrix into LAPACK column-major storage.
///
/// This allocation is necessary unless the caller already owns a contiguous
/// Fortran-order buffer, because LAPACK mutates its input in-place.
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
    let transa = trans_char(trans_a);
    let transb = trans_char(trans_b);

    unsafe {
        zgemm_(
            &transa,
            &transb,
            &m,
            &n,
            &k,
            &alpha,
            a_memory.as_ptr(),
            &lda,
            b_memory.as_ptr(),
            &ldb,
            &beta,
            result_memory.as_mut_ptr(),
            &ldc,
        );
    }

    result
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

pub fn zhseqr_schur(h: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
    let (n, m) = h.dim();
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
    let ilo = 1_i32;
    let ihi = n_i;

    let mut h_col = to_fortran_vec(h);
    let mut w = vec![zero(); n];
    let mut z = vec![zero(); n * n];

    let mut work_query = [zero(); 1];
    let mut info = 0_i32;

    unsafe {
        lapack::zhseqr(
            b'S',
            b'I',
            n_i,
            ilo,
            ihi,
            &mut h_col,
            n_i,
            &mut w,
            &mut z,
            n_i,
            &mut work_query,
            -1,
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

pub fn zgees_schur(a: &Array2<Complex64>) -> Result<SchurOutput, SchurError> {
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

    let n_i = n as i32;
    let mut a_col = to_fortran_vec(a);
    let mut w = vec![zero(); n];
    let mut z = vec![zero(); n * n];
    let mut rwork = vec![0.0; n];
    let mut bwork = vec![0_i32; n];

    let mut sdim = 0_i32;
    let mut work_query = [zero(); 1];
    let mut info = 0_i32;

    unsafe {
        lapack::zgees(
            b'V',
            b'N',
            None,
            n_i,
            &mut a_col,
            n_i,
            &mut sdim,
            &mut w,
            &mut z,
            n_i,
            &mut work_query,
            -1,
            &mut rwork,
            &mut bwork,
            &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
    }

    let lwork = (work_query[0].re as i32).max(2 * n_i).max(1);
    let mut work = vec![zero(); lwork as usize];
    let mut sdim = 0_i32;
    let mut info = 0_i32;

    unsafe {
        lapack::zgees(
            b'V', b'N', None, n_i, &mut a_col, n_i, &mut sdim, &mut w, &mut z, n_i, &mut work,
            lwork, &mut rwork, &mut bwork, &mut info,
        );
    }

    if info < 0 {
        return Err(SchurError::LapackIllegalArgument(-info));
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

    let mut select = vec![0_i32; dim];
    for &j in indices {
        select[j] = 1;
    }

    let mm = indices.len() as i32;
    let mut m_out = 0_i32;

    let mut vl_dummy = [zero(); 1];
    let mut vr_sel = vec![zero(); dim * indices.len()];
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

    let x_sel = fortran_view(dim, m_out as usize, &vr_sel)?;
    let z = fortran_view(dim, dim, &decomposition.z)?;

    // Единственная новая матрица здесь — результат Z * X.
    Ok(z.dot(&x_sel))
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
    let mut work_query = [zero(); 1];
    let mut info = 0;

    unsafe {
        lapack::zgeqrf(m, n, &mut q, lda, &mut tau, &mut work_query, -1, &mut info);
    }
    if info != 0 {
        return Err(format!("zgeqrf workspace query failed, info = {}", info));
    }

    let lwork = (work_query[0].re as i32).max(1);
    let mut work = vec![zero(); lwork as usize];

    unsafe {
        lapack::zgeqrf(m, n, &mut q, lda, &mut tau, &mut work, lwork, &mut info);
    }
    if info != 0 {
        return Err(format!("zgeqrf failed, info = {}", info));
    }

    work_query[0] = zero();
    unsafe {
        lapack::zungqr(m, n, k, &mut q, lda, &tau, &mut work_query, -1, &mut info);
    }
    if info != 0 {
        return Err(format!("zungqr workspace query failed, info = {}", info));
    }

    let lwork = (work_query[0].re as i32).max(1);
    let mut work = vec![zero(); lwork as usize];

    unsafe {
        lapack::zungqr(m, n, k, &mut q, lda, &tau, &mut work, lwork, &mut info);
    }
    if info != 0 {
        return Err(format!("zungqr failed, info = {}", info));
    }

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
    let mut work_query = [zero(); 1];
    let mut info = 0;

    unsafe {
        lapack::zgeqrf(m, n, &mut a, lda, &mut tau, &mut work_query, -1, &mut info);
    }
    if info != 0 {
        return Err(format!("zgeqrf workspace query failed, info = {info}"));
    }

    let lwork = (work_query[0].re as i32).max(n).max(1);
    let mut work = vec![zero(); lwork as usize];

    unsafe {
        lapack::zgeqrf(m, n, &mut a, lda, &mut tau, &mut work, lwork, &mut info);
    }
    if info != 0 {
        return Err(format!("zgeqrf failed, info = {info}"));
    }

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
    work_query[0] = zero();
    unsafe {
        lapack::zungqr(
            m,
            q_columns,
            q_columns,
            &mut a,
            lda,
            &tau,
            &mut work_query,
            -1,
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("zungqr workspace query failed, info = {info}"));
    }

    let lwork = (work_query[0].re as i32).max(q_columns).max(1);
    let mut work = vec![zero(); lwork as usize];
    unsafe {
        lapack::zungqr(
            m, q_columns, q_columns, &mut a, lda, &tau, &mut work, lwork, &mut info,
        );
    }
    if info != 0 {
        return Err(format!("zungqr failed, info = {info}"));
    }

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

    let mut work_query = [zero(); 1];
    let mut info = 0;

    unsafe {
        lapack::zgeqrf(
            m,
            n,
            &mut mat,
            lda,
            &mut tau,
            &mut work_query,
            -1,
            &mut info,
        );
    }
    if info != 0 {
        return Err(format!("zgeqrf workspace query failed, info = {}", info));
    }

    let lwork = work_query[0].re as i32;
    if lwork <= 0 {
        return Err(format!("zgeqrf returned invalid lwork = {}", lwork));
    }

    let mut work = vec![zero(); lwork as usize];

    unsafe {
        lapack::zgeqrf(m, n, &mut mat, lda, &mut tau, &mut work, lwork, &mut info);
    }
    if info != 0 {
        return Err(format!("zgeqrf failed, info = {}", info));
    }

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
