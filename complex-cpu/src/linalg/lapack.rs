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

    fn zlartg_(
        f: *const Complex64,
        g: *const Complex64,
        cs: *mut f64,
        sn: *mut Complex64,
        r: *mut Complex64,
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

    if shifts.is_empty() {
        return Ok((rotation, h));
    }

    for (shift_index, &shift) in shifts.iter().enumerate() {
        apply_implicit_shift(&mut h, &mut rotation, shift, shift_index);
        cleanup_hessenberg_roundoff(&mut h);
    }

    make_subdiagonal_real_nonnegative(&mut h, &mut rotation);
    deflate_small_subdiagonals(&mut h);
    cleanup_hessenberg_roundoff(&mut h);

    Ok((rotation, h))
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
    let mut c = 0.0;
    let mut s = Complex64::ZERO;
    let mut r = Complex64::ZERO;
    unsafe {
        zlartg_(&f, &g, &mut c, &mut s, &mut r);
    }

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
