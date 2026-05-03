use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder};
use num_complex::Complex64;
use std::os::raw::{c_char, c_int};

#[derive(Debug, Clone, Copy)]
pub enum ZgemvTranspose {
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

unsafe extern "C" {
    fn zlaqr5_(
        wantt: *const c_char,
        wantz: *const c_char,
        kacc22: *const c_int,
        n: *const c_int,
        ktop: *const c_int,
        kbot: *const c_int,
        nshfts: *const c_int,
        s: *mut Complex64,
        h: *mut Complex64,
        ldh: *const c_int,
        iloz: *const c_int,
        ihiz: *const c_int,
        z: *mut Complex64,
        ldz: *const c_int,
        v: *mut Complex64,
        ldv: *const c_int,
        u: *mut Complex64,
        ldu: *const c_int,
        nv: *const c_int,
        wv: *mut Complex64,
        ldwv: *const c_int,
        nh: *const c_int,
        wh: *mut Complex64,
        ldwh: *const c_int,
    );
}

pub fn zlaqr52(
    hessenberg: &mut Array2<Complex64>,
    shifts: &[Complex64],
) -> Result<(Array2<Complex64>, Array2<Complex64>), String> {
    let (h_rows, h_cols) = hessenberg.dim();
    if h_rows != h_cols {
        return Err("H must be square".into());
    }
    if shifts.len() < 2 {
        return Err("ZLAQR5 needs at least two shifts".into());
    }

    let n = h_rows;
    let n_i = n as c_int;

    let wantt: c_char = b'T' as c_char;
    let wantz: c_char = b'T' as c_char;
    let kacc22: c_int = 1;
    let ktop: c_int = 1;
    let kbot: c_int = n_i;
    let ldh: c_int = n_i;
    let ldz: c_int = n_i;

    let mut s = Vec::with_capacity(shifts.len() + (shifts.len() & 1));
    s.extend_from_slice(shifts);
    if s.len() % 2 == 1 {
        s.push(Complex64::ZERO);
    }

    let ns_even = s.len();
    let nshfts: c_int = ns_even as c_int;
    let nbmps = ns_even / 2;
    let kdu = 4 * nbmps;

    let ldv: c_int = 3;
    let ldu: c_int = kdu.max(1) as c_int;

    let nv: c_int = n_i;
    let ldwv: c_int = nv.max(1);

    let nh: c_int = n_i;
    let ldwh: c_int = kdu.max(1) as c_int;

    let mut h = to_fortran_vec(hessenberg);
    let mut z = identity_fortran_vec(n);

    let mut v = vec![zero(); (ldv as usize) * nbmps.max(1)];
    let mut u = vec![zero(); (ldu as usize) * kdu.max(1)];
    let mut wv = vec![zero(); (ldwv as usize) * kdu.max(1)];
    let mut wh = vec![zero(); (ldwh as usize) * (nh as usize)];

    let iloz: c_int = 1;
    let ihiz: c_int = n_i;

    unsafe {
        zlaqr5_(
            &wantt,
            &wantz,
            &kacc22,
            &n_i,
            &ktop,
            &kbot,
            &nshfts,
            s.as_mut_ptr(),
            h.as_mut_ptr(),
            &ldh,
            &iloz,
            &ihiz,
            z.as_mut_ptr(),
            &ldz,
            v.as_mut_ptr(),
            &ldv,
            u.as_mut_ptr(),
            &ldu,
            &nv,
            wv.as_mut_ptr(),
            &ldwv,
            &nh,
            wh.as_mut_ptr(),
            &ldwh,
        );
    }

    Ok((from_fortran_vec(n, n, z), from_fortran_vec(n, n, h)))
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
    fn zlaqr_test() {
        let mut h = array![
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

        let out = zlaqr52(
            &mut h,
            &[Complex64::new(3.0, 1.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap();

        let q = out.0;
        let q_star = q.t().mapv(|x| x.conj());
        let res = q.dot(&out.1).dot(&q_star);
        println!("res == {:.3?}", res);
    }
}
