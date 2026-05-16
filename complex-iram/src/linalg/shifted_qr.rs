//! Common implicit shifted QR filter used by all backends.
//!
//! The implementation intentionally stays backend-neutral. It operates on the
//! small Hessenberg matrix on the host, uses a row-major working copy for cache
//! locality, and returns Fortran-order arrays because the surrounding BLAS/LAPACK
//! wrappers expect column-major operands.

use ndarray::{Array2, ShapeBuilder, s};
use num_complex::Complex64;

unsafe extern "C" {
    fn zlartg_(
        f: *const Complex64,
        g: *const Complex64,
        cs: *mut f64,
        sn: *mut Complex64,
        r: *mut Complex64,
    );
}

#[inline]
fn zero() -> Complex64 {
    Complex64::ZERO
}

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

fn from_fortran_vec(rows: usize, cols: usize, data: Vec<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_vec((rows, cols).f(), data).expect("invalid Fortran buffer shape")
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

    // Work in row-major layout; the longest sweeps are along rows.
    let mut h = hessenberg.to_owned();
    let mut rotation = complex_identity(n);

    if shifts.is_empty() {
        let rotation_f = from_fortran_vec(n, n, to_fortran_vec(&rotation));
        let h_f = from_fortran_vec(n, n, to_fortran_vec(&h));
        return Ok((rotation_f, h_f));
    }

    let safe_threshold = safe_minimum_threshold(n);

    for (shift_index, shift) in shifts.iter().copied().enumerate() {
        apply_implicit_shift(
            &mut h,
            &mut rotation,
            shift,
            shift_index,
            safe_threshold,
        );

        cleanup_hessenberg_roundoff(&mut h);
    }

    make_subdiagonal_real_nonnegative(&mut h, &mut rotation);
    deflate_small_subdiagonals(&mut h, safe_threshold);
    cleanup_hessenberg_roundoff(&mut h);

    let rotation_f = from_fortran_vec(n, n, to_fortran_vec(&rotation));
    let h_f = from_fortran_vec(n, n, to_fortran_vec(&h));

    Ok((rotation_f, h_f))
}

fn complex_identity(n: usize) -> Array2<Complex64> {
    let mut eye = Array2::<Complex64>::zeros((n, n));

    for i in 0..n {
        eye[[i, i]] = Complex64::new(1.0, 0.0);
    }

    eye
}

fn apply_implicit_shift(
    h: &mut Array2<Complex64>,
    rotation: &mut Array2<Complex64>,
    shift: Complex64,
    shift_index: usize,
    safe_threshold: f64,
) {
    let n = h.nrows();

    if n < 2 {
        return;
    }

    let mut istart = 0usize;

    while istart + 1 < n {
        let mut iend = n - 1;

        for i in istart..n - 1 {
            if should_deflate_fast(h, i, safe_threshold) {
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

            if i + 1 < iend {
                f = h[[i + 1, i]];
                g = h[[i + 2, i]];
            }
        }

        istart = iend + 1;
    }
}

#[inline(always)]
fn zlartg(f: Complex64, g: Complex64) -> (f64, Complex64, Complex64) {
    let mut c = 0.0;
    let mut s = Complex64::ZERO;
    let mut r = Complex64::ZERO;

    unsafe {
        zlartg_(&f, &g, &mut c, &mut s, &mut r);
    }

    (c, s, r)
}

#[inline(always)]
fn apply_givens_from_left(
    h: &mut Array2<Complex64>,
    i: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();

    let (mut upper_row, mut lower_row) =
        h.multi_slice_mut((s![i, i..], s![i + 1, i..]));

    for (upper_ref, lower_ref) in upper_row.iter_mut().zip(lower_row.iter_mut()) {
        let upper = *upper_ref;
        let lower = *lower_ref;

        *upper_ref = c * upper + s * lower;
        *lower_ref = -s_conj * upper + c * lower;
    }
}

#[inline(always)]
fn apply_givens_from_right(
    h: &mut Array2<Complex64>,
    i: usize,
    iend: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();

    let last_row = usize::min(i + 2, iend);

    let (mut left_col, mut right_col) =
        h.multi_slice_mut((s![..=last_row, i], s![..=last_row, i + 1]));

    for (left_ref, right_ref) in left_col.iter_mut().zip(right_col.iter_mut()) {
        let left = *left_ref;
        let right = *right_ref;

        *left_ref = c * left + s_conj * right;
        *right_ref = -s * left + c * right;
    }
}

#[inline(always)]
fn accumulate_givens(
    rotation: &mut Array2<Complex64>,
    i: usize,
    shift_index: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();

    // Fast banded accumulation. It is enough for the restart filter and avoids
    // touching the full dense rotation matrix in early shifts.
    let row_count = usize::min(i + shift_index + 2, rotation.nrows());

    let (mut left_col, mut right_col) =
        rotation.multi_slice_mut((s![..row_count, i], s![..row_count, i + 1]));

    for (left_ref, right_ref) in left_col.iter_mut().zip(right_col.iter_mut()) {
        let left = *left_ref;
        let right = *right_ref;

        *left_ref = c * left + s_conj * right;
        *right_ref = -s * left + c * right;
    }
}

fn make_subdiagonal_real_nonnegative(
    h: &mut Array2<Complex64>,
    rotation: &mut Array2<Complex64>,
) {
    let n = h.nrows();

    for j in 0..n.saturating_sub(1) {
        let subdiagonal = h[[j + 1, j]];
        let magnitude = subdiagonal.norm();

        if magnitude == 0.0 || (subdiagonal.im == 0.0 && subdiagonal.re >= 0.0) {
            continue;
        }

        let phase = subdiagonal / magnitude;
        let phase_conj = phase.conj();

        {
            let mut row = h.slice_mut(s![j + 1, j..]);

            for value in row.iter_mut() {
                *value *= phase_conj;
            }
        }

        {
            let last_row = usize::min(j + 2, n - 1);
            let mut col = h.slice_mut(s![..=last_row, j + 1]);

            for value in col.iter_mut() {
                *value *= phase;
            }
        }

        {
            let mut rot_col = rotation.slice_mut(s![.., j + 1]);

            for value in rot_col.iter_mut() {
                *value *= phase;
            }
        }

        h[[j + 1, j]] = Complex64::new(magnitude, 0.0);
    }
}

fn deflate_small_subdiagonals(h: &mut Array2<Complex64>, safe_threshold: f64) {
    let n = h.nrows();

    for i in 0..n.saturating_sub(1) {
        if should_deflate_fast(h, i, safe_threshold) {
            h[[i + 1, i]] = Complex64::ZERO;
        }
    }
}

#[inline(always)]
fn should_deflate_fast(h: &Array2<Complex64>, i: usize, safe_threshold: f64) -> bool {
    let subdiagonal_norm = h[[i + 1, i]].norm();

    if subdiagonal_norm == 0.0 {
        return true;
    }

    let mut scale = h[[i, i]].norm() + h[[i + 1, i + 1]].norm();

    if scale == 0.0 {
        scale = hessenberg_one_norm(h);
    }

    subdiagonal_norm <= (f64::EPSILON * scale).max(safe_threshold)
}

fn safe_minimum_threshold(n: usize) -> f64 {
    f64::MIN_POSITIVE * (n.max(1) as f64 / f64::EPSILON)
}

fn hessenberg_one_norm(h: &Array2<Complex64>) -> f64 {
    (0..h.ncols())
        .map(|column| {
            let last_row = usize::min(column + 1, h.nrows().saturating_sub(1));
            (0..=last_row).map(|row| h[[row, column]].norm()).sum()
        })
        .fold(0.0, f64::max)
}

fn cleanup_hessenberg_roundoff(h: &mut Array2<Complex64>) {
    for row in 0..h.nrows() {
        for column in 0..row.saturating_sub(1) {
            h[[row, column]] = Complex64::ZERO;
        }
    }
}
