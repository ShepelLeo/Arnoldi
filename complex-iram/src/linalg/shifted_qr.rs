
use ndarray::{Array2, ShapeBuilder};
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

#[inline(always)]
fn zero() -> Complex64 {
    Complex64::ZERO
}

#[inline(always)]
fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

#[inline(always)]
fn idx(n: usize, row: usize, column: usize) -> usize {
    row * n + column
}

/// Reusable storage for the shifted QR filter.
///
/// The one-shot `shifted_qr_filter` creates this internally. If restart code is
/// later changed to keep a backend/workspace object alive, use
/// `shifted_qr_filter_with_workspace` to avoid reallocating these buffers on
/// every restart.
#[derive(Debug, Clone, Default)]
pub struct ShiftedQrWorkspace {
    n: usize,
    h_row_major: Vec<Complex64>,
    q_row_major: Vec<Complex64>,
    h_fortran: Vec<Complex64>,
    q_fortran: Vec<Complex64>,
}

impl ShiftedQrWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dimension(n: usize) -> Self {
        let mut workspace = Self::new();
        workspace.resize(n);
        workspace
    }

    fn resize(&mut self, n: usize) {
        self.n = n;
        let len = n.saturating_mul(n);
        self.h_row_major.resize(len, zero());
        self.q_row_major.resize(len, zero());
        self.h_fortran.resize(len, zero());
        self.q_fortran.resize(len, zero());
    }
}

/// Backward-compatible one-shot API.
pub fn shifted_qr_filter(
    hessenberg: &Array2<Complex64>,
    shifts: &[Complex64],
) -> Result<(Array2<Complex64>, Array2<Complex64>), String> {
    let mut workspace = ShiftedQrWorkspace::new();
    shifted_qr_filter_with_workspace(hessenberg, shifts, &mut workspace)
}

/// Slice-based shifted QR filter (column-major in/out).
///
/// Возвращает `(Q, H_filtered)` в column-major раскладке как обычные `Vec`-буферы
/// без `ndarray`. Это generic-примитив для ядра IRAM/Arnoldi.
pub fn shifted_qr_filter_slice(
    hessenberg: &[Complex64],
    n: usize,
    shifts: &[Complex64],
) -> Result<(Vec<Complex64>, Vec<Complex64>), String> {
    if hessenberg.len() != n * n {
        return Err("H buffer length must be n*n".into());
    }

    let mut workspace = ShiftedQrWorkspace::new();
    workspace.resize(n);

    copy_col_major_to_row_major(hessenberg, &mut workspace.h_row_major, n);
    fill_identity_row_major(&mut workspace.q_row_major, n);

    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    if !shifts.is_empty() {
        let safe_threshold = safe_minimum_threshold(n);
        let h_norm =
            hessenberg_one_norm_row_major(&workspace.h_row_major, n).max(safe_threshold);

        for (shift_index, shift) in shifts.iter().copied().enumerate() {
            apply_implicit_shift_row_major(
                &mut workspace.h_row_major,
                &mut workspace.q_row_major,
                n,
                shift,
                shift_index,
                safe_threshold,
                h_norm,
            );
        }

        make_subdiagonal_real_nonnegative_row_major(
            &mut workspace.h_row_major,
            &mut workspace.q_row_major,
            n,
        );
        deflate_small_subdiagonals_row_major(
            &mut workspace.h_row_major,
            n,
            safe_threshold,
            h_norm,
        );
    }

    cleanup_hessenberg_roundoff_row_major(&mut workspace.h_row_major, n);

    let mut q_fortran = vec![zero(); n * n];
    let mut h_fortran = vec![zero(); n * n];
    copy_row_major_to_fortran(&workspace.q_row_major, &mut q_fortran, n);
    copy_row_major_to_fortran(&workspace.h_row_major, &mut h_fortran, n);

    Ok((q_fortran, h_fortran))
}

fn copy_col_major_to_row_major(input: &[Complex64], output: &mut [Complex64], n: usize) {
    debug_assert_eq!(input.len(), n * n);
    debug_assert_eq!(output.len(), n * n);

    for column in 0..n {
        for row in 0..n {
            output[idx(n, row, column)] = input[row + column * n];
        }
    }
}

/// Workspace-aware shifted QR API.
///
/// Returns `(Q, H_filtered)` in Fortran/column-major ndarray layout.
pub fn shifted_qr_filter_with_workspace(
    hessenberg: &Array2<Complex64>,
    shifts: &[Complex64],
    workspace: &mut ShiftedQrWorkspace,
) -> Result<(Array2<Complex64>, Array2<Complex64>), String> {
    let (h_rows, h_cols) = hessenberg.dim();

    if h_rows != h_cols {
        return Err("H must be square".into());
    }

    let n = h_rows;
    workspace.resize(n);

    copy_ndarray_to_row_major(hessenberg, &mut workspace.h_row_major, n);
    fill_identity_row_major(&mut workspace.q_row_major, n);

    if n == 0 {
        return Ok((
            Array2::zeros((0, 0).f()),
            Array2::zeros((0, 0).f()),
        ));
    }

    if !shifts.is_empty() {
        let safe_threshold = safe_minimum_threshold(n);
        let h_norm = hessenberg_one_norm_row_major(&workspace.h_row_major, n)
            .max(safe_threshold);

        for (shift_index, shift) in shifts.iter().copied().enumerate() {
            apply_implicit_shift_row_major(
                &mut workspace.h_row_major,
                &mut workspace.q_row_major,
                n,
                shift,
                shift_index,
                safe_threshold,
                h_norm,
            );
        }

        make_subdiagonal_real_nonnegative_row_major(
            &mut workspace.h_row_major,
            &mut workspace.q_row_major,
            n,
        );
        deflate_small_subdiagonals_row_major(
            &mut workspace.h_row_major,
            n,
            safe_threshold,
            h_norm,
        );
    }

    cleanup_hessenberg_roundoff_row_major(&mut workspace.h_row_major, n);

    copy_row_major_to_fortran(&workspace.q_row_major, &mut workspace.q_fortran, n);
    copy_row_major_to_fortran(&workspace.h_row_major, &mut workspace.h_fortran, n);

    let q = Array2::from_shape_vec((n, n).f(), workspace.q_fortran.clone())
        .expect("invalid Fortran rotation buffer shape");
    let h = Array2::from_shape_vec((n, n).f(), workspace.h_fortran.clone())
        .expect("invalid Fortran Hessenberg buffer shape");

    Ok((q, h))
}

fn copy_ndarray_to_row_major(matrix: &Array2<Complex64>, output: &mut [Complex64], n: usize) {
    debug_assert_eq!(output.len(), n * n);

    for row in 0..n {
        for column in 0..n {
            output[idx(n, row, column)] = matrix[(row, column)];
        }
    }
}

fn copy_row_major_to_fortran(input: &[Complex64], output: &mut [Complex64], n: usize) {
    debug_assert_eq!(input.len(), n * n);
    debug_assert_eq!(output.len(), n * n);

    for column in 0..n {
        for row in 0..n {
            output[row + column * n] = input[idx(n, row, column)];
        }
    }
}

fn fill_identity_row_major(matrix: &mut [Complex64], n: usize) {
    debug_assert_eq!(matrix.len(), n * n);
    matrix.fill(zero());

    for i in 0..n {
        matrix[idx(n, i, i)] = one();
    }
}

fn apply_implicit_shift_row_major(
    h: &mut [Complex64],
    rotation: &mut [Complex64],
    n: usize,
    shift: Complex64,
    shift_index: usize,
    safe_threshold: f64,
    h_norm: f64,
) {
    if n < 2 {
        return;
    }

    let mut istart = 0usize;

    while istart + 1 < n {
        let mut iend = n - 1;

        for i in istart..n - 1 {
            if should_deflate_fast_row_major(h, n, i, safe_threshold, h_norm) {
                h[idx(n, i + 1, i)] = zero();
                iend = i;
                break;
            }
        }

        if istart == iend {
            istart = iend + 1;
            continue;
        }

        let mut f = h[idx(n, istart, istart)] - shift;
        let mut g = h[idx(n, istart + 1, istart)];

        for i in istart..iend {
            let (c, s, r) = zlartg(f, g);

            if i > istart {
                h[idx(n, i, i - 1)] = r;
                h[idx(n, i + 1, i - 1)] = zero();
            }

            apply_givens_from_left_row_major(h, n, i, c, s);
            apply_givens_from_right_row_major(h, n, i, iend, c, s);
            accumulate_givens_row_major(rotation, n, i, shift_index, c, s);

            // Keep the chased bulge explicitly clean. This replaces the old
            // full O(n^2) cleanup after every shift.
            if i > 0 && i + 1 < n {
                h[idx(n, i + 1, i - 1)] = zero();
            }

            if i + 1 < iend {
                f = h[idx(n, i + 1, i)];
                g = h[idx(n, i + 2, i)];
            }
        }

        istart = iend + 1;
    }
}

#[inline(always)]
fn zlartg(f: Complex64, g: Complex64) -> (f64, Complex64, Complex64) {
    let mut c = 0.0;
    let mut s = zero();
    let mut r = zero();

    unsafe {
        zlartg_(&f, &g, &mut c, &mut s, &mut r);
    }

    (c, s, r)
}

#[inline(always)]
fn apply_givens_from_left_row_major(
    h: &mut [Complex64],
    n: usize,
    i: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();

    for column in i..n {
        let upper_index = idx(n, i, column);
        let lower_index = idx(n, i + 1, column);

        let upper = h[upper_index];
        let lower = h[lower_index];

        h[upper_index] = c * upper + s * lower;
        h[lower_index] = -s_conj * upper + c * lower;
    }
}

#[inline(always)]
fn apply_givens_from_right_row_major(
    h: &mut [Complex64],
    n: usize,
    i: usize,
    iend: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();
    let last_row = usize::min(i + 2, iend);

    for row in 0..=last_row {
        let left_index = idx(n, row, i);
        let right_index = idx(n, row, i + 1);

        let left = h[left_index];
        let right = h[right_index];

        h[left_index] = c * left + s_conj * right;
        h[right_index] = -s * left + c * right;
    }
}

#[inline(always)]
fn accumulate_givens_row_major(
    rotation: &mut [Complex64],
    n: usize,
    i: usize,
    shift_index: usize,
    c: f64,
    s: Complex64,
) {
    let c = Complex64::new(c, 0.0);
    let s_conj = s.conj();

    // Same band-limited accumulation strategy as the previous implementation.
    // It avoids touching the whole dense rotation matrix during early shifts.
    let row_count = usize::min(i + shift_index + 2, n);

    for row in 0..row_count {
        let left_index = idx(n, row, i);
        let right_index = idx(n, row, i + 1);

        let left = rotation[left_index];
        let right = rotation[right_index];

        rotation[left_index] = c * left + s_conj * right;
        rotation[right_index] = -s * left + c * right;
    }
}

fn make_subdiagonal_real_nonnegative_row_major(
    h: &mut [Complex64],
    rotation: &mut [Complex64],
    n: usize,
) {
    for j in 0..n.saturating_sub(1) {
        let subdiagonal_index = idx(n, j + 1, j);
        let subdiagonal = h[subdiagonal_index];
        let magnitude = subdiagonal.norm();

        if magnitude == 0.0 || (subdiagonal.im == 0.0 && subdiagonal.re >= 0.0) {
            continue;
        }

        let phase = subdiagonal / magnitude;
        let phase_conj = phase.conj();

        // Left phase scaling: row j + 1, columns j..n.
        for column in j..n {
            h[idx(n, j + 1, column)] *= phase_conj;
        }

        // Right phase scaling: column j + 1, only rows that can be nonzero in
        // Hessenberg structure.
        let last_row = usize::min(j + 2, n - 1);
        for row in 0..=last_row {
            h[idx(n, row, j + 1)] *= phase;
        }

        // Keep accumulated Q consistent with the right phase scaling.
        for row in 0..n {
            rotation[idx(n, row, j + 1)] *= phase;
        }

        h[subdiagonal_index] = Complex64::new(magnitude, 0.0);
    }
}

fn deflate_small_subdiagonals_row_major(
    h: &mut [Complex64],
    n: usize,
    safe_threshold: f64,
    h_norm: f64,
) {
    for i in 0..n.saturating_sub(1) {
        if should_deflate_fast_row_major(h, n, i, safe_threshold, h_norm) {
            h[idx(n, i + 1, i)] = zero();
        }
    }
}

#[inline(always)]
fn should_deflate_fast_row_major(
    h: &[Complex64],
    n: usize,
    i: usize,
    safe_threshold: f64,
    h_norm: f64,
) -> bool {
    let subdiagonal_norm = h[idx(n, i + 1, i)].norm();

    if subdiagonal_norm == 0.0 {
        return true;
    }

    let mut scale = h[idx(n, i, i)].norm() + h[idx(n, i + 1, i + 1)].norm();

    if scale == 0.0 {
        scale = h_norm;
    }

    subdiagonal_norm <= (f64::EPSILON * scale).max(safe_threshold)
}

fn safe_minimum_threshold(n: usize) -> f64 {
    f64::MIN_POSITIVE * (n.max(1) as f64 / f64::EPSILON)
}

fn hessenberg_one_norm_row_major(h: &[Complex64], n: usize) -> f64 {
    let mut norm = 0.0;

    for column in 0..n {
        let last_row = usize::min(column + 1, n.saturating_sub(1));
        let mut column_sum = 0.0;

        for row in 0..=last_row {
            column_sum += h[idx(n, row, column)].norm();
        }

        if column_sum > norm {
            norm = column_sum;
        }
    }

    norm
}

fn cleanup_hessenberg_roundoff_row_major(h: &mut [Complex64], n: usize) {
    for row in 2..n {
        for column in 0..row - 1 {
            h[idx(n, row, column)] = zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn conjugate_transpose(matrix: &Array2<Complex64>) -> Array2<Complex64> {
        matrix.t().map(|value| value.conj()).to_owned()
    }

    fn max_abs_diff(left: &Array2<Complex64>, right: &Array2<Complex64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (*l - *r).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn shifted_qr_filter_preserves_similarity_for_small_hessenberg() {
        let h = arr2(&[
            [
                Complex64::new(2.0, 0.1),
                Complex64::new(1.0, -0.3),
                Complex64::new(0.2, 0.4),
            ],
            [
                Complex64::new(0.7, 0.0),
                Complex64::new(1.0, -0.2),
                Complex64::new(-0.5, 0.1),
            ],
            [
                Complex64::ZERO,
                Complex64::new(0.4, 0.0),
                Complex64::new(-1.0, 0.5),
            ],
        ]);
        let shifts = [Complex64::new(0.25, -0.4), Complex64::new(-0.1, 0.2)];

        let (q, filtered_h) = shifted_qr_filter(&h, &shifts).unwrap();
        let reconstructed = q.dot(&filtered_h).dot(&conjugate_transpose(&q));

        assert!(max_abs_diff(&h, &reconstructed) <= 1.0e-8);
    }
}
