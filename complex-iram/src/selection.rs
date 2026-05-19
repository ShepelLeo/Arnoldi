//! Ядро обработки чисел Ритца
//! Выбор wanted Ritz values и построение restart shifts.
//!
//! Если `inflation == None`, shifts выбираются как все unwanted Ritz values
//! (классические exact shifts).
//!
//! Если `inflation == Some(gamma)`, где `gamma >= 1`, shifts строятся как
//! корни масштабированного полинома Чебышёва для эллипса, покрывающего
//! unwanted Ritz values. Параметр `gamma` увеличивает полуоси этого эллипса:
//!
//!     a <- gamma * a,
//!     b <- gamma * b.
//!
//! Сами Chebyshev shifts лежат на фокальном отрезке эллипса:
//!
//!     mu_j = d + exp(i theta) * sqrt(a^2 - b^2)
//!            * cos((2j - 1) pi / (2p)),  j = 1..p.

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::config::SpectrumTarget;
use crate::error::IramError;

#[derive(Debug, Clone)]
pub struct SelectionOut {
    pub wanted: Vec<usize>,
    pub shifts: Vec<Complex64>,
}

/// Вход в селектор.
///
/// `inflation` задаёт режим построения shifts:
/// - `None`: exact shifts, то есть все unwanted Ritz values;
/// - `Some(gamma)`: Chebyshev shifts для эллипса unwanted Ritz values,
///   где `gamma >= 1` расширяет полуоси эллипса.
pub fn select_ritz_values(
    values: &[Complex64],
    target: SpectrumTarget,
    nev: usize,
    max_keep: usize,
    inflation: Option<f64>,
) -> Result<SelectionOut, IramError> {
    if values.is_empty() {
        return Ok(SelectionOut {
            wanted: Vec::new(),
            shifts: Vec::new(),
        });
    }

    let order = ranking(values, target);
    let retained_dimension = nev.min(values.len());

    if retained_dimension > max_keep {
        return Err(IramError::InvalidConfig(format!(
            "the requested nev={} requires retaining {} Ritz values, but only {} are available",
            nev, retained_dimension, max_keep,
        )));
    }

    let mut wanted = base_selection(values, target, retained_dimension, &order);
    wanted.sort_unstable();

    let unwanted = all_unwanted_values(values, &wanted);

    let shifts = match inflation {
        Some(gamma) => chebyshev_shifts_from_unwanted_ellipse(&unwanted, gamma)?,
        None => unwanted,
    };

    Ok(SelectionOut { wanted, shifts })
}

fn all_unwanted_values(values: &[Complex64], wanted: &[usize]) -> Vec<Complex64> {
    let mut is_wanted = vec![false; values.len()];

    for &index in wanted {
        is_wanted[index] = true;
    }

    values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            if is_wanted[index] {
                None
            } else {
                Some(value)
            }
        })
        .collect()
}

fn base_selection(
    values: &[Complex64],
    target: SpectrumTarget,
    nev: usize,
    ranking_order: &[usize],
) -> Vec<usize> {
    match target {
        SpectrumTarget::LargestMagnitude
        | SpectrumTarget::SmallestMagnitude
        | SpectrumTarget::LargestReal
        | SpectrumTarget::SmallestReal => ranking_order.iter().copied().take(nev).collect(),

        SpectrumTarget::BothEndsReal => {
            let ascending = sort_by_real(values);
            let left_count = nev / 2;
            let right_count = nev - left_count;

            let mut result = Vec::with_capacity(nev.min(values.len()));
            let mut selected = vec![false; values.len()];

            for &index in ascending.iter().rev().take(right_count) {
                result.push(index);
                selected[index] = true;
            }

            for &index in ascending.iter().take(left_count) {
                if !selected[index] {
                    result.push(index);
                    selected[index] = true;
                }
            }

            if result.len() < nev {
                for &index in ranking_order {
                    if result.len() == nev {
                        break;
                    }
                    if !selected[index] {
                        result.push(index);
                        selected[index] = true;
                    }
                }
            }

            result
        }
    }
}

/// Строит Chebyshev shifts по набору unwanted Ritz values.
///
/// Семантика `inflation`:
/// - `inflation == 1.0`: использовать эллипс, покрывающий unwanted values;
/// - `inflation > 1.0`: расширить обе полуоси эллипса в `inflation` раз.
///
/// Эллипс строится так:
/// 1. центр `d` берётся как среднее unwanted Ritz values;
/// 2. направление большой оси оценивается через 2D PCA по точкам `(Re z, Im z)`;
/// 3. полуоси оцениваются по bounding box в повернутых координатах;
/// 4. эллипс дополнительно масштабируется так, чтобы покрыть все unwanted values;
/// 5. обе полуоси умножаются на `inflation`.
fn chebyshev_shifts_from_unwanted_ellipse(
    unwanted: &[Complex64],
    inflation: f64,
) -> Result<Vec<Complex64>, IramError> {
    if unwanted.is_empty() {
        return Ok(Vec::new());
    }

    if !inflation.is_finite() || inflation < 1.0 {
        return Err(IramError::InvalidConfig(format!(
            "ritz-inflation must be finite and >= 1.0, got {}",
            inflation,
        )));
    }

    let p = unwanted.len();

    if p == 1 {
        return Ok(vec![unwanted[0]]);
    }

    let center = centroid(unwanted);
    let mut theta = principal_axis_angle(unwanted, center);

    let mut rotated = rotate_about_center(unwanted, center, -theta);
    let (mut a0, mut b0) = bounding_half_axes(&rotated);

    // Гарантируем, что a0 — большая полуось. PCA обычно уже даёт это,
    // но для малых или выбросных наборов max-extent может оказаться больше
    // в поперечном направлении.
    if b0 > a0 {
        theta += 0.5 * PI;
        rotated = rotate_about_center(unwanted, center, -theta);
        let axes = bounding_half_axes(&rotated);
        a0 = axes.0;
        b0 = axes.1;
    }

    let eps = 1e-14_f64;

    if a0 <= eps && b0 <= eps {
        return Ok(vec![center; p]);
    }

    // Масштабируем bounding-box полуоси до эллипса, который покрывает все точки.
    // Для каждой точки требуем (x/a)^2 + (y/b)^2 <= 1.
    let cover_scale = rotated
        .iter()
        .map(|z| {
            let x = if a0 > eps { z.re / a0 } else { 0.0 };
            let y = if b0 > eps { z.im / b0 } else { 0.0 };
            (x * x + y * y).sqrt()
        })
        .fold(0.0_f64, f64::max)
        .max(1.0);

    let a = inflation * cover_scale * a0;
    let b = inflation * cover_scale * b0;

    let c_abs = (a * a - b * b).max(0.0).sqrt();
    let direction = Complex64::from_polar(1.0, theta);

    let shifts = (1..=p)
        .map(|j| {
            let angle = (2.0 * j as f64 - 1.0) * PI / (2.0 * p as f64);
            center + direction * (c_abs * angle.cos())
        })
        .collect();

    Ok(shifts)
}

fn centroid(values: &[Complex64]) -> Complex64 {
    let sum = values
        .iter()
        .copied()
        .fold(Complex64::new(0.0, 0.0), |acc, z| acc + z);

    sum / values.len() as f64
}

fn principal_axis_angle(values: &[Complex64], center: Complex64) -> f64 {
    let n = values.len() as f64;

    let (sxx, syy, sxy) = values.iter().fold((0.0, 0.0, 0.0), |acc, z| {
        let x = z.re - center.re;
        let y = z.im - center.im;
        (acc.0 + x * x, acc.1 + y * y, acc.2 + x * y)
    });

    let sxx = sxx / n;
    let syy = syy / n;
    let sxy = sxy / n;

    if sxx == 0.0 && syy == 0.0 && sxy == 0.0 {
        0.0
    } else {
        0.5 * (2.0 * sxy).atan2(sxx - syy)
    }
}

fn rotate_about_center(values: &[Complex64], center: Complex64, angle: f64) -> Vec<Complex64> {
    let rot = Complex64::from_polar(1.0, angle);
    values.iter().map(|&z| rot * (z - center)).collect()
}

fn bounding_half_axes(rotated: &[Complex64]) -> (f64, f64) {
    let a = rotated.iter().map(|z| z.re.abs()).fold(0.0_f64, f64::max);

    let b = rotated.iter().map(|z| z.im.abs()).fold(0.0_f64, f64::max);

    (a, b)
}

/// Сортировка чисел Ритца.
fn ranking(values: &[Complex64], target: SpectrumTarget) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();

    match target {
        SpectrumTarget::LargestMagnitude => {
            indices.sort_unstable_by(|&left, &right| {
                values[right]
                    .norm_sqr()
                    .total_cmp(&values[left].norm_sqr())
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::SmallestMagnitude => {
            indices.sort_unstable_by(|&left, &right| {
                values[left]
                    .norm_sqr()
                    .total_cmp(&values[right].norm_sqr())
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::LargestReal => {
            indices.sort_unstable_by(|&left, &right| {
                values[right]
                    .re
                    .total_cmp(&values[left].re)
                    .then_with(|| values[right].im.abs().total_cmp(&values[left].im.abs()))
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::SmallestReal => {
            indices.sort_unstable_by(|&left, &right| {
                values[left]
                    .re
                    .total_cmp(&values[right].re)
                    .then_with(|| values[left].im.abs().total_cmp(&values[right].im.abs()))
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::BothEndsReal => {
            let min_real = values
                .iter()
                .map(|entry| entry.re)
                .fold(f64::INFINITY, f64::min);
            let max_real = values
                .iter()
                .map(|entry| entry.re)
                .fold(f64::NEG_INFINITY, f64::max);
            let center = 0.5 * (min_real + max_real);

            indices.sort_unstable_by(|&left, &right| {
                (values[right].re - center)
                    .abs()
                    .total_cmp(&(values[left].re - center).abs())
                    .then_with(|| left.cmp(&right))
            });
        }
    }

    indices
}

fn sort_by_real(values: &[Complex64]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        values[left]
            .re
            .total_cmp(&values[right].re)
            .then_with(|| left.cmp(&right))
    });
    indices
}
