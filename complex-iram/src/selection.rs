//! Ядро обработки чисел Ритца
//! Выбор и оценка невязки
use num_complex::Complex64;

use crate::config::SpectrumTarget;
use crate::error::IramError;

#[derive(Debug, Clone)]
pub struct SelectionOut {
    pub wanted: Vec<usize>,
    pub shifts: Vec<Complex64>,
}

/// Вход в селектор
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

    let shifts = match inflation {
        Some(inflation) => topology_cluster(values, &wanted, target, inflation),
        None => all_unwanted_values(values, &wanted),
    };
    Ok(SelectionOut {
        wanted,
        shifts,
    })
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


fn topology_cluster(
    values: &[Complex64], 
    wanted: &[usize], 
    target: SpectrumTarget, 
    inflation: f64) -> Vec<Complex64>{
    match target {
        SpectrumTarget::LargestMagnitude
        | SpectrumTarget::SmallestMagnitude => {
            let sum = wanted
                .iter()
                .fold(0.0, |acc, &i| acc + values[i].norm());

            let center = sum / wanted.len() as f64;

            // 6. Radius = максимальное расстояние от center до выбранных чисел
            let radius = wanted
                .iter()
                .map(|&i| (values[i].norm() - center).abs())
                .fold(0.0_f64, f64::max);

            // 7. Shifts = числа вне окружности
            let shifts: Vec<Complex64> = values
                .iter()
                .copied()
                .filter(|&z| (z.norm() - center).abs() > radius * inflation)
                .collect();

            shifts
        }

        SpectrumTarget::LargestReal
        | SpectrumTarget::SmallestReal => {
            let sum = wanted
                .iter()
                .fold(0.0, |acc, &i| acc + values[i].re);

            let center = sum / wanted.len() as f64;

            // 6. Radius = максимальное расстояние от center до выбранных чисел
            let radius = wanted
                .iter()
                .map(|&i| (values[i].re - center).abs())
                .fold(0.0_f64, f64::max);

            // 7. Shifts = числа вне окружности
            let shifts: Vec<Complex64> = values
                .iter()
                .copied()
                .filter(|&z| (z.re - center).abs() > radius * inflation)
                .collect();

            shifts
        }

        SpectrumTarget::BothEndsReal => {
            let sum = wanted
                .iter()
                .fold(0.0, |acc, &i| acc + values[i].re);

            let center = sum / wanted.len() as f64;

            // 6. Radius = максимальное расстояние от center до выбранных чисел
            let radius = wanted
                .iter()
                .map(|&i| (values[i].re - center).abs())
                .fold(f64::INFINITY, f64::min);

            // 7. Shifts = числа вне окружности
            let shifts: Vec<Complex64> = values
                .iter()
                .copied()
                .filter(|&z| (z.re - center).abs() < radius / inflation)
                .collect();

            shifts
        }

    }
}

/// Сортировка чисел Ритца
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

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::config::SpectrumTarget;

    use super::select_ritz_values;

    #[test]
    fn complex_selection_keeps_exactly_nev_values() {
        let values = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(-4.0, 0.0),
        ];

        let selection = select_ritz_values(&values, SpectrumTarget::LargestReal, 2, 4, Some(1.0))
            .expect("selection should succeed");

        assert_eq!(selection.wanted.len(), 2);
        assert_eq!(selection.shifts.len(), 2);
    }

    #[test]
    fn inflated_disk_keeps_extra_ritz_pairs_for_restart() {
        let values = vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let selection = select_ritz_values(&values, SpectrumTarget::LargestReal, 2, 4, Some(3.0))
            .expect("selection should succeed");

        assert_eq!(selection.wanted, vec![0, 1]);
        assert_eq!(selection.shifts, vec![Complex64::new(0.0, 0.0)]);
    }
}