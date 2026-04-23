//! Ядро обработки чисел Ритца
//! Выбор и оценка невязки
use num_complex::Complex64;

use crate::config::SpectrumTarget;
use crate::error::IramError;
use crate::linalg::small::RitzValue;
use crate::report::RitzEstimate;

#[derive(Debug, Clone)]
pub struct SelectionOutcome {
    pub wanted: Vec<RitzEstimate>,
    pub retained_dimension: usize,
    pub shifts: Vec<Complex64>,
}

/// Вход в селектор
pub fn select_ritz_values(
    values: &[RitzValue],
    target: SpectrumTarget,
    nev: usize,
    max_keep: usize,
) -> Result<SelectionOutcome, IramError> {
    if values.is_empty() {
        return Ok(SelectionOutcome {
            wanted: Vec::new(),
            retained_dimension: 0,
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

    let selected = base_selection(values, target, retained_dimension, &order);
    let mut keep_flags = vec![false; values.len()];
    selected.iter().for_each(|&index| keep_flags[index] = true);

    let wanted = selected
        .iter()
        .map(|&index| RitzEstimate {
            value: values[index].value,
            residual_estimate: values[index].residual_estimate,
        })
        .collect::<Vec<_>>();

    let shifts = order
        .iter()
        .copied()
        .filter(|&index| !keep_flags[index])
        .map(|index| values[index].value)
        .collect::<Vec<_>>();

    Ok(SelectionOutcome {
        wanted,
        retained_dimension,
        shifts,
    })
}

fn base_selection(
    values: &[RitzValue],
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

/// Сортировка чисел Ритца
fn ranking(values: &[RitzValue], target: SpectrumTarget) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();

    match target {
        SpectrumTarget::LargestMagnitude => {
            indices.sort_unstable_by(|&left, &right| {
                values[right]
                    .value
                    .norm()
                    .total_cmp(&values[left].value.norm())
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::SmallestMagnitude => {
            indices.sort_unstable_by(|&left, &right| {
                values[left]
                    .value
                    .norm()
                    .total_cmp(&values[right].value.norm())
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::LargestReal => {
            indices.sort_unstable_by(|&left, &right| {
                values[right]
                    .value
                    .re
                    .total_cmp(&values[left].value.re)
                    .then_with(|| {
                        values[right]
                            .value
                            .im
                            .abs()
                            .total_cmp(&values[left].value.im.abs())
                    })
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::SmallestReal => {
            indices.sort_unstable_by(|&left, &right| {
                values[left]
                    .value
                    .re
                    .total_cmp(&values[right].value.re)
                    .then_with(|| {
                        values[left]
                            .value
                            .im
                            .abs()
                            .total_cmp(&values[right].value.im.abs())
                    })
                    .then_with(|| left.cmp(&right))
            });
        }

        SpectrumTarget::BothEndsReal => {
            let min_real = values
                .iter()
                .map(|entry| entry.value.re)
                .fold(f64::INFINITY, f64::min);
            let max_real = values
                .iter()
                .map(|entry| entry.value.re)
                .fold(f64::NEG_INFINITY, f64::max);
            let center = 0.5 * (min_real + max_real);

            indices.sort_unstable_by(|&left, &right| {
                (values[right].value.re - center)
                    .abs()
                    .total_cmp(&(values[left].value.re - center).abs())
                    .then_with(|| left.cmp(&right))
            });
        }
    }

    indices
}

fn sort_by_real(values: &[RitzValue]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        values[left]
            .value
            .re
            .total_cmp(&values[right].value.re)
            .then_with(|| left.cmp(&right))
    });
    indices
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::config::SpectrumTarget;
    use crate::linalg::small::RitzValue;

    use super::select_ritz_values;

    #[test]
    fn complex_selection_keeps_exactly_nev_values() {
        let values = vec![
            RitzValue {
                value: Complex64::new(3.0, 0.0),
                residual_estimate: 1.0e-8,
            },
            RitzValue {
                value: Complex64::new(1.0, 2.0),
                residual_estimate: 1.0e-8,
            },
            RitzValue {
                value: Complex64::new(1.0, -2.0),
                residual_estimate: 1.0e-8,
            },
            RitzValue {
                value: Complex64::new(-4.0, 0.0),
                residual_estimate: 1.0e-8,
            },
        ];

        let selection = select_ritz_values(&values, SpectrumTarget::LargestReal, 2, 4)
            .expect("selection should succeed");

        assert_eq!(selection.retained_dimension, 2);
        assert_eq!(selection.wanted.len(), 2);
        assert_eq!(selection.shifts.len(), 2);
    }
}
