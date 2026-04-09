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
    // Составляем карту комплексно-сопряженных чисел Ритца
    let pair_map = build_pair_map(values);

    // Обьявляем порядок чисел в выбранной части спектра
    let order = ranking(values, target);

    // Выбираем 
    let base_wanted = base_selection(values, target, nev, &order);
    let mut keep_flags = vec![false; values.len()];
    for &index in &base_wanted {
        keep_flags[index] = true;
        if let Some(pair_index) = pair_map[index] {
            keep_flags[pair_index] = true;
        }
    }
    let retained_dimension = keep_flags.iter().filter(|&&flag| flag).count();

    if retained_dimension > max_keep {
        return Err(IramError::InvalidConfig(format!(
            "the requested nev={} together with complex-conjugate preservation requires {} retained Ritz values; increase ncv",
            nev, retained_dimension,
        )));
    }

    let mut wanted = Vec::with_capacity(retained_dimension);
    let mut shifts = Vec::with_capacity(values.len().saturating_sub(retained_dimension));

    for &index in &order {
        if keep_flags[index] {
            if wanted.len() < retained_dimension {
                wanted.push(RitzEstimate {
                    value: values[index].value,
                    residual_estimate: values[index].residual_estimate,
                });
            }
        } else {
            shifts.push(values[index].value);
        }
    }

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

/// Сортировка чисел Р
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

/// Собираем комплексносопряженные пары чисел Ритца в индекс -> индекс сопряженного
fn build_pair_map(values: &[RitzValue]) -> Vec<Option<usize>> {
    let tolerance = 1.0e-8;
    let mut pair_map = vec![None; values.len()];
    let mut used = vec![false; values.len()];

    values.iter().enumerate().for_each(|(index, entry)| {
        if used[index] || entry.value.im <= tolerance {
            return;
        }

        let conjugate = entry.value.conj();
        let candidate = values
            .iter()
            .enumerate()
            .filter(|(other_index, other_entry)| {
                *other_index != index && !used[*other_index] && other_entry.value.im < -tolerance
            })
            .min_by(|(_, left_entry), (_, right_entry)| {
                (left_entry.value - conjugate)
                    .norm()
                    .total_cmp(&(right_entry.value - conjugate).norm())
            });

        if let Some((pair_index, pair_entry)) = candidate {
            let scale = 1.0 + entry.value.norm().max(pair_entry.value.norm());
            if (pair_entry.value - conjugate).norm() <= tolerance * scale {
                pair_map[index] = Some(pair_index);
                pair_map[pair_index] = Some(index);
                used[index] = true;
                used[pair_index] = true;
            }
        }
    });

    pair_map
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::config::SpectrumTarget;
    use crate::linalg::small::RitzValue;

    use super::select_ritz_values;

    #[test]
    fn preserves_complex_conjugate_pairs() {
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

        let selection = select_ritz_values(&values, SpectrumTarget::LargestReal, 2, 3)
            .expect("selection should succeed");

        assert_eq!(selection.retained_dimension, 3);
        assert_eq!(selection.wanted.len(), 3);
        assert_eq!(selection.shifts.len(), 1);
    }
}
