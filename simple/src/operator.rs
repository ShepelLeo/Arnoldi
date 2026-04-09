//! Определение операторного типа
//! Пользовательские операторы
use std::fmt;
use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::IramError;

/// Трейт линейных операторов
pub trait LinearOperator: Send + Sync {
    /// Размерность задачи
    fn dimension(&self) -> usize;
    /// MatVec
    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError>;
    /// Буковки
    fn description(&self) -> String;
}


/// # Единичный оператор
#[derive(Debug, Clone)]
pub struct IdentityOperator {
    dimension: usize,
}

impl IdentityOperator {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl LinearOperator for IdentityOperator {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError> {
        validate_dimension(self.dimension, vector.len())?;
        Ok(vector.clone())
    }

    fn description(&self) -> String {
        format!("identity operator of dimension {}", self.dimension)
    }
}

/// # Матрица Тёплица
#[derive(Debug, Clone)]
pub struct GrcarOperator {
    dimension: usize,
    upper_bandwidth: usize,
}

impl GrcarOperator {
    pub fn new(dimension: usize, upper_bandwidth: usize) -> Self {
        Self {
            dimension,
            upper_bandwidth,
        }
    }
}

impl LinearOperator for GrcarOperator {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError> {
        validate_dimension(self.dimension, vector.len())?;

        Ok(Array1::from_iter((0..self.dimension).map(|row| {
            let diagonal = vector[row];
            let subdiagonal = if row > 0 { -vector[row - 1] } else { 0.0 };
            let superdiagonal_sum = (1..=self.upper_bandwidth)
                .filter_map(|offset| vector.get(row + offset).copied())
                .sum::<f64>();

            diagonal + subdiagonal + superdiagonal_sum
        })))
    }

    fn description(&self) -> String {
        format!(
            "Grcar operator of dimension {} with {} superdiagonals",
            self.dimension, self.upper_bandwidth,
        )
    }
}


/// # Разреженная матрица
#[derive(Debug, Clone)]
pub struct DenseMatrixOperator {
    matrix: Array2<f64>,
    label: String,
}

impl DenseMatrixOperator {
    pub fn from_text_file(path: impl AsRef<Path>) -> Result<Self, IramError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        let rows = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split_whitespace()
                    .map(|entry| {
                        entry.parse::<f64>().map_err(|error| {
                            IramError::Parse(format!(
                                "cannot parse matrix entry '{entry}': {error}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dimension = rows.len();
        let width = rows.first().map(Vec::len).unwrap_or(0);

        if dimension == 0 || width == 0 {
            return Err(IramError::Parse(
                "the dense matrix file is empty".to_string(),
            ));
        }

        if rows.iter().any(|row| row.len() != width) {
            return Err(IramError::Parse(
                "all dense matrix rows must have the same width".to_string(),
            ));
        }

        if dimension != width {
            return Err(IramError::Parse(format!(
                "the dense matrix must be square, got {dimension}x{width}"
            )));
        }

        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
        let matrix = Array2::from_shape_vec((dimension, width), flat)
            .map_err(|error| IramError::Parse(format!("cannot reshape dense matrix: {error}")))?;

        Ok(Self {
            matrix,
            label: format!("dense matrix loaded from {}", path.display()),
        })
    }
}

impl LinearOperator for DenseMatrixOperator {
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError> {
        validate_dimension(self.dimension(), vector.len())?;
        Ok(self.matrix.dot(vector))
    }

    fn description(&self) -> String {
        self.label.clone()
    }
}


/// # Матрица центральной разностной производной диффузионно-конвекционного оператора

#[derive(Debug, Clone)]
pub struct ConvectionDiffusionOperator {
    m: usize,
    rho: f64,
}

impl ConvectionDiffusionOperator {
    pub fn new(m: usize, rho: f64) -> Self {
        Self { m, rho }
    }

    fn h(&self) -> f64 {
        1.0 / (self.m as f64 + 1.0)
    }
}

impl LinearOperator for ConvectionDiffusionOperator {
    fn dimension(&self) -> usize {
        self.m * self.m
    }

    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError> {
        let n = self.dimension();
        validate_dimension(n, vector.len())?;

        let h = self.h();
        let inv_h2 = 1.0 / (h * h);
        let conv = self.rho / (2.0 * h);

        Ok(Array1::from_iter((0..n).map(|k| {
            let i = k % self.m;
            let j = k / self.m;

            let center = -4.0 * inv_h2 * vector[k];

            let left = (i > 0)
                .then(|| (inv_h2 - conv) * vector[k - 1])
                .unwrap_or(0.0);

            let right = (i + 1 < self.m)
                .then(|| (inv_h2 + conv) * vector[k + 1])
                .unwrap_or(0.0);

            let down = (j > 0).then(|| inv_h2 * vector[k - self.m]).unwrap_or(0.0);

            let up = (j + 1 < self.m)
                .then(|| inv_h2 * vector[k + self.m])
                .unwrap_or(0.0);

            center + left + right + down + up
        })))
    }

    fn description(&self) -> String {
        format!(
            "2D convection-diffusion operator on {}x{} interior grid, rho={}",
            self.m, self.m, self.rho
        )
    }
}

pub struct FnOperator<F>
where
    F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
{
    dimension: usize,
    name: String,
    matvec: F,
}

impl<F> FnOperator<F>
where
    F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
{
    pub fn new(dimension: usize, name: impl Into<String>, matvec: F) -> Self {
        Self {
            dimension,
            name: name.into(),
            matvec,
        }
    }
}

impl<F> fmt::Debug for FnOperator<F>
where
    F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnOperator")
            .field("dimension", &self.dimension)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<F> LinearOperator for FnOperator<F>
where
    F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, vector: &Array1<f64>) -> Result<Array1<f64>, IramError> {
        validate_dimension(self.dimension, vector.len())?;
        Ok((self.matvec)(vector))
    }

    fn description(&self) -> String {
        self.name.clone()
    }
}

fn validate_dimension(expected: usize, got: usize) -> Result<(), IramError> {
    (expected == got)
        .then_some(())
        .ok_or(IramError::DimensionMismatch { expected, got })
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::{ConvectionDiffusionOperator, LinearOperator};

    #[test]
    fn convection_diffusion_matches_laplacian_when_rho_is_zero() {
        let operator = ConvectionDiffusionOperator::new(2, 0.0);
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = operator
            .apply(&vector)
            .expect("convection-diffusion matvec should succeed");

        assert_eq!(result.to_vec(), vec![9.0, -27.0, -63.0, -99.0]);
    }
}