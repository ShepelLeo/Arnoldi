//! Определение операторного типа
//! Пользовательские операторы
use std::fmt;
use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::error::IramError;

/// Трейт линейных операторов
pub trait LinearOperator: Send + Sync {
    /// Размерность задачи
    fn dimension(&self) -> usize;
    /// MatVec
    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError>;
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

    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
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

    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
        validate_dimension(self.dimension, vector.len())?;

        Ok(Array1::from_iter((0..self.dimension).map(|row| {
            let diagonal = vector[row];
            let subdiagonal = if row > 0 {
                -vector[row - 1]
            } else {
                Complex64::new(0.0, 0.0)
            };
            let superdiagonal_sum = (1..=self.upper_bandwidth)
                .filter_map(|offset| vector.get(row + offset).copied())
                .sum::<Complex64>();

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

/// # Плотная матрица
#[derive(Debug, Clone)]
pub struct DenseMatrixOperator {
    matrix: Array2<Complex64>,
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
                    .map(parse_complex_token)
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

    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
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

    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
        let n = self.dimension();
        validate_dimension(n, vector.len())?;

        let h = self.h();
        let inv_h2 = 1.0 / (h * h);
        let conv = self.rho / (2.0 * h);

        Ok(Array1::from_iter((0..n).map(|k| {
            let i = k % self.m;
            let j = k / self.m;

            let center = vector[k] * Complex64::new(-4.0 * inv_h2, 0.0);

            let left = (i > 0)
                .then(|| vector[k - 1] * Complex64::new(inv_h2 - conv, 0.0))
                .unwrap_or(Complex64::new(0.0, 0.0));

            let right = (i + 1 < self.m)
                .then(|| vector[k + 1] * Complex64::new(inv_h2 + conv, 0.0))
                .unwrap_or(Complex64::new(0.0, 0.0));

            let down = (j > 0)
                .then(|| vector[k - self.m] * Complex64::new(inv_h2, 0.0))
                .unwrap_or(Complex64::new(0.0, 0.0));

            let up = (j + 1 < self.m)
                .then(|| vector[k + self.m] * Complex64::new(inv_h2, 0.0))
                .unwrap_or(Complex64::new(0.0, 0.0));

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
    F: Fn(&Array1<Complex64>) -> Array1<Complex64> + Send + Sync,
{
    dimension: usize,
    name: String,
    matvec: F,
}

impl<F> FnOperator<F>
where
    F: Fn(&Array1<Complex64>) -> Array1<Complex64> + Send + Sync,
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
    F: Fn(&Array1<Complex64>) -> Array1<Complex64> + Send + Sync,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnOperator")
            .field("dimension", &self.dimension)
            .field("name", &self.name)
            .finish()
    }
}

impl<F> LinearOperator for FnOperator<F>
where
    F: Fn(&Array1<Complex64>) -> Array1<Complex64> + Send + Sync,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
        validate_dimension(self.dimension, vector.len())?;
        Ok((self.matvec)(vector))
    }

    fn description(&self) -> String {
        self.name.clone()
    }
}

pub fn parse_complex_token(entry: &str) -> Result<Complex64, IramError> {
    let token = entry.trim();

    if token.is_empty() {
        return Err(IramError::Parse(
            "cannot parse an empty complex entry".to_string(),
        ));
    }

    if let Some(body) = token.strip_suffix('i').or_else(|| token.strip_suffix('j')) {
        return parse_imaginary_body(body, token);
    }

    token
        .parse::<f64>()
        .map(|value| Complex64::new(value, 0.0))
        .map_err(|error| IramError::Parse(format!("cannot parse complex entry '{token}': {error}")))
}

fn parse_imaginary_body(body: &str, original: &str) -> Result<Complex64, IramError> {
    if let Some(split_index) = find_complex_split(body) {
        let real_part = &body[..split_index];
        let imaginary_part = &body[split_index..];
        let real = parse_real_component(real_part, original)?;
        let imaginary = parse_imaginary_component(imaginary_part, original)?;
        Ok(Complex64::new(real, imaginary))
    } else {
        let imaginary = parse_imaginary_component(body, original)?;
        Ok(Complex64::new(0.0, imaginary))
    }
}

fn find_complex_split(body: &str) -> Option<usize> {
    let bytes = body.as_bytes();

    (1..bytes.len()).rev().find(|&index| {
        let current = bytes[index] as char;
        let previous = bytes[index - 1] as char;
        (current == '+' || current == '-') && previous != 'e' && previous != 'E'
    })
}

fn parse_real_component(component: &str, original: &str) -> Result<f64, IramError> {
    component.parse::<f64>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse real part of complex entry '{original}': {error}"
        ))
    })
}

fn parse_imaginary_component(component: &str, original: &str) -> Result<f64, IramError> {
    match component {
        "" | "+" => Ok(1.0),
        "-" => Ok(-1.0),
        value => value.parse::<f64>().map_err(|error| {
            IramError::Parse(format!(
                "cannot parse imaginary part of complex entry '{original}': {error}"
            ))
        }),
    }
}

fn validate_dimension(expected: usize, got: usize) -> Result<(), IramError> {
    (expected == got)
        .then_some(())
        .ok_or(IramError::DimensionMismatch { expected, got })
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::{ConvectionDiffusionOperator, LinearOperator, parse_complex_token};

    #[test]
    fn convection_diffusion_dimension_matches_grid() {
        let operator = ConvectionDiffusionOperator::new(4, 1.0);
        assert_eq!(operator.dimension(), 16);
    }

    #[test]
    fn complex_parser_supports_real_and_imaginary_entries() {
        assert_eq!(
            parse_complex_token("2.5").expect("real entry should parse"),
            Complex64::new(2.5, 0.0)
        );
        assert_eq!(
            parse_complex_token("-1.0+3.0i").expect("complex entry should parse"),
            Complex64::new(-1.0, 3.0)
        );
        assert_eq!(
            parse_complex_token("-i").expect("pure imaginary entry should parse"),
            Complex64::new(0.0, -1.0)
        );
    }
}
