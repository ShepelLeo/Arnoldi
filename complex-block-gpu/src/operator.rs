//! Определение операторного типа
//! Пользовательские операторы
use std::fmt;
use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};
use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{ZgemmTranspose, ZgemvTranspose, matmul_into, matvec_into};

/// Трейт линейных операторов
pub trait LinearOperator: Send + Sync {
    /// Размерность задачи
    fn dimension(&self) -> usize;
    /// MatVec
    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError>;
    fn apply(&self, vector: &Array1<Complex64>) -> Result<Array1<Complex64>, IramError> {
        let mut output = Array1::zeros(self.dimension());
        self.apply_into(vector.view(), output.view_mut())?;
        Ok(output)
    }
    /// MatMat для блочного метода. Реализация по умолчанию применяет MatVec к столбцам.
    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        mut output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension(), block.nrows())?;
        validate_dimension(self.dimension(), output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }

        for column in 0..block.ncols() {
            self.apply_into(block.column(column), output.column_mut(column))?;
        }

        Ok(())
    }
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

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, vector.len())?;
        validate_dimension(self.dimension, output.len())?;
        copy_vector_into(output, vector);
        Ok(())
    }

    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, block.nrows())?;
        validate_dimension(self.dimension, output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }
        copy_matrix_into(output, block);
        Ok(())
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

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, vector.len())?;
        validate_dimension(self.dimension, output.len())?;

        for row in 0..self.dimension {
            let diagonal = vector[row];
            let subdiagonal = if row > 0 {
                -vector[row - 1]
            } else {
                Complex64::new(0.0, 0.0)
            };
            let upper_end = (row + self.upper_bandwidth + 1).min(self.dimension);
            let superdiagonal_sum = ((row + 1)..upper_end)
                .map(|column| vector[column])
                .sum::<Complex64>();

            output[row] = diagonal + subdiagonal + superdiagonal_sum;
        }

        Ok(())
    }

    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        mut output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, block.nrows())?;
        validate_dimension(self.dimension, output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }

        for column in 0..block.ncols() {
            let input = block.column(column);
            let mut target = output.column_mut(column);
            for row in 0..self.dimension {
                let upper_end = (row + self.upper_bandwidth + 1).min(self.dimension);
                let mut acc = input[row];
                if row > 0 {
                    acc -= input[row - 1];
                }
                for upper in (row + 1)..upper_end {
                    acc += input[upper];
                }
                target[row] = acc;
            }
        }

        Ok(())
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

        let mut matrix = Array2::zeros((dimension, width).f());
        for (row_index, row) in rows.iter().enumerate() {
            for (column_index, &value) in row.iter().enumerate() {
                matrix[[row_index, column_index]] = value;
            }
        }

        Ok(Self {
            matrix,
            label: format!("dense matrix loaded from {}", path.display()),
        })
    }
}

#[derive(Debug, Clone)]
pub struct MatrixMarketOperator {
    dimension: usize,
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<Complex64>,
    label: String,
}

#[derive(Debug, Clone, Copy)]
struct MatrixMarketEntry {
    row: usize,
    column: usize,
    value: Complex64,
}

impl MatrixMarketOperator {
    pub fn from_text_file(path: impl AsRef<Path>) -> Result<Self, IramError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;

        parse_matrix_market(
            &content,
            format!("Matrix Market matrix loaded from {}", path.display()),
        )
    }
}

pub fn matrix_operator_from_text_file(
    path: impl AsRef<Path>,
) -> Result<Box<dyn LinearOperator>, IramError> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)?;

    if is_matrix_market_content(&content) {
        return parse_matrix_market(
            &content,
            format!("Matrix Market matrix loaded from {}", path.display()),
        )
        .map(|operator| Box::new(operator) as Box<dyn LinearOperator>);
    }

    DenseMatrixOperator::from_text_file(path)
        .map(|operator| Box::new(operator) as Box<dyn LinearOperator>)
}

fn is_matrix_market_content(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_start().starts_with("%%MatrixMarket"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixMarketFormat {
    Coordinate,
    Array,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixMarketField {
    Real,
    Integer,
    Complex,
    Pattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixMarketSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

fn parse_matrix_market(content: &str, label: String) -> Result<MatrixMarketOperator, IramError> {
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| IramError::Parse("the Matrix Market file is empty".to_string()))?;
    let (format, field, symmetry) = parse_matrix_market_header(header)?;
    let data_lines = lines
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('%'))
        .collect::<Vec<_>>();
    let size_line = data_lines
        .first()
        .ok_or_else(|| IramError::Parse("Matrix Market file is missing a size line".to_string()))?;
    let tokens = data_lines
        .iter()
        .skip(1)
        .flat_map(|line| line.split_whitespace())
        .collect::<Vec<_>>();
    let (dimension, entries) = match format {
        MatrixMarketFormat::Coordinate => {
            parse_matrix_market_coordinate(size_line, &tokens, field, symmetry)
        }
        MatrixMarketFormat::Array => parse_matrix_market_array(size_line, &tokens, field, symmetry),
    }?;

    Ok(build_matrix_market_operator(dimension, entries, label))
}

fn parse_matrix_market_header(
    header: &str,
) -> Result<(MatrixMarketFormat, MatrixMarketField, MatrixMarketSymmetry), IramError> {
    let parts = header.split_whitespace().collect::<Vec<_>>();

    if parts.len() != 5
        || !parts[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !parts[1].eq_ignore_ascii_case("matrix")
    {
        return Err(IramError::Parse(
            "Matrix Market header must be '%%MatrixMarket matrix <format> <field> <symmetry>'"
                .to_string(),
        ));
    }

    let format = match parts[2].to_ascii_lowercase().as_str() {
        "coordinate" => MatrixMarketFormat::Coordinate,
        "array" => MatrixMarketFormat::Array,
        other => {
            return Err(IramError::Parse(format!(
                "unsupported Matrix Market storage format '{other}'"
            )));
        }
    };

    let field = match parts[3].to_ascii_lowercase().as_str() {
        "real" => MatrixMarketField::Real,
        "integer" => MatrixMarketField::Integer,
        "complex" => MatrixMarketField::Complex,
        "pattern" => MatrixMarketField::Pattern,
        other => {
            return Err(IramError::Parse(format!(
                "unsupported Matrix Market field '{other}'"
            )));
        }
    };

    let symmetry = match parts[4].to_ascii_lowercase().as_str() {
        "general" => MatrixMarketSymmetry::General,
        "symmetric" => MatrixMarketSymmetry::Symmetric,
        "skew-symmetric" => MatrixMarketSymmetry::SkewSymmetric,
        "hermitian" => MatrixMarketSymmetry::Hermitian,
        other => {
            return Err(IramError::Parse(format!(
                "unsupported Matrix Market symmetry '{other}'"
            )));
        }
    };

    Ok((format, field, symmetry))
}

fn build_matrix_market_operator(
    dimension: usize,
    mut entries: Vec<MatrixMarketEntry>,
    label: String,
) -> MatrixMarketOperator {
    entries.sort_unstable_by_key(|entry| (entry.row, entry.column));

    let mut row_offsets = vec![0usize; dimension + 1];
    entries
        .iter()
        .for_each(|entry| row_offsets[entry.row + 1] += 1);
    (1..=dimension).for_each(|row| row_offsets[row] += row_offsets[row - 1]);

    let columns = entries.iter().map(|entry| entry.column).collect::<Vec<_>>();
    let values = entries.iter().map(|entry| entry.value).collect::<Vec<_>>();

    MatrixMarketOperator {
        dimension,
        row_offsets,
        columns,
        values,
        label,
    }
}

fn parse_matrix_market_coordinate(
    size_line: &str,
    tokens: &[&str],
    field: MatrixMarketField,
    symmetry: MatrixMarketSymmetry,
) -> Result<(usize, Vec<MatrixMarketEntry>), IramError> {
    let size = size_line.split_whitespace().collect::<Vec<_>>();

    if size.len() != 3 {
        return Err(IramError::Parse(
            "Matrix Market coordinate size line must contain rows, columns, and nnz".to_string(),
        ));
    }

    let rows = parse_usize(size[0], "Matrix Market row count")?;
    let columns = parse_usize(size[1], "Matrix Market column count")?;
    let nnz = parse_usize(size[2], "Matrix Market nonzero count")?;
    ensure_square_matrix(rows, columns, "Matrix Market coordinate matrix")?;

    let mut entries = Vec::with_capacity(nnz);
    let mut cursor = 0usize;

    for entry_index in 0..nnz {
        let row_token = next_token(tokens, &mut cursor, "coordinate row")?;
        let column_token = next_token(tokens, &mut cursor, "coordinate column")?;
        let row = parse_matrix_market_index(row_token, rows, "row")?;
        let column = parse_matrix_market_index(column_token, columns, "column")?;
        let value = read_matrix_market_value(tokens, &mut cursor, field)?;

        add_matrix_market_entry(&mut entries, row, column, value, symmetry).map_err(|message| {
            IramError::Parse(format!(
                "invalid Matrix Market coordinate entry {}: {message}",
                entry_index + 1,
            ))
        })?;
    }

    ensure_no_extra_tokens(tokens, cursor, "Matrix Market coordinate data")?;
    Ok((rows, entries))
}

fn parse_matrix_market_array(
    size_line: &str,
    tokens: &[&str],
    field: MatrixMarketField,
    symmetry: MatrixMarketSymmetry,
) -> Result<(usize, Vec<MatrixMarketEntry>), IramError> {
    if field == MatrixMarketField::Pattern {
        return Err(IramError::Parse(
            "Matrix Market array format cannot use pattern field".to_string(),
        ));
    }

    let size = size_line.split_whitespace().collect::<Vec<_>>();

    if size.len() != 2 {
        return Err(IramError::Parse(
            "Matrix Market array size line must contain rows and columns".to_string(),
        ));
    }

    let rows = parse_usize(size[0], "Matrix Market row count")?;
    let columns = parse_usize(size[1], "Matrix Market column count")?;
    ensure_square_matrix(rows, columns, "Matrix Market array matrix")?;

    let mut entries = Vec::new();
    let mut cursor = 0usize;

    match symmetry {
        MatrixMarketSymmetry::General => {
            for column in 0..columns {
                for row in 0..rows {
                    let value = read_matrix_market_value(tokens, &mut cursor, field)?;
                    add_matrix_market_entry(
                        &mut entries,
                        row,
                        column,
                        value,
                        MatrixMarketSymmetry::General,
                    )
                    .map_err(IramError::Parse)?;
                }
            }
        }
        MatrixMarketSymmetry::Symmetric | MatrixMarketSymmetry::Hermitian => {
            for column in 0..columns {
                for row in column..rows {
                    let value = read_matrix_market_value(tokens, &mut cursor, field)?;
                    add_matrix_market_entry(&mut entries, row, column, value, symmetry)
                        .map_err(IramError::Parse)?;
                }
            }
        }
        MatrixMarketSymmetry::SkewSymmetric => {
            for column in 0..columns {
                for row in (column + 1)..rows {
                    let value = read_matrix_market_value(tokens, &mut cursor, field)?;
                    add_matrix_market_entry(&mut entries, row, column, value, symmetry)
                        .map_err(IramError::Parse)?;
                }
            }
        }
    }

    ensure_no_extra_tokens(tokens, cursor, "Matrix Market array data")?;
    Ok((rows, entries))
}

fn read_matrix_market_value(
    tokens: &[&str],
    cursor: &mut usize,
    field: MatrixMarketField,
) -> Result<Complex64, IramError> {
    match field {
        MatrixMarketField::Real | MatrixMarketField::Integer => {
            let token = next_token(tokens, cursor, "Matrix Market value")?;
            parse_f64(token, "Matrix Market value").map(|value| Complex64::new(value, 0.0))
        }
        MatrixMarketField::Complex => {
            let real_token = next_token(tokens, cursor, "Matrix Market real part")?;
            let imaginary_token = next_token(tokens, cursor, "Matrix Market imaginary part")?;
            let real = parse_f64(real_token, "Matrix Market real part")?;
            let imaginary = parse_f64(imaginary_token, "Matrix Market imaginary part")?;
            Ok(Complex64::new(real, imaginary))
        }
        MatrixMarketField::Pattern => Ok(Complex64::new(1.0, 0.0)),
    }
}

fn add_matrix_market_entry(
    entries: &mut Vec<MatrixMarketEntry>,
    row: usize,
    column: usize,
    value: Complex64,
    symmetry: MatrixMarketSymmetry,
) -> Result<(), String> {
    if row == column && symmetry == MatrixMarketSymmetry::SkewSymmetric && value != Complex64::ZERO
    {
        return Err("skew-symmetric diagonal entries must be zero".to_string());
    }

    if row == column && symmetry == MatrixMarketSymmetry::Hermitian && value.im != 0.0 {
        return Err("Hermitian diagonal entries must be real".to_string());
    }

    if value != Complex64::ZERO {
        entries.push(MatrixMarketEntry { row, column, value });
    }

    if row == column {
        return Ok(());
    }

    match symmetry {
        MatrixMarketSymmetry::General => {}
        MatrixMarketSymmetry::Symmetric => {
            if value != Complex64::ZERO {
                entries.push(MatrixMarketEntry {
                    row: column,
                    column: row,
                    value,
                });
            }
        }
        MatrixMarketSymmetry::SkewSymmetric => {
            if value != Complex64::ZERO {
                entries.push(MatrixMarketEntry {
                    row: column,
                    column: row,
                    value: -value,
                });
            }
        }
        MatrixMarketSymmetry::Hermitian => {
            if value != Complex64::ZERO {
                entries.push(MatrixMarketEntry {
                    row: column,
                    column: row,
                    value: value.conj(),
                });
            }
        }
    }

    Ok(())
}

fn ensure_square_matrix(rows: usize, columns: usize, context: &str) -> Result<(), IramError> {
    if rows == 0 || columns == 0 {
        return Err(IramError::Parse(format!("{context} is empty")));
    }

    if rows != columns {
        return Err(IramError::Parse(format!(
            "{context} must be square, got {rows}x{columns}"
        )));
    }

    Ok(())
}

fn parse_matrix_market_index(
    token: &str,
    dimension: usize,
    axis: &str,
) -> Result<usize, IramError> {
    let index = parse_usize(token, axis)?;

    if index == 0 || index > dimension {
        return Err(IramError::Parse(format!(
            "Matrix Market {axis} index {index} is outside 1..={dimension}"
        )));
    }

    Ok(index - 1)
}

fn next_token<'a>(
    tokens: &'a [&str],
    cursor: &mut usize,
    context: &str,
) -> Result<&'a str, IramError> {
    if *cursor >= tokens.len() {
        return Err(IramError::Parse(format!(
            "unexpected end of Matrix Market file while reading {context}"
        )));
    }

    let token = tokens[*cursor];
    *cursor += 1;
    Ok(token)
}

fn ensure_no_extra_tokens(tokens: &[&str], cursor: usize, context: &str) -> Result<(), IramError> {
    if cursor != tokens.len() {
        return Err(IramError::Parse(format!(
            "{context} has {} extra token(s)",
            tokens.len() - cursor,
        )));
    }

    Ok(())
}

fn parse_usize(token: &str, context: &str) -> Result<usize, IramError> {
    token.parse::<usize>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse {context} '{token}' as an unsigned integer: {error}"
        ))
    })
}

fn parse_f64(token: &str, context: &str) -> Result<f64, IramError> {
    token.parse::<f64>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse {context} '{token}' as a real number: {error}"
        ))
    })
}

impl LinearOperator for DenseMatrixOperator {
    fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension(), vector.len())?;
        validate_dimension(self.dimension(), output.len())?;
        matvec_into(
            ZgemvTranspose::None,
            self.matrix.view(),
            Complex64::new(1.0, 0.0),
            vector,
            Complex64::ZERO,
            output.view_mut(),
        );
        Ok(())
    }

    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        mut output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension(), block.nrows())?;
        validate_dimension(self.dimension(), output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }
        matmul_into(
            ZgemmTranspose::None,
            ZgemmTranspose::None,
            Complex64::new(1.0, 0.0),
            self.matrix.view(),
            block,
            Complex64::ZERO,
            output.view_mut(),
        );
        Ok(())
    }

    fn description(&self) -> String {
        self.label.clone()
    }
}

impl LinearOperator for MatrixMarketOperator {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, vector.len())?;
        validate_dimension(self.dimension, output.len())?;

        for row in 0..self.dimension {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];

            output[row] = self.columns[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(|(&column, &value)| value * vector[column])
                .sum::<Complex64>();
        }

        Ok(())
    }

    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        mut output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, block.nrows())?;
        validate_dimension(self.dimension, output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }

        output.fill(Complex64::ZERO);
        for row in 0..self.dimension {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for entry in start..end {
                let source = self.columns[entry];
                let value = self.values[entry];
                for column in 0..block.ncols() {
                    output[[row, column]] += value * block[[source, column]];
                }
            }
        }

        Ok(())
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

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        let n = self.dimension();
        validate_dimension(n, vector.len())?;
        validate_dimension(n, output.len())?;

        let m = self.m;
        let h = self.h();
        let inv_h2 = 1.0 / (h * h);
        let conv = self.rho / (2.0 * h);

        let center_scale = Complex64::new(-4.0 * inv_h2, 0.0);
        let left_scale = Complex64::new(inv_h2 - conv, 0.0);
        let right_scale = Complex64::new(inv_h2 + conv, 0.0);
        let vertical_scale = Complex64::new(inv_h2, 0.0);

        for j in 0..m {
            let row_start = j * m;
            for i in 0..m {
                let k = row_start + i;

                let mut acc = vector[k] * center_scale;

                if i > 0 {
                    acc += vector[k - 1] * left_scale;
                }
                if i + 1 < m {
                    acc += vector[k + 1] * right_scale;
                }
                if j > 0 {
                    acc += vector[k - m] * vertical_scale;
                }
                if j + 1 < m {
                    acc += vector[k + m] * vertical_scale;
                }

                output[k] = acc;
            }
        }

        Ok(())
    }

    fn apply_block_into(
        &self,
        block: ArrayView2<'_, Complex64>,
        mut output: ArrayViewMut2<'_, Complex64>,
    ) -> Result<(), IramError> {
        let n = self.dimension();
        validate_dimension(n, block.nrows())?;
        validate_dimension(n, output.nrows())?;
        if block.ncols() != output.ncols() {
            return Err(IramError::DimensionMismatch {
                expected: block.ncols(),
                got: output.ncols(),
            });
        }

        let m = self.m;
        let h = self.h();
        let inv_h2 = 1.0 / (h * h);
        let conv = self.rho / (2.0 * h);

        let center_scale = Complex64::new(-4.0 * inv_h2, 0.0);
        let left_scale = Complex64::new(inv_h2 - conv, 0.0);
        let right_scale = Complex64::new(inv_h2 + conv, 0.0);
        let vertical_scale = Complex64::new(inv_h2, 0.0);

        for column in 0..block.ncols() {
            let input = block.column(column);
            let mut target = output.column_mut(column);
            for j in 0..m {
                let row_start = j * m;
                for i in 0..m {
                    let k = row_start + i;
                    let mut acc = input[k] * center_scale;

                    if i > 0 {
                        acc += input[k - 1] * left_scale;
                    }
                    if i + 1 < m {
                        acc += input[k + 1] * right_scale;
                    }
                    if j > 0 {
                        acc += input[k - m] * vertical_scale;
                    }
                    if j + 1 < m {
                        acc += input[k + m] * vertical_scale;
                    }

                    target[k] = acc;
                }
            }
        }

        Ok(())
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
    F: for<'a, 'b> Fn(ArrayView1<'a, Complex64>, ArrayViewMut1<'b, Complex64>) + Send + Sync,
{
    dimension: usize,
    name: String,
    matvec: F,
}

impl<F> FnOperator<F>
where
    F: for<'a, 'b> Fn(ArrayView1<'a, Complex64>, ArrayViewMut1<'b, Complex64>) + Send + Sync,
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
    F: for<'a, 'b> Fn(ArrayView1<'a, Complex64>, ArrayViewMut1<'b, Complex64>) + Send + Sync,
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
    F: for<'a, 'b> Fn(ArrayView1<'a, Complex64>, ArrayViewMut1<'b, Complex64>) + Send + Sync,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, vector.len())?;
        validate_dimension(self.dimension, output.len())?;
        (self.matvec)(vector, output);
        Ok(())
    }

    fn description(&self) -> String {
        self.name.clone()
    }
}

fn copy_vector_into(mut output: ArrayViewMut1<'_, Complex64>, input: ArrayView1<'_, Complex64>) {
    assert_eq!(output.len(), input.len());
    if let (Some(output_slice), Some(input_slice)) = (
        output.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        output_slice.copy_from_slice(input_slice);
        return;
    }

    output.zip_mut_with(&input, |target, value| *target = *value);
}

fn copy_matrix_into(mut output: ArrayViewMut2<'_, Complex64>, input: ArrayView2<'_, Complex64>) {
    assert_eq!(output.dim(), input.dim());
    if let (Some(output_slice), Some(input_slice)) = (
        output.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
    ) {
        output_slice.copy_from_slice(input_slice);
        return;
    }

    output.zip_mut_with(&input, |target, value| *target = *value);
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
    use ndarray::{Array1, Array2, ShapeBuilder};
    use num_complex::Complex64;

    use super::{
        ConvectionDiffusionOperator, DenseMatrixOperator, GrcarOperator, LinearOperator,
        parse_complex_token,
    };

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

    #[test]
    fn matrix_market_coordinate_hermitian_uses_sparse_matvec() {
        let content = "\
%%MatrixMarket matrix coordinate complex hermitian
3 3 4
1 1 2.0 0.0
2 1 3.0 4.0
2 2 5.0 0.0
3 2 6.0 -1.0
";
        let operator = super::parse_matrix_market(content, "test matrix".to_string())
            .expect("Hermitian Matrix Market matrix should parse");
        let vector = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
        ]);
        let result = operator
            .apply(&vector)
            .expect("Matrix Market matvec should succeed");

        assert_eq!(
            result.to_vec(),
            vec![
                Complex64::new(32.0, -40.0),
                Complex64::new(653.0, 104.0),
                Complex64::new(60.0, -10.0),
            ]
        );
    }

    #[test]
    fn block_apply_matches_column_matvecs_for_builtin_operators() {
        assert_block_apply_matches_column_matvecs(&GrcarOperator::new(7, 3));
        assert_block_apply_matches_column_matvecs(&ConvectionDiffusionOperator::new(3, 1.5));
        let dense = DenseMatrixOperator {
            matrix: Array2::from_shape_fn((4, 4).f(), |(row, column)| {
                Complex64::new((row + 2 * column + 1) as f64, (row + column) as f64 / 5.0)
            }),
            label: "test dense matrix".to_string(),
        };
        assert_block_apply_matches_column_matvecs(&dense);

        let content = "\
%%MatrixMarket matrix coordinate complex general
4 4 6
1 1 2.0 0.0
1 3 -1.0 0.5
2 2 3.0 0.0
3 1 0.0 2.0
4 2 -4.0 0.0
4 4 5.0 -1.0
";
        let matrix_market = super::parse_matrix_market(content, "test matrix".to_string())
            .expect("Matrix Market matrix should parse");
        assert_block_apply_matches_column_matvecs(&matrix_market);
    }

    fn assert_block_apply_matches_column_matvecs(operator: &dyn LinearOperator) {
        let dimension = operator.dimension();
        let block = Array2::from_shape_fn((dimension, 3).f(), |(row, column)| {
            Complex64::new((row + 1) as f64 / 7.0, -((column + 2) as f64) / 11.0)
        });
        let mut block_output = Array2::zeros((dimension, block.ncols()).f());
        operator
            .apply_block_into(block.view(), block_output.view_mut())
            .expect("block application should succeed");

        for column in 0..block.ncols() {
            let mut scalar_output = Array1::zeros(dimension);
            operator
                .apply_into(block.column(column), scalar_output.view_mut())
                .expect("scalar application should succeed");
            for row in 0..dimension {
                assert!(
                    (block_output[[row, column]] - scalar_output[row]).norm() <= 1.0e-12,
                    "block column {column}, row {row} differs"
                );
            }
        }
    }
}
