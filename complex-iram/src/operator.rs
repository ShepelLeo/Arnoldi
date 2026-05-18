//! Operator definitions and CSR input handling.
//!
//! External sparse matrices are accepted either as explicit CSR text or Matrix
//! Market text. Matrix Market input is converted once into the same canonical
//! CSR representation before the selected backend prepares the operator.

use std::fmt;
use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use num_complex::Complex64;

use crate::error::IramError;

/// Canonical compressed sparse row matrix.
///
/// The format used by `from_text_file` is whitespace based:
///
/// ```text
/// # comments are optional
/// rows cols nnz
/// row_offsets[0] ... row_offsets[rows]
/// columns[0] ... columns[nnz-1]
/// values[0] ... values[nnz-1]
/// ```
///
/// Indices are zero-based. Complex values use the same token parser as start
/// vectors, for example `1`, `-2.5`, `3+4i`, `-i`.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl CsrMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<Complex64>,
    ) -> Result<Self, IramError> {
        validate_csr(rows, columns, &row_offsets, &column_indices, &values)?;
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
        })
    }

    pub fn from_text_file(path: impl AsRef<Path>) -> Result<Self, IramError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        Self::from_auto_text(&content).map_err(|error| match error {
            IramError::Parse(message) => {
                IramError::Parse(format!("{}: {message}", path.display()))
            }
            other => other,
        })
    }

    pub fn from_auto_text(content: &str) -> Result<Self, IramError> {
        if is_matrix_market_content(content) {
            Self::from_matrix_market_text(content)
        } else {
            Self::from_text(content)
        }
    }

    pub fn from_matrix_market_text(content: &str) -> Result<Self, IramError> {
        parse_matrix_market_as_csr(content)
    }

    pub fn from_text(content: &str) -> Result<Self, IramError> {
        let tokens = content
            .lines()
            .map(|line| line.split('#').next().unwrap_or(""))
            .flat_map(str::split_whitespace)
            .collect::<Vec<_>>();

        let mut cursor = 0usize;
        let rows = next_usize(&tokens, &mut cursor, "CSR row count")?;
        let columns = next_usize(&tokens, &mut cursor, "CSR column count")?;
        let nnz = next_usize(&tokens, &mut cursor, "CSR nnz")?;

        if rows == 0 || columns == 0 {
            return Err(IramError::Parse(
                "CSR matrix dimensions must be strictly positive".to_string(),
            ));
        }
        if rows != columns {
            return Err(IramError::Parse(format!(
                "CSR operator matrix must be square, got {rows}x{columns}",
            )));
        }

        let mut row_offsets = Vec::with_capacity(rows + 1);
        for index in 0..=rows {
            row_offsets.push(next_usize(
                &tokens,
                &mut cursor,
                &format!("CSR row_offsets[{index}]"),
            )?);
        }

        let mut column_indices = Vec::with_capacity(nnz);
        for index in 0..nnz {
            column_indices.push(next_usize(
                &tokens,
                &mut cursor,
                &format!("CSR column_indices[{index}]"),
            )?);
        }

        let mut values = Vec::with_capacity(nnz);
        for index in 0..nnz {
            let token = next_token(&tokens, &mut cursor, &format!("CSR values[{index}]"))?;
            values.push(parse_complex_token(token)?);
        }

        if cursor != tokens.len() {
            return Err(IramError::Parse(format!(
                "CSR file has {} extra token(s)",
                tokens.len() - cursor,
            )));
        }

        Self::new(rows, columns, row_offsets, column_indices, values)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    pub fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.columns, vector.len())?;
        validate_dimension(self.rows, output.len())?;

        for row in 0..self.rows {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            output[row] = self.column_indices[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(|(&column, &value)| value * vector[column])
                .sum();
        }

        Ok(())
    }
}

fn validate_csr(
    rows: usize,
    columns: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[Complex64],
) -> Result<(), IramError> {
    if row_offsets.len() != rows + 1 {
        return Err(IramError::Parse(format!(
            "CSR row_offsets length must be rows + 1 = {}, got {}",
            rows + 1,
            row_offsets.len(),
        )));
    }
    if column_indices.len() != values.len() {
        return Err(IramError::Parse(format!(
            "CSR column_indices length ({}) must match values length ({})",
            column_indices.len(),
            values.len(),
        )));
    }
    if row_offsets.first().copied() != Some(0) {
        return Err(IramError::Parse(
            "CSR row_offsets[0] must be zero".to_string(),
        ));
    }
    if row_offsets.last().copied() != Some(values.len()) {
        return Err(IramError::Parse(format!(
            "CSR row_offsets[rows] must equal nnz {}, got {}",
            values.len(),
            row_offsets.last().copied().unwrap_or(usize::MAX),
        )));
    }
    for row in 0..rows {
        if row_offsets[row] > row_offsets[row + 1] {
            return Err(IramError::Parse(format!(
                "CSR row_offsets must be monotonically nondecreasing; row_offsets[{row}] > row_offsets[{}]",
                row + 1,
            )));
        }
    }
    for (index, &column) in column_indices.iter().enumerate() {
        if column >= columns {
            return Err(IramError::Parse(format!(
                "CSR column_indices[{index}]={column} is outside 0..{columns}",
            )));
        }
    }
    Ok(())
}

fn next_token<'a>(tokens: &'a [&str], cursor: &mut usize, context: &str) -> Result<&'a str, IramError> {
    let token = tokens.get(*cursor).copied().ok_or_else(|| {
        IramError::Parse(format!("unexpected end of CSR file while reading {context}"))
    })?;
    *cursor += 1;
    Ok(token)
}

fn next_usize(tokens: &[&str], cursor: &mut usize, context: &str) -> Result<usize, IramError> {
    let token = next_token(tokens, cursor, context)?;
    token.parse::<usize>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse {context} '{token}' as an unsigned integer: {error}",
        ))
    })
}

#[derive(Debug, Clone, Copy)]
struct MatrixMarketEntry {
    row: usize,
    column: usize,
    value: Complex64,
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

fn is_matrix_market_content(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_start().starts_with("%%MatrixMarket"))
}

fn parse_matrix_market_as_csr(content: &str) -> Result<CsrMatrix, IramError> {
    let mut lines = content.lines();
    let header = lines
        .by_ref()
        .find(|line| !line.trim().is_empty())
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

    build_csr_from_entries(dimension, entries)
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
                "unsupported Matrix Market storage format '{other}'",
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
                "unsupported Matrix Market field '{other}'",
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
                "unsupported Matrix Market symmetry '{other}'",
            )));
        }
    };

    if field == MatrixMarketField::Pattern && format == MatrixMarketFormat::Array {
        return Err(IramError::Parse(
            "Matrix Market array format cannot use pattern field".to_string(),
        ));
    }

    Ok((format, field, symmetry))
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

    let rows = parse_plain_usize(size[0], "Matrix Market row count")?;
    let columns = parse_plain_usize(size[1], "Matrix Market column count")?;
    let nnz = parse_plain_usize(size[2], "Matrix Market nonzero count")?;
    ensure_square_matrix(rows, columns, "Matrix Market coordinate matrix")?;

    let mut entries = Vec::with_capacity(nnz);
    let mut cursor = 0usize;

    for entry_index in 0..nnz {
        let row_token = mm_next_token(tokens, &mut cursor, "coordinate row")?;
        let column_token = mm_next_token(tokens, &mut cursor, "coordinate column")?;
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
    let size = size_line.split_whitespace().collect::<Vec<_>>();

    if size.len() != 2 {
        return Err(IramError::Parse(
            "Matrix Market array size line must contain rows and columns".to_string(),
        ));
    }

    let rows = parse_plain_usize(size[0], "Matrix Market row count")?;
    let columns = parse_plain_usize(size[1], "Matrix Market column count")?;
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

fn build_csr_from_entries(
    dimension: usize,
    mut entries: Vec<MatrixMarketEntry>,
) -> Result<CsrMatrix, IramError> {
    entries.sort_unstable_by_key(|entry| (entry.row, entry.column));

    let mut row_offsets = Vec::with_capacity(dimension + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);

    let mut cursor = 0usize;
    for row in 0..dimension {
        while cursor < entries.len() && entries[cursor].row == row {
            let column = entries[cursor].column;
            let mut value = Complex64::ZERO;
            while cursor < entries.len()
                && entries[cursor].row == row
                && entries[cursor].column == column
            {
                value += entries[cursor].value;
                cursor += 1;
            }
            if value != Complex64::ZERO {
                column_indices.push(column);
                values.push(value);
            }
        }
        row_offsets.push(column_indices.len());
    }

    CsrMatrix::new(dimension, dimension, row_offsets, column_indices, values)
}

fn read_matrix_market_value(
    tokens: &[&str],
    cursor: &mut usize,
    field: MatrixMarketField,
) -> Result<Complex64, IramError> {
    match field {
        MatrixMarketField::Real | MatrixMarketField::Integer => {
            let token = mm_next_token(tokens, cursor, "Matrix Market value")?;
            parse_f64(token, "Matrix Market value").map(|value| Complex64::new(value, 0.0))
        }
        MatrixMarketField::Complex => {
            let real_token = mm_next_token(tokens, cursor, "Matrix Market real part")?;
            let imaginary_token = mm_next_token(tokens, cursor, "Matrix Market imaginary part")?;
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
            "{context} must be square, got {rows}x{columns}",
        )));
    }

    Ok(())
}

fn parse_matrix_market_index(
    token: &str,
    dimension: usize,
    axis: &str,
) -> Result<usize, IramError> {
    let index = parse_plain_usize(token, axis)?;

    if index == 0 || index > dimension {
        return Err(IramError::Parse(format!(
            "Matrix Market {axis} index {index} is outside 1..={dimension}",
        )));
    }

    Ok(index - 1)
}

fn mm_next_token<'a>(
    tokens: &'a [&str],
    cursor: &mut usize,
    context: &str,
) -> Result<&'a str, IramError> {
    if *cursor >= tokens.len() {
        return Err(IramError::Parse(format!(
            "unexpected end of Matrix Market file while reading {context}",
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

fn parse_plain_usize(token: &str, context: &str) -> Result<usize, IramError> {
    token.parse::<usize>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse {context} '{token}' as an unsigned integer: {error}",
        ))
    })
}

fn parse_f64(token: &str, context: &str) -> Result<f64, IramError> {
    token.parse::<f64>().map_err(|error| {
        IramError::Parse(format!(
            "cannot parse {context} '{token}' as a real number: {error}",
        ))
    })
}

/// Linear operator interface retained for host-side construction and LAPACK.
/// Accelerated backends should call `to_csr`/`as_csr` during preparation and
/// execute MatVec through their backend-specific implementation.
pub trait LinearOperator: Send + Sync {
    fn dimension(&self) -> usize;

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

    fn description(&self) -> String;

    fn as_csr(&self) -> Option<&CsrMatrix> {
        None
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct CsrOperator {
    matrix: CsrMatrix,
    label: String,
}

impl CsrOperator {
    pub fn from_text_file(path: impl AsRef<Path>) -> Result<Self, IramError> {
        let path = path.as_ref();
        Ok(Self {
            matrix: CsrMatrix::from_text_file(path)?,
            label: format!("CSR matrix loaded from {}", path.display()),
        })
    }

    pub fn new(matrix: CsrMatrix, label: impl Into<String>) -> Self {
        Self {
            matrix,
            label: label.into(),
        }
    }

    pub fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }
}

impl LinearOperator for CsrOperator {
    fn dimension(&self) -> usize {
        self.matrix.rows()
    }

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        self.matrix.apply_into(vector, output)
    }

    fn description(&self) -> String {
        self.label.clone()
    }

    fn as_csr(&self) -> Option<&CsrMatrix> {
        Some(&self.matrix)
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        Some(self.matrix.clone())
    }
}

pub fn csr_operator_from_text_file(path: impl AsRef<Path>) -> Result<Box<dyn LinearOperator>, IramError> {
    CsrOperator::from_text_file(path).map(|operator| Box::new(operator) as Box<dyn LinearOperator>)
}

/// Backward-compatible name. Input may be explicit CSR text or Matrix Market;
/// either way the returned operator owns a canonical CSR matrix.
pub fn matrix_operator_from_text_file(path: impl AsRef<Path>) -> Result<Box<dyn LinearOperator>, IramError> {
    csr_operator_from_text_file(path)
}

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
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension, vector.len())?;
        validate_dimension(self.dimension, output.len())?;
        output.assign(&vector);
        Ok(())
    }

    fn description(&self) -> String {
        format!("identity operator of dimension {}", self.dimension)
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        let row_offsets = (0..=self.dimension).collect::<Vec<_>>();
        let column_indices = (0..self.dimension).collect::<Vec<_>>();
        let values = vec![Complex64::new(1.0, 0.0); self.dimension];
        CsrMatrix::new(self.dimension, self.dimension, row_offsets, column_indices, values).ok()
    }
}

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
                Complex64::ZERO
            };
            let upper_end = (row + self.upper_bandwidth + 1).min(self.dimension);
            let superdiagonal_sum = ((row + 1)..upper_end)
                .map(|column| vector[column])
                .sum::<Complex64>();

            output[row] = diagonal + subdiagonal + superdiagonal_sum;
        }

        Ok(())
    }

    fn description(&self) -> String {
        format!(
            "Grcar operator of dimension {} with {} superdiagonals",
            self.dimension, self.upper_bandwidth,
        )
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        let mut row_offsets = Vec::with_capacity(self.dimension + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);

        for row in 0..self.dimension {
            if row > 0 {
                column_indices.push(row - 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            column_indices.push(row);
            values.push(Complex64::new(1.0, 0.0));

            let upper_end = (row + self.upper_bandwidth + 1).min(self.dimension);
            for column in (row + 1)..upper_end {
                column_indices.push(column);
                values.push(Complex64::new(1.0, 0.0));
            }
            row_offsets.push(column_indices.len());
        }

        CsrMatrix::new(
            self.dimension,
            self.dimension,
            row_offsets,
            column_indices,
            values,
        )
        .ok()
    }
}

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
            return Err(IramError::Parse("the dense matrix file is empty".to_string()));
        }
        if rows.iter().any(|row| row.len() != width) {
            return Err(IramError::Parse(
                "all dense matrix rows must have the same width".to_string(),
            ));
        }
        if dimension != width {
            return Err(IramError::Parse(format!(
                "the dense matrix must be square, got {dimension}x{width}",
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

    fn apply_into(
        &self,
        vector: ArrayView1<'_, Complex64>,
        mut output: ArrayViewMut1<'_, Complex64>,
    ) -> Result<(), IramError> {
        validate_dimension(self.dimension(), vector.len())?;
        validate_dimension(self.dimension(), output.len())?;
        output.assign(&self.matrix.dot(&vector));
        Ok(())
    }

    fn description(&self) -> String {
        self.label.clone()
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        let n = self.dimension();
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);

        for row in 0..n {
            for column in 0..n {
                let value = self.matrix[(row, column)];
                if value != Complex64::ZERO {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_offsets.push(column_indices.len());
        }

        CsrMatrix::new(n, n, row_offsets, column_indices, values).ok()
    }
}

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

    fn stencil_scales(&self) -> (Complex64, Complex64, Complex64, Complex64) {
        let h = self.h();
        let inv_h2 = 1.0 / (h * h);
        let conv = self.rho / (2.0 * h);

        (
            Complex64::new(-4.0 * inv_h2, 0.0),
            Complex64::new(inv_h2 - conv, 0.0),
            Complex64::new(inv_h2 + conv, 0.0),
            Complex64::new(inv_h2, 0.0),
        )
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
        let (center_scale, left_scale, right_scale, vertical_scale) = self.stencil_scales();

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

    fn description(&self) -> String {
        format!(
            "2D convection-diffusion operator on {}x{} interior grid, rho={}",
            self.m, self.m, self.rho,
        )
    }

    fn to_csr(&self) -> Option<CsrMatrix> {
        let n = self.dimension();
        let m = self.m;
        let (center_scale, left_scale, right_scale, vertical_scale) = self.stencil_scales();
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);

        for j in 0..m {
            for i in 0..m {
                let k = j * m + i;

                if j > 0 {
                    column_indices.push(k - m);
                    values.push(vertical_scale);
                }
                if i > 0 {
                    column_indices.push(k - 1);
                    values.push(left_scale);
                }
                column_indices.push(k);
                values.push(center_scale);
                if i + 1 < m {
                    column_indices.push(k + 1);
                    values.push(right_scale);
                }
                if j + 1 < m {
                    column_indices.push(k + m);
                    values.push(vertical_scale);
                }

                row_offsets.push(column_indices.len());
            }
        }

        CsrMatrix::new(n, n, row_offsets, column_indices, values).ok()
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
            .finish_non_exhaustive()
    }
}

impl<F> LinearOperator for FnOperator<F>
where
    F: Fn(&Array1<Complex64>) -> Array1<Complex64> + Send + Sync,
{
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
        let result = (self.matvec)(&vector.to_owned());
        validate_dimension(self.dimension, result.len())?;
        output.assign(&result);
        Ok(())
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
            "cannot parse real part of complex entry '{original}': {error}",
        ))
    })
}

fn parse_imaginary_component(component: &str, original: &str) -> Result<f64, IramError> {
    match component {
        "" | "+" => Ok(1.0),
        "-" => Ok(-1.0),
        value => value.parse::<f64>().map_err(|error| {
            IramError::Parse(format!(
                "cannot parse imaginary part of complex entry '{original}': {error}",
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
    use ndarray::Array1;
    use num_complex::Complex64;

    use super::{ConvectionDiffusionOperator, CsrMatrix, LinearOperator, parse_complex_token};

    #[test]
    fn convection_diffusion_dimension_matches_grid() {
        let operator = ConvectionDiffusionOperator::new(4, 1.0);
        assert_eq!(operator.dimension(), 16);
    }

    #[test]
    fn complex_parser_accepts_real_and_imaginary_forms() {
        assert_eq!(parse_complex_token("3.5").unwrap(), Complex64::new(3.5, 0.0));
        assert_eq!(parse_complex_token("2-4i").unwrap(), Complex64::new(2.0, -4.0));
        assert_eq!(parse_complex_token("-i").unwrap(), Complex64::new(0.0, -1.0));
    }

    #[test]
    fn matrix_market_coordinate_converts_to_csr() {
        let matrix = CsrMatrix::from_matrix_market_text(
            r#"
            %%MatrixMarket matrix coordinate complex hermitian
            % row column real imaginary
            3 3 3
            1 1 2.0 0.0
            2 1 3.0 4.0
            3 3 5.0 0.0
            "#,
        )
        .unwrap();

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.columns(), 3);
        assert_eq!(matrix.row_offsets(), &[0, 2, 3, 4]);
        assert_eq!(matrix.column_indices(), &[0, 1, 0, 2]);
        assert_eq!(matrix.values()[0], Complex64::new(2.0, 0.0));
        assert_eq!(matrix.values()[1], Complex64::new(3.0, -4.0));
        assert_eq!(matrix.values()[2], Complex64::new(3.0, 4.0));
        assert_eq!(matrix.values()[3], Complex64::new(5.0, 0.0));
    }

    #[test]
    fn matrix_market_duplicate_entries_are_summed() {
        let matrix = CsrMatrix::from_matrix_market_text(
            r#"
            %%MatrixMarket matrix coordinate real general
            2 2 3
            1 2 1.5
            1 2 2.5
            2 1 -1.0
            "#,
        )
        .unwrap();

        assert_eq!(matrix.row_offsets(), &[0, 1, 2]);
        assert_eq!(matrix.column_indices(), &[1, 0]);
        assert_eq!(matrix.values(), &[Complex64::new(4.0, 0.0), Complex64::new(-1.0, 0.0)]);
    }

    #[test]
    fn csr_parser_accepts_canonical_arrays() {
        let matrix = CsrMatrix::from_text(
            r#"
            # 3x3 with 4 nonzeros
            3 3 4
            0 2 3 4
            0 2 1 2
            1 2 3+1i -4
            "#,
        )
        .unwrap();

        let input = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let mut output = Array1::zeros(3);
        matrix.apply_into(input.view(), output.view_mut()).unwrap();

        assert_eq!(output[0], Complex64::new(7.0, 0.0));
        assert_eq!(output[1], Complex64::new(6.0, 2.0));
        assert_eq!(output[2], Complex64::new(-12.0, 0.0));
    }

    #[test]
    fn convection_diffusion_csr_matches_operator_apply() {
        let operator = ConvectionDiffusionOperator::new(3, 2.0);
        let csr = operator.to_csr().unwrap();
        let vector = Array1::from_iter((0..operator.dimension()).map(|i| Complex64::new(i as f64, 0.0)));
        let direct = operator.apply(&vector).unwrap();
        let mut sparse = Array1::zeros(operator.dimension());
        csr.apply_into(vector.view(), sparse.view_mut()).unwrap();

        for (a, b) in direct.iter().zip(sparse.iter()) {
            assert!((*a - *b).norm() <= 1.0e-12);
        }
    }
}
