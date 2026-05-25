//! Абстракция бэкенда для общего ядра IRAM/Arnoldi.
//!
//! Бэкенд предоставляет ядру только базовые численные примитивы: ортогональные
//! операции над собственным базисом Крылова, плотный/разрежённый matvec и
//! matmul, malloc-style управление крупными буферами и обобщённые операции
//! численной линейной алгебры (multishift QR на маленьком Хессенберге,
//! спектральная задача на малом плотном блоке). Ничего специфичного для
//! IRAM/Арнольди здесь нет.
//!
//! Все матричные данные передаются в виде непрерывных срезов `&[Complex64]`
//! column-major + явное `ld`. Векторы — простые срезы `&[Complex64]`. Никакой
//! `ndarray` в этом интерфейсе не используется.

use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::{OrthogonalizedVector, Trans};
use crate::operator::LinearOperator;

pub mod lapack;
#[cfg(feature = "magma")]
pub mod magma;

pub use lapack::LapackBackend;
#[cfg(feature = "magma")]
pub use magma::MagmaBackend;

/// Опционная плотная матрица (column-major). Используется только для
/// возвращаемых ядру результатов фиксированно небольшой размерности
/// (n×n или n×k, где n = `ncv`).
pub struct DenseColMajor {
    pub data: Vec<Complex64>,
    pub rows: usize,
    pub cols: usize,
}

impl DenseColMajor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Complex64::ZERO; rows.saturating_mul(cols)],
            rows,
            cols,
        }
    }

    #[inline]
    pub fn ld(&self) -> usize {
        self.rows.max(1)
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row + col * self.rows] = value;
    }

    #[inline]
    pub fn column(&self, col: usize) -> &[Complex64] {
        let start = col * self.rows;
        &self.data[start..start + self.rows]
    }

    #[inline]
    pub fn column_mut(&mut self, col: usize) -> &mut [Complex64] {
        let start = col * self.rows;
        &mut self.data[start..start + self.rows]
    }
}

/// Результат малой плотной спектральной задачи: собственные значения и правые
/// собственные векторы.
pub struct SmallEig {
    pub values: Vec<Complex64>,
    /// Правые собственные векторы в column-major раскладке, ld = `dim`.
    pub vectors: Vec<Complex64>,
    pub dim: usize,
}

/// Универсальная абстракция бэкенда. Содержит только независимые от алгоритма
/// примитивы.
pub trait Backend {
    /// Дескриптор оператора, подготовленный для быстрого matvec
    /// (например, CSR-копия на устройстве).
    type OperatorHandle;

    /// Дескриптор крупного базиса Крылова (поля size `m × (ncv+1)`,
    /// column-major). Может жить на CPU или GPU.
    type BasisHandle;

    /// Дескриптор вектора длины `m`. Может жить на CPU или GPU.
    type VectorHandle;

    /// Имя бэкенда для логирования.
    fn name(&self) -> &'static str;

    // ---------- Подготовка операторов и буферов ----------

    /// Превращает абстрактный `LinearOperator` в дескриптор бэкенда.
    /// Для CPU это no-op; для GPU — копия CSR на устройство.
    fn prepare_operator(
        &mut self,
        operator: &dyn LinearOperator,
    ) -> Result<Self::OperatorHandle, IramError>;

    /// Аллоцирует базис на стороне бэкенда. `dimension` — высота
    /// (= размерность задачи), `capacity` — число столбцов под весь Крылов
    /// (`ncv + 1` обычно).
    fn alloc_basis(
        &mut self,
        dimension: usize,
        capacity: usize,
    ) -> Result<Self::BasisHandle, IramError>;

    /// Аллоцирует одиночный вектор.
    fn alloc_vector(&mut self, dimension: usize) -> Result<Self::VectorHandle, IramError>;

    /// Загружает host-вектор в столбец базиса.
    fn write_basis_column(
        &mut self,
        basis: &mut Self::BasisHandle,
        column: usize,
        values: &[Complex64],
    );

    /// Считывает столбец базиса в host-срез.
    fn read_basis_column(
        &mut self,
        basis: &Self::BasisHandle,
        column: usize,
        out: &mut [Complex64],
    );

    /// Загружает host-вектор в `VectorHandle`.
    fn write_vector(&mut self, vector: &mut Self::VectorHandle, values: &[Complex64]);

    /// Считывает `VectorHandle` в host-срез.
    fn read_vector(&mut self, vector: &Self::VectorHandle, out: &mut [Complex64]);

    // ---------- Разрежённый matvec ----------

    /// y = A * x, где A — подготовленный оператор. `x` — столбец `column`
    /// базиса, `y` — результат, размещаемый в `out_vector` (GPU/CPU).
    fn spmv_basis_column(
        &mut self,
        operator: &mut Self::OperatorHandle,
        operator_obj: &dyn LinearOperator,
        basis: &Self::BasisHandle,
        column: usize,
        out_vector: &mut Self::VectorHandle,
        out_host_mirror: &mut [Complex64],
    ) -> Result<(), IramError>;

    // ---------- Векторные примитивы (без участия base) ----------

    /// 2-норма host-среза.
    fn nrm2(&mut self, vector: &[Complex64]) -> f64 {
        crate::linalg::ops::nrm2(vector)
    }

    /// x *= alpha (host).
    fn scal(&mut self, vector: &mut [Complex64], alpha: Complex64) {
        crate::linalg::ops::scal(vector, alpha);
    }

    /// y += alpha * x (host).
    fn axpy(&mut self, target: &mut [Complex64], alpha: Complex64, source: &[Complex64]) {
        crate::linalg::ops::axpy(target, alpha, source);
    }

    /// Нормализация host-среза.
    fn normalize(&mut self, vector: &mut [Complex64], context: &'static str) -> Result<f64, IramError> {
        crate::linalg::ops::normalize(vector, context)
    }

    // ---------- Ортогонализация против полей бэкенда ----------

    /// Классический Gram-Schmidt с реортогонализацией.
    ///
    /// Базис — это первые `basis_columns` колонок дескриптора `basis`
    /// (хранимого на стороне бэкенда). На вход — host-копия кандидата
    /// (`candidate_host`) и его же копия на стороне бэкенда (`candidate_vec`).
    /// Бэкенд обновляет оба, а также накапливает поправки в `h_column`
    /// (host-срез, длина = `basis_columns`).
    fn orthogonalize_against_basis(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        candidate_host: &mut [Complex64],
        candidate_vec: &mut Self::VectorHandle,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector;

    /// Ортогонализация host-вектора против host-матрицы (column-major).
    /// Используется в логике рестарта против уже повернутого базиса.
    fn orthogonalize_against_host_basis(
        &mut self,
        residual: &mut [Complex64],
        basis: &[Complex64],
        basis_rows: usize,
        basis_columns: usize,
        h_column: &mut [Complex64],
        reference_norm: f64,
        breakdown_tol: f64,
    ) -> OrthogonalizedVector;

    // ---------- Плотные операции на маленьких матрицах ----------

    /// C = op(A) * op(B). Все матрицы — column-major, host-стороны.
    fn gemm(
        &mut self,
        trans_a: Trans,
        trans_b: Trans,
        m: usize,
        n: usize,
        k: usize,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        c: &mut [Complex64],
        ldc: usize,
    );

    /// Multishift QR / Schur-like primitive: применяет неявный шифтованный QR к
    /// маленькой верхнехессенберговой `n×n` матрице, накапливает унитарное
    /// преобразование `Q`. Это обобщённый numerical-LA примитив; алгоритм
    /// рестарта дальше сам разбирает результат.
    fn multishift_qr_filter(
        &mut self,
        hessenberg: &DenseColMajor,
        shifts: &[Complex64],
    ) -> Result<(DenseColMajor, DenseColMajor), IramError>;

    /// Спектральная задача на маленьком плотном блоке: собственные значения
    /// + правые собственные векторы. Вход — column-major матрица,
    /// `dim × dim`.
    fn small_eig(&mut self, matrix: &DenseColMajor) -> Result<SmallEig, IramError>;
}
