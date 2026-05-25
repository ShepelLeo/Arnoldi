//! Абстракция бэкенда для общего ядра IRAM/Arnoldi.
//!
//! Бэкенд предоставляет ядру только базовые численные примитивы и владение
//! крупными буферами (базис Крылова, вектор-кандидат). Ничего о шагах
//! Арнольди, ортогонализации, рестартах, Ритц-парах бэкенд не знает —
//! всё это решает ядро.
//!
//! Все матричные данные передаются в виде непрерывных срезов `&[Complex64]`
//! column-major + явное `ld`. Векторы — простые срезы `&[Complex64]`. Никакой
//! `ndarray` в этом интерфейсе не используется.

use num_complex::Complex64;

use crate::error::IramError;
use crate::linalg::ops::Trans;
use crate::operator::LinearOperator;

pub mod lapack;
#[cfg(feature = "magma")]
pub mod magma;

pub use lapack::LapackBackend;
#[cfg(feature = "magma")]
pub use magma::MagmaBackend;

/// Плотная матрица в column-major раскладке. Используется для маленьких
/// `n×n`/`n×k` блоков, передаваемых между ядром и бэкендом.
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

/// Абстракция бэкенда. Только владение крупными буферами + generic numerical
/// primitives. Никаких высокоуровневых алгоритмических методов (CGS,
/// `arnoldi_step`, восстановление Ритца) здесь нет — это ответственность ядра.
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

    /// `y = A * x`, где `A` — подготовленный оператор. `x` — столбец `column`
    /// базиса, `y` — результат, размещаемый в `out_vector` (GPU/CPU). Зеркало
    /// `out_host_mirror` синхронно обновляется, чтобы ядро могло считать
    /// норму на host без отдельной D2H-копии.
    fn spmv_basis_column(
        &mut self,
        operator: &mut Self::OperatorHandle,
        operator_obj: &dyn LinearOperator,
        basis: &Self::BasisHandle,
        column: usize,
        out_vector: &mut Self::VectorHandle,
        out_host_mirror: &mut [Complex64],
    ) -> Result<(), IramError>;

    // ---------- Низкоуровневые vector ops над VectorHandle ----------

    /// `||v||_2` — 2-норма device/host вектора.
    fn vector_nrm2(&mut self, vector: &Self::VectorHandle) -> f64;

    /// `v *= alpha` — масштабирование device/host вектора in-place.
    fn vector_scale(&mut self, vector: &mut Self::VectorHandle, alpha: Complex64);

    // ---------- Низкоуровневые basis ops для CGS, реализуемой в ядре ----------

    /// `proj := V[:, 0..n]^H * x`, где `V` — первые `n` колонок дескриптора
    /// `basis`, `x` — `vector`. Результат — host-срез длиной `n`.
    ///
    /// Это основной primitive, на котором ядро строит классический
    /// Gram-Schmidt: бэкенд знает, на host или GPU лежит базис, но не знает,
    /// для чего эта проекция нужна алгоритму.
    fn basis_prefix_conj_dot_vector(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        vector: &Self::VectorHandle,
        out_projection: &mut [Complex64],
    );

    /// `x -= V[:, 0..n] * proj`, где `V` — первые `n` колонок `basis`,
    /// `x` — `vector`, `proj` — host-срез длиной `n`.
    fn basis_prefix_sub_mul(
        &mut self,
        basis: &Self::BasisHandle,
        basis_columns: usize,
        projection: &[Complex64],
        vector: &mut Self::VectorHandle,
    );

    // ---------- Плотные операции на маленьких матрицах ----------

    /// `C = op(A) * op(B)`. Все матрицы — column-major, host-стороны.
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
    /// преобразование `Q`. Обобщённый numerical-LA примитив; алгоритм
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
