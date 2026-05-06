//! Пользовательский сценарий
//! Здесь содержится описание CLI, сборка периферии для входа в алгоритм, обработка вывода

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use complex_block_cpu::config::{SolverConfig, SpectrumTarget};
use complex_block_cpu::linalg::ops::{normalize, normalized_random_unitary_matrix};
use complex_block_cpu::memory;
use complex_block_cpu::operator::{
    ConvectionDiffusionOperator, GrcarOperator, IdentityOperator, LinearOperator,
    matrix_operator_from_text_file, parse_complex_token,
};
use complex_block_cpu::{IramError, solve_block};
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// CLI
#[derive(Debug, Parser)]
#[command(author, version, about = "Complex IRAM written with ndarray")]
struct Cli {
    /// Размерность задачи
    #[arg(long, default_value_t = 32)]
    dimension: usize,

    /// Искомое количество собственных значений
    #[arg(long, default_value_t = 1)]
    nev: usize,

    /// Размер стартового и последующих блочных пополнений; по умолчанию равен nev
    #[arg(long)]
    block_size: Option<usize>,

    /// Количество блочных итераций Arnoldi
    #[arg(long)]
    ncv: usize,

    /// Максимальное количество рестартов
    #[arg(long, default_value_t = 40)]
    max_restarts: usize,

    /// Невязка Ритц-пары
    #[arg(long, default_value_t = 1.0e-10)]
    tol: f64,

    /// Стоп-значение невязки ортогонализации при пополнении базиса Крылова
    #[arg(long, default_value_t = 1.0e-12)]
    breakdown_tol: f64,

    /// Расширение окружности для выбора Ritz-пар, удерживаемых на толстом рестарте
    #[arg(long, default_value_t = 1.0)]
    ritz_inflation: f64,

    /// Искомая часть спектра
    #[arg(long, value_enum, default_value_t = TargetArg::LargestMagnitude)]
    target: TargetArg,

    /// Название оператора
    #[arg(long, value_enum, default_value_t = OperatorArg::Identity)]
    operator: OperatorArg,

    /// Файл с плотной матрицей
    #[arg(long)]
    matrix_file: Option<PathBuf>,

    /// Стартовый вектор
    #[arg(long)]
    start_vector: Option<PathBuf>,

    /// Параметр матрицы (1)
    #[arg(long, default_value_t = 3)]
    grcar_upper: usize,

    /// Параметр матрицы (2)
    #[arg(long, default_value_t = 100.0)]
    rho: f64,

    /// Сид генерации стартового вектора
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Файл отчёта
    #[arg(long, default_value = "iram_report.txt")]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetArg {
    LargestMagnitude,
    SmallestMagnitude,
    LargestReal,
    SmallestReal,
    BothEndsReal,
}

impl From<TargetArg> for SpectrumTarget {
    fn from(value: TargetArg) -> Self {
        match value {
            TargetArg::LargestMagnitude => Self::LargestMagnitude,
            TargetArg::SmallestMagnitude => Self::SmallestMagnitude,
            TargetArg::LargestReal => Self::LargestReal,
            TargetArg::SmallestReal => Self::SmallestReal,
            TargetArg::BothEndsReal => Self::BothEndsReal,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OperatorArg {
    Identity,
    Grcar,
    ConvectionDiffusion,
}

/// Запуск программы
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), IramError> {
    let cli = Cli::parse();
    let operator = build_operator(&cli)?;
    let block_size = cli.block_size.unwrap_or(cli.nev);
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let (start_block, start_description) =
        build_start_block(&cli, block_size, operator.dimension(), &mut rng)?;
    let config = SolverConfig {
        nev: cli.nev,
        block_size,
        ncv: cli.ncv,
        max_restarts: cli.max_restarts,
        tol: cli.tol,
        breakdown_tol: cli.breakdown_tol,
        ritz_inflation: cli.ritz_inflation,
        target: cli.target.into(),
    };

    memory::reset_peak();
    let solve_timer = Instant::now();
    let mut report = solve_block(operator.as_ref(), start_block, config, start_description)?;
    report.elapsed_seconds = solve_timer.elapsed().as_secs_f64();
    fs::write(&cli.output, report.render_text())?;

    println!("report written to {}", cli.output.display());
    println!(
        "converged {} / {} eigenvalues in {} restart cycles with {} matvecs\npeak tracked heap memory = {} bytes\nelapsed wall-clock time = {:.6} s",
        report.converged,
        report.config.nev,
        report.total_restarts,
        report.total_matvecs,
        report.peak_memory_bytes,
        report.elapsed_seconds,
    );

    if let Some(note) = &report.note {
        println!("note: {note}");
    }

    Ok(())
}

fn build_operator(cli: &Cli) -> Result<Box<dyn LinearOperator>, IramError> {
    if let Some(matrix_file) = &cli.matrix_file {
        return matrix_operator_from_text_file(matrix_file);
    }

    match cli.operator {
        OperatorArg::Identity => Ok(Box::new(IdentityOperator::new(cli.dimension))),
        OperatorArg::Grcar => Ok(Box::new(GrcarOperator::new(cli.dimension, cli.grcar_upper))),
        OperatorArg::ConvectionDiffusion => Ok(Box::new(ConvectionDiffusionOperator::new(
            cli.dimension,
            cli.rho,
        ))),
    }
}

fn build_start_block(
    cli: &Cli,
    block_size: usize,
    dimension: usize,
    rng: &mut StdRng,
) -> Result<(Array2<Complex64>, String), IramError> {
    if let Some(path) = &cli.start_vector {
        if block_size != 1 {
            return Err(IramError::InvalidConfig(
                "--start-vector can only be used with --block-size 1".to_string(),
            ));
        }

        let content = fs::read_to_string(path)?;
        let entries = content
            .split_whitespace()
            .map(parse_complex_token)
            .collect::<Result<Vec<_>, _>>()?;

        if entries.len() != dimension {
            return Err(IramError::DimensionMismatch {
                expected: dimension,
                got: entries.len(),
            });
        }

        let mut vector = Array1::from_vec(entries);
        normalize(&mut vector, "user-supplied start vector")?;
        let block = vector.insert_axis(Axis(1));
        return Ok((block, format!("loaded from {}", path.display())));
    }

    normalized_random_unitary_matrix(dimension, block_size, rng).map(|block| {
        (
            block,
            format!(
                "random orthonormal block {}x{} with seed {}",
                dimension, block_size, cli.seed
            ),
        )
    })
}
