//! Пользовательский сценарий
//! Здесь содержится описание CLI, сборка периферии для входа в алгоритм, обработка вывода

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use simple::config::{SolverConfig, SpectrumTarget, recommended_ncv};
use simple::linalg::ops::{normalize, normalized_random_vector};
use simple::memory;
use simple::operator::{
    ConvectionDiffusionOperator, DenseMatrixOperator, GrcarOperator, IdentityOperator,
    LinearOperator,
};
use simple::{IramError, solve};


/// CLI
#[derive(Debug, Parser)]
#[command(author, version, about = "Real nonsymmetric IRAM written with ndarray")]
struct Cli {
    /// Размерность задачи
    #[arg(long, default_value_t = 32)]
    dimension: usize,

    /// Размерность базиса Крылова
    #[arg(long, default_value_t = 1)]
    nev: usize,

    /// Искомое количество собственных значений
    #[arg(long)]
    ncv: Option<usize>,

    /// Максимальное количество рестартов
    #[arg(long, default_value_t = 40)]
    max_restarts: usize,

    /// Невязка Ритц-пары
    #[arg(long, default_value_t = 1.0e-10)]
    tol: f64,

    /// Стоп-значение невязки ортогонализации при пополнении базиса Крылова
    #[arg(long, default_value_t = 1.0e-12)]
    breakdown_tol: f64,

    /// Искомая часть спектра
    /// См. [TargetArg]
    #[arg(long, value_enum, default_value_t = TargetArg::LargestMagnitude)]
    target: TargetArg,

    /// Название оператора
    /// См. [build_operator], [OperatorArg] и [simple::operator]
    #[arg(long, value_enum, default_value_t = OperatorArg::Identity)]
    operator: OperatorArg,

    /// Файл с разреженной матрицы
    #[arg(long)]
    matrix_file: Option<PathBuf>,

    /// Стартовый вектор
    #[arg(long)]
    start_vector: Option<PathBuf>,

    /// Параметр матрицы (1)
    /// См. [GrcarOperator]
    #[arg(long, default_value_t = 3)]
    grcar_upper: usize,

    /// Параметр матрицы (2)
    /// См. [ConvectionDiffusionOperator]
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
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let (start_vector, start_description) =
        build_start_vector(&cli, operator.dimension(), &mut rng)?;
    let config = SolverConfig {
        nev: cli.nev,
        ncv: cli
            .ncv
            .unwrap_or_else(|| recommended_ncv(cli.nev, operator.dimension())),
        max_restarts: cli.max_restarts,
        tol: cli.tol,
        breakdown_tol: cli.breakdown_tol,
        target: cli.target.into(),
    };

    memory::reset_peak();
    let solve_timer = Instant::now();
    let mut report = solve(operator.as_ref(), start_vector, config, start_description)?; // Вход в алгоритм
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
        return DenseMatrixOperator::from_text_file(matrix_file)
            .map(|operator| Box::new(operator) as Box<dyn LinearOperator>);
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

fn build_start_vector(
    cli: &Cli,
    dimension: usize,
    rng: &mut StdRng,
) -> Result<(Array1<f64>, String), IramError> {
    if let Some(path) = &cli.start_vector {
        let content = fs::read_to_string(path)?;
        let entries = content
            .split_whitespace()
            .map(|entry| {
                entry.parse::<f64>().map_err(|error| {
                    IramError::Parse(format!(
                        "cannot parse start-vector entry '{entry}': {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if entries.len() != dimension {
            return Err(IramError::DimensionMismatch {
                expected: dimension,
                got: entries.len(),
            });
        }

        let mut vector = Array1::from_vec(entries);
        normalize(&mut vector, "user-supplied start vector")?;
        return Ok((vector, format!("loaded from {}", path.display())));
    }

    normalized_random_vector(dimension, rng)
        .map(|vector| (vector, format!("random vector with seed {}", cli.seed)))
}
