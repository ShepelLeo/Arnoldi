//! Формат отчета
use std::fmt::Write;

use num_complex::Complex64;

use crate::config::SolverConfig;

#[derive(Debug, Clone)]
pub struct RitzEstimate {
    pub value: Complex64,
    pub residual_estimate: f64,
}

#[derive(Debug, Clone)]
pub struct IterationLog {
    pub restart: usize,
    pub krylov_dimension: usize,
    pub retained_dimension: usize,
    pub converged: usize,
    pub total_matvecs: usize,
    pub peak_memory_bytes: usize,
    pub happy_breakdown: bool,
    pub wanted: Vec<RitzEstimate>,
    pub shifts: Vec<Complex64>,
}

#[derive(Debug, Clone)]
pub struct SolveReport {
    pub operator_description: String,
    pub start_description: String,
    pub dimension: usize,
    pub config: SolverConfig,
    pub elapsed_seconds: f64,
    pub total_restarts: usize,
    pub total_matvecs: usize,
    pub peak_memory_bytes: usize,
    pub converged: usize,
    pub fully_converged: bool,
    pub happy_breakdown: bool,
    pub final_values: Vec<RitzEstimate>,
    pub history: Vec<IterationLog>,
    pub note: Option<String>,
}

impl SolveReport {
    pub fn render_text(&self) -> String {
        let mut output = String::new();

        writeln!(&mut output, "IRAM convergence report").ok();
        writeln!(&mut output, "=======================").ok();
        writeln!(&mut output, "operator: {}", self.operator_description).ok();
        writeln!(&mut output, "dimension: {}", self.dimension).ok();
        writeln!(&mut output, "start vector: {}", self.start_description).ok();
        writeln!(&mut output, "target: {}", self.config.target).ok();
        writeln!(&mut output, "requested eigenvalues: {}", self.config.nev).ok();
        writeln!(
            &mut output,
            "Arnoldi subspace dimension (ncv): {}",
            self.config.ncv
        )
        .ok();
        writeln!(&mut output, "tolerance: {:.3e}", self.config.tol).ok();
        writeln!(&mut output, "max restarts: {}", self.config.max_restarts).ok();
        writeln!(&mut output).ok();

        writeln!(&mut output, "summary").ok();
        writeln!(&mut output, "-------").ok();
        writeln!(
            &mut output,
            "converged eigenvalues: {} / {}",
            self.converged, self.config.nev,
        )
        .ok();
        writeln!(
            &mut output,
            "elapsed wall-clock time: {:.6} s",
            self.elapsed_seconds,
        )
        .ok();
        writeln!(&mut output, "restart cycles: {}", self.total_restarts).ok();
        writeln!(&mut output, "matvec operations: {}", self.total_matvecs).ok();
        writeln!(
            &mut output,
            "peak tracked heap memory: {} bytes ({:.3} MiB)",
            self.peak_memory_bytes,
            self.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        )
        .ok();
        writeln!(
            &mut output,
            "happy breakdown: {}",
            yes_no(self.happy_breakdown)
        )
        .ok();
        writeln!(
            &mut output,
            "fully converged: {}",
            yes_no(self.fully_converged),
        )
        .ok();

        if let Some(note) = &self.note {
            writeln!(&mut output, "note: {note}").ok();
        }

        writeln!(&mut output).ok();
        writeln!(&mut output, "final Ritz values").ok();
        writeln!(&mut output, "-----------------").ok();

        self.final_values
            .iter()
            .enumerate()
            .for_each(|(index, estimate)| {
                writeln!(
                    &mut output,
                    "{:>2}. lambda = {:>24} | residual estimate = {:.3e}",
                    index + 1,
                    format_complex(estimate.value),
                    estimate.residual_estimate,
                )
                .ok();
            });

        writeln!(&mut output).ok();
        writeln!(&mut output, "history").ok();
        writeln!(&mut output, "-------").ok();

        self.history.iter().for_each(|log| {
            writeln!(
                &mut output,
                "restart {:>2}: krylov_dim={}, kept_dim={}, converged={}, matvecs={}, peak_memory={} bytes, happy_breakdown={}",
                log.restart,
                log.krylov_dimension,
                log.retained_dimension,
                log.converged,
                log.total_matvecs,
                log.peak_memory_bytes,
                yes_no(log.happy_breakdown),
            )
            .ok();

            log.wanted.iter().enumerate().for_each(|(index, estimate)| {
                writeln!(
                    &mut output,
                    "      wanted {:>2}: lambda = {:>24} | residual estimate = {:.3e}",
                    index + 1,
                    format_complex(estimate.value),
                    estimate.residual_estimate,
                )
                .ok();
            });

            if log.shifts.is_empty() {
                writeln!(&mut output, "      shifts: none").ok();
            } else {
                writeln!(&mut output, "      shifts:").ok();
                log.shifts.iter().enumerate().for_each(|(index, shift)| {
                    writeln!(
                        &mut output,
                        "        {:>2}. {}",
                        index + 1,
                        format_complex(*shift),
                    )
                    .ok();
                });
            }
        });

        output
    }
}

pub fn format_complex(value: Complex64) -> String {
    if value.im.abs() <= 1.0e-12 {
        format!("{:.12e}", value.re)
    } else {
        format!("{:.12e} {:+.12e}i", value.re, value.im)
    }
}

fn yes_no(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}
