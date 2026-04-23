use std::fmt;

use crate::error::IramError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumTarget {
    LargestMagnitude,
    SmallestMagnitude,
    LargestReal,
    SmallestReal,
    BothEndsReal,
}

impl SpectrumTarget {
    pub fn description(self) -> &'static str {
        match self {
            Self::LargestMagnitude => "largest magnitude",
            Self::SmallestMagnitude => "smallest magnitude",
            Self::LargestReal => "largest algebraic value (real part)",
            Self::SmallestReal => "smallest algebraic value (real part)",
            Self::BothEndsReal => "half from each edge of the spectrum (real part)",
        }
    }
}

impl fmt::Display for SpectrumTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.description())
    }
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub nev: usize,
    pub ncv: usize,
    pub max_restarts: usize,
    pub tol: f64,
    pub breakdown_tol: f64,
    pub target: SpectrumTarget,
}

impl SolverConfig {
    pub fn validate(&self, dimension: usize) -> Result<(), IramError> {
        if dimension == 0 {
            return Err(IramError::InvalidConfig(
                "the operator dimension must be strictly positive".to_string(),
            ));
        }

        if self.nev == 0 {
            return Err(IramError::InvalidConfig(
                "nev must be strictly positive".to_string(),
            ));
        }

        if self.nev >= dimension {
            return Err(IramError::InvalidConfig(format!(
                "nev ({}) must be smaller than the operator dimension ({dimension})",
                self.nev,
            )));
        }

        if self.ncv <= self.nev {
            return Err(IramError::InvalidConfig(format!(
                "ncv ({}) must be larger than nev ({})",
                self.ncv, self.nev,
            )));
        }

        if self.ncv > dimension {
            return Err(IramError::InvalidConfig(format!(
                "ncv ({}) cannot exceed the operator dimension ({dimension})",
                self.ncv,
            )));
        }

        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(IramError::InvalidConfig(
                "tol must be a positive finite number".to_string(),
            ));
        }

        if !self.breakdown_tol.is_finite() || self.breakdown_tol <= 0.0 {
            return Err(IramError::InvalidConfig(
                "breakdown_tol must be a positive finite number".to_string(),
            ));
        }

        Ok(())
    }
}

pub fn recommended_ncv(nev: usize, dimension: usize) -> usize {
    let lower_bound = nev.saturating_add(2);
    let heuristic = nev.saturating_mul(2).saturating_add(8);

    heuristic.max(lower_bound).min(dimension)
}
