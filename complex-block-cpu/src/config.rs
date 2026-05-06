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
    pub block_size: usize,
    pub ncv: usize,
    pub max_restarts: usize,
    pub tol: f64,
    pub breakdown_tol: f64,
    pub ritz_inflation: f64,
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

        if self.block_size == 0 {
            return Err(IramError::InvalidConfig(
                "block_size must be strictly positive".to_string(),
            ));
        }

        if self.block_size > dimension {
            return Err(IramError::InvalidConfig(format!(
                "block_size ({}) cannot exceed the operator dimension ({dimension})",
                self.block_size,
            )));
        }

        if self.ncv == 0 {
            return Err(IramError::InvalidConfig(
                "ncv must be strictly positive".to_string(),
            ));
        }

        let krylov_capacity = self.ncv.checked_mul(self.block_size).ok_or_else(|| {
            IramError::InvalidConfig("ncv * block_size overflows usize".to_string())
        })?;

        let effective_capacity = krylov_capacity.min(dimension);
        if effective_capacity <= self.nev {
            return Err(IramError::InvalidConfig(format!(
                "min(ncv * block_size, dimension) ({effective_capacity}) must be larger than nev ({})",
                self.nev,
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

        if !self.ritz_inflation.is_finite() || self.ritz_inflation < 1.0 {
            return Err(IramError::InvalidConfig(
                "ritz_inflation must be a finite number >= 1".to_string(),
            ));
        }

        Ok(())
    }
}
