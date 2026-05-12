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

        if self.block_size > self.nev {
            return Err(IramError::InvalidConfig(format!(
                "block_size ({}) should not exceed nev ({})",
                self.block_size, self.nev,
            )));
        }

        if self.block_size > dimension {
            return Err(IramError::InvalidConfig(format!(
                "block_size ({}) cannot exceed the operator dimension ({dimension})",
                self.block_size,
            )));
        }

        if self.ncv < 2 {
            return Err(IramError::InvalidConfig(format!(
                "ncv ({}) must contain at least two Arnoldi blocks for restart",
                self.ncv,
            )));
        }

        let restart_capacity = self
            .ncv
            .saturating_sub(1)
            .checked_mul(self.block_size)
            .ok_or_else(|| {
                IramError::InvalidConfig("(ncv - 1) * block_size overflows usize".to_string())
            })?;
        if restart_capacity < self.nev {
            return Err(IramError::InvalidConfig(format!(
                "(ncv - 1) * block_size must be at least nev so restart can retain the wanted Ritz values, got ({} - 1) * {} = {} < {}",
                self.ncv, self.block_size, restart_capacity, self.nev,
            )));
        }

        let basis_blocks = self
            .ncv
            .checked_add(1)
            .ok_or_else(|| IramError::InvalidConfig("ncv + 1 overflows usize".to_string()))?;
        let extended_basis_dimension =
            basis_blocks.checked_mul(self.block_size).ok_or_else(|| {
                IramError::InvalidConfig("(ncv + 1) * block_size overflows usize".to_string())
            })?;
        if extended_basis_dimension > dimension {
            return Err(IramError::InvalidConfig(format!(
                "(ncv + 1) * block_size ({} * {} = {}) cannot exceed the operator dimension ({dimension})",
                basis_blocks, self.block_size, extended_basis_dimension,
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
