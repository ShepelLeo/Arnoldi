pub mod arnoldi;
pub mod config;
pub mod error;
pub mod iram;
pub mod linalg;
pub mod memory;
pub mod operator;
pub mod report;
pub mod selection;

#[global_allocator]
static GLOBAL_ALLOCATOR: memory::TrackingAllocator = memory::TrackingAllocator;

pub use config::{SolverConfig, SpectrumTarget, recommended_ncv};
pub use error::IramError;
pub use iram::solve;
pub use operator::{
    ConvectionDiffusionOperator, DenseMatrixOperator, FnOperator, GrcarOperator, IdentityOperator,
    LinearOperator,
};
pub use report::{IterationLog, RitzEstimate, SolveReport};
