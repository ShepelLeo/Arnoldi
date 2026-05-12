pub mod block_arnoldi;
pub mod block_iram;
pub mod config;
pub mod error;
pub mod linalg;
pub mod memory;
pub mod operator;
pub mod report;
pub mod selection;

#[global_allocator]
static GLOBAL_ALLOCATOR: memory::TrackingAllocator = memory::TrackingAllocator;

pub use block_iram::solve_block;
pub use config::{SolverConfig, SpectrumTarget};
pub use error::IramError;
pub use operator::{
    ConvectionDiffusionOperator, DenseMatrixOperator, FnOperator, GrcarOperator, IdentityOperator,
    LinearOperator, MatrixMarketOperator, matrix_operator_from_text_file, parse_complex_token,
};
pub use report::{IterationLog, RitzEstimate, SolveReport};
