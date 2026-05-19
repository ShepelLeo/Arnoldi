pub mod arnoldi;
pub mod backend;
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

pub use backend::LapackBackend;
#[cfg(feature = "magma")]
pub use backend::MagmaBackend;
pub use config::{SolverConfig, SpectrumTarget, recommended_ncv};
pub use error::IramError;
pub use iram::{solve, solve_with_backend};
pub use operator::{
    ConvectionDiffusionOperator, CsrMatrix, DenseMatrixOperator, FnOperator, GrcarOperator,
    IdentityOperator, LinearOperator, MatrixMarketOperator, matrix_operator_from_text_file,
    parse_complex_token,
};
pub use report::{IterationLog, RitzEstimate, SolveReport};
