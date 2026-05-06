use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum IramError {
    InvalidConfig(String),
    DimensionMismatch { expected: usize, got: usize },
    ZeroVector(&'static str),
    Io(std::io::Error),
    Parse(String),
    Spectral(String),
}

impl fmt::Display for IramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid configuration: {message}"),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::ZeroVector(context) => write!(formatter, "zero vector encountered in {context}"),
            Self::Io(error) => write!(formatter, "i/o error: {error}"),
            Self::Parse(message) => write!(formatter, "parse error: {message}"),
            Self::Spectral(message) => write!(formatter, "spectral error: {message}"),
        }
    }
}

impl Error for IramError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IramError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
