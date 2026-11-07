//! CLI error type and exit-code map.
//!
//! 1 a check failed (`erploc e2e`: the run completed and at least one line did not pass), 2 usage (bad command line), 3 invalid input (anything `io` rejects before a library call, including
//! `BaError::Shape`), 4 numerical / library failure (every other library `Err`), 5 I/O (file cannot be read or
//! written), 70 internal error (a panic caught by `main`; a bug by the CLI error model).
use erploc_est::ba_types::BaError;
use erploc_est::resection::ResectError;
use erploc_linalg::LinalgError;
use erploc_place::placement::PlaceError;

#[derive(Debug, Clone, PartialEq)]
pub enum CliError {
    Usage(String),
    Input(String),
    Numeric(String),
    Io(String),
    Internal(String),
    /// `erploc e2e`: the computation ran, but a check line failed.
    CheckFailed(String),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::CheckFailed(_) => 1,
            CliError::Usage(_) => 2,
            CliError::Input(_) => 3,
            CliError::Numeric(_) => 4,
            CliError::Io(_) => 5,
            CliError::Internal(_) => 70,
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Usage(m) => write!(f, "usage error: {m}"),
            CliError::Input(m) => write!(f, "invalid input: {m}"),
            CliError::Numeric(m) => write!(f, "computation failed: {m}"),
            CliError::Io(m) => write!(f, "i/o error: {m}"),
            CliError::Internal(m) => write!(f, "internal error: {m}"),
            CliError::CheckFailed(m) => write!(f, "check failed: {m}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<LinalgError> for CliError {
    fn from(e: LinalgError) -> Self {
        CliError::Numeric(e.to_string())
    }
}

impl From<ResectError> for CliError {
    fn from(e: ResectError) -> Self {
        match e {
            ResectError::InvalidInput(_) => CliError::Input(e.to_string()),
            _ => CliError::Numeric(e.to_string()),
        }
    }
}

impl From<BaError> for CliError {
    fn from(e: BaError) -> Self {
        match e {
            BaError::Shape(_) => CliError::Input(e.to_string()),
            _ => CliError::Numeric(e.to_string()),
        }
    }
}

impl From<PlaceError> for CliError {
    fn from(e: PlaceError) -> Self {
        match e {
            PlaceError::InvalidInput(_) => CliError::Input(e.to_string()),
            _ => CliError::Numeric(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_map() {
        assert_eq!(CliError::CheckFailed("x".into()).exit_code(), 1);
        assert_eq!(CliError::Usage("x".into()).exit_code(), 2);
        assert_eq!(CliError::Input("x".into()).exit_code(), 3);
        assert_eq!(CliError::Numeric("x".into()).exit_code(), 4);
        assert_eq!(CliError::Io("x".into()).exit_code(), 5);
        assert_eq!(CliError::Internal("x".into()).exit_code(), 70);
    }

    #[test]
    fn library_errors_map() {
        assert_eq!(CliError::from(BaError::Shape("rp".into())).exit_code(), 3);
        assert_eq!(CliError::from(BaError::NoSeedFrame).exit_code(), 4);
        assert_eq!(CliError::from(ResectError::NoKnownMarkers).exit_code(), 4);
        assert_eq!(CliError::from(PlaceError::Site("bad magic".into())).exit_code(), 4);
        assert_eq!(CliError::from(PlaceError::InvalidInput("x".into())).exit_code(), 3);
        assert_eq!(CliError::from(ResectError::InvalidInput("x".into())).exit_code(), 3);
        assert_eq!(CliError::from(LinalgError::Singular).exit_code(), 4);
        assert!(CliError::Input("boom".into()).to_string().contains("boom"));
    }
}
