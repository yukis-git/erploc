//! Small dense linear algebra for the erploc port. Row-major f64, no external dependencies, no unsafe.
//! Conventions follow numpy 2.0.2 where the Python code depends on them (see each module).

pub mod chol;
pub mod dmat;
pub mod eig;
pub mod lu;
pub mod mat;
pub mod rng;
pub mod svd;

#[cfg(kani)]
mod proofs;

pub use dmat::DMat;
pub use mat::{cross, dot, norm, Mat, V3};

/// Numerical failures of the factorisations (the only `Err` cases). Shape errors (non-square input, mismatched
/// right-hand side) are programming errors and panic with a message instead.
/// `Singular` = an exactly zero LU pivot (numpy `LinAlgError`); `NotPositiveDefinite` = a Cholesky pivot not > 0;
/// `NoConvergence` = Jacobi sweep limit reached, or non-finite input to the SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinalgError {
    Singular,
    NotPositiveDefinite,
    NoConvergence,
    /// A matrix of the wrong shape for the operation (not square, or too small).
    Shape,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::Singular => write!(f, "singular matrix (zero LU pivot)"),
            LinalgError::NotPositiveDefinite => write!(f, "matrix is not positive definite (Cholesky pivot <= 0)"),
            LinalgError::NoConvergence => write!(f, "no convergence (Jacobi sweep limit reached or non-finite input)"),
            LinalgError::Shape => write!(f, "matrix has the wrong shape for this operation"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// Power of two `2^e ≤ max_abs` (exact scaling factor) for pre-scaling eig/SVD inputs; 1.0 for 0 or non-finite.
pub(crate) fn pow2_scale(max_abs: f64) -> f64 {
    if max_abs == 0.0 || !max_abs.is_finite() {
        1.0
    } else if max_abs < f64::MIN_POSITIVE {
        f64::MIN_POSITIVE
    } else {
        f64::from_bits(max_abs.to_bits() & 0x7FF0_0000_0000_0000)
    }
}

#[cfg(test)]
mod display_tests {
    use super::LinalgError;

    #[test]
    fn display_nonempty() {
        for e in [LinalgError::Singular, LinalgError::NotPositiveDefinite, LinalgError::NoConvergence] {
            assert!(!e.to_string().is_empty());
            let b: Box<dyn std::error::Error> = Box::new(e);
            assert!(std::error::Error::source(&*b).is_none());
        }
    }
}
