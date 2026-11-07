//! erploc-core: equirectangular localisation core (pure Rust port of the Python package `erploc`).
pub use erploc_linalg::{DMat, LinalgError, Mat};

pub type V2 = [f64; 2];
pub type V3 = [f64; 3];
pub type M3 = Mat<3, 3>;

pub mod erp;
pub mod jacobian;
pub mod lie;
pub mod markers;
#[cfg(kani)]
mod proofs;
pub mod rot;
