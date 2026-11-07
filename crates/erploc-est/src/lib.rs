//! Estimator layer of the erploc port: LM, Fisher information, synthetic data, resection.

pub mod ba;
pub mod ba_init;
pub mod ba_types;
pub mod corners;
pub mod fisher;
pub mod lm;
#[cfg(kani)]
mod proofs;
pub mod resection;
pub mod synth;
