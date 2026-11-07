//! Placement optimisation: scene geometry, marker visibility, the placement model and its searches (greedy with 1-swap and
//! pruning, the IFT minimal-count search), the site model, plus numpy-compatibility helpers for the parity tests. No external
//! dependencies, no unsafe.

pub mod fast_eval;
#[cfg(test)]
mod fast_eval_ref;
pub mod npcompat;
pub mod occlusion;
pub mod partial;
pub mod placement;
pub mod scene;
pub mod search;
pub mod sigma10;
pub mod site;
pub mod site_synth;
pub mod visibility;
pub mod zone;

#[cfg(kani)]
mod proofs;
