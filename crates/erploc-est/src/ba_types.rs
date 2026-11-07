//! BA problem and state types. Port of `erploc/ba_types.py`.
//!
//! Python's string `noise_model` / `cam_dof` arguments are enums, so the `NotImplementedError` cases are
//! unrepresentable. `BaError` also carries the data-dependent failures of `ba_init` and `bundle_adjust`.

use crate::resection::{CamDof, ResectError};
use crate::synth::Observation;
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::V3;
use erploc_linalg::LinalgError;
use std::collections::BTreeMap;

/// BA input. `Yaw`: `rp[f]` = known (roll, pitch) of frame f (constraint). `Full`: `rp` is only a hint for the
/// initial values.
#[derive(Clone, Debug)]
pub struct BaProblem {
    pub frames: Vec<Vec<Observation>>,
    pub rp: Vec<[f64; 2]>,
    /// id -> (p_TS, sigma_TS); `BTreeMap` order = Python `sorted()`.
    pub surveyed: BTreeMap<i64, (V3, f64)>,
    pub a: f64,
    pub cam_model: ErpCamera,
    pub sigma_px: f64,
    pub noise_model: NoiseModel,
    pub cam_dof: CamDof,
}

/// `cams` has one entry per frame (`None` for a frame without observations); `markers` holds all markers.
#[derive(Clone, Debug)]
pub struct BaState {
    pub cams: Vec<Option<CamPose>>,
    pub markers: BTreeMap<i64, MarkerPose>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BaError {
    /// `ba_init`: no frame sees >= 2 surveyed markers.
    NoSeedFrame,
    /// `ba_init`: graph not connected / surveyed marker unobserved.
    Disconnected {
        cams: Vec<usize>,
        markers: Vec<i64>,
    },
    /// `bundle_adjust` (a), Yaw: x0 roll/pitch differs from `rp`.
    X0RollPitchMismatch {
        frame: usize,
    },
    /// `bundle_adjust` (a), Full: x0 camera has no valid rotation.
    X0InvalidRotation {
        frame: usize,
    },
    /// `bundle_adjust` (b).
    TooFewSurveyed(usize),
    /// `bundle_adjust` (c).
    SurveyedMissingInX0(Vec<i64>),
    /// `bundle_adjust`: LM ended non-finite (Python assert).
    NonFinite,
    /// Malformed `BaProblem` / `BaState` (wrong lengths, duplicate ids, non-positive sigma ...); the message names the field.
    Shape(String),
    Resect(ResectError),
    Linalg(LinalgError),
}

impl std::fmt::Display for BaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaError::NoSeedFrame => write!(f, "ba_init: no frame sees at least 2 surveyed markers"),
            BaError::Disconnected { cams, markers } => {
                write!(f, "ba_init: graph not connected; unreached cameras {cams:?}, markers {markers:?}")
            }
            BaError::X0RollPitchMismatch { frame } => write!(f, "bundle_adjust: x0 roll/pitch of frame {frame} differs from rp"),
            BaError::X0InvalidRotation { frame } => write!(f, "bundle_adjust: x0 camera of frame {frame} has no valid rotation"),
            BaError::TooFewSurveyed(n) => write!(f, "bundle_adjust: {n} surveyed marker(s), need at least 2"),
            BaError::SurveyedMissingInX0(ids) => write!(f, "bundle_adjust: surveyed markers {ids:?} missing in x0"),
            BaError::NonFinite => write!(f, "bundle_adjust: Levenberg-Marquardt ended non-finite"),
            BaError::Shape(msg) => write!(f, "invalid problem shape: {msg}"),
            BaError::Resect(e) => write!(f, "bundle adjustment: {e}"),
            BaError::Linalg(e) => write!(f, "bundle adjustment: linear algebra failure: {e}"),
        }
    }
}

impl std::error::Error for BaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BaError::Resect(e) => Some(e),
            BaError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ResectError> for BaError {
    fn from(e: ResectError) -> Self {
        BaError::Resect(e)
    }
}

impl From<LinalgError> for BaError {
    fn from(e: LinalgError) -> Self {
        BaError::Linalg(e)
    }
}
