//! JSON types, validation and conversion to library types. ALL input validation of the CLI lives here: after a value has
//! passed through this module it cannot trigger a library `assert!` / index panic (the CLI error model: a panic reachable from
//! CLI input is a bug; `main` still catches it and exits 70). The obligations are the ones recorded in `verify/README.md`
//! (the CLI validation obligations), plus the finiteness of every number.
//!
//! Conventions (see `README` of the CLI in `verify/README.md`): angles in radians, 3x3 rotations row-major
//! `[[..],[..],[..]]`, pixel coordinates in the `erploc` convention (pixel centres are integers, `u` in `[-0.5, W-0.5)`), which
//! is the convention of the eqar detector's `corner_px` (`ray_to_px`: `(lon + pi) / 2pi * W - 0.5`).
//! Non-finite floats are written as `null` (serde_json); `null` is read as NaN only for the `uv` of an invalid corner.
//!
//! Thresholds stated here: rotation matrices must satisfy `max |R^T R - I| <= ROT_TOL = 1e-6` and `det R > 0`; valid-corner
//! pixel coordinates must lie in `[-W, 2W] x [-H, 2H]` (a loose sanity bound that keeps every later square finite).
use crate::error::CliError;
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::{DMat, Mat, M3, V3};
use erploc_est::lm::Loss;
use erploc_est::resection::CamDof;
use erploc_est::synth::Observation;
use erploc_place::placement::PlacementSpec;
use erploc_place::scene::{cells, Candidate, Obstacle, Workspace};
use erploc_place::visibility::{Sector, VisParams};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ROT_TOL: f64 = 1e-6;
/// Sanity limits (the dense Levenberg-Marquardt of `resect` / `ba_init` returns `Err(NoConvergence)` on a non-finite `J^T J`, so
/// magnitudes that can overflow a square, or a weight `1 / sigma` beyond these ranges, are rejected here with a clear input error
/// instead of reaching that library error):
/// positions and camera centres `|x| <= MAX_COORD` m; marker side `a`, `sigma_px`, `corner_sigma_scale`, `sigma_TS`, `huber_delta`
/// within the ranges below.
pub const MAX_COORD: f64 = 1e6;
pub const A_RANGE: (f64, f64) = (1e-6, 1e3);
pub const SIGMA_PX_RANGE: (f64, f64) = (1e-6, 1e6);
pub const SCALE_RANGE: (f64, f64) = (1e-6, 1e6);
pub const SIGMA_TS_RANGE: (f64, f64) = (1e-9, 1e9);
pub const HUBER_RANGE: (f64, f64) = (1e-9, 1e9);
pub const MAX_CANDS: usize = 4096;
pub const MAX_CELLS: usize = 100_000;
pub const MAX_KN: usize = 20_000_000;
pub const MAX_GRID_POINTS: f64 = 4.0e6;
pub const MAX_FRAMES: usize = 100_000;
/// Largest dense bundle-adjustment Hessian, in entries: `(w * frames + 6 * markers)^2` with `w = 4` (yaw) or `6` (full camera pose).
/// The adjustment holds several matrices of that size (information, factorisation, covariance), so about 16 million entries (128 MB
/// each) is the ceiling; a larger request is an input error instead of a failed allocation. It admits about 1000 yaw frames.
pub const MAX_BA_HESSIAN: usize = 16_000_000;

/// Rejects a bundle adjustment whose dense Hessian `(w * frames + 6 * markers)^2` would exceed [`MAX_BA_HESSIAN`] entries.
pub fn check_ba_size(frames: usize, markers: usize, full_dof: bool) -> Result<(), CliError> {
    let w = if full_dof { 6 } else { 4 };
    let n = frames.saturating_mul(w).saturating_add(markers.saturating_mul(6));
    if n.saturating_mul(n) > MAX_BA_HESSIAN {
        return Err(CliError::Input(format!(
            "bundle adjustment of {frames} frames and {markers} markers has {n} unknowns; the dense Hessian is limited to {MAX_BA_HESSIAN} entries (about {} unknowns)",
            (MAX_BA_HESSIAN as f64).sqrt() as usize
        )));
    }
    Ok(())
}

fn bad<T>(msg: impl Into<String>) -> Result<T, CliError> {
    Err(CliError::Input(msg.into()))
}

pub fn parse<T: serde::de::DeserializeOwned>(s: &str) -> Result<T, CliError> {
    serde_json::from_str(s).map_err(|e| CliError::Input(format!("json: {e}")))
}

pub fn to_json<T: Serialize>(v: &T) -> Result<String, CliError> {
    serde_json::to_string(v).map_err(|e| CliError::Internal(format!("serialise: {e}")))
}

// ------------------------------------------------------------------------------------------------ scalar checks
pub fn finite(x: f64, what: &str) -> Result<f64, CliError> {
    if x.is_finite() {
        Ok(x)
    } else {
        bad(format!("{what} must be finite, got {x}"))
    }
}

pub fn positive(x: f64, what: &str) -> Result<f64, CliError> {
    if x.is_finite() && x > 0.0 {
        Ok(x)
    } else {
        bad(format!("{what} must be finite and > 0, got {x}"))
    }
}

pub fn nonneg(x: f64, what: &str) -> Result<f64, CliError> {
    if x.is_finite() && x >= 0.0 {
        Ok(x)
    } else {
        bad(format!("{what} must be finite and >= 0, got {x}"))
    }
}

pub fn within(x: f64, range: (f64, f64), what: &str) -> Result<f64, CliError> {
    if x.is_finite() && x >= range.0 && x <= range.1 {
        Ok(x)
    } else {
        bad(format!("{what} must be in [{:e}, {:e}], got {x}", range.0, range.1))
    }
}

fn finite_all(xs: &[f64], what: &str) -> Result<(), CliError> {
    match xs.iter().find(|x| !x.is_finite()) {
        Some(x) => bad(format!("{what} must be finite, got {x}")),
        None => Ok(()),
    }
}

/// A position: finite and `|x| <= MAX_COORD`.
pub fn v3(p: &[f64; 3], what: &str) -> Result<V3, CliError> {
    finite_all(p, what)?;
    if p.iter().any(|x| x.abs() > MAX_COORD) {
        return bad(format!("{what} must lie within {MAX_COORD:e} m of the origin, got {p:?}"));
    }
    Ok(*p)
}

/// `max |R^T R - I| <= ROT_TOL`, `det R > 0`, all entries finite.
pub fn rotation(r: &[[f64; 3]; 3], what: &str) -> Result<M3, CliError> {
    for row in r {
        finite_all(row, what)?;
    }
    let m = Mat(*r);
    let mut dev = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = r.iter().map(|row| row[i] * row[j]).sum();
            dev = dev.max((s - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if !(dev.is_finite() && dev <= ROT_TOL && det.is_finite() && det > 0.0) {
        return bad(format!("{what} is not a rotation matrix (max |R^T R - I| = {dev:e}, det = {det}; tolerance {ROT_TOL:e})"));
    }
    Ok(m)
}

pub fn mat_rows(m: &DMat) -> Vec<Vec<f64>> {
    let c = m.cols().max(1);
    m.as_slice().chunks(c).map(|r| r.to_vec()).collect()
}

// ------------------------------------------------------------------------------------------------ enums
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NoiseJson {
    Tangent,
    Pixel,
}

impl NoiseJson {
    pub fn get(self) -> NoiseModel {
        match self {
            NoiseJson::Tangent => NoiseModel::Tangent,
            NoiseJson::Pixel => NoiseModel::Pixel,
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LossJson {
    Linear,
    Huber,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DofJson {
    Yaw,
    Full,
}

impl DofJson {
    pub fn get(self) -> CamDof {
        match self {
            DofJson::Yaw => CamDof::Yaw,
            DofJson::Full => CamDof::Full,
        }
    }
}

pub fn loss(l: Option<LossJson>, delta: Option<f64>) -> Result<Loss, CliError> {
    match l.unwrap_or(LossJson::Linear) {
        LossJson::Linear => Ok(Loss::Linear),
        LossJson::Huber => Ok(Loss::Huber { delta: within(delta.unwrap_or(1.5), HUBER_RANGE, "huber_delta")? }),
    }
}

// ------------------------------------------------------------------------------------------------ camera
fn one() -> i64 {
    1
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct CameraJson {
    pub w: u64,
    pub h: u64,
    #[serde(default = "one")]
    pub model: i64,
}

pub fn camera(c: &CameraJson) -> Result<ErpCamera, CliError> {
    if c.h == 0 || c.h > (1 << 20) || c.h.checked_mul(2) != Some(c.w) {
        return bad(format!("camera needs 0 < h <= 2^20 and w == 2h (got w={}, h={})", c.w, c.h));
    }
    if c.model != 1 && c.model != -1 {
        return bad(format!("camera model (lam_sign) must be 1 or -1, got {}", c.model));
    }
    Ok(ErpCamera::new(c.w as usize, c.h as usize, c.model as i32))
}

// ------------------------------------------------------------------------------------------------ markers and cameras
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MarkerJson {
    pub id: i64,
    pub p: [f64; 3],
    pub r: [[f64; 3]; 3],
}

pub fn marker_pose(m: &MarkerJson) -> Result<MarkerPose, CliError> {
    if m.id < 0 {
        return bad(format!("marker id must be >= 0, got {}", m.id));
    }
    Ok(MarkerPose { id: m.id, p: v3(&m.p, &format!("marker {} p", m.id))?, r: rotation(&m.r, &format!("marker {} r", m.id))? })
}

pub fn marker_map(list: &[MarkerJson]) -> Result<BTreeMap<i64, MarkerPose>, CliError> {
    let mut out = BTreeMap::new();
    for m in list {
        if out.insert(m.id, marker_pose(m)?).is_some() {
            return bad(format!("duplicate marker id {}", m.id));
        }
    }
    Ok(out)
}

pub fn marker_out(m: &MarkerPose) -> MarkerJson {
    MarkerJson { id: m.id, p: m.p, r: m.r.0 }
}

fn zero() -> f64 {
    0.0
}

/// Camera pose input (`x0` of `resect` / a camera of a `ba` start state). `r` (row-major) wins over the Euler angles in `r()`.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct CamIn {
    pub c: [f64; 3],
    #[serde(default = "zero")]
    pub yaw: f64,
    #[serde(default = "zero")]
    pub roll: f64,
    #[serde(default = "zero")]
    pub pitch: f64,
    pub r: Option<[[f64; 3]; 3]>,
}

pub fn cam_pose(c: &CamIn, what: &str) -> Result<CamPose, CliError> {
    let cc = v3(&c.c, &format!("{what} c"))?;
    let (yaw, roll, pitch) = (finite(c.yaw, "yaw")?, finite(c.roll, "roll")?, finite(c.pitch, "pitch")?);
    let r_mat = match &c.r {
        Some(r) => Some(rotation(r, &format!("{what} r"))?),
        None => None,
    };
    Ok(CamPose { c: cc, yaw, roll, pitch, r_mat })
}

#[derive(Serialize, Clone, Debug)]
pub struct CamOut {
    pub c: [f64; 3],
    pub yaw: f64,
    pub roll: f64,
    pub pitch: f64,
    pub r: [[f64; 3]; 3],
}

pub fn cam_out(c: &CamPose) -> CamOut {
    CamOut { c: c.c, yaw: c.yaw, roll: c.roll, pitch: c.pitch, r: c.r().0 }
}

// ------------------------------------------------------------------------------------------------ detections
/// `bool` or the integers / floats 0 and 1.
#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(untagged)]
pub enum Flag {
    B(bool),
    I(i64),
    F(f64),
}

impl Flag {
    fn get(self, what: &str) -> Result<bool, CliError> {
        match self {
            Flag::B(b) => Ok(b),
            Flag::I(0) => Ok(false),
            Flag::I(1) => Ok(true),
            Flag::F(0.0) => Ok(false),
            Flag::F(1.0) => Ok(true),
            other => bad(format!("{what} must be a boolean or 0/1, got {other:?}")),
        }
    }
}

/// Detector marker entry (`bench/eqar_ffi.py::_marker_to_dict`): `corners` TL, TR, BR, BL; `null` allowed for the `uv` of an invalid corner.
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DetectionJson {
    pub id: i64,
    pub corners: [[Option<f64>; 2]; 4],
    pub corner_valid: Option<Vec<Flag>>,
    pub corner_sigma_scale: Option<Vec<Option<f64>>>,
    pub ok: Option<Flag>,
}

/// `Ok(None)`: the entry is dropped (`ok == false` or `id < 0`; the caller counts it).
pub fn observation(d: &DetectionJson, cam: &ErpCamera, ctx: &str) -> Result<Option<Observation>, CliError> {
    let dropped = match d.ok {
        Some(f) => !f.get(&format!("{ctx}: ok"))?,
        None => false,
    } || d.id < 0;
    if dropped {
        return Ok(None);
    }
    let what = format!("{ctx}: marker {}", d.id);
    let mut mask = [true; 4];
    if let Some(v) = &d.corner_valid {
        if v.len() != 4 {
            return bad(format!("{what}: corner_valid needs 4 entries, got {}", v.len()));
        }
        for (m, f) in mask.iter_mut().zip(v) {
            *m = f.get(&format!("{what}: corner_valid"))?;
        }
    }
    let mut scale = [1.0; 4];
    if let Some(s) = &d.corner_sigma_scale {
        if s.len() != 4 {
            return bad(format!("{what}: corner_sigma_scale needs 4 entries, got {}", s.len()));
        }
        for (o, v) in scale.iter_mut().zip(s) {
            match v {
                Some(x) if x.is_finite() && *x >= SCALE_RANGE.0 && *x <= SCALE_RANGE.1 => *o = *x,
                other => {
                    return bad(format!(
                        "{what}: corner_sigma_scale must be in [{:e}, {:e}] (also for invalid corners), got {other:?}",
                        SCALE_RANGE.0, SCALE_RANGE.1
                    ))
                }
            }
        }
    }
    let (w, h) = (cam.w() as f64, cam.h() as f64);
    let mut uv = [[f64::NAN; 2]; 4];
    for k in 0..4 {
        match (d.corners[k][0], d.corners[k][1]) {
            (Some(u), Some(v)) if u.is_finite() && v.is_finite() => {
                if mask[k] && !(u >= -w && u <= 2.0 * w && v >= -h && v <= 2.0 * h) {
                    return bad(format!("{what}: corner {k} pixel ({u}, {v}) outside [-W, 2W] x [-H, 2H]"));
                }
                uv[k] = [u, v];
            }
            _ if mask[k] => return bad(format!("{what}: corner {k} is valid but its pixel is null or not finite")),
            _ => {}
        }
    }
    Ok(Some(Observation { id: d.id, uv, corner_mask: mask, corner_sigma_scale: scale }))
}

/// One frame: converted observations and the number of dropped entries. Duplicate kept ids are rejected.
pub fn frame(list: &[DetectionJson], cam: &ErpCamera, ctx: &str) -> Result<(Vec<Observation>, usize), CliError> {
    let mut out = Vec::with_capacity(list.len());
    let mut seen = BTreeSet::new();
    let mut dropped = 0;
    for d in list {
        match observation(d, cam, ctx)? {
            Some(o) => {
                if !seen.insert(o.id) {
                    return bad(format!("{ctx}: duplicate marker id {}", o.id));
                }
                out.push(o);
            }
            None => dropped += 1,
        }
    }
    Ok((out, dropped))
}

// ------------------------------------------------------------------------------------------------ placement spec
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceJson {
    pub poly: Vec<[f64; 2]>,
    pub spacing: f64,
    pub z_cam: Vec<f64>,
    pub headings: Vec<f64>,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ObstacleJson {
    pub poly: Vec<[f64; 2]>,
    pub height: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct CandidateJson {
    pub xy: [f64; 2],
    pub z: f64,
    pub azimuth: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SectorJson {
    pub lam_min: f64,
    pub lam_max: f64,
    pub phi_min: f64,
    pub phi_max: f64,
}

fn eps_default() -> f64 {
    1e-3
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct VisJson {
    pub psi_max: f64,
    pub px_min: f64,
    #[serde(default = "eps_default")]
    pub eps: f64,
    #[serde(default)]
    pub sectors: Vec<SectorJson>,
}

fn k_min_default() -> usize {
    2
}
fn cap_default() -> f64 {
    10.0
}
fn wk_default() -> f64 {
    1.0
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PlaceSpecJson {
    pub workspace: WorkspaceJson,
    #[serde(default)]
    pub obstacles: Vec<ObstacleJson>,
    pub candidates: Vec<CandidateJson>,
    pub n_max: usize,
    pub a: f64,
    pub camera: CameraJson,
    pub sigma_px: f64,
    pub noise_model: NoiseJson,
    pub vis: VisJson,
    pub tau: f64,
    #[serde(default = "k_min_default")]
    pub k_min: usize,
    pub tau_loo: Option<f64>,
    #[serde(default = "cap_default")]
    pub sigma_cap: f64,
    #[serde(default = "wk_default")]
    pub w_k: f64,
}

fn polygon(p: &[[f64; 2]], what: &str) -> Result<Vec<[f64; 2]>, CliError> {
    if p.len() < 3 {
        return bad(format!("{what} needs at least 3 vertices, got {}", p.len()));
    }
    for v in p {
        finite_all(v, what)?;
        if v[0].abs() > 1e6 || v[1].abs() > 1e6 {
            return bad(format!("{what} vertex {v:?} is beyond 1e6 m"));
        }
    }
    Ok(p.to_vec())
}

pub fn place_spec(j: &PlaceSpecJson) -> Result<PlacementSpec, CliError> {
    let ws = &j.workspace;
    let poly = polygon(&ws.poly, "workspace.poly")?;
    let spacing = positive(ws.spacing, "workspace.spacing")?;
    if ws.z_cam.is_empty() || ws.z_cam.len() > 16 {
        return bad(format!("workspace.z_cam needs 1..=16 heights, got {}", ws.z_cam.len()));
    }
    finite_all(&ws.z_cam, "workspace.z_cam")?;
    if ws.headings.is_empty() || ws.headings.len() > 64 {
        return bad(format!("workspace.headings needs 1..=64 entries, got {}", ws.headings.len()));
    }
    finite_all(&ws.headings, "workspace.headings")?;
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for v in &poly {
        x0 = x0.min(v[0]);
        x1 = x1.max(v[0]);
        y0 = y0.min(v[1]);
        y1 = y1.max(v[1]);
    }
    let pts = ((x1 - x0) / spacing).ceil().max(1.0) * ((y1 - y0) / spacing).ceil().max(1.0);
    if pts.is_nan() || pts > MAX_GRID_POINTS {
        return bad(format!("workspace grid would have {pts:e} points (limit {MAX_GRID_POINTS:e}); increase workspace.spacing"));
    }
    let mut obstacles = Vec::with_capacity(j.obstacles.len());
    for (i, o) in j.obstacles.iter().enumerate() {
        obstacles.push(Obstacle { poly: polygon(&o.poly, &format!("obstacles[{i}].poly"))?, height: finite(o.height, "obstacle height")? });
    }
    let n = j.candidates.len();
    if n == 0 || n > MAX_CANDS {
        return bad(format!("candidates needs 1..={MAX_CANDS} entries, got {n}"));
    }
    let mut candidates = Vec::with_capacity(n);
    for (i, c) in j.candidates.iter().enumerate() {
        let p = v3(&[c.xy[0], c.xy[1], c.z], &format!("candidates[{i}] (x, y, z)"))?;
        candidates.push(Candidate { xy: [p[0], p[1]], z: p[2], azimuth: finite(c.azimuth, "candidate azimuth")? });
    }
    if j.n_max == 0 || j.n_max > n {
        return bad(format!("n_max must be in 1..={n} (the number of candidates), got {}", j.n_max));
    }
    let vis = &j.vis;
    let mut sectors = Vec::with_capacity(vis.sectors.len());
    for s in &vis.sectors {
        finite_all(&[s.lam_min, s.lam_max, s.phi_min, s.phi_max], "vis.sectors")?;
        sectors.push(Sector { lam_min: s.lam_min, lam_max: s.lam_max, phi_min: s.phi_min, phi_max: s.phi_max });
    }
    let spec = PlacementSpec {
        ws: Workspace { poly, spacing, z_cam: ws.z_cam.clone(), headings: ws.headings.clone() },
        obstacles,
        candidates,
        n_max: j.n_max,
        a: within(j.a, A_RANGE, "a")?,
        cam_model: camera(&j.camera)?,
        sigma_px: within(j.sigma_px, SIGMA_PX_RANGE, "sigma_px")?,
        noise_model: j.noise_model.get(),
        vp: VisParams {
            psi_max: positive(vis.psi_max, "vis.psi_max")?,
            px_min: nonneg(vis.px_min, "vis.px_min")?,
            sectors,
            eps: nonneg(vis.eps, "vis.eps")?,
        },
        tau: positive(j.tau, "tau")?,
        k_min: j.k_min,
        tau_loo: match j.tau_loo {
            Some(t) => Some(positive(t, "tau_loo")?),
            None => None,
        },
        sigma_cap: positive(j.sigma_cap, "sigma_cap")?,
        w_k: nonneg(j.w_k, "w_k")?,
    };
    let k_xy = cells(&spec.ws, &spec.obstacles)?.len();
    let k = k_xy * spec.ws.z_cam.len();
    if k == 0 {
        return bad("no evaluation cell lies inside the workspace polygon and outside the obstacles");
    }
    if k > MAX_CELLS || k.saturating_mul(n) > MAX_KN {
        return bad(format!("{k} cells x {n} candidates exceeds the limits ({MAX_CELLS} cells, {MAX_KN} cell-candidate pairs)"));
    }
    Ok(spec)
}

// ------------------------------------------------------------------------------------------------ site
/// `Grid::new` asserts that the cells lie on a 1 m grid (`np.allclose`); the same test without the panic.
pub fn cells_on_1m_grid(cells: &[V3]) -> bool {
    if cells.is_empty() || cells.iter().any(|c| !(c[0].is_finite() && c[1].is_finite())) {
        return false;
    }
    let x0 = cells.iter().fold(f64::INFINITY, |m, c| m.min(c[0]));
    let y0 = cells.iter().fold(f64::INFINITY, |m, c| m.min(c[1]));
    let close = |d: f64| (d.round_ties_even() - d).abs() <= 1e-8 + 1e-5 * d.abs();
    cells.iter().all(|c| close(c[0] - x0) && close(c[1] - y0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_checks() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(rotation(&id, "r").is_ok());
        let mut r = id;
        r[0][0] = 1.001;
        assert!(rotation(&r, "r").is_err());
        let refl = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(rotation(&refl, "r").is_err());
        let mut n = id;
        n[2][1] = f64::NAN;
        assert!(rotation(&n, "r").is_err());
    }

    #[test]
    fn flag_values() {
        assert_eq!(Flag::I(1).get("x").ok(), Some(true));
        assert_eq!(Flag::F(0.0).get("x").ok(), Some(false));
        assert!(Flag::I(2).get("x").is_err());
    }

    #[test]
    fn camera_checks() {
        assert!(camera(&CameraJson { w: 6528, h: 3264, model: 1 }).is_ok());
        assert!(camera(&CameraJson { w: 6528, h: 3000, model: 1 }).is_err());
        assert!(camera(&CameraJson { w: 0, h: 0, model: 1 }).is_err());
        assert!(camera(&CameraJson { w: 8, h: 4, model: 0 }).is_err());
        assert!(camera(&CameraJson { w: u64::MAX, h: u64::MAX / 2 + 1, model: 1 }).is_err());
    }
}
