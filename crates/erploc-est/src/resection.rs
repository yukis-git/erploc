//! Single-frame resection (weighted nonlinear least squares), 4-dof (x, y, z, yaw) and 6-dof (c, R).
//! Port of `erploc/resection.py`. Python's `ValueError` cases are [`ResectError`]; the `cam_rot_dof`
//! tuple is the enum [`CamDof`], so the `NotImplementedError` case is unrepresentable.
//!
//! Per-corner masks: only the valid corners of an observation are used (bearings, rows); an observation with no valid
//! corner is dropped like an unknown marker; the 6-dof planar-PnP seeds come from all-4-valid observations only; residual
//! rows of valid corners are whitened by `1 / (sigma * scale)`; the covariance is the plain whitened `JᵀJ` at the solution
//! for every loss (`plain_info`), so under Huber `info`/`cov` are the Gaussian CRLB (the Huber estimator's own asymptotic
//! variance is about 1.037 times larger at delta = 1.5).

use crate::corners::{n_valid, row_select, safe_uv, RowSel};
use crate::fisher::crlb;
use crate::lm::{levenberg_marquardt, vec_retract, LmOptions, Loss, Solver};
use crate::synth::Observation;
use erploc_core::erp::{lamphi_to_bearing, px_to_lamphi, residual_lamphi, wrap_pi, ErpCamera, NoiseModel};
use erploc_core::jacobian::{jac_marker, predict_marker, Axis, CamPose};
use erploc_core::markers::{corners_world, planar_pnp, MarkerPose};
use erploc_core::rot::{exp_so3, mat3_vec, rot_zyx};
use erploc_core::{M3, V2, V3};
use erploc_linalg::{lu, svd, DMat, LinalgError};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// 5 deg x 72 = 360 deg.
pub const N_YAW_STEPS: usize = 72;
/// 6-dof initialisation: coarse yaw scan (15 deg steps) with roll/pitch hints.
pub const N_YAW_HINT_STEPS: usize = 24;
/// Minimum fraction of corners with `D_k . (X_k - c) > 0`.
pub const FORWARD_FRAC_MIN: f64 = 0.75;

/// Python `cam_rot_dof` ('z',) / ('x','y','z').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamDof {
    Yaw,
    Full,
}

impl CamDof {
    pub fn axes(self) -> &'static [Axis] {
        match self {
            CamDof::Yaw => &[Axis::Z],
            CamDof::Full => &[Axis::X, Axis::Y, Axis::Z],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResectError {
    /// No observation refers to a known marker (Python `ValueError`).
    NoKnownMarkers,
    /// No yaw / pose candidate passed the forward test (Python `ValueError`).
    NoForwardCandidate,
    Linalg(LinalgError),
    /// An observation of a known marker has a `corner_sigma_scale` that is not finite and > 0 (also for a masked corner).
    InvalidInput(String),
}

impl std::fmt::Display for ResectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResectError::NoKnownMarkers => write!(f, "resection: no observation refers to a known marker"),
            ResectError::NoForwardCandidate => write!(f, "resection: no pose candidate passed the forward test"),
            ResectError::Linalg(e) => write!(f, "resection: linear algebra failure: {e}"),
            ResectError::InvalidInput(m) => write!(f, "resection: invalid input: {m}"),
        }
    }
}

impl std::error::Error for ResectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ResectError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for ResectError {
    fn from(e: LinalgError) -> Self {
        ResectError::Linalg(e)
    }
}

#[derive(Clone, Debug)]
pub struct ResectOptions {
    pub x0: Option<CamPose>,
    pub sigma_px: f64,
    pub noise_model: NoiseModel,
    pub loss: Loss,
    pub dof: CamDof,
}

/// `cov`/`info` are 4x4 for [`CamDof::Yaw`] (x, y, z, yaw) and 6x6 for [`CamDof::Full`] (dc, dtheta).
#[derive(Clone, Debug)]
pub struct ResectionResult {
    pub cam: CamPose,
    pub cov: DMat,
    pub info: DMat,
    pub rms_px: f64,
    pub n_markers: usize,
    pub converged: bool,
}

type Pairs<'a> = Vec<(&'a Observation, &'a MarkerPose)>;

fn valid_pairs<'a>(obs: &'a [Observation], markers: &'a BTreeMap<i64, MarkerPose>) -> Result<Pairs<'a>, ResectError> {
    let pairs: Pairs<'a> = obs.iter().filter(|ob| n_valid(ob) > 0).filter_map(|ob| markers.get(&ob.id).map(|mk| (ob, mk))).collect();
    if pairs.is_empty() {
        return Err(ResectError::NoKnownMarkers);
    }
    if let Some((ob, _)) = pairs.iter().find(|(ob, _)| !ob.corner_sigma_scale.iter().all(|s| s.is_finite() && *s > 0.0)) {
        return Err(ResectError::InvalidInput(format!(
            "marker {}: corner_sigma_scale must be finite and > 0 for all four corners, got {:?}",
            ob.id, ob.corner_sigma_scale
        )));
    }
    Ok(pairs)
}

fn sub3(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: &V3) -> f64 {
    dot3(a, a).sqrt()
}

/// Camera-frame bearings and world corners of the VALID corners of all pairs, concatenated.
fn stack(pairs: &Pairs, a: f64, cam_model: &ErpCamera) -> (Vec<V3>, Vec<V3>) {
    let mut d_all = Vec::new();
    let mut x_all = Vec::new();
    for (ob, mk) in pairs {
        let d = lamphi_to_bearing(&px_to_lamphi(&safe_uv(ob), cam_model));
        let x = corners_world(&mk.p, &mk.r, a);
        for k in 0..4 {
            if ob.corner_mask[k] {
                d_all.push(d[k]);
                x_all.push(x[k]);
            }
        }
    }
    (d_all, x_all)
}

/// `D_k = R d_k` (row-vector convention `d @ R.T`).
fn world_dirs(r: &M3, d_all: &[V3]) -> Vec<V3> {
    d_all.iter().map(|d| mat3_vec(r, d)).collect()
}

/// Least-squares camera centre for fixed world directions: `sum (I - D D^T) c = sum (I - D D^T) X`
/// (3x3, `lstsq` fallback on a singular LU only).
fn centre_for(dw: &[V3], x_all: &[V3], x_sum: &V3) -> Result<V3, LinalgError> {
    let n_pts = dw.len() as f64;
    let mut m = DMat::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for d in dw {
                s += d[i] * d[j];
            }
            m[(i, j)] = if i == j { n_pts } else { 0.0 } - s;
        }
    }
    let mut dtd = [0.0; 3];
    for (d, x) in dw.iter().zip(x_all) {
        let dx = dot3(d, x);
        for i in 0..3 {
            dtd[i] += d[i] * dx;
        }
    }
    let b = [x_sum[0] - dtd[0], x_sum[1] - dtd[1], x_sum[2] - dtd[2]];
    let c = match lu::solve(&m, &b) {
        Ok(c) => c,
        Err(LinalgError::Singular) => svd::lstsq(&m, &b)?,
        Err(e) => return Err(e),
    };
    Ok([c[0], c[1], c[2]])
}

/// Mean angular residual of a pose candidate; `None` if rejected by the forward test or `dist < 1e-12`.
fn score_pose_candidate(r: &M3, c: &V3, d_all: &[V3], x_all: &[V3]) -> Option<f64> {
    let dw = world_dirs(r, d_all);
    let diff: Vec<V3> = x_all.iter().map(|x| sub3(x, c)).collect();
    let n_fwd = dw.iter().zip(&diff).filter(|(d, df)| dot3(d, df) > 0.0).count();
    if (n_fwd as f64) / (dw.len() as f64) < FORWARD_FRAC_MIN {
        return None;
    }
    let dist: Vec<f64> = diff.iter().map(norm3).collect();
    if dist.iter().any(|&v| v < 1e-12) {
        return None;
    }
    Some(mean_angle(&dw, &diff, &dist))
}

fn mean_angle(dw: &[V3], diff: &[V3], dist: &[f64]) -> f64 {
    let mut s = 0.0;
    for ((d, df), n) in dw.iter().zip(diff).zip(dist) {
        let u = [df[0] / n, df[1] / n, df[2] / n];
        s += dot3(d, &u).clamp(-1.0, 1.0).acos();
    }
    s / dw.len() as f64
}

fn resect_init_4dof(pairs: &Pairs, a: f64, cam_model: &ErpCamera, roll: f64, pitch: f64) -> Result<CamPose, ResectError> {
    let (d_all, x_all) = stack(pairs, a, cam_model);
    let mut x_sum = [0.0; 3];
    for x in &x_all {
        for i in 0..3 {
            x_sum[i] += x[i];
        }
    }
    let mut best: Option<(f64, f64, V3)> = None;
    for i in 0..N_YAW_STEPS {
        let yaw = (5.0 * i as f64).to_radians();
        let r = rot_zyx(yaw, pitch, roll);
        let dw = world_dirs(&r, &d_all);
        let c = centre_for(&dw, &x_all, &x_sum)?;
        let diff: Vec<V3> = x_all.iter().map(|x| sub3(x, &c)).collect();
        let n_fwd = dw.iter().zip(&diff).filter(|(d, df)| dot3(d, df) > 0.0).count();
        if (n_fwd as f64) / (dw.len() as f64) < FORWARD_FRAC_MIN {
            continue;
        }
        let dist: Vec<f64> = diff.iter().map(norm3).collect();
        let score = mean_angle(&dw, &diff, &dist);
        if best.map_or(true, |b| score < b.0) {
            best = Some((score, yaw, c));
        }
    }
    let (_, yaw, c) = best.ok_or(ResectError::NoForwardCandidate)?;
    Ok(CamPose::euler(c, yaw, roll, pitch))
}

fn resect_init_6dof(pairs: &Pairs, a: f64, cam_model: &ErpCamera, roll_hint: f64, pitch_hint: f64) -> Result<CamPose, ResectError> {
    let (d_all, x_all) = stack(pairs, a, cam_model);
    let mut candidates: Vec<(M3, V3)> = Vec::new();
    for (ob, mk) in pairs {
        if n_valid(ob) < 4 {
            continue; // planar PnP seeds only from all-4-valid observations
        }
        let d = lamphi_to_bearing(&px_to_lamphi(&ob.uv, cam_model));
        let d4 = [d[0], d[1], d[2], d[3]];
        let Ok((t_cm, r_cm)) = planar_pnp(&d4, a) else { continue };
        let r_cam = mk.r.mul(&r_cm.transpose());
        let rc = mat3_vec(&r_cam, &t_cm);
        candidates.push((r_cam, sub3(&mk.p, &rc)));
        let n_cm = [r_cm.0[0][2], r_cm.0[1][2], r_cm.0[2][2]];
        let t_norm = norm3(&t_cm);
        if t_norm > 1e-9 {
            let u_hat = [t_cm[0] / t_norm, t_cm[1] / t_norm, t_cm[2] / t_norm];
            let axis = erploc_linalg::cross(&n_cm, &u_hat);
            let axis_norm = norm3(&axis);
            if axis_norm > 1e-6 {
                // Python: axis = axis / axis_norm, then exp_so3(np.pi * axis) (same rounding order)
                let ax = [PI * (axis[0] / axis_norm), PI * (axis[1] / axis_norm), PI * (axis[2] / axis_norm)];
                let r_cm_flip = exp_so3(&ax).mul(&r_cm);
                let r_cam_flip = mk.r.mul(&r_cm_flip.transpose());
                let rcf = mat3_vec(&r_cam_flip, &t_cm);
                candidates.push((r_cam_flip, sub3(&mk.p, &rcf)));
            }
        }
    }
    let mut x_sum = [0.0; 3];
    for x in &x_all {
        for i in 0..3 {
            x_sum[i] += x[i];
        }
    }
    for i in 0..N_YAW_HINT_STEPS {
        let yaw = 2.0 * PI * i as f64 / N_YAW_HINT_STEPS as f64;
        let r = rot_zyx(yaw, pitch_hint, roll_hint);
        let dw = world_dirs(&r, &d_all);
        let c = centre_for(&dw, &x_all, &x_sum)?;
        candidates.push((r, c));
    }
    let mut best: Option<(f64, M3, V3)> = None;
    for (r, c) in candidates {
        let Some(score) = score_pose_candidate(&r, &c, &d_all, &x_all) else { continue };
        if best.map_or(true, |b| score < b.0) {
            best = Some((score, r, c));
        }
    }
    let (_, r, c) = best.ok_or(ResectError::NoForwardCandidate)?;
    Ok(CamPose::from_r(c, r))
}

/// Closed-form initial pose (4-dof: yaw scan; 6-dof: planar PnP candidates + yaw-hint scan).
/// For `CamDof::Full`, `roll`/`pitch` are only hints.
pub fn resect_init(
    obs: &[Observation],
    markers: &BTreeMap<i64, MarkerPose>,
    a: f64,
    cam_model: &ErpCamera,
    roll: f64,
    pitch: f64,
    dof: CamDof,
) -> Result<CamPose, ResectError> {
    let pairs = valid_pairs(obs, markers)?;
    match dof {
        CamDof::Yaw => resect_init_4dof(&pairs, a, cam_model, roll, pitch),
        CamDof::Full => resect_init_6dof(&pairs, a, cam_model, roll, pitch),
    }
}

struct Ctx<'a> {
    pairs: &'a Pairs<'a>,
    lp_obs: Vec<[V2; 4]>,
    /// Kept rows and 1/scale weights per pair.
    sel: Vec<RowSel>,
    a: f64,
    cam_model: &'a ErpCamera,
    sigma: f64,
    nm: NoiseModel,
}

impl Ctx<'_> {
    fn n_rows(&self) -> usize {
        self.sel.iter().map(|s| s.n).sum()
    }

    /// Valid rows only, `(r * w) / sigma`.
    fn residual(&self, cam: &CamPose) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_rows());
        for (((_, mk), lp_obs), sel) in self.pairs.iter().zip(&self.lp_obs).zip(&self.sel) {
            let pred = predict_marker(cam, mk, self.a);
            let (r, _) = residual_lamphi(lp_obs, &pred, self.cam_model, self.nm);
            // 4 corners x (lam, phi) on the stack: no per-pair allocation (the only allocation left is the returned vector)
            let mut r8 = [0.0f64; 8];
            for (q, v) in r.iter().flatten().enumerate() {
                r8[q] = *v;
            }
            for j in 0..sel.n {
                out.push((r8[sel.idx[j]] * sel.w[j]) / self.sigma);
            }
        }
        out
    }

    fn jac(&self, cam: &CamPose, dof: CamDof) -> DMat {
        let nc = 3 + dof.axes().len();
        let mut j = DMat::zeros(self.n_rows(), nc);
        let mut row0 = 0;
        for (((_, mk), lp_obs), sel) in self.pairs.iter().zip(&self.lp_obs).zip(&self.sel) {
            let pred = predict_marker(cam, mk, self.a);
            let (_, d) = residual_lamphi(lp_obs, &pred, self.cam_model, self.nm);
            let d8: Vec<f64> = d.iter().flatten().copied().collect();
            let (jc, _) = jac_marker(cam, mk, self.a, dof.axes());
            for q in 0..sel.n {
                let src = sel.idx[q];
                for col in 0..nc {
                    j[(row0 + q, col)] = -((d8[src] * jc[(src, col)]) * sel.w[q]) / self.sigma;
                }
            }
            row0 += sel.n;
        }
        j
    }

    /// RMS of the residuals in pixels before whitening: `((r * sigma) / w)^2` averaged (Python `rms_px`).
    fn rms_px(&self, r: &[f64]) -> f64 {
        let w_all = self.sel.iter().flat_map(|s| s.w[..s.n].iter().copied());
        let s: f64 = r
            .iter()
            .zip(w_all)
            .map(|(v, w)| {
                let t = (v * self.sigma) / w;
                t * t
            })
            .sum();
        (s / r.len() as f64).sqrt()
    }
}

/// Information used for the covariance: the plain whitened `JᵀJ` at the solution (Gaussian CRLB), for any loss. With the
/// linear loss it is LM's own `info` (identical); under Huber LM's `info` is the IRLS-weighted `JᵀWJ`, so `J` is rebuilt.
fn plain_info(lm_info: DMat, loss: Loss, jac: impl FnOnce() -> DMat) -> DMat {
    match loss {
        Loss::Linear => lm_info,
        Loss::Huber { .. } => {
            let j = jac();
            j.transpose().matmul(&j)
        }
    }
}

/// Single-frame resection. `x0 = None` uses [`resect_init`].
pub fn resect(
    obs: &[Observation],
    markers: &BTreeMap<i64, MarkerPose>,
    a: f64,
    cam_model: &ErpCamera,
    roll: f64,
    pitch: f64,
    opts: &ResectOptions,
) -> Result<ResectionResult, ResectError> {
    let pairs = valid_pairs(obs, markers)?;
    let ctx = Ctx {
        pairs: &pairs,
        lp_obs: pairs
            .iter()
            .map(|(ob, _)| {
                let lp = px_to_lamphi(&safe_uv(ob), cam_model);
                [lp[0], lp[1], lp[2], lp[3]]
            })
            .collect(),
        sel: pairs.iter().map(|(ob, _)| row_select(ob)).collect(),
        a,
        cam_model,
        sigma: opts.sigma_px,
        nm: opts.noise_model,
    };
    let lm_opts = LmOptions { loss: opts.loss, max_iter: 100, solver: Solver::Lu, ..LmOptions::default() };
    let (cam, info, r, converged) = match opts.dof {
        CamDof::Yaw => {
            let x0 = match opts.x0 {
                Some(x0) => x0,
                None => resect_init(obs, markers, a, cam_model, roll, pitch, CamDof::Yaw)?,
            };
            let x0_vec = vec![x0.c[0], x0.c[1], x0.c[2], x0.yaw];
            let cam_of = |x: &Vec<f64>| CamPose::euler([x[0], x[1], x[2]], x[3], roll, pitch);
            let res = levenberg_marquardt(
                &mut |x: &Vec<f64>| ctx.residual(&cam_of(x)),
                &mut |x: &Vec<f64>| ctx.jac(&cam_of(x), CamDof::Yaw),
                x0_vec,
                &vec_retract,
                &lm_opts,
            )?;
            let cam = CamPose::euler([res.x[0], res.x[1], res.x[2]], wrap_pi(res.x[3]), roll, pitch);
            let info = plain_info(res.info, opts.loss, || ctx.jac(&cam_of(&res.x), CamDof::Yaw));
            (cam, info, res.r, res.converged)
        }
        CamDof::Full => {
            let x0 = match opts.x0 {
                Some(x0) if x0.r_mat.is_some() => x0,
                Some(x0) => CamPose::from_r(x0.c, x0.r()),
                None => resect_init(obs, markers, a, cam_model, roll, pitch, CamDof::Full)?,
            };
            let res = levenberg_marquardt(
                &mut |cam: &CamPose| ctx.residual(cam),
                &mut |cam: &CamPose| ctx.jac(cam, CamDof::Full),
                x0,
                &|cam: &CamPose, dx: &[f64]| cam.retract(&[dx[0], dx[1], dx[2]], &[dx[3], dx[4], dx[5]]),
                &lm_opts,
            )?;
            let info = plain_info(res.info, opts.loss, || ctx.jac(&res.x, CamDof::Full));
            (res.x, info, res.r, res.converged)
        }
    };
    let cov = crlb(&info)?;
    let rms = ctx.rms_px(&r);
    Ok(ResectionResult { cam, cov, info, rms_px: rms, n_markers: pairs.len(), converged })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{make_markers, render, MarkerLayout, SynthConfig};
    use erploc_core::rot::log_so3;
    use erploc_linalg::rng::Pcg64;

    const A: f64 = 0.16;
    const SIG: f64 = 0.3;

    fn cam_model() -> ErpCamera {
        ErpCamera::new(6528, 3264, 1)
    }

    fn cfg() -> SynthConfig {
        SynthConfig { cam: cam_model(), a: A, sigma_px: SIG, noise_model: NoiseModel::Tangent, seed: 0 }
    }

    fn base_markers() -> Vec<MarkerPose> {
        let mut rng = Pcg64::new(20260911);
        make_markers(MarkerLayout::Walls, 6, &[-4.0, 4.0, -3.0, 3.0], [1.0, 1.8], &mut rng)
    }

    fn by_id(mks: &[MarkerPose]) -> BTreeMap<i64, MarkerPose> {
        mks.iter().map(|m| (m.id, *m)).collect()
    }

    fn opts(dof: CamDof) -> ResectOptions {
        ResectOptions { x0: None, sigma_px: SIG, noise_model: NoiseModel::Tangent, loss: Loss::Linear, dof }
    }

    fn zenith() -> MarkerPose {
        // columns x = (1,0,0), y = (0,-1,0), n = (0,0,-1)
        let r = erploc_linalg::Mat::<3, 3>([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);
        MarkerPose { id: 100, p: [0.0, 0.0, 3.0], r }
    }

    fn err4(res: &ResectionResult, truth: &CamPose) -> (f64, f64) {
        let dp = norm3(&sub3(&res.cam.c, &truth.c));
        (dp, wrap_pi(res.cam.yaw - truth.yaw).abs())
    }

    fn err6(res: &ResectionResult, truth: &CamPose) -> (f64, f64) {
        let dp = norm3(&sub3(&res.cam.c, &truth.c));
        let dr = norm3(&log_so3(&res.cam.r().mul(&truth.r().transpose())));
        (dp, dr)
    }

    fn seam_cam(mks: &[MarkerPose], roll: f64, pitch: f64) -> CamPose {
        let c_b = [0.0, 0.0, 1.2];
        let beta = (mks[0].p[1] - c_b[1]).atan2(mks[0].p[0] - c_b[0]);
        CamPose::euler(c_b, wrap_pi(beta - PI), roll, pitch)
    }

    fn assert_small(v: f64, tol: f64, what: &str) {
        assert!(v.is_finite(), "{what} not finite: {v}");
        assert!(v < tol, "{what} = {v:e} >= {tol:e}");
        eprintln!("{what} = {v:.3e}");
    }

    /// Fix: a non-finite J^T J in the dense LM was an assert (panic); it is now an `Err` (former panic input: a start at 1e300).
    #[test]
    fn nonfinite_jtj_is_err_not_panic() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        for dof in [CamDof::Yaw, CamDof::Full] {
            let mut o = opts(dof);
            o.x0 = Some(CamPose::euler([0.4, 1e300, 1.2], 0.7, 0.0, 0.0));
            assert!(resect(&obs, &by_id(&mks), A, &cam_model(), 0.0, 0.0, &o).is_err(), "{dof:?}");
        }
        // an iterate on a marker corner (zero bearing)
        let corner = erploc_core::markers::corners_world(&mks[0].p, &mks[0].r, A)[1];
        let mut o = opts(CamDof::Yaw);
        o.x0 = Some(CamPose::euler(corner, 0.7, 0.0, 0.0));
        assert!(resect(&obs, &by_id(&mks), A, &cam_model(), 0.0, 0.0, &o).is_err());
    }

    #[test]
    fn a1_cold_start() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let res = resect(&obs, &by_id(&mks), A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap();
        let (dp, dy) = err4(&res, &truth);
        assert_small(dp, 1e-6, "a1 pos");
        assert_small(dy, 1e-8, "a1 yaw");
        assert_eq!(res.n_markers, 6);
        assert_eq!(res.cov.rows(), 4);
    }

    #[test]
    fn a1b_seam() {
        let mks = base_markers();
        let truth = seam_cam(&mks, 0.0, 0.0);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let ob0 = obs.iter().find(|o| o.id == mks[0].id).expect("marker 0 visible");
        let us: Vec<f64> = ob0.uv.iter().map(|p| p[0]).collect();
        let span = us.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - us.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(us.iter().all(|u| u.is_finite()));
        assert!(span > 0.5 * 6528.0, "u span {span}");
        let res = resect(&obs, &by_id(&mks), A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap();
        let (dp, dy) = err4(&res, &truth);
        assert_small(dp, 1e-6, "a1b pos");
        assert_small(dy, 1e-8, "a1b yaw");
    }

    #[test]
    fn a1c_zenith() {
        let mk = zenith();
        let truth = CamPose::euler([0.05, -0.02, 1.2], 0.5, 0.0, 0.0);
        let obs = render(&[truth], &[mk], &cfg(), None).remove(0);
        let lp = px_to_lamphi(&obs[0].uv, &cam_model());
        assert!(lp.iter().all(|p| p[1].is_finite()));
        let elev_min = lp.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min).to_degrees();
        assert!(elev_min > 80.0, "elev {elev_min}");
        let res = resect(&obs, &by_id(&[mk]), A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap();
        let (dp, dy) = err4(&res, &truth);
        assert_small(dp, 1e-6, "a1c pos");
        assert_small(dy, 1e-8, "a1c yaw");
    }

    #[test]
    fn a1d_single() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let one = obs.iter().map(|o| o.id).min().unwrap();
        let known: BTreeMap<i64, MarkerPose> = mks.iter().filter(|m| m.id == one).map(|m| (m.id, *m)).collect();
        let res = resect(&obs, &known, A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap();
        let (dp, dy) = err4(&res, &truth);
        assert_small(dp, 1e-6, "a1d pos");
        assert_small(dy, 1e-8, "a1d yaw");
        assert_eq!(res.n_markers, 1);
    }

    #[test]
    fn a1e_empty() {
        let known = by_id(&base_markers());
        let e = resect_init(&[], &known, A, &cam_model(), 0.0, 0.0, CamDof::Yaw).unwrap_err();
        assert_eq!(e, ResectError::NoKnownMarkers);
        let e = resect(&[], &known, A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap_err();
        assert_eq!(e, ResectError::NoKnownMarkers);
    }

    #[test]
    fn roll_pitch_known() {
        let mks = base_markers();
        let (roll, pitch) = (5.0f64.to_radians(), (-3.0f64).to_radians());
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, roll, pitch);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let res = resect(&obs, &by_id(&mks), A, &cam_model(), roll, pitch, &opts(CamDof::Yaw)).unwrap();
        let (dp, dy) = err4(&res, &truth);
        assert_small(dp, 1e-6, "rp pos");
        assert_small(dy, 1e-8, "rp yaw");
    }

    #[test]
    fn a61_cold_6dof() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 4.0f64.to_radians(), (-2.0f64).to_radians());
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let mut ids: Vec<i64> = obs.iter().map(|o| o.id).collect();
        ids.sort();
        ids.dedup();
        for n in [6usize, 3, 2, 1] {
            let sel = &ids[..n.min(ids.len())];
            let obs_n: Vec<Observation> = obs.iter().filter(|o| sel.contains(&o.id)).cloned().collect();
            let known: BTreeMap<i64, MarkerPose> = mks.iter().filter(|m| sel.contains(&m.id)).map(|m| (m.id, *m)).collect();
            let res = resect(&obs_n, &known, A, &cam_model(), 0.0, 0.0, &opts(CamDof::Full)).unwrap();
            let (dp, dr) = err6(&res, &truth);
            assert_small(dp, 1e-9, &format!("a61 n={n} pos"));
            assert_small(dr, 1e-9, &format!("a61 n={n} rot"));
            assert_eq!(res.cov.rows(), 6);
        }
    }

    #[test]
    fn a61b_seam_6dof() {
        let mks = base_markers();
        let truth = seam_cam(&mks, 3.0f64.to_radians(), 1.5f64.to_radians());
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let res = resect(&obs, &by_id(&mks), A, &cam_model(), 0.0, 0.0, &opts(CamDof::Full)).unwrap();
        let (dp, dr) = err6(&res, &truth);
        assert_small(dp, 1e-9, "a61b pos");
        assert_small(dr, 1e-9, "a61b rot");
    }

    #[test]
    fn a61c_zenith_6dof() {
        let mk = zenith();
        let truth = CamPose::euler([0.05, -0.02, 1.2], 0.5, (-2.0f64).to_radians(), 1.0f64.to_radians());
        let obs = render(&[truth], &[mk], &cfg(), None).remove(0);
        let res = resect(&obs, &by_id(&[mk]), A, &cam_model(), 0.0, 0.0, &opts(CamDof::Full)).unwrap();
        let (dp, dr) = err6(&res, &truth);
        assert_small(dp, 1e-9, "a61c pos");
        assert_small(dr, 1e-9, "a61c rot");
    }

    /// Masked corners are never read (NaN uv), an observation without a valid corner is absent, exact data stays exact.
    #[test]
    fn masked_corners_never_read() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
        let obs = render(&[truth], &mks, &cfg(), None).remove(0);
        let mut masked = obs.clone();
        masked[0].corner_mask = [true, false, true, true];
        masked[0].corner_sigma_scale = [1.0, 3.0, 2.0, 1.5];
        masked[1].corner_mask = [false, false, true, true];
        masked[2].corner_mask = [false; 4];
        for k in 0..4 {
            if !masked[0].corner_mask[k] {
                masked[0].uv[k] = [f64::NAN, f64::NAN];
            }
            if !masked[1].corner_mask[k] {
                masked[1].uv[k] = [f64::NAN, f64::INFINITY];
            }
            masked[2].uv[k] = [f64::NAN, f64::NAN];
        }
        let map = by_id(&mks);
        for dof in [CamDof::Yaw, CamDof::Full] {
            let res = resect(&masked, &map, A, &cam_model(), 0.0, 0.0, &opts(dof)).unwrap();
            let (dp, d2) = if dof == CamDof::Yaw { err4(&res, &truth) } else { err6(&res, &truth) };
            assert_small(dp, 1e-6, &format!("masked {dof:?} pos"));
            assert_small(d2, 1e-6, &format!("masked {dof:?} rot"));
            assert_eq!(res.n_markers, obs.len() - 1, "0-valid observation is dropped");
            assert!(res.cov.as_slice().iter().all(|v| v.is_finite()) && res.rms_px < 1e-6);
        }
        // only the 0-valid observation of a known marker left -> no known markers
        let e = resect(&masked[2..3], &map, A, &cam_model(), 0.0, 0.0, &opts(CamDof::Yaw)).unwrap_err();
        assert_eq!(e, ResectError::NoKnownMarkers);
    }

    /// The covariance is the plain whitened JtJ for every loss. On exact data Huber == linear (no residual is large).
    #[test]
    fn huber_covariance_is_plain_jtj() {
        let mks = base_markers();
        let truth = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
        let mut obs = render(&[truth], &mks, &cfg(), None).remove(0);
        obs[0].uv[0][0] += 40.0; // gross outlier: Huber down-weights it, the information must not
        let map = by_id(&mks);
        let mut o = opts(CamDof::Yaw);
        o.loss = Loss::Huber { delta: 1.5 };
        let hub = resect(&obs, &map, A, &cam_model(), 0.0, 0.0, &o).unwrap();
        // rebuild JtJ at the Huber solution through a linear resection started there with zero iterations of effect
        let lp: Vec<[V2; 4]> = obs
            .iter()
            .map(|ob| {
                let l = px_to_lamphi(&ob.uv, &cam_model());
                [l[0], l[1], l[2], l[3]]
            })
            .collect();
        let pairs: Pairs = obs.iter().map(|ob| (ob, &map[&ob.id])).collect();
        let ctx = Ctx {
            pairs: &pairs,
            lp_obs: lp,
            sel: obs.iter().map(row_select).collect(),
            a: A,
            cam_model: &cam_model(),
            sigma: SIG,
            nm: NoiseModel::Tangent,
        };
        let j = ctx.jac(&hub.cam, CamDof::Yaw);
        let plain = j.transpose().matmul(&j);
        let rel = hub.info.sub(&plain).max_abs() / plain.max_abs();
        assert!(rel < 1e-12, "info is not the plain JtJ: {rel:e}");
    }
}
