//! Bundle adjustment model: layout, residuals, normal equations, Jacobian, retract.
//! Port of `erploc/ba.py` (`Layout`, `_iter_obs`, `residuals`, `jacobian`, `retract`).
//!
//! Column order: cameras of non-`None` frames in frame order (4 cols for Yaw, 6 for Full), then markers in id
//! order (6 cols = [p, dw]). Row order: per valid observation the rows of its valid corners (2 per corner, corner order,
//! 8 for a fully valid observation) whitened by `1 / (sigma * scale)`, then 3 prior rows per surveyed id. An observation
//! with no valid corner is absent. The covariance is the plain whitened `JᵀJ` for every loss.

use crate::ba_init::initialize;
use crate::ba_types::{BaError, BaProblem, BaState};
use crate::corners::{n_valid, row_select, safe_uv, RowSel};
use crate::fisher::nullspace;
use crate::lm::{levenberg_marquardt_normal, LmOptions, Loss, Solver};
use crate::resection::CamDof;
use erploc_core::erp::{px_to_lamphi, residual_lamphi, wrap_pi};
use erploc_core::jacobian::{jac_marker, predict_marker, CamPose};
use erploc_core::markers::MarkerPose;
use erploc_core::rot::exp_so3;
use erploc_linalg::{eig::eigvalsh, lu, DMat};
use std::collections::BTreeMap;
use std::ops::Range;

/// Column assignment of the state vector.
#[derive(Clone, Debug)]
pub struct Layout {
    cam: Vec<Option<Range<usize>>>,
    mk: BTreeMap<i64, Range<usize>>,
    n_cam_cols: usize,
    n: usize,
}

impl Layout {
    pub fn new(state: &BaState, cam_dof: CamDof) -> Layout {
        let width = match cam_dof {
            CamDof::Yaw => 4,
            CamDof::Full => 6,
        };
        let present: Vec<bool> = state.cams.iter().map(|c| c.is_some()).collect();
        let ids: Vec<i64> = state.markers.keys().copied().collect();
        Layout::from_parts(&present, &ids, width)
    }

    /// Column assignment of `Layout::new` from the present-frame pattern and the sorted marker ids.
    pub(crate) fn from_parts(present: &[bool], ids: &[i64], width: usize) -> Layout {
        let (cam, mk_ranges, n_cam_cols, n) = layout_columns(present, ids.len(), width);
        let mk = ids.iter().copied().zip(mk_ranges).collect();
        Layout { cam, mk, n_cam_cols, n }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn n_cam_cols(&self) -> usize {
        self.n_cam_cols
    }

    /// `None` for a frame that is not in the state.
    pub fn cam_cols(&self, f: usize) -> Option<Range<usize>> {
        self.cam.get(f).cloned().flatten()
    }

    /// Panics on an unknown id (Python `KeyError`).
    pub fn mk_cols(&self, id: i64) -> Range<usize> {
        self.mk.get(&id).unwrap_or_else(|| panic!("Layout::mk_cols: unknown marker id {id}")).clone()
    }
}

pub(crate) type LayoutCols = (Vec<Option<Range<usize>>>, Vec<Range<usize>>, usize, usize);

/// Float-free column arithmetic: cameras of present frames in order (`width` columns each), then one 6-column
/// block per marker, in id order. Returns `(camera ranges, marker ranges, n_cam_cols, n)`.
pub(crate) fn layout_columns(present: &[bool], n_mk: usize, width: usize) -> LayoutCols {
    let mut col = 0;
    let mut cam = Vec::with_capacity(present.len());
    for &p in present {
        if p {
            cam.push(Some(col..col + width));
            col += width;
        } else {
            cam.push(None);
        }
    }
    let n_cam_cols = col;
    let mut mk = Vec::with_capacity(n_mk);
    for _ in 0..n_mk {
        mk.push(col..col + 6);
        col += 6;
    }
    (cam, mk, n_cam_cols, col)
}

/// Valid observations `(f, ob, cam, mk)` in frame order, then in-frame order (observations without a valid corner skipped).
fn iter_obs<'a>(
    state: &'a BaState,
    problem: &'a BaProblem,
    layout: &'a Layout,
) -> impl Iterator<Item = (usize, &'a crate::synth::Observation, &'a CamPose, &'a MarkerPose)> + 'a {
    state.cams.iter().enumerate().flat_map(move |(f, cam)| {
        let cam = if layout.cam_cols(f).is_some() { cam.as_ref() } else { None };
        cam.into_iter().flat_map(move |cam| {
            problem.frames[f].iter().filter(|ob| n_valid(ob) > 0).filter_map(move |ob| state.markers.get(&ob.id).map(|mk| (f, ob, cam, mk)))
        })
    })
}

/// Residual of one observation: the rows of its valid corners, `(r * w) / sigma_px` (compact, first `sel.n` entries of
/// the array), the row selection, and the residual-scale diagonal of all 8 rows.
fn obs_residual(cam: &CamPose, mk: &MarkerPose, ob: &crate::synth::Observation, problem: &BaProblem) -> (RowSel, [f64; 8], [f64; 8]) {
    let lp_obs = px_to_lamphi(&safe_uv(ob), &problem.cam_model);
    let lp_pred = predict_marker(cam, mk, problem.a);
    let (r, d) = residual_lamphi(&lp_obs, &lp_pred, &problem.cam_model, problem.noise_model);
    let sel = row_select(ob);
    let mut r_sel = [0.0; 8];
    let mut d8 = [0.0; 8];
    for k in 0..4 {
        for i in 0..2 {
            d8[2 * k + i] = d[k][i];
        }
    }
    for (j, out) in r_sel.iter_mut().enumerate().take(sel.n) {
        let src = sel.idx[j];
        *out = (r[src / 2][src % 2] * sel.w[j]) / problem.sigma_px;
    }
    (sel, r_sel, d8)
}

/// Residual vector: 8 per valid observation, then 3 prior rows per surveyed id (id order).
/// Panics if a surveyed id is not in `state.markers`.
pub fn residuals(state: &BaState, problem: &BaProblem, layout: &Layout) -> Vec<f64> {
    let mut out = Vec::new();
    for (_, ob, cam, mk) in iter_obs(state, problem, layout) {
        let (sel, r_sel, _) = obs_residual(cam, mk, ob, problem);
        out.extend_from_slice(&r_sel[..sel.n]);
    }
    for (mid, (p_ts, sigma_ts)) in &problem.surveyed {
        let p_m = state.markers[mid].p;
        for i in 0..3 {
            out.push((p_m[i] - p_ts[i]) / sigma_ts);
        }
    }
    out
}

/// Weighted Jacobian blocks of one observation, valid-corner rows only: `-((d8 ⊙rows J) * w) / sigma` for camera
/// (n×nc) and marker (n×6), the compact residual, and the row selection.
fn obs_jacobian(cam: &CamPose, mk: &MarkerPose, ob: &crate::synth::Observation, problem: &BaProblem) -> (DMat, DMat, [f64; 8], RowSel) {
    let (sel, r_sel, d8) = obs_residual(cam, mk, ob, problem);
    let (jc, jm) = jac_marker(cam, mk, problem.a, problem.cam_dof.axes());
    let scale = |j: &DMat| DMat::from_fn(sel.n, j.cols(), |i, k| -((d8[sel.idx[i]] * j[(sel.idx[i], k)]) * sel.w[i]) / problem.sigma_px);
    (scale(&jc), scale(&jm), r_sel, sel)
}

/// Which part of the observation block a cell belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Blk {
    /// `A[cc]`, local `(p, q)` in camera x camera
    Cc,
    /// `A[cm]`, local `(p, q)` in camera x marker
    Cm,
    /// `A[mc]` (transpose of `Cm`), local `(p, q)` as in `Cm`
    Mc,
    /// `A[mm]`, local `(p, q)` in marker x marker
    Mm,
    /// `g_c[p]` (`q` unused, 0)
    Gc,
    /// `g_m[p]` (`q` unused, 0)
    Gm,
}

/// Float-free index walk of one observation's scatter: calls `f(kind, row, col, p, q)` for every cell touched, with
/// the absolute matrix position and the local block indices. For `Gc`/`Gm` only `row` is meaningful (the entry of `g`).
pub(crate) fn walk_obs_cells(cam_cols: &Range<usize>, mk_cols: &Range<usize>, mut f: impl FnMut(Blk, usize, usize, usize, usize)) {
    let (nc, nm) = (cam_cols.len(), mk_cols.len());
    for p in 0..nc {
        f(Blk::Gc, cam_cols.start + p, 0, p, 0);
        for q in 0..nc {
            f(Blk::Cc, cam_cols.start + p, cam_cols.start + q, p, q);
        }
        for q in 0..nm {
            f(Blk::Cm, cam_cols.start + p, mk_cols.start + q, p, q);
            f(Blk::Mc, mk_cols.start + q, cam_cols.start + p, p, q);
        }
    }
    for p in 0..nm {
        f(Blk::Gm, mk_cols.start + p, 0, p, 0);
        for q in 0..nm {
            f(Blk::Mm, mk_cols.start + p, mk_cols.start + q, p, q);
        }
    }
}

/// Add one observation's block contribution to `(A, g)`: `A[cc] += Jcᵀ diag(w²) Jc`, `A[cm] += …`,
/// `A[mc] = A[cm]ᵀ`, `A[mm] += …`, `g_c += Jcᵀ (w² ⊙ r)`, `g_m += …`. `r` and `w` are the residual and the sqrt weights of
/// the observation's rows (`jc.rows()` of them: 8 when all corners are valid).
/// The index walk is [`walk_obs_cells`]; each cell accumulates over the rows in order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_obs_block(
    a: &mut DMat,
    g: &mut [f64],
    cam_cols: Range<usize>,
    mk_cols: Range<usize>,
    jc: &DMat,
    jm: &DMat,
    r8: &[f64],
    w8: &[f64],
) {
    for i in 0..jc.rows() {
        let w2 = w8[i] * w8[i];
        walk_obs_cells(&cam_cols, &mk_cols, |blk, row, col, p, q| match blk {
            Blk::Gc => g[row] += (w2 * jc[(i, p)]) * r8[i],
            Blk::Gm => g[row] += (w2 * jm[(i, p)]) * r8[i],
            Blk::Cc => a[(row, col)] += (w2 * jc[(i, p)]) * jc[(i, q)],
            Blk::Cm | Blk::Mc => a[(row, col)] += (w2 * jc[(i, p)]) * jm[(i, q)],
            Blk::Mm => a[(row, col)] += (w2 * jm[(i, p)]) * jm[(i, q)],
        });
    }
}

/// `(A, g)` of the weighted normal equations `A = JwᵀJw`, `g = Jwᵀrw`, assembled block-wise; `r` must be
/// `residuals(..)` and `w` the sqrt weight per row (same order), or `None` for all ones.
pub fn normal_equations(state: &BaState, problem: &BaProblem, layout: &Layout, r: &[f64], w: Option<&[f64]>) -> (DMat, Vec<f64>) {
    normal_impl(state, problem, layout, r, w, true)
}

/// Same as [`normal_equations`] without the prior rows (Python `J_obs`); `r` is still the full residual vector.
fn normal_equations_obs(state: &BaState, problem: &BaProblem, layout: &Layout, r: &[f64], w: Option<&[f64]>) -> (DMat, Vec<f64>) {
    normal_impl(state, problem, layout, r, w, false)
}

fn normal_impl(state: &BaState, problem: &BaProblem, layout: &Layout, r: &[f64], w: Option<&[f64]>, with_prior: bool) -> (DMat, Vec<f64>) {
    let n = layout.n();
    let mut a = DMat::zeros(n, n);
    let mut g = vec![0.0; n];
    let mut row = 0;
    for (f, ob, cam, mk) in iter_obs(state, problem, layout) {
        let (jc, jm, _, sel) = obs_jacobian(cam, mk, ob, problem);
        let n = sel.n;
        let ones = [1.0; 8];
        let w_obs: &[f64] = match w {
            Some(w) => &w[row..row + n],
            None => &ones[..n],
        };
        let cc = layout.cam_cols(f).expect("iter_obs yields only frames with camera columns");
        scatter_obs_block(&mut a, &mut g, cc, layout.mk_cols(ob.id), &jc, &jm, &r[row..row + n], w_obs);
        row += n;
    }
    for (mid, (_, sigma_ts)) in problem.surveyed.iter().filter(|_| with_prior) {
        let cols = layout.mk_cols(*mid);
        for i in 0..3 {
            let wi = w.map_or(1.0, |w| w[row + i]);
            let c = cols.start + i;
            a[(c, c)] += (wi / sigma_ts) * (wi / sigma_ts);
            g[c] += wi * wi * r[row + i] / sigma_ts;
        }
        row += 3;
    }
    if !with_prior {
        row += 3 * problem.surveyed.len();
    }
    assert_eq!(row, r.len(), "normal_equations: r has the wrong length");
    (a, g)
}

/// Dense Jacobian (M × n), same row order as [`residuals`].
pub fn dense_jacobian(state: &BaState, problem: &BaProblem, layout: &Layout) -> DMat {
    let n_rows: usize = iter_obs(state, problem, layout).map(|(_, ob, _, _)| row_select(ob).n).sum();
    let m = n_rows + 3 * problem.surveyed.len();
    let mut j = DMat::zeros(m, layout.n());
    let mut row = 0;
    for (f, ob, cam, mk) in iter_obs(state, problem, layout) {
        let (jc, jm, _, sel) = obs_jacobian(cam, mk, ob, problem);
        let cc = layout.cam_cols(f).expect("iter_obs yields only frames with camera columns");
        let mc = layout.mk_cols(ob.id);
        for i in 0..sel.n {
            for (k, col) in cc.clone().enumerate() {
                j[(row + i, col)] = jc[(i, k)];
            }
            for (k, col) in mc.clone().enumerate() {
                j[(row + i, col)] = jm[(i, k)];
            }
        }
        row += sel.n;
    }
    for (mid, (_, sigma_ts)) in &problem.surveyed {
        let cols = layout.mk_cols(*mid);
        for i in 0..3 {
            j[(row + i, cols.start + i)] = 1.0 / sigma_ts;
        }
        row += 3;
    }
    j
}

/// State update. Yaw: `c += d[0..3]`, `yaw = wrap_pi(yaw + d[3])`, roll/pitch kept. Full: `cam.retract`
/// (world left perturbation). Markers: `p += d[0..3]`, `R <- R exp(d[3..6])`. `None` frames stay `None`.
/// The input state is not modified. Panics if `dx` is shorter than the layout.
pub fn retract(state: &BaState, dx: &[f64], cam_dof: CamDof) -> BaState {
    let layout = Layout::new(state, cam_dof);
    assert!(dx.len() >= layout.n(), "retract: dx too short");
    let cams = state
        .cams
        .iter()
        .enumerate()
        .map(|(f, cam)| {
            let cols = layout.cam_cols(f)?;
            let cam = cam.as_ref()?;
            let d = &dx[cols];
            Some(match cam_dof {
                CamDof::Full => cam.retract(&[d[0], d[1], d[2]], &[d[3], d[4], d[5]]),
                CamDof::Yaw => {
                    CamPose::euler([cam.c[0] + d[0], cam.c[1] + d[1], cam.c[2] + d[2]], wrap_pi(cam.yaw + d[3]), cam.roll, cam.pitch)
                }
            })
        })
        .collect();
    let markers = state
        .markers
        .iter()
        .map(|(&id, mk)| {
            let d = &dx[layout.mk_cols(id)];
            let p = [mk.p[0] + d[0], mk.p[1] + d[1], mk.p[2] + d[2]];
            let r = mk.r.mul(&exp_so3(&[d[3], d[4], d[5]]));
            (id, MarkerPose { id, p, r })
        })
        .collect();
    BaState { cams, markers }
}

#[derive(Clone, Debug)]
pub struct Gauge {
    /// Null-space dimension of the information matrix without the prior rows.
    pub null_dim_noprior: usize,
    /// `w_min / w_max` of the eigenvalues of the information matrix with priors.
    pub eig_min_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct BaResult {
    pub state: BaState,
    /// 6x6 per marker id.
    pub cov_markers: BTreeMap<i64, DMat>,
    /// 4x4 (Yaw) or 6x6 (Full) per frame; `None` where Python fills NaN (frame without camera).
    pub cov_cams: Vec<Option<DMat>>,
    /// Unweighted `JᵀJ` with priors.
    pub info: DMat,
    /// RMS of the observation residuals in pixels (before weighting).
    pub rms_px: f64,
    pub gauge: Gauge,
    pub converged: bool,
}

fn block(m: &DMat, r: &Range<usize>) -> DMat {
    DMat::from_fn(r.len(), r.len(), |i, j| m[(r.start + i, r.start + j)])
}

/// Up-front shape and range validation of a user-built problem (and optional start), so that no later index or assert panics.
/// Called first by `bundle_adjust` and `initialize`; a valid problem passes untouched.
pub(crate) fn check_shapes(problem: &BaProblem, x0: Option<&BaState>) -> Result<(), BaError> {
    check_shapes_impl(problem, x0, true)
}

/// `strict_sigma_ts`: `bundle_adjust` weights the survey prior by 1 / sigma_TS, so it needs sigma_TS > 0; `initialize` never reads
/// sigma_TS (the Python self-checks start from sigma_TS = 0), so it only needs it non-negative and finite.
pub(crate) fn check_shapes_impl(problem: &BaProblem, x0: Option<&BaState>, strict_sigma_ts: bool) -> Result<(), BaError> {
    let shape = |s: String| Err(BaError::Shape(s));
    if problem.rp.len() != problem.frames.len() {
        return shape(format!("rp has {} entries but frames has {}", problem.rp.len(), problem.frames.len()));
    }
    if let Some(f) = problem.rp.iter().position(|r| !(r[0].is_finite() && r[1].is_finite())) {
        return shape(format!("rp[{f}] (roll, pitch) must be finite, got {:?}", problem.rp[f]));
    }
    if let Some(x0) = x0 {
        if x0.cams.len() != problem.frames.len() {
            return shape(format!("x0.cams has {} entries but frames has {}", x0.cams.len(), problem.frames.len()));
        }
    }
    if !(problem.a.is_finite() && problem.a > 0.0) {
        return shape(format!("a must be finite and > 0, got {}", problem.a));
    }
    if !(problem.sigma_px.is_finite() && problem.sigma_px > 0.0) {
        return shape(format!("sigma_px must be finite and > 0, got {}", problem.sigma_px));
    }
    for (id, (p, sigma)) in &problem.surveyed {
        if !p.iter().all(|v| v.is_finite()) {
            return shape(format!("surveyed[{id}] position is not finite: {p:?}"));
        }
        if !(sigma.is_finite() && if strict_sigma_ts { *sigma > 0.0 } else { *sigma >= 0.0 }) {
            let rel = if strict_sigma_ts { "> 0" } else { ">= 0" };
            return shape(format!("surveyed[{id}] sigma_TS must be finite and {rel}, got {sigma}"));
        }
    }
    for (f, obs) in problem.frames.iter().enumerate() {
        let mut seen = std::collections::BTreeSet::new();
        for ob in obs {
            if ob.id == i64::MIN {
                return shape(format!("frames[{f}] observation id i64::MIN is not supported"));
            }
            if !seen.insert(ob.id) {
                return shape(format!("frames[{f}] contains marker id {} more than once", ob.id));
            }
            if !ob.corner_sigma_scale.iter().all(|s| s.is_finite() && *s > 0.0) {
                return shape(format!(
                    "frames[{f}] marker {} corner_sigma_scale must be finite and > 0: {:?}",
                    ob.id, ob.corner_sigma_scale
                ));
            }
        }
    }
    Ok(())
}

/// Bundle adjustment. Python `bundle_adjust`: `x0 = None` starts from `initialize(problem, 1)`. Validation order:
/// (a) Yaw: x0 roll/pitch equals `rp` (1e-12); Full: x0 rotations finite, (b) at least 2 surveyed markers,
/// (c) every surveyed id is in `x0.markers`. Solves with LM (Schur solver); `info`/covariances use the
/// UNWEIGHTED Jacobian at the solution, even with Huber.
pub fn bundle_adjust(problem: &BaProblem, x0: Option<BaState>, loss: Loss, max_iter: usize) -> Result<BaResult, BaError> {
    check_shapes(problem, x0.as_ref())?;
    let x0 = match x0 {
        Some(x) => x,
        None => initialize(problem, 1)?,
    };
    match problem.cam_dof {
        CamDof::Full => {
            for (f, cam) in x0.cams.iter().enumerate() {
                if let Some(cam) = cam {
                    if !cam.r().to_dmat().as_slice().iter().all(|v| v.is_finite()) {
                        return Err(BaError::X0InvalidRotation { frame: f });
                    }
                }
            }
        }
        CamDof::Yaw => {
            for (f, cam) in x0.cams.iter().enumerate() {
                if let Some(cam) = cam {
                    // negated so that NaN fails the check, like Python's `not (abs(..) < 1e-12 and ..)`
                    if !((cam.roll - problem.rp[f][0]).abs() < 1e-12 && (cam.pitch - problem.rp[f][1]).abs() < 1e-12) {
                        return Err(BaError::X0RollPitchMismatch { frame: f });
                    }
                }
            }
        }
    }
    if problem.surveyed.len() < 2 {
        return Err(BaError::TooFewSurveyed(problem.surveyed.len()));
    }
    let missing: Vec<i64> = problem.surveyed.keys().filter(|id| !x0.markers.contains_key(id)).copied().collect();
    if !missing.is_empty() {
        return Err(BaError::SurveyedMissingInX0(missing));
    }

    let layout = Layout::new(&x0, problem.cam_dof);
    let cam_width = match problem.cam_dof {
        CamDof::Yaw => 4,
        CamDof::Full => 6,
    };
    let opts = LmOptions { loss, max_iter, solver: Solver::Schur { n_cam_cols: layout.n_cam_cols(), cam_width }, ..LmOptions::default() };
    let lm = levenberg_marquardt_normal(
        &mut |s: &BaState| residuals(s, problem, &layout),
        &mut |s: &BaState, r: &[f64], w: Option<&[f64]>| normal_equations(s, problem, &layout, r, w),
        x0.clone(),
        &|s: &BaState, dx: &[f64]| retract(s, dx, problem.cam_dof),
        &opts,
    )?;
    if !(lm.cost.is_finite() && lm.r.iter().all(|v| v.is_finite())) {
        return Err(BaError::NonFinite);
    }
    let state = lm.x;

    let (info, _) = normal_equations(&state, problem, &layout, &lm.r, None);
    let cov = lu::inv(&info)?;
    let cov_markers = state.markers.keys().map(|&id| (id, block(&cov, &layout.mk_cols(id)))).collect();
    let cov_cams = (0..state.cams.len()).map(|f| layout.cam_cols(f).map(|c| block(&cov, &c))).collect();

    let n_obs_rows = lm.r.len() - 3 * problem.surveyed.len();
    // RMS in pixels before whitening: ((r * sigma) / w)^2 with w = 1 / scale per row
    let rms_px = if n_obs_rows > 0 {
        let w_obs = iter_obs(&state, problem, &layout).flat_map(|(_, ob, _, _)| {
            let sel = row_select(ob);
            (0..sel.n).map(move |j| sel.w[j])
        });
        let ss: f64 = lm.r[..n_obs_rows]
            .iter()
            .zip(w_obs)
            .map(|(v, w)| {
                let t = (v * problem.sigma_px) / w;
                t * t
            })
            .sum();
        (ss / n_obs_rows as f64).sqrt()
    } else {
        0.0
    };

    let (i_obs, _) = normal_equations_obs(&state, problem, &layout, &lm.r, None);
    let (null_dim_noprior, _) = nullspace(&i_obs, 1e-9)?;
    let sym = info.add(&info.transpose()).scale(0.5);
    let w = eigvalsh(&sym)?;
    let eig_min_ratio = w[0] / w[w.len() - 1];

    Ok(BaResult { state, cov_markers, cov_cams, info, rms_px, gauge: Gauge { null_dim_noprior, eig_min_ratio }, converged: lm.converged })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ba_types::BaProblem;
    use crate::lm::huber_sqrt_w;
    use crate::resection::ResectError;
    use crate::synth::{render, SynthConfig};
    use erploc_core::erp::{ErpCamera, NoiseModel};
    use erploc_core::markers::facing_pose;
    use erploc_linalg::LinalgError;

    const A: f64 = 0.16;

    fn cfg() -> SynthConfig {
        SynthConfig { cam: ErpCamera::new(3840, 1920, 1), a: A, sigma_px: 1.0, noise_model: NoiseModel::Tangent, seed: 0 }
    }

    /// 4 frames, 4 markers, markers facing the room centre; returns (truth state, problem).
    fn scene(dof: CamDof) -> (BaState, BaProblem) {
        let centre = [0.0, 0.0, 1.2];
        let mpos = [[3.0, 0.5, 1.0], [-0.5, 3.0, 1.4], [-3.0, -0.5, 0.9], [0.5, -3.0, 1.5]];
        let markers: Vec<MarkerPose> = mpos
            .iter()
            .enumerate()
            .map(|(i, p)| facing_pose(i as i64 + 10, p, &centre, 0.2 * i as f64, 0.7 * i as f64, 0.1 * i as f64))
            .collect();
        let cams = vec![
            CamPose::euler([0.2, 0.1, 1.2], 0.3, 0.02, -0.03),
            CamPose::euler([-0.3, 0.4, 1.3], -1.0, -0.01, 0.02),
            CamPose::euler([0.5, -0.4, 1.1], 1.2, 0.03, 0.01),
            CamPose::euler([-0.2, -0.3, 1.25], 0.1, 0.0, -0.02),
        ];
        let c = cfg();
        let frames = render(&cams, &markers, &c, None);
        let surveyed = markers[..3].iter().map(|m| (m.id, ([m.p[0] + 0.01, m.p[1] - 0.01, m.p[2] + 0.02], 0.02))).collect();
        let problem = BaProblem {
            frames,
            rp: cams.iter().map(|c| [c.roll, c.pitch]).collect(),
            surveyed,
            a: A,
            cam_model: c.cam,
            sigma_px: 1.0,
            noise_model: NoiseModel::Tangent,
            cam_dof: dof,
        };
        let state = BaState { cams: cams.into_iter().map(Some).collect(), markers: markers.into_iter().map(|m| (m.id, m)).collect() };
        (state, problem)
    }

    fn perturbed(state: &BaState, dof: CamDof, amp: f64) -> BaState {
        let layout = Layout::new(state, dof);
        let dx: Vec<f64> = (0..layout.n()).map(|i| amp * ((i as f64 * 1.7).sin())).collect();
        retract(state, &dx, dof)
    }

    #[test]
    fn skip_rules() {
        let (truth, problem) = scene(CamDof::Yaw);
        let n_obs: Vec<usize> = problem.frames.iter().map(|f| f.len()).collect();
        let n_prior = 3 * problem.surveyed.len();
        let check = |state: &BaState, problem: &BaProblem, expect_obs: usize, expect_cols: usize| {
            let layout = Layout::new(state, CamDof::Yaw);
            let r = residuals(state, problem, &layout);
            assert_eq!(r.len(), 8 * expect_obs + n_prior);
            assert_eq!(layout.n(), expect_cols);
            let (a, g) = normal_equations(state, problem, &layout, &r, None);
            assert_eq!((a.rows(), a.cols(), g.len()), (expect_cols, expect_cols, expect_cols));
            assert!(a.as_slice().iter().all(|v| v.is_finite()));
        };
        let total: usize = n_obs.iter().sum();
        assert!(n_obs[1] > 0 && total > n_obs[1]);
        check(&truth, &problem, total, 4 * 4 + 24);

        // observation of an id that is not in state.markers is skipped
        let mut p2 = problem.clone();
        let mut ob = p2.frames[0][0].clone();
        ob.id = 99;
        p2.frames[0].push(ob);
        check(&truth, &p2, total, 4 * 4 + 24);

        // a frame without camera contributes no rows and no columns, although it has observations
        let mut s3 = truth.clone();
        s3.cams[1] = None;
        check(&s3, &problem, total - n_obs[1], 3 * 4 + 24);
    }

    #[test]
    fn layout_order() {
        for (dof, w) in [(CamDof::Yaw, 4), (CamDof::Full, 6)] {
            let (mut s, _) = scene(dof);
            s.cams[1] = None;
            let l = Layout::new(&s, dof);
            assert_eq!(l.n_cam_cols(), 3 * w);
            assert_eq!(l.n(), 3 * w + 6 * 4);
            assert_eq!(l.cam_cols(0), Some(0..w));
            assert_eq!(l.cam_cols(1), None);
            assert_eq!(l.cam_cols(2), Some(w..2 * w));
            assert_eq!(l.cam_cols(3), Some(2 * w..3 * w));
            assert_eq!(l.cam_cols(9), None);
            for (k, id) in [10, 11, 12, 13].iter().enumerate() {
                assert_eq!(l.mk_cols(*id), 3 * w + 6 * k..3 * w + 6 * (k + 1));
            }
        }
        let (s, _) = scene(CamDof::Yaw);
        assert_eq!(Layout::new(&s, CamDof::Yaw).n(), 4 * 4 + 24);
        assert_eq!(Layout::new(&s, CamDof::Full).n(), 6 * 4 + 24);
    }

    #[test]
    fn normal_matches_dense() {
        for dof in [CamDof::Yaw, CamDof::Full] {
            let (truth, problem) = scene(dof);
            let st = perturbed(&truth, dof, 2e-2);
            let layout = Layout::new(&st, dof);
            let r = residuals(&st, &problem, &layout);
            let j = dense_jacobian(&st, &problem, &layout);
            assert_eq!(j.rows(), r.len());
            assert!(r.iter().all(|v| v.is_finite()) && j.as_slice().iter().all(|v| v.is_finite()));
            let huber: Vec<f64> = r.iter().map(|&x| huber_sqrt_w(x, 1.345)).collect();
            assert!(huber.iter().any(|&w| w < 1.0), "test needs active Huber weights");
            for w in [None, Some(huber)] {
                let (a, g) = normal_equations(&st, &problem, &layout, &r, w.as_deref());
                let wv = |i: usize| w.as_ref().map_or(1.0, |w| w[i]);
                let jw = DMat::from_fn(j.rows(), j.cols(), |i, k| wv(i) * j[(i, k)]);
                let rw: Vec<f64> = r.iter().enumerate().map(|(i, x)| wv(i) * x).collect();
                let a_ref = jw.transpose().matmul(&jw);
                let g_ref = jw.transpose().matvec(&rw);
                assert!(a.as_slice().iter().all(|v| v.is_finite()) && g.iter().all(|v| v.is_finite()));
                let da = a.sub(&a_ref).max_abs() / a_ref.max_abs();
                let dg = g.iter().zip(&g_ref).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
                    / g_ref.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
                println!("normal_matches_dense {dof:?} weighted={}: dA={da:e} dg={dg:e}", w.is_some());
                assert!(da < 1e-12, "dA {da}");
                assert!(dg < 1e-12, "dg {dg}");
            }
        }
    }

    #[test]
    fn jacobian_matches_numeric() {
        for dof in [CamDof::Yaw, CamDof::Full] {
            let (truth, problem) = scene(dof);
            let st = perturbed(&truth, dof, 1e-3);
            let layout = Layout::new(&st, dof);
            let j = dense_jacobian(&st, &problem, &layout);
            let h = 1e-6;
            let mut max_err = 0.0f64;
            for k in 0..layout.n() {
                let mut dx = vec![0.0; layout.n()];
                dx[k] = h;
                let rp = residuals(&retract(&st, &dx, dof), &problem, &layout);
                dx[k] = -h;
                let rm = residuals(&retract(&st, &dx, dof), &problem, &layout);
                for i in 0..rp.len() {
                    let num = (rp[i] - rm[i]) / (2.0 * h);
                    assert!(num.is_finite() && j[(i, k)].is_finite());
                    max_err = max_err.max((num - j[(i, k)]).abs());
                }
            }
            let rel = max_err / j.max_abs();
            println!("jacobian_matches_numeric {dof:?}: max err {max_err:e}, rel {rel:e}");
            assert!(rel < 1e-6, "rel {rel}");
        }
    }

    #[test]
    fn retract_semantics() {
        let (s, _) = scene(CamDof::Yaw);
        // Yaw: roll/pitch kept, yaw wraps, c shifts, marker right-perturbed.
        let layout = Layout::new(&s, CamDof::Yaw);
        let mut dx = vec![0.0; layout.n()];
        dx[0..4].copy_from_slice(&[0.1, -0.2, 0.3, 3.0]); // frame 0 yaw 0.3 + 3.0 wraps
        let mc = layout.mk_cols(11);
        dx[mc.start..mc.end].copy_from_slice(&[0.01, 0.02, 0.03, 0.1, -0.2, 0.3]);
        let t = retract(&s, &dx, CamDof::Yaw);
        let c0 = t.cams[0].unwrap();
        let o0 = s.cams[0].unwrap();
        assert_eq!(c0.c, [o0.c[0] + 0.1, o0.c[1] - 0.2, o0.c[2] + 0.3]);
        assert_eq!(c0.roll, o0.roll);
        assert_eq!(c0.pitch, o0.pitch);
        assert!(c0.r_mat.is_none());
        assert_eq!(c0.yaw, wrap_pi(0.3 + 3.0));
        assert!(c0.yaw < 0.0, "yaw wrapped into (-pi, pi]");
        assert_eq!(t.cams[1].unwrap(), s.cams[1].unwrap());
        let m0 = s.markers[&11];
        let m1 = t.markers[&11];
        assert_eq!(m1.p, [m0.p[0] + 0.01, m0.p[1] + 0.02, m0.p[2] + 0.03]);
        assert_eq!(m1.r, m0.r.mul(&exp_so3(&[0.1, -0.2, 0.3])));
        assert_eq!(t.markers[&10], s.markers[&10]);

        // Full: matches CamPose::retract; None frames stay None.
        let (mut s, _) = scene(CamDof::Full);
        s.cams[2] = None;
        let layout = Layout::new(&s, CamDof::Full);
        let dx: Vec<f64> = (0..layout.n()).map(|i| 0.01 * (i as f64 + 1.0).cos()).collect();
        let t = retract(&s, &dx, CamDof::Full);
        assert!(t.cams[2].is_none());
        let cols = layout.cam_cols(3).unwrap();
        let d = &dx[cols];
        assert_eq!(t.cams[3].unwrap(), s.cams[3].unwrap().retract(&[d[0], d[1], d[2]], &[d[3], d[4], d[5]]));
    }

    /// Masked rows are compacted, 0-valid observations are absent, masked uv is never read (NaN), scales whiten.
    #[test]
    fn masked_rows() {
        for dof in [CamDof::Yaw, CamDof::Full] {
            let (truth, mut problem) = scene(dof);
            let total: usize = problem.frames.iter().map(|f| f.len()).sum();
            let n_prior = 3 * problem.surveyed.len();
            // frame 0: obs 0 has 3 valid corners with scales, obs 1 has 2 valid corners, obs 2 none (if present)
            let f0 = problem.frames[0].len();
            assert!(f0 >= 3);
            problem.frames[0][0].corner_mask = [true, false, true, true];
            problem.frames[0][0].corner_sigma_scale = [1.0, 2.0, 1.5, 1.0];
            problem.frames[0][1].corner_mask = [false, true, false, true];
            problem.frames[0][2].corner_mask = [false; 4];
            for ob in problem.frames[0][..3].iter_mut() {
                for k in 0..4 {
                    if !ob.corner_mask[k] {
                        ob.uv[k] = [f64::NAN, f64::NAN];
                    }
                }
            }
            let layout = Layout::new(&truth, dof);
            let r = residuals(&truth, &problem, &layout);
            assert_eq!(r.len(), 8 * (total - 3) + 6 + 4 + n_prior);
            assert!(r.iter().all(|v| v.is_finite()));
            let j = dense_jacobian(&truth, &problem, &layout);
            assert_eq!(j.rows(), r.len());
            let (a, g) = normal_equations(&truth, &problem, &layout, &r, None);
            let a_ref = j.transpose().matmul(&j);
            assert!(a.sub(&a_ref).max_abs() / a_ref.max_abs() < 1e-12);
            assert!(a.as_slice().iter().all(|v| v.is_finite()) && g.iter().all(|v| v.is_finite()));
            // exact data stays exact: truth has a zero observation residual, so rms is 0
            let rms: f64 = r[..r.len() - n_prior].iter().map(|v| v * v).sum::<f64>();
            assert!(rms < 1e-18, "{rms}");
            // numeric Jacobian on a perturbed state
            let st = perturbed(&truth, dof, 1e-3);
            let jn = dense_jacobian(&st, &problem, &layout);
            let h = 1e-6;
            let mut max_err = 0.0f64;
            for k in 0..layout.n() {
                let mut dx = vec![0.0; layout.n()];
                dx[k] = h;
                let rp = residuals(&retract(&st, &dx, dof), &problem, &layout);
                dx[k] = -h;
                let rm = residuals(&retract(&st, &dx, dof), &problem, &layout);
                for i in 0..rp.len() {
                    max_err = max_err.max(((rp[i] - rm[i]) / (2.0 * h) - jn[(i, k)]).abs());
                }
            }
            assert!(max_err / jn.max_abs() < 1e-6, "{max_err:e}");
        }
    }

    fn is_shape(r: Result<BaResult, BaError>, field: &str) {
        match r {
            Err(BaError::Shape(msg)) => assert!(msg.contains(field), "message {msg:?} must name {field}"),
            other => panic!("expected Shape({field}), got {:?}", other.map(|_| ())),
        }
    }

    /// M-8: a user-built problem with a short `rp` is an error, not an index panic (Yaw indexes `rp[f]`).
    #[test]
    fn ba_shape_rp_short() {
        for dof in [CamDof::Yaw, CamDof::Full] {
            let (truth, mut problem) = scene(dof);
            problem.rp.pop();
            is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "rp");
        }
    }

    #[test]
    fn ba_shape_x0_longer() {
        let (mut truth, problem) = scene(CamDof::Yaw);
        let extra = truth.cams[0];
        truth.cams.push(extra);
        is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "x0.cams");
    }

    #[test]
    fn ba_shape_x0_longer_full() {
        let (mut truth, problem) = scene(CamDof::Full);
        let extra = truth.cams[0];
        truth.cams.push(extra);
        is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "x0.cams");
    }

    /// NaN / inf roll or pitch in `rp` is a Shape error on both paths (it would reach the dense LM otherwise).
    #[test]
    fn ba_shape_rp_nonfinite() {
        for bad in [f64::NAN, f64::INFINITY] {
            for dof in [CamDof::Yaw, CamDof::Full] {
                let (truth, mut problem) = scene(dof);
                problem.rp[1][1] = bad;
                is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "rp[1]");
                assert!(matches!(crate::ba_init::initialize(&problem, 1), Err(BaError::Shape(_))));
            }
        }
    }

    /// `initialize` never reads sigma_TS, so sigma_TS = 0 is accepted there (but rejected by `bundle_adjust`).
    #[test]
    fn initialize_accepts_sigma_ts_zero() {
        let (_, mut problem) = scene(CamDof::Yaw);
        for v in problem.surveyed.values_mut() {
            v.1 = 0.0;
        }
        assert!(!matches!(crate::ba_init::initialize(&problem, 1), Err(BaError::Shape(_))));
        let (truth, _) = scene(CamDof::Yaw);
        is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "sigma_TS");
    }

    #[test]
    fn ba_shape_id_i64_min() {
        let (truth, mut problem) = scene(CamDof::Yaw);
        problem.frames[0][0].id = i64::MIN;
        is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "i64::MIN");
        assert!(matches!(crate::ba_init::initialize(&problem, 1), Err(BaError::Shape(_))));
    }

    #[test]
    fn ba_shape_obs_ids_dup() {
        let (truth, mut problem) = scene(CamDof::Yaw);
        let dup = problem.frames[1][0].clone();
        problem.frames[1].push(dup);
        is_shape(bundle_adjust(&problem, Some(truth.clone()), Loss::Linear, 5), "more than once");
        assert!(matches!(crate::ba_init::initialize(&problem, 1), Err(BaError::Shape(_))));
    }

    #[test]
    fn ba_shape_surveyed_sigma() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (truth, mut problem) = scene(CamDof::Yaw);
            problem.surveyed.values_mut().next().unwrap().1 = bad;
            is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "sigma_TS");
        }
        let (truth, mut problem) = scene(CamDof::Yaw);
        problem.surveyed.values_mut().next().unwrap().0[1] = f64::NAN;
        is_shape(bundle_adjust(&problem, Some(truth), Loss::Linear, 5), "position");
    }

    #[test]
    fn ba_shape_scalars_and_scales() {
        let (truth, mut p) = scene(CamDof::Yaw);
        p.sigma_px = 0.0;
        is_shape(bundle_adjust(&p, Some(truth.clone()), Loss::Linear, 5), "sigma_px");
        let (_, mut p) = scene(CamDof::Yaw);
        p.a = -0.1;
        is_shape(bundle_adjust(&p, Some(truth.clone()), Loss::Linear, 5), "a must");
        let (_, mut p) = scene(CamDof::Yaw);
        p.frames[0][0].corner_sigma_scale = [1.0, 0.0, 1.0, 1.0];
        is_shape(bundle_adjust(&p, Some(truth), Loss::Linear, 5), "corner_sigma_scale");
    }

    /// A valid problem passes `check_shapes` and still solves.
    #[test]
    fn ba_shape_valid_passes() {
        let (truth, problem) = scene(CamDof::Yaw);
        check_shapes(&problem, Some(&truth)).unwrap();
        assert!(bundle_adjust(&problem, Some(truth), Loss::Linear, 20).is_ok());
    }

    #[test]
    fn display_nonempty() {
        use std::error::Error;
        let lin = [LinalgError::Singular, LinalgError::NotPositiveDefinite, LinalgError::NoConvergence];
        let res = [ResectError::NoKnownMarkers, ResectError::NoForwardCandidate, ResectError::Linalg(LinalgError::Singular)];
        let mut ba = vec![
            BaError::NoSeedFrame,
            BaError::Disconnected { cams: vec![3, 7], markers: vec![41, 42] },
            BaError::X0RollPitchMismatch { frame: 5 },
            BaError::X0InvalidRotation { frame: 6 },
            BaError::TooFewSurveyed(1),
            BaError::SurveyedMissingInX0(vec![9, 11]),
            BaError::NonFinite,
            BaError::Shape("rp has 1 entries".into()),
        ];
        ba.push(BaError::Resect(ResectError::NoKnownMarkers));
        ba.push(BaError::Linalg(LinalgError::Singular));
        for e in lin {
            assert!(!e.to_string().is_empty());
        }
        for e in &res {
            assert!(!e.to_string().is_empty());
        }
        for e in &ba {
            assert!(!e.to_string().is_empty(), "{e:?}");
        }
        let d = BaError::Disconnected { cams: vec![3, 7], markers: vec![41, 42] }.to_string();
        assert!(d.contains('3') && d.contains('7') && d.contains("41") && d.contains("42"), "{d}");
        assert!(BaError::TooFewSurveyed(1).to_string().contains('1'));
        assert!(BaError::SurveyedMissingInX0(vec![9, 11]).to_string().contains("11"));
        assert!(BaError::X0RollPitchMismatch { frame: 5 }.to_string().contains('5'));
        assert!(BaError::Shape("rp has 1 entries".into()).to_string().contains("rp has 1 entries"));
        // source chain
        let s = BaError::Linalg(LinalgError::Singular);
        assert_eq!(s.source().unwrap().to_string(), LinalgError::Singular.to_string());
        let s = BaError::Resect(ResectError::Linalg(LinalgError::NoConvergence));
        let inner = s.source().unwrap();
        assert_eq!(inner.to_string(), ResectError::Linalg(LinalgError::NoConvergence).to_string());
        assert_eq!(inner.source().unwrap().to_string(), LinalgError::NoConvergence.to_string());
        assert!(ResectError::NoKnownMarkers.source().is_none() && BaError::NonFinite.source().is_none());
        let boxed: Box<dyn Error> = Box::new(BaError::NonFinite);
        assert!(!boxed.to_string().is_empty());
    }
}
