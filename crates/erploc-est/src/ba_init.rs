//! Incremental BA initialiser: planar PnP -> seed frame -> support-first propagation -> refit rounds.
//! Port of `erploc/ba_init.py` .
//!
//! Python's `ValueError`s are `BaError::{NoSeedFrame, Disconnected}`; a failing `planar_pnp` (Python raises) is
//! `BaError::Linalg`. `cam_dof` is read from the problem (`CamDof`), so the `NotImplementedError` case is unrepresentable.
//!
//! Per-corner masks: observations without a valid corner are absent everywhere; planar PnP (hence the seed frame, the
//! seed pose, the seed markers and the single-view marker start) uses all-4-valid observations only; "support" and the
//! observation count of the ordering key count VALID CORNERS; a frame is a resect candidate with at least `min_cam_corners`
//! supporting corners (Yaw 3, Full 4 * min_cam_support = 8); a frame whose total valid corners are below that threshold
//! stays `None` (the complement of the resect condition), a frame at or above it that is still unresolved is
//! `Disconnected`.

use crate::ba_types::{BaError, BaProblem, BaState};
use crate::corners::{n_valid, row_select, safe_uv, RowSel};
use crate::lm::{levenberg_marquardt, LmOptions, Loss};
use crate::resection::{resect, CamDof, ResectOptions};
use crate::synth::Observation;
use erploc_core::erp::{lamphi_to_bearing, px_to_lamphi, residual_lamphi};
use erploc_core::jacobian::{jac_marker, predict_marker, Axis, CamPose};
use erploc_core::markers::{planar_pnp, MarkerPose};
use erploc_core::rot::{exp_so3, log_so3, mat3_vec, rot_zyx};
use erploc_core::{M3, V2, V3};
use erploc_linalg::DMat;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Minimum number of markers for a trustworthy 6-DoF cold start (Python `MIN_MARKERS_FULL_CAM`).
pub const MIN_MARKERS_FULL_CAM: usize = 2;

/// Ordering key of the support-first propagation: (support, n_obs, kind (1 = camera, 0 = marker), -index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PropKey {
    pub support: usize,
    pub n_obs: usize,
    pub kind: u8,
    pub neg_index: i64,
}

/// Deterministic selection: the candidate with the maximum key; the first one on equal keys.
pub(crate) fn select_best(cands: &[(PropKey, bool, i64)]) -> Option<(PropKey, bool, i64)> {
    let mut best: Option<(PropKey, bool, i64)> = None;
    for &c in cands {
        if best.map_or(true, |(bk, _, _)| c.0 > bk) {
            best = Some(c);
        }
    }
    best
}

fn add3(a: &V3, b: &V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn four(v: Vec<V2>) -> [V2; 4] {
    [v[0], v[1], v[2], v[3]]
}

/// `(f, id) -> (t_cm, R_cm)`.
pub type PnpMap = BTreeMap<(usize, i64), (V3, M3)>;

/// Planar PnP of every observation with all 4 corners valid: `(f, id) -> (t_cm, R_cm)` (a repeated id in a frame: the
/// last one wins).
pub fn pnp_all(problem: &BaProblem) -> Result<PnpMap, BaError> {
    let mut out = BTreeMap::new();
    for (f, obs) in problem.frames.iter().enumerate() {
        for ob in obs {
            if n_valid(ob) < 4 {
                continue;
            }
            let d = lamphi_to_bearing(&px_to_lamphi(&ob.uv, &problem.cam_model));
            let res = planar_pnp(&[d[0], d[1], d[2], d[3]], problem.a)?;
            out.insert((f, ob.id), res);
        }
    }
    Ok(out)
}

fn min_cam_support_of(dof: CamDof) -> usize {
    match dof {
        CamDof::Yaw => 1,
        CamDof::Full => MIN_MARKERS_FULL_CAM,
    }
}

/// Valid corners a frame needs to be resected: 3 corners (6 rows > 4 unknowns) for `min_cam_support == 1`, else
/// `4 * min_cam_support` (all corners of that many markers). Full validity gives `4k >= threshold` iff `k >= support`.
fn min_cam_corners_of(min_cam_support: usize) -> usize {
    if min_cam_support == 1 {
        3
    } else {
        4 * min_cam_support
    }
}

/// Shared read-only data of one initialisation run.
struct Ctx<'a> {
    problem: &'a BaProblem,
    pnp: PnpMap,
    /// id -> frames observing it with at least one valid corner (one entry per observation, ascending frame order).
    frames_of: BTreeMap<i64, Vec<usize>>,
    /// Every observation (the last one of a repeated id wins), including those without a valid corner.
    obs_of: BTreeMap<(usize, i64), &'a Observation>,
    /// Valid corners per frame.
    nv_frame: Vec<usize>,
    /// Valid corners per marker id over all frames.
    nv_mk: BTreeMap<i64, usize>,
    need_mk: BTreeSet<i64>,
    need_cam: BTreeSet<usize>,
}

impl<'a> Ctx<'a> {
    fn new(problem: &'a BaProblem) -> Result<Self, BaError> {
        let pnp = pnp_all(problem)?;
        let mut frames_of: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        let mut obs_of = BTreeMap::new();
        let mut nv_mk: BTreeMap<i64, usize> = BTreeMap::new();
        for (f, obs) in problem.frames.iter().enumerate() {
            for ob in obs {
                if n_valid(ob) > 0 {
                    frames_of.entry(ob.id).or_default().push(f);
                }
                obs_of.insert((f, ob.id), ob);
                *nv_mk.entry(ob.id).or_default() += n_valid(ob);
            }
        }
        let nv_frame: Vec<usize> = problem.frames.iter().map(|obs| obs.iter().map(n_valid).sum()).collect();
        let mut need_mk: BTreeSet<i64> = frames_of.keys().copied().collect();
        need_mk.extend(problem.surveyed.keys().copied());
        let need_cam = (0..problem.frames.len()).filter(|&f| nv_frame[f] > 0).collect();
        Ok(Ctx { problem, pnp, frames_of, obs_of, nv_frame, nv_mk, need_mk, need_cam })
    }

    /// Distinct surveyed ids of frame `f` that have a planar PnP (an all-4-valid observation).
    fn n_distinct_surveyed(&self, f: usize) -> usize {
        let ids: BTreeSet<i64> = self.problem.frames[f]
            .iter()
            .filter(|ob| self.problem.surveyed.contains_key(&ob.id) && self.pnp.contains_key(&(f, ob.id)))
            .map(|ob| ob.id)
            .collect();
        ids.len()
    }

    /// Frame with the most distinct surveyed ids (first on ties, strict `>`); fewer than 2 -> `NoSeedFrame`.
    fn seed_frame(&self) -> Result<usize, BaError> {
        let (mut best_f, mut best_n) = (None, 0);
        for f in 0..self.problem.frames.len() {
            let n = self.n_distinct_surveyed(f);
            if n > best_n {
                best_f = Some(f);
                best_n = n;
            }
        }
        match best_f {
            Some(f) if best_n >= 2 => Ok(f),
            _ => Err(BaError::NoSeedFrame),
        }
    }

    /// Seed pose: yaw from the complex mean over surveyed pairs, c from the mean of `p_TS - R t`.
    fn seed_pose(&self, f_seed: usize) -> CamPose {
        let p = self.problem;
        let ids: BTreeSet<i64> = p.frames[f_seed]
            .iter()
            .filter(|ob| p.surveyed.contains_key(&ob.id) && self.pnp.contains_key(&(f_seed, ob.id)))
            .map(|ob| ob.id)
            .collect();
        let ids: Vec<i64> = ids.into_iter().collect();
        let [roll, pitch] = p.rp[f_seed];
        let r_rp = rot_zyx(0.0, pitch, roll);
        let (mut sc, mut ss) = (0.0, 0.0);
        for a in 0..ids.len() {
            for b in a + 1..ids.len() {
                let (i, j) = (ids[a], ids[b]);
                let t_i = self.pnp[&(f_seed, i)].0;
                let t_j = self.pnp[&(f_seed, j)].0;
                let v = mat3_vec(&r_rp, &sub3(&t_j, &t_i));
                let dp = sub3(&p.surveyed[&j].0, &p.surveyed[&i].0);
                let yaw_ij = dp[1].atan2(dp[0]) - v[1].atan2(v[0]);
                sc += yaw_ij.cos();
                ss += yaw_ij.sin();
            }
        }
        let yaw = ss.atan2(sc);
        let r = rot_zyx(yaw, pitch, roll);
        let mut sum = [0.0; 3];
        for i in &ids {
            let rt = mat3_vec(&r, &self.pnp[&(f_seed, *i)].0);
            sum = add3(&sum, &sub3(&p.surveyed[i].0, &rt));
        }
        let n = ids.len() as f64;
        CamPose::euler([sum[0] / n, sum[1] / n, sum[2] / n], yaw, roll, pitch)
    }

    /// LM fit of one marker from known cameras (position fixed -> 3 DoF, else 6).
    fn fit_marker(&self, pairs: &[(&Observation, CamPose)], x0: &MarkerPose, fix_position: bool) -> Result<MarkerPose, BaError> {
        let p = self.problem;
        let (a, cam_model, sigma, nm) = (p.a, &p.cam_model, p.sigma_px, p.noise_model);
        let p0 = x0.p;
        let r0 = x0.r;
        let ndim = if fix_position { 3 } else { 6 };
        let lp_list: Vec<[V2; 4]> = pairs.iter().map(|(ob, _)| four(px_to_lamphi(&safe_uv(ob), cam_model))).collect();
        let sel: Vec<RowSel> = pairs.iter().map(|(ob, _)| row_select(ob)).collect();
        let n_rows: usize = sel.iter().map(|s| s.n).sum();
        let mk_of = |x: &Vec<f64>| -> MarkerPose {
            if fix_position {
                MarkerPose { id: x0.id, p: p0, r: r0.mul(&exp_so3(&[x[0], x[1], x[2]])) }
            } else {
                MarkerPose { id: x0.id, p: [p0[0] + x[0], p0[1] + x[1], p0[2] + x[2]], r: r0.mul(&exp_so3(&[x[3], x[4], x[5]])) }
            }
        };
        let mut residual = |x: &Vec<f64>| -> Vec<f64> {
            let mk = mk_of(x);
            let mut out = Vec::with_capacity(n_rows);
            for (((_, cam), lp_obs), sl) in pairs.iter().zip(&lp_list).zip(&sel) {
                let lp_pred = predict_marker(cam, &mk, a);
                let (r, _) = residual_lamphi(lp_obs, &lp_pred, cam_model, nm);
                for j in 0..sl.n {
                    let src = sl.idx[j];
                    out.push((r[src / 2][src % 2] * sl.w[j]) / sigma);
                }
            }
            out
        };
        let mut jac = |x: &Vec<f64>| -> DMat {
            let mk = mk_of(x);
            let mut j = DMat::zeros(n_rows, ndim);
            let mut row0 = 0;
            for (((_, cam), lp_obs), sl) in pairs.iter().zip(&lp_list).zip(&sel) {
                let lp_pred = predict_marker(cam, &mk, a);
                let (_, d) = residual_lamphi(lp_obs, &lp_pred, cam_model, nm);
                let (_, j_mk) = jac_marker(cam, &mk, a, &[Axis::Z]);
                let d8: Vec<f64> = d.iter().flat_map(|v| [v[0], v[1]]).collect();
                let c0 = if fix_position { 3 } else { 0 };
                for q in 0..sl.n {
                    let src = sl.idx[q];
                    for col in 0..ndim {
                        j[(row0 + q, col)] = -((d8[src] * j_mk[(src, c0 + col)]) * sl.w[q]) / sigma;
                    }
                }
                row0 += sl.n;
            }
            j
        };
        let retract = |x: &Vec<f64>, dx: &[f64]| -> Vec<f64> {
            let w = |o: usize| log_so3(&exp_so3(&[x[o], x[o + 1], x[o + 2]]).mul(&exp_so3(&[dx[o], dx[o + 1], dx[o + 2]])));
            if fix_position {
                w(0).to_vec()
            } else {
                let r = w(3);
                vec![x[0] + dx[0], x[1] + dx[1], x[2] + dx[2], r[0], r[1], r[2]]
            }
        };
        let opts = LmOptions { max_iter: 50, ..LmOptions::default() };
        let res = levenberg_marquardt(&mut residual, &mut jac, vec![0.0; ndim], &retract, &opts)?;
        Ok(mk_of(&res.x))
    }

    /// Does some known camera see marker `mid` with a planar PnP?
    fn has_known_pnp(&self, mid: i64, cams: &BTreeMap<usize, CamPose>) -> bool {
        self.frames_of.get(&mid).is_some_and(|fs| fs.iter().any(|f| cams.contains_key(f) && self.pnp.contains_key(&(*f, mid))))
    }

    /// Resect frame `f` with all known markers it sees.
    fn resect_frame(&self, f: usize, mks: &BTreeMap<i64, MarkerPose>, x0: Option<CamPose>) -> Result<CamPose, BaError> {
        let p = self.problem;
        let [roll, pitch] = p.rp[f];
        let known: BTreeMap<i64, MarkerPose> = p.frames[f].iter().filter_map(|ob| mks.get(&ob.id).map(|m| (ob.id, *m))).collect();
        let opts = ResectOptions { x0, sigma_px: p.sigma_px, noise_model: p.noise_model, loss: Loss::Linear, dof: p.cam_dof };
        Ok(resect(&p.frames[f], &known, p.a, &p.cam_model, roll, pitch, &opts)?.cam)
    }

    /// Fit marker `mid` from all known cameras that see it. `x0 = None`: nearest known frame's single-view PnP.
    fn fit_marker_from_cams(&self, mid: i64, cams: &BTreeMap<usize, CamPose>, x0: Option<&MarkerPose>) -> Result<MarkerPose, BaError> {
        let known_f: Vec<usize> =
            self.frames_of.get(&mid).map(|v| v.iter().copied().filter(|f| cams.contains_key(f)).collect()).unwrap_or_default();
        let is_ts = self.problem.surveyed.contains_key(&mid);
        let x0_owned;
        let x0 = match x0 {
            Some(x) => x,
            None => {
                // nearest known frame among those with a planar PnP (the first on ties)
                let with_pnp: Vec<usize> = known_f.iter().copied().filter(|f| self.pnp.contains_key(&(*f, mid))).collect();
                let mut f0 = with_pnp[0];
                let mut best = norm3(&self.pnp[&(f0, mid)].0);
                for &f in &with_pnp[1..] {
                    let n = norm3(&self.pnp[&(f, mid)].0);
                    if n < best {
                        best = n;
                        f0 = f;
                    }
                }
                let (t_cm, r_cm) = self.pnp[&(f0, mid)];
                let cam0 = cams[&f0];
                let p0 = if is_ts { self.problem.surveyed[&mid].0 } else { add3(&cam0.c, &mat3_vec(&cam0.r(), &t_cm)) };
                x0_owned = MarkerPose { id: mid, p: p0, r: cam0.r().mul(&r_cm) };
                &x0_owned
            }
        };
        let pairs: Vec<(&Observation, CamPose)> = known_f.iter().map(|&f| (self.obs_of[&(f, mid)], cams[&f])).collect();
        self.fit_marker(&pairs, x0, is_ts)
    }
}

/// Mutable propagation state shared by `initialize` and the naive BFS.
struct Prop {
    cams: BTreeMap<usize, CamPose>,
    mks: BTreeMap<i64, MarkerPose>,
    sup_cam: BTreeMap<usize, usize>,
    sup_mk: BTreeMap<i64, usize>,
}

impl Prop {
    fn new(ctx: &Ctx) -> Self {
        Prop {
            cams: BTreeMap::new(),
            mks: BTreeMap::new(),
            sup_cam: ctx.need_cam.iter().map(|&f| (f, 0)).collect(),
            sup_mk: ctx.need_mk.iter().map(|&m| (m, 0)).collect(),
        }
    }

    fn commit_cam(&mut self, ctx: &Ctx, f: usize, cam: CamPose) {
        self.cams.insert(f, cam);
        for ob in &ctx.problem.frames[f] {
            if n_valid(ob) == 0 {
                continue; // an observation without a valid corner is absent
            }
            *self.sup_mk.get_mut(&ob.id).expect("marker id in need_mk") += n_valid(ob);
        }
    }

    fn commit_mk(&mut self, ctx: &Ctx, mid: i64, mk: MarkerPose) {
        self.mks.insert(mid, mk);
        if let Some(fs) = ctx.frames_of.get(&mid) {
            for f in fs {
                *self.sup_cam.get_mut(f).expect("frame in need_cam") += n_valid(ctx.obs_of[&(*f, mid)]);
            }
        }
    }

    /// Seed frame, its surveyed markers (oriented by `R_seed R_cm`), then the seed frame re-resected.
    fn seed(&mut self, ctx: &Ctx, f_seed: usize) -> Result<(), BaError> {
        let pose = ctx.seed_pose(f_seed);
        let r_seed = pose.r();
        self.cams.insert(f_seed, pose);
        self.commit_cam(ctx, f_seed, pose);
        for ob in &ctx.problem.frames[f_seed] {
            if let (Some((p_ts, _)), Some((_, r_cm))) = (ctx.problem.surveyed.get(&ob.id), ctx.pnp.get(&(f_seed, ob.id))) {
                self.commit_mk(ctx, ob.id, MarkerPose { id: ob.id, p: *p_ts, r: r_seed.mul(r_cm) });
            }
        }
        let cam = ctx.resect_frame(f_seed, &self.mks, None)?;
        self.cams.insert(f_seed, cam);
        Ok(())
    }

    fn missing(&self, ctx: &Ctx, min_cam_corners: usize) -> Result<(), BaError> {
        let missing_mk: Vec<i64> = ctx.need_mk.iter().copied().filter(|m| !self.mks.contains_key(m)).collect();
        // frames with fewer valid corners than the resect threshold can never be resected: left None (the complement of the
        // resect condition); at or above it, an unresolved frame is a true disconnection
        let missing_cam: Vec<usize> =
            ctx.need_cam.iter().copied().filter(|f| !self.cams.contains_key(f) && ctx.nv_frame[*f] >= min_cam_corners).collect();
        if missing_mk.is_empty() && missing_cam.is_empty() {
            Ok(())
        } else {
            Err(BaError::Disconnected { cams: missing_cam, markers: missing_mk })
        }
    }

    fn into_state(self, n_frames: usize) -> BaState {
        BaState { cams: (0..n_frames).map(|f| self.cams.get(&f).copied()).collect(), markers: self.mks }
    }
}

/// Incremental initialisation: PnP of all observations -> seed frame -> support-first propagation -> `refine_rounds`
/// rounds of (refit all markers, re-resect all cameras). Python default `refine_rounds = 1`.
pub fn initialize(problem: &BaProblem, refine_rounds: usize) -> Result<BaState, BaError> {
    crate::ba::check_shapes_impl(problem, None, false)?;
    let min_cam_corners = min_cam_corners_of(min_cam_support_of(problem.cam_dof));
    let n_frames = problem.frames.len();
    let ctx = Ctx::new(problem)?;
    let f_seed = ctx.seed_frame()?;
    let mut st = Prop::new(&ctx);
    st.seed(&ctx, f_seed)?;

    loop {
        // (key, is_cam, index): the maximum key wins; keys are unique, so iteration order is irrelevant
        let mut cands: Vec<(PropKey, bool, i64)> = Vec::new();
        for &f in &ctx.need_cam {
            let sup = st.sup_cam[&f];
            if st.cams.contains_key(&f) || sup < min_cam_corners {
                continue;
            }
            let key = PropKey { support: sup, n_obs: ctx.nv_frame[f], kind: 1, neg_index: -(f as i64) };
            cands.push((key, true, f as i64));
        }
        for &mid in &ctx.need_mk {
            let sup = st.sup_mk[&mid];
            if st.mks.contains_key(&mid) || sup == 0 {
                continue;
            }
            // a marker seen only by observations with missing corners has no initial pose without a known frame's PnP
            if !ctx.has_known_pnp(mid, &st.cams) {
                continue;
            }
            let n_obs = ctx.nv_mk.get(&mid).copied().unwrap_or(0);
            let key = PropKey { support: sup, n_obs, kind: 0, neg_index: mid.checked_neg().expect("marker id i64::MIN") };
            cands.push((key, false, mid));
        }
        let best = select_best(&cands);
        let Some((_, is_cam, k)) = best else { break };
        if is_cam {
            let cam = ctx.resect_frame(k as usize, &st.mks, None)?;
            st.commit_cam(&ctx, k as usize, cam);
        } else {
            let mk = ctx.fit_marker_from_cams(k, &st.cams, None)?;
            st.commit_mk(&ctx, k, mk);
        }
    }

    st.missing(&ctx, min_cam_corners)?;

    for _ in 0..refine_rounds {
        let mut new_mks = BTreeMap::new();
        for (&mid, mk) in &st.mks {
            new_mks.insert(mid, ctx.fit_marker_from_cams(mid, &st.cams, Some(mk))?);
        }
        st.mks = new_mks;
        let mut new_cams = BTreeMap::new();
        for (&f, &cam) in &st.cams {
            new_cams.insert(f, ctx.resect_frame(f, &st.mks, Some(cam))?);
        }
        st.cams = new_cams;
    }
    Ok(st.into_state(n_frames))
}

/// Verification only (Python `_naive_bfs_initialize`): FIFO order, Full DoF semantics (min support 2), no Phase 2.
#[doc(hidden)]
pub fn naive_bfs_initialize(problem: &BaProblem) -> Result<BaState, BaError> {
    let min_cam_corners = min_cam_corners_of(MIN_MARKERS_FULL_CAM);
    let n_frames = problem.frames.len();
    let ctx = Ctx::new(problem)?;
    let f_seed = ctx.seed_frame()?;
    let mut st = Prop::new(&ctx);
    let mut queue: VecDeque<(bool, i64)> = VecDeque::new();
    let mut queued: BTreeSet<(bool, i64)> = BTreeSet::new();

    // Same seed procedure as `Prop::seed`, with the enqueueing of Python's `_commit_*` interleaved.
    let pose = ctx.seed_pose(f_seed);
    let r_seed = pose.r();
    st.cams.insert(f_seed, pose);
    let enqueue_after_cam = |st: &Prop, f: usize, queue: &mut VecDeque<(bool, i64)>, queued: &mut BTreeSet<(bool, i64)>| {
        for ob in &problem.frames[f] {
            if n_valid(ob) == 0 {
                continue;
            }
            let key = (false, ob.id);
            if !st.mks.contains_key(&ob.id) && !queued.contains(&key) && st.sup_mk[&ob.id] > 0 && ctx.has_known_pnp(ob.id, &st.cams) {
                queue.push_back(key);
                queued.insert(key);
            }
        }
    };
    let enqueue_after_mk = |st: &Prop, mid: i64, queue: &mut VecDeque<(bool, i64)>, queued: &mut BTreeSet<(bool, i64)>| {
        if let Some(fs) = ctx.frames_of.get(&mid) {
            for &f in fs {
                let key = (true, f as i64);
                if !st.cams.contains_key(&f) && !queued.contains(&key) && st.sup_cam[&f] >= min_cam_corners {
                    queue.push_back(key);
                    queued.insert(key);
                }
            }
        }
    };
    st.commit_cam(&ctx, f_seed, pose);
    enqueue_after_cam(&st, f_seed, &mut queue, &mut queued);
    for ob in &problem.frames[f_seed] {
        if let (Some((p_ts, _)), Some((_, r_cm))) = (problem.surveyed.get(&ob.id), ctx.pnp.get(&(f_seed, ob.id))) {
            st.commit_mk(&ctx, ob.id, MarkerPose { id: ob.id, p: *p_ts, r: r_seed.mul(r_cm) });
            enqueue_after_mk(&st, ob.id, &mut queue, &mut queued);
        }
    }
    let cam = ctx.resect_frame(f_seed, &st.mks, None)?;
    st.cams.insert(f_seed, cam);

    while let Some((is_cam, k)) = queue.pop_front() {
        if is_cam {
            let f = k as usize;
            if st.cams.contains_key(&f) {
                continue;
            }
            let cam = ctx.resect_frame(f, &st.mks, None)?;
            st.commit_cam(&ctx, f, cam);
            enqueue_after_cam(&st, f, &mut queue, &mut queued);
        } else {
            if st.mks.contains_key(&k) {
                continue;
            }
            let mk = ctx.fit_marker_from_cams(k, &st.cams, None)?;
            st.commit_mk(&ctx, k, mk);
            enqueue_after_mk(&st, k, &mut queue, &mut queued);
        }
    }
    // the naive variant uses the Full threshold regardless of cam_dof
    let missing_mk: Vec<i64> = ctx.need_mk.iter().copied().filter(|m| !st.mks.contains_key(m)).collect();
    let missing_cam: Vec<usize> =
        ctx.need_cam.iter().copied().filter(|f| !st.cams.contains_key(f)).filter(|&f| ctx.nv_frame[f] >= min_cam_corners).collect();
    if !missing_mk.is_empty() || !missing_cam.is_empty() {
        return Err(BaError::Disconnected { cams: missing_cam, markers: missing_mk });
    }
    Ok(st.into_state(n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{add_noise, make_markers, make_trajectory, render, survey, MarkerLayout, Observation, SynthConfig, TrajectoryKind};
    use erploc_core::erp::{wrap_pi, ErpCamera, NoiseModel};
    use erploc_linalg::rng::Pcg64;

    const A: f64 = 0.16;
    const SIG: f64 = 0.3;
    const SEED: u64 = 20260911;
    const ROOM: [f64; 4] = [-4.0, 4.0, -3.0, 3.0];

    fn cam_model() -> ErpCamera {
        ErpCamera::new(6528, 3264, 1)
    }

    fn cfg() -> SynthConfig {
        SynthConfig { cam: cam_model(), a: A, sigma_px: SIG, noise_model: NoiseModel::Tangent, seed: SEED }
    }

    fn scene(seed: u64) -> (Vec<MarkerPose>, Vec<CamPose>, Pcg64) {
        let mut rng = Pcg64::new(seed);
        let mks = make_markers(MarkerLayout::Walls, 12, &ROOM, [0.8, 1.8], &mut rng);
        let cams = make_trajectory(TrajectoryKind::Circle, 40, &ROOM, 1.2, &mut rng);
        (mks, cams, rng)
    }

    fn problem(frames: Vec<Vec<Observation>>, rng: &mut Pcg64, sigma_ts: f64, mks: &[MarkerPose], dof: CamDof) -> BaProblem {
        let surveyed = survey(mks, &[0, 4, 8], sigma_ts, rng);
        let n = frames.len();
        BaProblem {
            frames,
            rp: vec![[0.0, 0.0]; n],
            surveyed,
            a: A,
            cam_model: cam_model(),
            sigma_px: SIG,
            noise_model: NoiseModel::Tangent,
            cam_dof: dof,
        }
    }

    fn with_frames(p: &BaProblem, frames: Vec<Vec<Observation>>) -> BaProblem {
        BaProblem { frames, ..p.clone() }
    }

    fn assert_small(v: f64, tol: f64, what: &str) {
        assert!(v.is_finite(), "{what} not finite: {v}");
        assert!(v < tol, "{what} = {v:e} >= {tol:e}");
    }

    fn cam_err(truth: &CamPose, est: &CamPose) -> (f64, f64) {
        (norm3(&sub3(&est.c, &truth.c)), wrap_pi(est.yaw - truth.yaw).abs())
    }

    /// Clean problem (seed 20260911, sigma_TS = 0 but the survey draws are still consumed).
    fn clean() -> (Vec<MarkerPose>, Vec<CamPose>, BaProblem) {
        let (mks, cams, mut rng) = scene(SEED);
        let frames = render(&cams, &mks, &cfg(), None);
        let p = problem(frames, &mut rng, 0.0, &mks, CamDof::Yaw);
        (mks, cams, p)
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn zero_noise_exact() {
        let (mks, cams, p) = clean();
        let st = initialize(&p, 1).unwrap();
        let (mut dp, mut dyaw) = (0.0f64, 0.0f64);
        for (ct, ce) in cams.iter().zip(&st.cams) {
            let (a, b) = cam_err(ct, ce.as_ref().unwrap());
            assert!(a.is_finite() && b.is_finite());
            dp = dp.max(a);
            dyaw = dyaw.max(b);
        }
        assert_small(dp, 1e-6, "cam position");
        assert_small(dyaw, 1e-8, "cam yaw");
        let (mut mp, mut mr) = (0.0f64, 0.0f64);
        for (id, me) in &st.markers {
            let t = mks.iter().find(|m| m.id == *id).unwrap();
            let a = norm3(&sub3(&me.p, &t.p));
            let b = norm3(&log_so3(&me.r.transpose().mul(&t.r)));
            assert!(a.is_finite() && b.is_finite());
            mp = mp.max(a);
            mr = mr.max(b);
        }
        eprintln!("zero_noise_exact: cam {dp:.2e} m, yaw {dyaw:.2e} rad, marker {mp:.2e} m, rot {mr:.2e} rad");
        assert_small(mp, 1e-6, "marker position");
        assert_small(mr, 1e-7, "marker rotation");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn no_seed_frame() {
        let (_, _, p) = clean();
        let mut surveyed = BTreeMap::new();
        surveyed.insert(0, p.surveyed[&0]);
        let p4 = BaProblem { surveyed, ..p };
        assert_eq!(initialize(&p4, 1).unwrap_err(), BaError::NoSeedFrame);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn unobserved_surveyed() {
        let (_, _, p) = clean();
        let frames = p.frames.iter().map(|o| o.iter().filter(|ob| ob.id != 8).cloned().collect()).collect();
        match initialize(&with_frames(&p, frames), 1) {
            Err(BaError::Disconnected { markers, .. }) => assert!(markers.contains(&8)),
            other => panic!("expected Disconnected, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn empty_frame() {
        let (_, cams, p) = clean();
        let empty_f = p.frames.iter().position(|o| !o.is_empty()).unwrap();
        let mut frames = p.frames.clone();
        frames[empty_f].clear();
        let st = initialize(&with_frames(&p, frames), 1).unwrap();
        assert!(st.cams[empty_f].is_none());
        let (mut dp, mut dyaw) = (0.0f64, 0.0f64);
        for f in (0..cams.len()).filter(|&f| f != empty_f) {
            let (a, b) = cam_err(&cams[f], st.cams[f].as_ref().expect("other frame initialised"));
            assert!(a.is_finite() && b.is_finite());
            dp = dp.max(a);
            dyaw = dyaw.max(b);
        }
        eprintln!("empty_frame: cam {dp:.2e} m, yaw {dyaw:.2e} rad");
        assert_small(dp, 1e-6, "cam position");
        assert_small(dyaw, 1e-8, "cam yaw");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn full_single_marker_frame_none() {
        let (mks, cams, mut rng) = scene(SEED + 900);
        let mut nrng = Pcg64::new(SEED + 901);
        let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut nrng);
        let f_one = frames.iter().position(|o| o.len() >= 2).unwrap();
        let mut frames = frames;
        frames[f_one].truncate(1);
        let p = problem(frames, &mut rng, 0.005, &mks, CamDof::Full);
        let st = initialize(&p, 1).unwrap();
        assert!(st.cams[f_one].is_none());
        for f in 0..cams.len() {
            if f != f_one && !p.frames[f].is_empty() {
                assert!(st.cams[f].is_some(), "frame {f} not initialised");
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // full initialize() (pnp_all + LM fits): > 400 s each under Miri
    fn naive_bfs_matches_support_first_when_noise_free() {
        let (mks, cams, mut rng) = scene(SEED);
        let frames = render(&cams, &mks, &cfg(), None);
        let p = problem(frames, &mut rng, 0.0, &mks, CamDof::Full);
        let st = naive_bfs_initialize(&p).unwrap();
        for (ct, ce) in cams.iter().zip(&st.cams) {
            let (dp, _) = cam_err(ct, ce.as_ref().unwrap());
            assert_small(dp, 1e-6, "naive cam position");
        }
    }

    #[test]
    fn prop_key_order() {
        let k = |support, n_obs, kind, neg_index| PropKey { support, n_obs, kind, neg_index };
        assert!(k(3, 0, 0, -9) > k(2, 99, 1, 0));
        assert!(k(2, 5, 0, -9) > k(2, 4, 1, 0));
        assert!(k(2, 5, 1, -9) > k(2, 5, 0, 0));
        assert!(k(2, 5, 1, -1) > k(2, 5, 1, -2));
    }
}
