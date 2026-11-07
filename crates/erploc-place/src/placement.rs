//! Placement model (port of `erploc/placement.py`): spec, precomputed tables, per-selection cell statistics, penalties, the
//! lexicographic keys and the greedy / 1-swap / prune / survey-triplet search.
//!
//! NaN rule: Python's `np.maximum/np.minimum/max/min` propagate NaN, `f64::max/min` do not; the local `nmax`/`nmin`
//! helpers reproduce numpy. Keys compare like Python tuples (NaN comparisons are false).
use crate::npcompat::np_sum;
use crate::scene::{candidates_from_walls, cells, Candidate, Obstacle, Workspace};
use crate::site::CandKind;
use crate::visibility::{visible_batch, VisParams, FRONT_GAP_STAND, FRONT_GAP_WALL};
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::{Axis, CamPose};
use erploc_core::markers::{upright_pose, MarkerPose};
use erploc_core::V3;
use erploc_est::fisher::{info_marker, schur_markers, sigma_pos, SIGMA_RTOL};
use erploc_linalg::eig::eigvalsh;
use erploc_linalg::{lu, DMat, LinalgError};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct PlacementSpec {
    pub ws: Workspace,
    pub obstacles: Vec<Obstacle>,
    pub candidates: Vec<Candidate>,
    pub n_max: usize,
    pub a: f64,
    pub cam_model: ErpCamera,
    pub sigma_px: f64,
    pub noise_model: NoiseModel,
    pub vp: VisParams,
    pub tau: f64,
    /// Python default 2.
    pub k_min: usize,
    pub tau_loo: Option<f64>,
    /// Python default 10.0.
    pub sigma_cap: f64,
    /// Python default 1.0.
    pub w_k: f64,
}

impl PlacementSpec {
    /// `_tau_loo`: `tau` unless `tau_loo` is set.
    pub fn tau_loo_eff(&self) -> f64 {
        self.tau_loo.unwrap_or(self.tau)
    }
    /// `dataclasses.replace(spec, n_max=k)`.
    pub fn with_n_max(&self, k: usize) -> PlacementSpec {
        PlacementSpec { n_max: k, ..self.clone() }
    }
}

/// `vis[(k*hd + h)*n + j]`, `info[((k*n + j)*4 + a)*4 + b]` (camera 4x4 block `I_cc`, row-major).
#[derive(Clone, Debug)]
pub struct Tables {
    pub cells: Vec<V3>,
    pub k: usize,
    pub hd: usize,
    pub n: usize,
    pub vis: Vec<bool>,
    pub info: Vec<f64>,
}

/// Index into `Tables::vis`: `(k*hd + h)*n + j`.
pub(crate) fn vis_index(k: usize, hd: usize, h: usize, n: usize, j: usize) -> usize {
    (k * hd + h) * n + j
}

/// Start of the 4x4 block `I[k, j]` in `Tables::info`: `(k*n + j)*16`.
pub(crate) fn info_offset(k: usize, n: usize, j: usize) -> usize {
    info_entry(k, n, j, 0, 0)
}

/// Entry `(a, b)` of the block `I[k, j]`: `((k*n + j)*4 + a)*4 + b`.
pub(crate) fn info_entry(k: usize, n: usize, j: usize, a: usize, b: usize) -> usize {
    ((k * n + j) * 4 + a) * 4 + b
}

impl Tables {
    /// Checked constructor for tables imported from flat arrays.
    ///
    /// Layout: `vis[(k*hd + h)*n + j]`, `info[((k*n + j)*4 + a)*4 + b]`. `info` must hold the FULL symmetric
    /// 4x4 camera block per entry, row-major (`sigma_pos`/`eigvalsh` read all 16 entries); the 10 upper-triangle
    /// entries that a FastEval-style table keeps are not enough and must be expanded by the caller.
    ///
    /// # Panics
    /// Unless `cells.len() * hd * n == vis.len()` and `info.len() == cells.len() * n * 16` (a shape error; the signature returns no
    /// `Result` so that tables built from trusted flat arrays stay free of error plumbing).
    pub fn from_raw(cells: Vec<V3>, hd: usize, n: usize, vis: Vec<bool>, info: Vec<f64>) -> Tables {
        let k = cells.len();
        assert_eq!(vis.len(), k * hd * n, "Tables::from_raw: vis length != cells*hd*n");
        assert_eq!(info.len(), k * n * 16, "Tables::from_raw: info length != cells*n*16");
        Tables { cells, k, hd, n, vis, info }
    }
    pub fn vis_at(&self, k: usize, h: usize, j: usize) -> bool {
        self.vis[vis_index(k, self.hd, h, self.n, j)]
    }
    /// The 16 entries of `I[k, j]`, row-major.
    pub fn info_at(&self, k: usize, j: usize) -> &[f64] {
        let o = info_offset(k, self.n, j);
        &self.info[o..o + 16]
    }
    /// `np.all(vis == vis[:, :1, :])`.
    pub fn heading_independent(&self) -> bool {
        (0..self.k).all(|k| (0..self.n).all(|j| (1..self.hd).all(|h| self.vis_at(k, h, j) == self.vis_at(k, 0, j))))
    }
}

#[derive(Clone, Debug)]
pub struct CellStats {
    pub sigma: Vec<f64>,
    pub n_vis: Vec<f64>,
    pub sigma_loo: Vec<f64>,
    pub feasible: Vec<bool>,
    pub margin: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoryStep {
    Greedy {
        sel: Vec<usize>,
        n_infeasible: usize,
        penalty: f64,
    },
    OneSwap {
        sel: Vec<usize>,
        n_infeasible: usize,
        penalty: f64,
        neg_margin: f64,
    },
    Prune {
        sel: Vec<usize>,
        min_margin: f64,
    },
    Survey {
        ids: Vec<usize>,
        score: f64,
        n_skip: usize,
    },
    /// One `ift_det` size of `optimize_ift` (Python `{'step': 'ift', 'k', 'n_infeasible', 'penalty'}`).
    Ift {
        k: usize,
        n_infeasible: usize,
        penalty: f64,
    },
}

#[derive(Clone, Debug)]
pub struct PlacementResult {
    pub sel: Vec<usize>,
    pub stats: CellStats,
    /// Up to 3.
    pub survey_ids: Vec<usize>,
    pub history: Vec<HistoryStep>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key2 {
    pub n_infeasible: usize,
    pub penalty_sum: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key3 {
    pub n_infeasible: usize,
    pub penalty_sum: f64,
    pub neg_margin: f64,
}

impl Key2 {
    /// Python tuple `<`: `a0 < b0` or (`a0 == b0` and `a1 < b1`).
    pub fn lt(&self, o: &Key2) -> bool {
        self.n_infeasible < o.n_infeasible || (self.n_infeasible == o.n_infeasible && self.penalty_sum < o.penalty_sum)
    }
}

impl Key3 {
    /// 3-term lexicographic `<` with Python tuple semantics (a NaN term makes `==` and `<` false).
    pub fn lt(&self, o: &Key3) -> bool {
        if self.n_infeasible != o.n_infeasible {
            return self.n_infeasible < o.n_infeasible;
        }
        if self.penalty_sum != o.penalty_sum {
            return self.penalty_sum < o.penalty_sum;
        }
        self.neg_margin < o.neg_margin
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlaceError {
    Linalg(LinalgError),
    NonFiniteSchur,
    /// Site table file: bad magic, version, length, index or reference value.
    Site(String),
    /// `FastEval::new` on tables whose visibility depends on the heading (`get_eval` then uses the slow evaluator).
    NotHeadingIndependent,
    /// The tables' heading axis differs from `spec.ws.headings` (site tables are heading-collapsed, `hd = 1`, against 8 headings): a
    /// consumer that sums over headings (`frame_blocks`, `choose_survey_triplet`) would silently give a different score.
    HeadingMismatch {
        tables: usize,
        spec: usize,
    },
    /// An argument that the function cannot work with: an empty polygon or candidate list, a non-positive spacing, an index or
    /// subset size out of range. Returned instead of a panic by the public functions that return a `Result`.
    InvalidInput(String),
}

impl std::fmt::Display for PlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaceError::Linalg(e) => write!(f, "placement: linear algebra failure: {e}"),
            PlaceError::NonFiniteSchur => write!(f, "placement: non-finite Schur complement"),
            PlaceError::Site(msg) => write!(f, "placement: site table error: {msg}"),
            PlaceError::NotHeadingIndependent => write!(f, "placement: visibility tables depend on the heading"),
            PlaceError::HeadingMismatch { tables, spec } => {
                write!(f, "placement: tables have {tables} headings but the spec has {spec}")
            }
            PlaceError::InvalidInput(msg) => write!(f, "placement: invalid input: {msg}"),
        }
    }
}

impl std::error::Error for PlaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlaceError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for PlaceError {
    fn from(e: LinalgError) -> Self {
        PlaceError::Linalg(e)
    }
}

/// `np.maximum` of two scalars: NaN propagates.
pub(crate) fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// `np.minimum` of two scalars: NaN propagates.
pub(crate) fn nmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// `v.max()` / `v.min()` over a non-empty array, NaN-propagating.
fn red_min(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |m, &x| nmin(m, x))
}

/// id = index; `upright_pose(j, [x, y, z], azimuth)`.
pub fn markers_from_candidates(c: &[Candidate]) -> Vec<MarkerPose> {
    c.iter().enumerate().map(|(j, cd)| upright_pose(j as i64, &[cd.xy[0], cd.xy[1], cd.z], cd.azimuth)).collect()
}

/// `_cap`: non-finite -> `cap`, then `min(., cap)`.
pub fn cap(x: f64, cap: f64) -> f64 {
    nmin(if x.is_finite() { x } else { cap }, cap)
}

pub fn precompute(spec: &PlacementSpec) -> Result<Tables, PlaceError> {
    precompute_with_gaps(spec, None)
}

/// `precompute` with the per-marker front gap of the site model: `FRONT_GAP_STAND` for stands, `FRONT_GAP_WALL` for walls
///. With all-wall `kinds` it equals `precompute` bit for bit.
/// Python reference: `visible_batch(..., gaps=...)` + `info_marker`, composed by the exporter of the IFT fixtures (not included).
pub fn precompute_typed(spec: &PlacementSpec, kinds: &[CandKind]) -> Result<Tables, PlaceError> {
    if kinds.len() != spec.candidates.len() {
        return Err(PlaceError::InvalidInput(format!(
            "precompute_typed: {} kinds for {} candidates (one kind per candidate)",
            kinds.len(),
            spec.candidates.len()
        )));
    }
    let gaps: Vec<f64> = kinds.iter().map(|k| if *k == CandKind::Stand { FRONT_GAP_STAND } else { FRONT_GAP_WALL }).collect();
    precompute_with_gaps(spec, Some(&gaps))
}

fn precompute_with_gaps(spec: &PlacementSpec, gaps: Option<&[f64]>) -> Result<Tables, PlaceError> {
    spec.ws.validate()?;
    let xy = cells(&spec.ws, &spec.obstacles)?;
    // z-major stacking (np.concatenate over z_cam)
    let mut c: Vec<V3> = Vec::with_capacity(xy.len() * spec.ws.z_cam.len());
    for &z in &spec.ws.z_cam {
        c.extend(xy.iter().map(|p| [p[0], p[1], z]));
    }
    let mks = markers_from_candidates(&spec.candidates);
    let vis = visible_batch(&c, &spec.ws.headings, &mks, spec.a, &spec.cam_model, &spec.obstacles, &spec.vp, true, gaps);
    let (k, n) = (c.len(), mks.len());
    let mut info = Vec::with_capacity(k * n * 16);
    for ck in &c {
        let cam = CamPose::euler(*ck, 0.0, 0.0, 0.0);
        for mk in &mks {
            let m = info_marker(&cam, mk, spec.a, &spec.cam_model, spec.sigma_px, spec.noise_model, None, &[Axis::Z]);
            info.extend_from_slice(m.icc.as_slice());
        }
    }
    Ok(Tables { cells: c, k, hd: spec.ws.headings.len(), n, vis, info })
}

/// `np.einsum('kj,kjab->kab')` for one cell: sequential left-to-right sum over `jj` (measured bit-identical to numpy on
/// the demo scene for 4, 5, 7, 8 and 10 markers; the reversed order is not).
fn sum_weighted(w: &[f64], blocks: &[&[f64]]) -> [f64; 16] {
    let mut acc = [0.0; 16];
    for (wj, b) in w.iter().zip(blocks) {
        for (a, x) in acc.iter_mut().zip(b.iter()) {
            *a += wj * x;
        }
    }
    acc
}

fn dmat4(v: &[f64; 16]) -> DMat {
    DMat::from_rows(4, 4, v)
}

/// Cell statistics of the selection `sel`.
///
/// # Errors
/// `PlaceError::InvalidInput` when an index of `sel` is not a candidate of `tb` (`>= tb.n`).
pub fn evaluate(tb: &Tables, sel: &[usize], spec: &PlacementSpec) -> Result<CellStats, PlaceError> {
    if let Some(&j) = sel.iter().find(|&&j| j >= tb.n) {
        return Err(PlaceError::InvalidInput(format!("evaluate: candidate index {j} out of range (N = {})", tb.n)));
    }
    let (k, hd) = (tb.k, tb.hd);
    let cp = spec.sigma_cap;
    let (sigma, n_vis, sigma_loo);
    if sel.is_empty() {
        sigma = vec![cp; k];
        n_vis = vec![0.0; k];
        sigma_loo = vec![cp; k];
    } else {
        let mut sig_h = vec![vec![0.0; k]; hd];
        let mut nv_h = vec![vec![0.0; k]; hd];
        let mut loo_h = vec![vec![0.0; k]; hd];
        // per-(heading, cell) scratch, hoisted out of the hot loop
        let mut w: Vec<f64> = Vec::with_capacity(sel.len());
        let mut blocks: Vec<&[f64]> = Vec::with_capacity(sel.len());
        for h in 0..hd {
            for ki in 0..k {
                w.clear();
                w.extend(sel.iter().map(|&j| if tb.vis_at(ki, h, j) { 1.0 } else { 0.0 }));
                blocks.clear();
                blocks.extend(sel.iter().map(|&j| tb.info_at(ki, j)));
                let ih = sum_weighted(&w, &blocks);
                sig_h[h][ki] = cap(sigma_pos(&dmat4(&ih), SIGMA_RTOL)?, cp);
                let nv = np_sum(&w);
                nv_h[h][ki] = nv;
                let mut loo = 0.0;
                for jj in 0..sel.len() {
                    if w[jj] > 0.0 {
                        let mut d = ih;
                        for (x, y) in d.iter_mut().zip(blocks[jj]) {
                            *x -= y;
                        }
                        loo = nmax(loo, cap(sigma_pos(&dmat4(&d), SIGMA_RTOL)?, cp));
                    }
                }
                if nv == 0.0 {
                    loo = cp;
                }
                loo_h[h][ki] = loo;
            }
        }
        let fold = |hs: &Vec<Vec<f64>>, f: fn(f64, f64) -> f64| -> Vec<f64> {
            (0..k).map(|ki| hs[1..].iter().fold(hs[0][ki], |m, row| f(m, row[ki]))).collect()
        };
        sigma = fold(&sig_h, nmax);
        n_vis = fold(&nv_h, nmin);
        sigma_loo = fold(&loo_h, nmax);
    }
    let tl = spec.tau_loo_eff();
    let feasible = (0..k).map(|i| sigma[i] <= spec.tau && n_vis[i] >= spec.k_min as f64 && sigma_loo[i] <= tl).collect();
    let margin = (0..k).map(|i| nmin(1.0 - sigma[i] / spec.tau, 1.0 - sigma_loo[i] / tl)).collect();
    Ok(CellStats { sigma, n_vis, sigma_loo, feasible, margin })
}

pub fn penalty_of(st: &CellStats, spec: &PlacementSpec) -> Vec<f64> {
    let tl = spec.tau_loo_eff();
    (0..st.sigma.len())
        .map(|i| {
            nmax(0.0, st.sigma[i] / spec.tau - 1.0)
                + nmax(0.0, st.sigma_loo[i] / tl - 1.0)
                + spec.w_k * nmax(0.0, spec.k_min as f64 - st.n_vis[i])
        })
        .collect()
}

pub fn key2(st: &CellStats, spec: &PlacementSpec) -> Key2 {
    Key2 { n_infeasible: st.feasible.iter().filter(|&&f| !f).count(), penalty_sum: np_sum(&penalty_of(st, spec)) }
}

pub fn key3(st: &CellStats, spec: &PlacementSpec) -> Key3 {
    let k2 = key2(st, spec);
    Key3 { n_infeasible: k2.n_infeasible, penalty_sum: k2.penalty_sum, neg_margin: -red_min(&st.margin) }
}

/// Lexicographic `itertools.combinations(range(n), k)`.
pub struct Combinations {
    n: usize,
    idx: Vec<usize>,
    first: bool,
    done: bool,
}

pub fn combinations(n: usize, k: usize) -> Combinations {
    Combinations { n, idx: (0..k).collect(), first: true, done: k > n }
}

impl Iterator for Combinations {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done {
            return None;
        }
        let k = self.idx.len();
        if self.first {
            self.first = false;
            return Some(self.idx.clone());
        }
        let mut i = k;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.idx[i] != i + self.n - k {
                break;
            }
        }
        self.idx[i] += 1;
        for j in i + 1..k {
            self.idx[j] = self.idx[j - 1] + 1;
        }
        Some(self.idx.clone())
    }
}

/// Greedy forward selection by `key2`; stops when every cell is feasible or `n_max` is reached.
pub fn greedy(tb: &Tables, spec: &PlacementSpec) -> Result<(Vec<usize>, Vec<HistoryStep>), PlaceError> {
    let mut sel: Vec<usize> = Vec::new();
    let mut history = Vec::new();
    while sel.len() < spec.n_max {
        let mut best: Option<(Key2, usize)> = None;
        for j in 0..tb.n {
            if sel.contains(&j) {
                continue;
            }
            let mut t = sel.clone();
            t.push(j);
            let key = key2(&evaluate(tb, &t, spec)?, spec);
            if best.as_ref().map_or(true, |b| key.lt(&b.0)) {
                best = Some((key, j));
            }
        }
        let Some((key, j)) = best else { break };
        sel.push(j);
        history.push(HistoryStep::Greedy { sel: sel.clone(), n_infeasible: key.n_infeasible, penalty: key.penalty_sum });
        if key.n_infeasible == 0 {
            break;
        }
    }
    Ok((sel, history))
}

/// First-improvement 1-swap on `key3` (position kept), sweeps restart after every accepted swap.
///
/// # Errors
/// `PlaceError::InvalidInput` when an index of `sel` is out of range (see [`evaluate`]).
pub fn local_search_1swap(tb: &Tables, spec: &PlacementSpec, sel: &[usize]) -> Result<(Vec<usize>, Vec<HistoryStep>), PlaceError> {
    let mut sel = sel.to_vec();
    let mut history = Vec::new();
    let mut cur = key3(&evaluate(tb, &sel, spec)?, spec);
    loop {
        let mut improved = false;
        'sweep: for i in 0..sel.len() {
            for j in 0..tb.n {
                if sel.contains(&j) {
                    continue;
                }
                let mut trial = sel.clone();
                trial[i] = j;
                let key = key3(&evaluate(tb, &trial, spec)?, spec);
                if key.lt(&cur) {
                    sel = trial;
                    cur = key;
                    history.push(HistoryStep::OneSwap {
                        sel: sel.clone(),
                        n_infeasible: key.n_infeasible,
                        penalty: key.penalty_sum,
                        neg_margin: key.neg_margin,
                    });
                    improved = true;
                    break 'sweep;
                }
            }
        }
        if !improved {
            break;
        }
    }
    Ok((sel, history))
}

/// Remove markers one at a time while every cell stays feasible (best resulting min margin first).
///
/// # Errors
/// `PlaceError::InvalidInput` when an index of `sel` is out of range (see [`evaluate`]).
pub fn prune(tb: &Tables, spec: &PlacementSpec, sel: &[usize]) -> Result<(Vec<usize>, Vec<HistoryStep>), PlaceError> {
    let mut sel = sel.to_vec();
    let mut history = Vec::new();
    if !evaluate(tb, &sel, spec)?.feasible.iter().all(|&f| f) {
        return Ok((sel, history));
    }
    while sel.len() > 1 {
        let mut best: Option<(f64, Vec<usize>)> = None;
        for i in 0..sel.len() {
            let trial: Vec<usize> = sel.iter().enumerate().filter(|&(k, _)| k != i).map(|(_, &m)| m).collect();
            let st = evaluate(tb, &trial, spec)?;
            if st.feasible.iter().all(|&f| f) {
                let mm = red_min(&st.margin);
                if best.as_ref().map_or(true, |b| mm > b.0) {
                    best = Some((mm, trial));
                }
            }
        }
        let Some((mm, trial)) = best else { break };
        sel = trial;
        history.push(HistoryStep::Prune { sel: sel.clone(), min_margin: mm });
    }
    Ok((sel, history))
}

fn red_max(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |m, &x| nmax(m, x))
}

/// Exhaustive search (small N only). `Some(k)`: best `(n_infeasible, max capped sigma)` over k-subsets.
/// `None`: smallest size with an all-feasible subset (min max sigma), else best `n_max`-subset.
///
/// # Errors
/// `PlaceError::InvalidInput` when `k` (or `spec.n_max` for `None` without an all-feasible subset) is larger than the number of
/// candidates, so that no subset exists.
pub fn brute_force(tb: &Tables, spec: &PlacementSpec, k: Option<usize>) -> Result<Vec<usize>, PlaceError> {
    let n = tb.n;
    if let Some(k) = k {
        if k > n {
            return Err(PlaceError::InvalidInput(format!("brute_force: subset size {k} exceeds the {n} candidates")));
        }
    }
    let capped_max = |st: &CellStats| red_max(&st.sigma.iter().map(|&s| cap(s, spec.sigma_cap)).collect::<Vec<_>>());
    if let Some(k) = k {
        let mut best: Option<((usize, f64), Vec<usize>)> = None;
        for comb in combinations(n, k) {
            let st = evaluate(tb, &comb, spec)?;
            let key = (st.feasible.iter().filter(|&&f| !f).count(), capped_max(&st));
            let better = best.as_ref().map_or(true, |b| key.0 < b.0 .0 || (key.0 == b.0 .0 && key.1 < b.0 .1));
            if better {
                best = Some((key, comb));
            }
        }
        return Ok(best.ok_or_else(|| PlaceError::InvalidInput("brute_force: no combinations".into()))?.1);
    }
    for kk in 1..=spec.n_max {
        let mut best: Option<(f64, Vec<usize>)> = None;
        for comb in combinations(n, kk) {
            let st = evaluate(tb, &comb, spec)?;
            if st.feasible.iter().all(|&f| f) {
                let v = capped_max(&st);
                if best.as_ref().map_or(true, |b| v < b.0) {
                    best = Some((v, comb));
                }
            }
        }
        if let Some((_, c)) = best {
            return Ok(c);
        }
    }
    brute_force(tb, spec, Some(spec.n_max))
}

/// Virtual-frame blocks for the survey search (`_frame_blocks`).
#[derive(Clone, Debug)]
pub struct FrameBlocks {
    pub icc: Vec<DMat>,
    pub icm: Vec<DMat>,
    pub imm: DMat,
    pub col: BTreeMap<usize, usize>,
    pub n_skip: usize,
}

pub fn frame_blocks(tb: &Tables, spec: &PlacementSpec, sel: &[usize], mks: &[MarkerPose], rcond: f64) -> Result<FrameBlocks, PlaceError> {
    if tb.hd != spec.ws.headings.len() {
        return Err(PlaceError::HeadingMismatch { tables: tb.hd, spec: spec.ws.headings.len() });
    }
    let n = sel.len();
    let col: BTreeMap<usize, usize> = sel.iter().enumerate().map(|(i, &m)| (m, 6 * i)).collect();
    let mut icc_b = Vec::new();
    let mut icm_b = Vec::new();
    let mut imm = vec![0.0; 36 * n * n];
    let mut n_skip = 0;
    for k in 0..tb.k {
        for h in 0..tb.hd {
            let vis_m: Vec<usize> = sel.iter().copied().filter(|&m| tb.vis_at(k, h, m)).collect();
            if vis_m.is_empty() {
                continue;
            }
            let cam = CamPose::euler(tb.cells[k], spec.ws.headings[h], 0.0, 0.0);
            let mut i_cc = vec![0.0; 16];
            let mut i_cm = vec![0.0; 4 * 6 * n];
            let mut blk: Vec<(usize, DMat)> = Vec::new();
            for &m in &vis_m {
                let mi = info_marker(&cam, &mks[m], spec.a, &spec.cam_model, spec.sigma_px, spec.noise_model, None, &[Axis::Z]);
                for (x, y) in i_cc.iter_mut().zip(mi.icc.as_slice()) {
                    *x += y;
                }
                let c0 = col[&m];
                for r in 0..4 {
                    for c in 0..6 {
                        i_cm[r * 6 * n + c0 + c] += mi.icm.as_slice()[r * 6 + c];
                    }
                }
                blk.push((c0, mi.imm));
            }
            let icc = DMat::from_rows(4, 4, &i_cc);
            let sym = icc.add(&icc.transpose()).scale(0.5);
            let w = eigvalsh(&sym)?;
            let (w0, wl) = (w[0], w[w.len() - 1]);
            if wl <= 0.0 || w0 <= rcond * wl {
                n_skip += 1;
                continue;
            }
            for (c0, m) in &blk {
                for r in 0..6 {
                    for c in 0..6 {
                        let o = (c0 + r) * 6 * n + c0 + c;
                        imm[o] += m.as_slice()[r * 6 + c];
                    }
                }
            }
            icc_b.push(icc);
            icm_b.push(DMat::from_rows(4, 6 * n, &i_cm));
        }
    }
    Ok(FrameBlocks { icc: icc_b, icm: icm_b, imm: DMat::from_rows(6 * n, 6 * n, &imm), col, n_skip })
}

/// Survey triplet: minimises the worst `sqrt(tr(S_T^-1[p_m]))` over unsurveyed markers (ties: larger lambda_min).
/// Returns `(ids, score, n_skip)`. Errors with `HeadingMismatch` when `tb.hd` differs from the spec's heading count (site tables).
pub fn choose_survey_triplet(
    tb: &Tables,
    spec: &PlacementSpec,
    sel: &[usize],
    sigma_ts: f64,
) -> Result<(Vec<usize>, f64, usize), PlaceError> {
    if tb.hd != spec.ws.headings.len() {
        return Err(PlaceError::HeadingMismatch { tables: tb.hd, spec: spec.ws.headings.len() });
    }
    if sel.len() < 3 {
        return Ok((sel.to_vec(), f64::INFINITY, 0));
    }
    let mks = markers_from_candidates(&spec.candidates);
    let fb = frame_blocks(tb, spec, sel, &mks, 1e-9)?;
    if fb.icc.is_empty() {
        return Ok((sel[..3].to_vec(), f64::INFINITY, fb.n_skip));
    }
    let s_mm = schur_markers(&fb.icc, &fb.icm, &fb.imm)?;
    if s_mm.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(PlaceError::NonFiniteSchur);
    }
    let dim = s_mm.rows();
    let prior = 1.0 / (sigma_ts * sigma_ts);
    let mut best: Option<((f64, f64), Vec<usize>)> = None;
    for tc in combinations(sel.len(), 3) {
        let t: Vec<usize> = tc.iter().map(|&i| sel[i]).collect();
        let mut sv = s_mm.as_slice().to_vec();
        for m in &t {
            let c0 = fb.col[m];
            for d in 0..3 {
                sv[(c0 + d) * dim + c0 + d] += prior;
            }
        }
        let s = DMat::from_rows(dim, dim, &sv);
        let sinv = match lu::inv(&s) {
            Ok(x) => x,
            Err(LinalgError::Singular) => continue,
            Err(e) => return Err(e.into()),
        };
        let sym = s.add(&s.transpose()).scale(0.5);
        let lam_min = eigvalsh(&sym)?[0];
        let mut score = 0.0;
        let mut first = true;
        for &m in sel.iter().filter(|m| !t.contains(m)) {
            let c0 = fb.col[&m];
            let si = sinv.as_slice();
            let tr = si[c0 * dim + c0] + si[(c0 + 1) * dim + c0 + 1] + si[(c0 + 2) * dim + c0 + 2];
            let v = if tr > 0.0 { tr.sqrt() } else { f64::INFINITY };
            score = if first { v } else { nmax(score, v) };
            first = false;
        }
        let key = (score, -lam_min);
        let better = best.as_ref().map_or(true, |b| key.0 < b.0 .0 || (key.0 == b.0 .0 && key.1 < b.0 .1));
        if better {
            best = Some((key, t));
        }
    }
    match best {
        Some((key, t)) => Ok((t, key.0, fb.n_skip)),
        None => Ok((sel[..3].to_vec(), f64::INFINITY, fb.n_skip)),
    }
}

/// precompute -> greedy -> 1-swap -> prune -> survey triplet.
///
/// # Errors
/// `PlaceError::InvalidInput` for an empty workspace polygon, headings, `z_cam` or candidate list, or a spacing that is not positive and
/// finite.
pub fn optimize(spec: &PlacementSpec, sigma_ts: f64) -> Result<PlacementResult, PlaceError> {
    if spec.candidates.is_empty() {
        return Err(PlaceError::InvalidInput("optimize: no candidates".into()));
    }
    let tb = precompute(spec)?;
    let (sel, h1) = greedy(&tb, spec)?;
    let (sel, h2) = local_search_1swap(&tb, spec, &sel)?;
    let (sel, h3) = prune(&tb, spec, &sel)?;
    let stats = evaluate(&tb, &sel, spec)?;
    let (ids, score, n_skip) = choose_survey_triplet(&tb, spec, &sel, sigma_ts)?;
    let mut history = h1;
    history.extend(h2);
    history.extend(h3);
    history.push(HistoryStep::Survey { ids: ids.clone(), score, n_skip });
    Ok(PlacementResult { sel, stats, survey_ids: ids, history })
}

/// Text of `placement.csv` (the caller writes the file).
pub fn report_csv(result: &PlacementResult, spec: &PlacementSpec) -> String {
    let mut s = String::from("id,x,y,z,azimuth_deg,surveyed\n");
    for &m in &result.sel {
        let cd = &spec.candidates[m];
        let surveyed = result.survey_ids.contains(&m);
        s.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:.2},{}\n",
            m,
            cd.xy[0],
            cd.xy[1],
            cd.z,
            cd.azimuth * (180.0 / std::f64::consts::PI),
            if surveyed { 1 } else { 0 }
        ));
    }
    s
}

const ROOM: [[f64; 2]; 4] = [[0., 0.], [8., 0.], [8., 6.], [0., 6.]];
const WORK: [[f64; 2]; 4] = [[1., 1.], [7., 1.], [7., 5.], [1., 5.]];
const BOX: [[f64; 2]; 4] = [[3., 2.5], [5., 2.5], [5., 3.5], [3., 3.5]];

/// demo scene (Python `demo_spec`).
pub fn demo_spec(n_max: usize) -> PlacementSpec {
    let ws = Workspace {
        poly: WORK.to_vec(),
        spacing: 0.5,
        z_cam: vec![1.2],
        headings: (0..4).map(|k| std::f64::consts::PI * k as f64 / 2.0).collect(),
    };
    let candidates =
        candidates_from_walls(&[Obstacle { poly: ROOM.to_vec(), height: 3.0 }], 3.0, 1.5, 0.05, true).expect("demo room walls");
    PlacementSpec {
        ws,
        obstacles: vec![Obstacle { poly: BOX.to_vec(), height: 1.5 }],
        candidates,
        n_max,
        a: 0.16,
        cam_model: ErpCamera::new(6528, 3264, 1),
        sigma_px: 0.3,
        noise_model: NoiseModel::Tangent,
        vp: VisParams::new(70.0_f64.to_radians(), 12.0),
        tau: 0.02,
        k_min: 2,
        tau_loo: Some(0.5),
        sigma_cap: 10.0,
        w_k: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_est::fisher::info_camera;
    use erploc_linalg::rng::Pcg64;
    use std::error::Error;

    #[test]
    fn display_nonempty() {
        let all = [
            PlaceError::Linalg(LinalgError::Singular),
            PlaceError::NonFiniteSchur,
            PlaceError::Site("bad magic".into()),
            PlaceError::NotHeadingIndependent,
            PlaceError::HeadingMismatch { tables: 1, spec: 8 },
        ];
        for e in &all {
            assert!(!e.to_string().is_empty(), "{e:?}");
        }
        assert!(all[2].to_string().contains("bad magic"));
        let h = all[4].to_string();
        assert!(h.contains('1') && h.contains('8'), "{h}");
        assert_eq!(all[0].source().unwrap().to_string(), LinalgError::Singular.to_string());
        assert!(all[1].source().is_none());
    }

    const GREEDY: [usize; 4] = [6, 7, 1, 2];
    /// x -> 8-x mirror of `GREEDY` (5<->7, 0<->2).
    const GREEDY_MIRROR: [usize; 4] = [6, 5, 1, 0];

    fn tables() -> (PlacementSpec, Tables) {
        let spec = demo_spec(10);
        let tb = precompute(&spec).unwrap();
        (spec, tb)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn demo_tables() {
        let (_, tb) = tables();
        assert_eq!((tb.k, tb.hd, tb.n), (88, 4, 10));
        assert_eq!(tb.vis.iter().filter(|&&v| v).count(), 2416);
        assert_eq!(2416.0 / 3520.0, 0.6863636363636364);
        assert_eq!(tb.info.len(), 88 * 10 * 16);
        assert!(tb.info.iter().all(|x| x.is_finite()));
        assert_eq!(tb.cells.len(), 88);
        assert_eq!(tb.heading_independent(), (0..88).all(|k| (0..10).all(|j| (0..4).all(|h| tb.vis_at(k, h, j) == tb.vis_at(k, 0, j)))));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_selection_stats() {
        let (spec, tb) = tables();
        let st = evaluate(&tb, &[], &spec).unwrap();
        let want = nmin(1.0 - 10.0 / spec.tau, 1.0 - 10.0 / 0.5);
        assert!(st.sigma.iter().all(|&s| s == 10.0));
        assert!(st.sigma_loo.iter().all(|&s| s == 10.0));
        assert!(st.n_vis.iter().all(|&s| s == 0.0));
        assert!(st.feasible.iter().all(|&f| !f));
        assert!(st.margin.iter().all(|&m| m == want));
    }

    #[test]
    fn key_lexicographic() {
        let k2 = |a, b| Key2 { n_infeasible: a, penalty_sum: b };
        assert!(k2(1, 5.0).lt(&k2(2, 0.0)));
        assert!(!k2(2, 0.0).lt(&k2(1, 5.0)));
        assert!(k2(2, 1.0).lt(&k2(2, 2.0)));
        assert!(!k2(2, 2.0).lt(&k2(2, 2.0)));
        assert!(!k2(2, f64::NAN).lt(&k2(2, 1.0)));
        assert!(!k2(2, 1.0).lt(&k2(2, f64::NAN)));
        assert!(k2(1, f64::NAN).lt(&k2(2, f64::NAN)));
        let k3 = |a, b, c| Key3 { n_infeasible: a, penalty_sum: b, neg_margin: c };
        assert!(k3(0, 1.0, 9.0).lt(&k3(1, 0.0, 0.0)));
        assert!(k3(1, 1.0, 9.0).lt(&k3(1, 2.0, 0.0)));
        assert!(k3(1, 1.0, -1.0).lt(&k3(1, 1.0, 0.0)));
        assert!(!k3(1, 1.0, 0.0).lt(&k3(1, 1.0, 0.0)));
        assert!(!k3(1, 1.0, f64::NAN).lt(&k3(1, 1.0, 0.0)));
        assert!(!k3(1, 1.0, 0.0).lt(&k3(1, 1.0, f64::NAN)));
        assert!(!k3(1, f64::NAN, 0.0).lt(&k3(1, f64::NAN, 1.0)));
        assert!(!k3(1, f64::NAN, 0.0).lt(&k3(1, 1.0, 1.0)));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn greedy_selection_key() {
        let (spec, tb) = tables();
        let st = evaluate(&tb, &GREEDY, &spec).unwrap();
        let k = key3(&st, &spec);
        assert_eq!(k.n_infeasible, 0);
        assert!(k.penalty_sum.abs() < 1e-9, "{}", k.penalty_sum);
        assert!((k.neg_margin - (-0.4451234311772042)).abs() < 1e-9, "{}", k.neg_margin);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn penalty_formula() {
        let mut spec = demo_spec(10);
        spec.tau = 0.02;
        spec.tau_loo = Some(0.5);
        spec.k_min = 2;
        spec.w_k = 1.0;
        let st = CellStats {
            sigma: vec![0.04, 0.01, 0.01],
            sigma_loo: vec![0.1, 1.0, 0.1],
            n_vis: vec![3.0, 3.0, 0.0],
            feasible: vec![false; 3],
            margin: vec![0.0; 3],
        };
        let p = penalty_of(&st, &spec);
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!((p[1] - 1.0).abs() < 1e-12);
        assert!((p[2] - 2.0).abs() < 1e-12);
        spec.tau_loo = None; // falls back to tau = 0.02
        assert_eq!(spec.tau_loo_eff(), 0.02);
        let p = penalty_of(&st, &spec);
        assert!((p[0] - (1.0 + 4.0)).abs() < 1e-12);
        let nan = CellStats { sigma: vec![f64::NAN], sigma_loo: vec![0.0], n_vis: vec![3.0], feasible: vec![false], margin: vec![0.0] };
        assert!(penalty_of(&nan, &spec)[0].is_nan());
        let k = key2(&st, &spec);
        assert_eq!(k.n_infeasible, 3);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn python_key_parity() {
        // key3 values recorded from Python (demo_spec, numpy on arm64); count exact, floats to 1e-9 relative.
        let (spec, tb) = tables();
        let cases: [(&[usize], usize, f64, f64); 3] = [
            (&[6, 7, 1, 2], 0, 0.0, -0.4451234311772042),
            (&[0, 3, 5], 28, 422.84125067191536, 28.86444958414683),
            (&[4], 88, 14567.319478123498, 499.0),
        ];
        for (sel, n, pen, nm) in cases {
            let k = key3(&evaluate(&tb, sel, &spec).unwrap(), &spec);
            assert_eq!(k.n_infeasible, n);
            // Not bit-exact: the I table differs from numpy at the last ulp in places, amplified by the
            // ill-conditioned leave-one-out inverses (observed relative gaps <= 5e-11).
            assert!((k.penalty_sum - pen).abs() <= 1e-9 * pen.abs().max(1.0), "{sel:?} penalty {}", k.penalty_sum);
            assert!((k.neg_margin - nm).abs() <= 1e-9 * nm.abs().max(1.0), "{sel:?} margin {}", k.neg_margin);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cap_semantics() {
        assert_eq!(cap(f64::NAN, 10.0), 10.0);
        assert_eq!(cap(f64::INFINITY, 10.0), 10.0);
        assert_eq!(cap(3.0, 10.0), 3.0);
        assert_eq!(cap(30.0, 10.0), 10.0);
        assert_eq!(demo_spec(3).with_n_max(7).n_max, 7);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn additivity_yaw_invariance() {
        let (spec, tb) = tables();
        let mks = markers_from_candidates(&spec.candidates);
        let mut rng = Pcg64::new(0);
        let mut idx: Vec<usize> = Vec::new();
        while idx.len() < 5 {
            let i = (rng.next_u64() % tb.k as u64) as usize;
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        let mut worst = 0.0f64;
        let mut checked = 0;
        for &k in &idx {
            for h in 0..tb.hd {
                let vis: Vec<MarkerPose> = GREEDY.iter().filter(|&&m| tb.vis_at(k, h, m)).map(|&m| mks[m]).collect();
                if vis.is_empty() {
                    continue;
                }
                let cam = CamPose::euler(tb.cells[k], spec.ws.headings[h], 0.0, 0.0);
                let iref = info_camera(&cam, &vis, spec.a, &spec.cam_model, spec.sigma_px, spec.noise_model, &[Axis::Z]);
                let w: Vec<f64> = GREEDY.iter().map(|&m| if tb.vis_at(k, h, m) { 1.0 } else { 0.0 }).collect();
                let blocks: Vec<&[f64]> = GREEDY.iter().map(|&m| tb.info_at(k, m)).collect();
                let itab = dmat4(&sum_weighted(&w, &blocks));
                let d = (sigma_pos(&iref, SIGMA_RTOL).unwrap() - sigma_pos(&itab, SIGMA_RTOL).unwrap()).abs();
                assert!(
                    d.is_finite()
                        || (sigma_pos(&iref, SIGMA_RTOL).unwrap().is_infinite() && sigma_pos(&itab, SIGMA_RTOL).unwrap().is_infinite())
                );
                if d.is_finite() {
                    worst = worst.max(d);
                }
                checked += 1;
            }
        }
        assert!(checked > 0);
        assert!(worst < 1e-9, "max dsigma = {worst:e}");
    }

    // ---- edge cases ----

    #[test]
    #[cfg_attr(miri, ignore)]
    fn penalty_weight_on_nvis_term() {
        let mut spec = demo_spec(10);
        spec.w_k = 2.5;
        let st = CellStats {
            sigma: vec![0.01, 0.01],
            sigma_loo: vec![0.1, 0.1],
            n_vis: vec![0.0, 1.0],
            feasible: vec![false; 2],
            margin: vec![0.0; 2],
        };
        let p = penalty_of(&st, &spec);
        assert!((p[0] - 2.5 * 2.0).abs() < 1e-12, "{}", p[0]);
        assert!((p[1] - 2.5 * 1.0).abs() < 1e-12, "{}", p[1]);
    }

    #[test]
    fn np_sum_88_bitwise_vs_naive() {
        // 8-lane pairwise reference written independently of npcompat for n = 88 (n < 128).
        let mut v = vec![0.0f64; 88];
        for (i, x) in v.iter_mut().enumerate() {
            *x = if i % 8 == 0 {
                1e16
            } else if i % 8 == 1 {
                -1e16
            } else {
                1.0 + 0.1 * i as f64
            };
        }
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&v[..8]);
        for c in 1..11 {
            for l in 0..8 {
                r[l] += v[c * 8 + l];
            }
        }
        let want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        let naive: f64 = v.iter().fold(0.0, |a, &x| a + x);
        assert_eq!(np_sum(&v).to_bits(), want.to_bits());
        assert_ne!(want.to_bits(), naive.to_bits(), "crafted data must separate pairwise from naive");
    }

    #[test]
    fn sum_weighted_is_left_to_right() {
        let b0 = [1e16; 16];
        let b1 = [1.0; 16];
        let b2 = [1.0; 16];
        let acc = sum_weighted(&[1.0, 1.0, 1.0], &[&b0, &b1, &b2]);
        // left to right: ((1e16 + 1) + 1) = 1e16; reversed would give 1e16 + 2
        assert!(acc.iter().all(|&x| x == 1e16));
        let acc = sum_weighted(&[1.0, 0.0, 1.0], &[&b1, &b0, &b2]);
        assert!(acc.iter().all(|&x| x == 2.0));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn heading_independent_false_with_sectors() {
        use crate::visibility::Sector;
        let (_, tb) = tables();
        assert!(tb.heading_independent());
        let mut spec = demo_spec(10);
        spec.vp.sectors = vec![Sector { lam_min: -0.5, lam_max: 0.5, phi_min: -1.5, phi_max: 1.5 }];
        let tb2 = precompute(&spec).unwrap();
        assert!(!tb2.heading_independent());
    }

    // ---- search and survey ----

    fn sub_tables(tb: &Tables, m: usize) -> Tables {
        let mut vis = Vec::new();
        let mut info = Vec::new();
        for k in 0..tb.k {
            for h in 0..tb.hd {
                for j in 0..m {
                    vis.push(tb.vis_at(k, h, j));
                }
            }
            for j in 0..m {
                info.extend_from_slice(tb.info_at(k, j));
            }
        }
        Tables { cells: tb.cells.clone(), k: tb.k, hd: tb.hd, n: m, vis, info }
    }

    #[test]
    fn combinations_lexicographic() {
        let c: Vec<Vec<usize>> = combinations(4, 2).collect();
        assert_eq!(c, vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3]]);
        assert_eq!(combinations(5, 3).count(), 10);
        assert_eq!(combinations(3, 0).count(), 1);
        assert_eq!(combinations(2, 3).count(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn greedy_demo() {
        let (spec, tb) = tables();
        let (sel, h) = greedy(&tb, &spec).unwrap();
        // Candidates 5<->7 and 0<->2 are mirror images (x -> 8-x). Their key2 values tie to ~1e-13 relative, so which
        // one a strict `<` keeps is decided by last-ulp noise: Python picks 7 (and 2), this port picks 5 (and 0).
        if erploc_testkit::strict() {
            assert!(sel == GREEDY || sel == GREEDY_MIRROR, "{sel:?}");
        }
        // another libm may settle a tie of the key on another layout: only the key of every step is compared there (below)
        assert_eq!(h.len(), 4);
        let want = [(88usize, 13869.546674904634), (47, 8919.43427380697), (20, 409.9381944430356), (0, 0.0)];
        for (st, (ninf, pen)) in h.iter().zip(want) {
            let HistoryStep::Greedy { n_infeasible, penalty, .. } = st else { panic!() };
            assert_eq!(*n_infeasible, ninf);
            assert!((penalty - pen).abs() <= 1e-9 * pen.max(1.0), "{penalty} vs {pen}");
        }
        match h.last().unwrap() {
            HistoryStep::Greedy { n_infeasible, .. } => assert_eq!(*n_infeasible, 0),
            _ => panic!(),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn oneswap_never_worse() {
        let (spec, tb) = tables();
        for start in [vec![0usize, 3, 5], vec![4, 9, 8, 0]] {
            let k0 = key3(&evaluate(&tb, &start, &spec).unwrap(), &spec);
            let (out, hist) = local_search_1swap(&tb, &spec, &start).unwrap();
            assert_eq!(out.len(), start.len());
            let k1 = key3(&evaluate(&tb, &out, &spec).unwrap(), &spec);
            assert!(!k0.lt(&k1));
            let mut cur = k0;
            for st in &hist {
                match st {
                    HistoryStep::OneSwap { n_infeasible, penalty, neg_margin, .. } => {
                        let k = Key3 { n_infeasible: *n_infeasible, penalty_sum: *penalty, neg_margin: *neg_margin };
                        assert!(k.lt(&cur));
                        cur = k;
                    }
                    _ => panic!(),
                }
            }
        }
        // the greedy result is a local optimum
        let (out, hist) = local_search_1swap(&tb, &spec, &GREEDY).unwrap();
        assert_eq!(out, GREEDY);
        assert!(hist.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prune_keeps_feasible() {
        let (spec, tb) = tables();
        // start from a feasible superset
        let start = vec![6usize, 7, 1, 2, 0, 3];
        assert!(evaluate(&tb, &start, &spec).unwrap().feasible.iter().all(|&f| f));
        let (out, hist) = prune(&tb, &spec, &start).unwrap();
        assert!(out.len() < start.len());
        assert!(evaluate(&tb, &out, &spec).unwrap().feasible.iter().all(|&f| f));
        // independent re-derivation of every step
        let mut cur = start.clone();
        for st in &hist {
            let HistoryStep::Prune { sel, min_margin } = st else { panic!() };
            let mut best: Option<(f64, usize)> = None;
            for i in 0..cur.len() {
                let mut t = cur.clone();
                t.remove(i);
                let s = evaluate(&tb, &t, &spec).unwrap();
                if s.feasible.iter().all(|&f| f) {
                    let mm = s.margin.iter().cloned().fold(f64::INFINITY, f64::min);
                    if best.map_or(true, |b| mm > b.0) {
                        best = Some((mm, i));
                    }
                }
            }
            let (mm, i) = best.unwrap();
            cur.remove(i);
            assert_eq!(&cur, sel);
            assert_eq!(mm, *min_margin);
        }
        assert_eq!(cur, out);
        // no further single removal is feasible
        for i in 0..out.len() {
            let mut t = out.clone();
            t.remove(i);
            assert!(out.len() == 1 || !evaluate(&tb, &t, &spec).unwrap().feasible.iter().all(|&f| f));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prune_noop_when_infeasible() {
        let (spec, tb) = tables();
        let (out, hist) = prune(&tb, &spec, &[4]).unwrap();
        assert_eq!(out, vec![4]);
        assert!(hist.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn brute_force_small() {
        let (spec0, tb0) = tables();
        let tb = sub_tables(&tb0, 6);
        let spec = spec0.with_n_max(4);
        let got = brute_force(&tb, &spec, Some(2)).unwrap();
        let mut best: Option<((usize, f64), Vec<usize>)> = None;
        for i in 0..6 {
            for j in i + 1..6 {
                let st = evaluate(&tb, &[i, j], &spec).unwrap();
                let key =
                    (st.feasible.iter().filter(|&&f| !f).count(), st.sigma.iter().map(|&s| cap(s, 10.0)).fold(f64::NEG_INFINITY, f64::max));
                if best.as_ref().map_or(true, |b| key.0 < b.0 .0 || (key.0 == b.0 .0 && key.1 < b.0 .1)) {
                    best = Some((key, vec![i, j]));
                }
            }
        }
        assert_eq!(got, best.unwrap().1);
        // k = None: minimal feasible size, or the n_max best
        let r = brute_force(&tb, &spec, None).unwrap();
        let feas = |s: &[usize]| evaluate(&tb, s, &spec).unwrap().feasible.iter().all(|&f| f);
        if feas(&r) {
            for kk in 1..r.len() {
                assert!(combinations(6, kk).all(|c| !feas(&c)));
            }
        } else {
            assert_eq!(r.len(), 4);
            assert_eq!(r, brute_force(&tb, &spec, Some(4)).unwrap());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn triplet_basic() {
        let (spec, tb) = tables();
        let (ids, score, n_skip) = choose_survey_triplet(&tb, &spec, &GREEDY, 0.005).unwrap();
        assert_eq!(ids.len(), 3);
        let mut u = ids.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), 3);
        assert!(ids.iter().all(|i| GREEDY.contains(i)));
        assert!(score.is_finite());
        assert_eq!(n_skip, 0);
        assert_eq!(ids, vec![6, 7, 2]);
        assert!((score - 0.005974825439844254).abs() <= 1e-9 * 0.006, "{score}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn triplet_short_selection() {
        let (spec, tb) = tables();
        let (ids, score, n) = choose_survey_triplet(&tb, &spec, &[6, 7], 0.005).unwrap();
        assert_eq!((ids, n), (vec![6, 7], 0));
        assert!(score.is_infinite());
        let (ids, score, _) = choose_survey_triplet(&tb, &spec, &[6, 7, 2], 0.005).unwrap();
        assert_eq!(ids, vec![6, 7, 2]);
        assert_eq!(score, 0.0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn survey_rejects_heading_mismatch() {
        let (spec, tb0) = tables();
        let hs = spec.ws.headings.len();
        assert!(hs > 1 && tb0.hd == hs);
        // heading-collapsed tables (as the site tables, hd = 1) against the multi-heading spec
        let tb = Tables::from_raw(tb0.cells.clone(), 1, tb0.n, vec![true; tb0.k * tb0.n], tb0.info.clone());
        let want = PlaceError::HeadingMismatch { tables: 1, spec: hs };
        let mks = markers_from_candidates(&spec.candidates);
        match frame_blocks(&tb, &spec, &[6, 7, 1], &mks, 1e-9) {
            Err(e) => assert_eq!(format!("{e:?}"), format!("{want:?}")),
            Ok(_) => panic!("frame_blocks accepted heading-collapsed tables"),
        }
        match choose_survey_triplet(&tb, &spec, &[6, 7, 1], 0.01) {
            Err(e) => assert_eq!(format!("{e:?}"), format!("{want:?}")),
            Ok(_) => panic!("choose_survey_triplet accepted heading-collapsed tables"),
        }
        // the multi-heading tables still work
        assert!(choose_survey_triplet(&tb0, &spec, &[6, 7, 1], 0.01).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn frame_blocks_skip_rule() {
        let (spec, tb0) = tables();
        let mks = markers_from_candidates(&spec.candidates);
        // one cell, one heading, only marker 6 visible
        let mut vis = vec![false; 4 * 10];
        vis[6] = true;
        let tb = Tables::from_raw(vec![tb0.cells[0]], 4, 10, vis, tb0.info[..160].to_vec());
        // a single marker gives a full-rank but poorly conditioned 4-dof frame: with the default rcond it is kept,
        // with rcond = 1 (w0 <= w_max always) it is skipped and contributes nothing
        assert_eq!(frame_blocks(&tb, &spec, &[6, 7, 1], &mks, 1e-9).unwrap().n_skip, 0);
        let fb = frame_blocks(&tb, &spec, &[6, 7, 1], &mks, 1.0).unwrap();
        assert_eq!(fb.n_skip, 1);
        assert!(fb.icc.is_empty() && fb.icm.is_empty());
        assert!(fb.imm.as_slice().iter().all(|&x| x == 0.0));
        assert_eq!(fb.col.get(&7), Some(&6));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn report_csv_format() {
        let (spec, tb) = tables();
        let sel = GREEDY.to_vec();
        let stats = evaluate(&tb, &sel, &spec).unwrap();
        let res = PlacementResult { sel: sel.clone(), stats, survey_ids: vec![6, 7, 2], history: vec![] };
        let csv = report_csv(&res, &spec);
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "id,x,y,z,azimuth_deg,surveyed");
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 4);
        for (r, m) in rows.iter().zip(&sel) {
            let f: Vec<&str> = r.split(',').collect();
            assert_eq!(f.len(), 6);
            assert_eq!(f[0], m.to_string());
            assert_eq!(f[5], if [6, 7, 2].contains(m) { "1" } else { "0" });
        }
    }
}
