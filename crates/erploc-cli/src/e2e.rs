//! End-to-end checks E2E-1..4 (+ the run-time line) of `erploc/e2e.py`, ported to Rust (`erploc e2e`).
//!
//! Pipeline (Python `run_self_checks`): demo room (`demo_spec`, `precompute`), IFT selection and survey ids
//! (`optimize_ift`), test frames = every cell x every heading (352), `render` with the `render_mask` visibility, noise,
//! resection with the Huber loss (`LOSS = 'huber'`, e2e.py:LOSS), bundle adjustment maps, leave-one-marker-out.
//! Thresholds are copied unchanged from `e2e.py` (the line numbers are quoted next to each constant / comparison).
//!
//! KNOWN-FRAGILE: E2E-3 ratio. `RMS(|e|)/RMS(sigma_pos)` in [0.8, 1.25] is unstable across seeds in Python itself: over
//! SEED..SEED+30 the unmodified Python check misses on 9 of 31 seeds (ratio median 0.992, range 0.775-1.931; SEED passes at 1.2304,
//! just under the limit), because ~96 two-marker frames become weakly conditioned one-marker resections whose heavy tail dominates
//! the sum of squares. The Rust distribution matches (KS p = 0.36). The threshold and `run_e2e` / the CLI
//! (exit 1 on a miss) stay faithful; the fixed-seed tests do not assert this ratio, `e2e3_ratio_distribution` does so as a
//! distribution. Left as is: the IFT selection uses `Budget::Wall` (it finishes in ~0.02 s, so the clock does not
//! matter in practice) and the E2E-3 loop builds a `Stats` per resection to read one row.
//!
//! Divergences from Python (all listed here, none retuned):
//! 1. RNG: `Pcg64` is not numpy's `Generator` (rng.rs, "not stream-compatible with numpy by design"), so every noise draw,
//!    BA cell/yaw draw, survey draw and `choice` differs from the Python stream. The statistics are therefore compared by
//!    their thresholds, not draw-by-draw. `rng.integers(0, K)` is `IftRng::integers` (modulo reduction), `rng.uniform` is
//!    `Pcg64::uniform`, `rng.choice(n, size, replace=False)` is `IftRng::choice` (partial Fisher-Yates), `rng.normal` is the
//!    polar-method `Pcg64::normal`.
//! 2. Stream layout: seeds are the Python seeds (`SEED + 1`, `+2`, `+100 + rlz`, `+3000`, `+4000`) but each one seeds a `Pcg64`.
//!    Python's `_build_map` hands `seed` to `default_rng` and the per-frame test noise comes from the same continued stream; the same
//!    structure is kept.
//! 3. A resection that returns `Err(NoConvergence)` (non-finite `J^T J` in the dense LM) has no pose, so it is counted in
//!    `n_nonconv` and has no row in the error statistics (Python's LM never returns an error). Any `LinalgError::NoConvergence` is
//!    counted this way, including a genuine Jacobi sweep-limit failure inside `resect`, where Python would raise. A frame with an `Err` other than
//!    `NoConvergence` aborts the run (`PlaceError`), as an exception aborts Python.
//! 4. Mirror selections of the IFT search are accepted only as tie-equivalent (equal `Key3`); the selection and key are recorded in
//!    the report, and the run uses whatever `optimize_ift` returns.
//! 5. `Cov_map = inv(info)` uses LU `inv`; Python's `np.linalg.inv` is also LU, but a non-finite result is an `Err` here, an
//!    `assert` there.
use erploc_core::erp::NoiseModel;
use erploc_core::jacobian::{Axis, CamPose};
use erploc_core::markers::MarkerPose;
use erploc_est::ba::{bundle_adjust, Layout};
use erploc_est::ba_types::BaProblem;
use erploc_est::fisher::{info_marker, sigma_pos, SIGMA_RTOL};
use erploc_est::lm::Loss;
use erploc_est::resection::{resect, CamDof, ResectError, ResectOptions, ResectionResult};
use erploc_est::synth::{add_noise, render, survey, Observation, SynthConfig};
use erploc_linalg::rng::Pcg64;
use erploc_linalg::{lu, DMat, LinalgError};
use erploc_place::placement::{demo_spec, markers_from_candidates, precompute, PlaceError, PlacementSpec, Tables};
use erploc_place::search::{optimize_ift, Budget, IftRng};
use erploc_place::visibility::render_mask;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::time::Instant;

/// e2e.py:SIGMA_TS.
pub const SIGMA_TS: f64 = 0.005;
/// Huber delta of Python `loss='huber'` (lm.py default, resection/BA).
pub const HUBER: Loss = Loss::Huber { delta: 1.5 };
/// Python `ift.optimize_ift(spec, tbudget=120.0)` (ift.py:576), the budget of the Python run.
pub const PY_IFT_BUDGET_S: f64 = 120.0;
/// e2e.py run-time criterion \[s\] (`elapsed < 90.0`). Python's own time: 39.9 s (e2e_selfcheck.txt).
pub const RUNTIME_LIMIT_S: f64 = 90.0;

/// One line of the result table.
#[derive(Clone, Debug)]
pub struct CheckLine {
    pub name: String,
    pub passed: bool,
    pub measured: String,
    pub criterion: String,
}

/// Constants of e2e.py (defaults = the Python values: SEED, SIGMA_TS, TAU_LOO, N_BA_FRAMES, N_TEST_BA, N_MAP_REALIZATIONS, LOO_FRAC).
#[derive(Clone, Debug)]
pub struct E2eConsts {
    pub seed: u64,
    pub sigma_ts: f64,
    pub tau_loo: f64,
    pub n_ba_frames: usize,
    pub n_test_ba: usize,
    pub n_map_realizations: usize,
    pub loo_frac: f64,
}

impl Default for E2eConsts {
    fn default() -> Self {
        E2eConsts {
            seed: 20260912,
            sigma_ts: SIGMA_TS,
            tau_loo: 0.5,
            n_ba_frames: 60,
            n_test_ba: 40,
            n_map_realizations: 20,
            loo_frac: 0.5,
        }
    }
}

/// Selection, key and timings of one run (for the report).
#[derive(Clone, Debug, Default)]
pub struct E2eInfo {
    pub sel: Vec<usize>,
    pub survey_ids: Vec<usize>,
    pub key: (usize, f64, f64),
    pub t_ift: f64,
    pub t_e1: f64,
    pub t_e2: f64,
    pub t_e3: f64,
    pub t_e4: f64,
    pub t_total: f64,
    /// E2E-3 fraction within `tau_loo` and RMS ratio (for the distribution test).
    pub e3_frac: f64,
    pub e3_ratio: f64,
}

fn site_err(msg: impl Into<String>) -> PlaceError {
    PlaceError::Site(format!("e2e: {}", msg.into()))
}

/// `c.seed + off`, as an error (never a panic or a wrap) when the sum leaves `u64`.
fn seed_at(c: &E2eConsts, off: u64) -> Result<u64, PlaceError> {
    c.seed.checked_add(off).ok_or_else(|| site_err(format!("seed-base {} + {off} overflows u64", c.seed)))
}

/// Checks that every seed `run_e2e` derives from `c.seed` fits in `u64` (the CLI calls this before running).
pub fn check_seed_base(c: &E2eConsts) -> Result<(), PlaceError> {
    let max_off = (100 + (c.n_map_realizations as u64).saturating_sub(1)).max(4000);
    seed_at(c, max_off).map(|_| ())
}

fn resect_err(e: ResectError) -> PlaceError {
    match e {
        ResectError::Linalg(l) => PlaceError::Linalg(l),
        other => site_err(other.to_string()),
    }
}

// ---------------------------------------------------------------- statistics helpers

pub fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// numpy `median` (mean of the two middle values for an even count).
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn norm3(e: &[f64; 3]) -> f64 {
    (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
}

/// Squared Mahalanobis distance `e^T cov^-1 e` of a 3-vector.
pub fn mahalanobis2(e: &[f64; 3], cov3: &DMat) -> Result<f64, LinalgError> {
    let x = lu::solve(cov3, e)?;
    Ok(e[0] * x[0] + e[1] * x[1] + e[2] * x[2])
}

fn top3(m: &DMat) -> DMat {
    DMat::from_fn(3, 3, |i, j| m.as_slice()[i * m.cols() + j])
}

fn trace3(m: &DMat) -> f64 {
    let c = m.cols();
    let s = m.as_slice();
    s[0] + s[c + 1] + s[2 * c + 2]
}

/// Accumulated resection statistics (Python `_resect_stats` outputs).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub e: Vec<[f64; 3]>,
    pub sigma: Vec<f64>,
    pub d2: Vec<f64>,
    pub n_skip: usize,
    pub n_nonconv: usize,
}

impl Stats {
    pub fn e_norms(&self) -> Vec<f64> {
        self.e.iter().map(norm3).collect()
    }

    /// Adds one resection outcome. `Err(NoConvergence)` and `converged == false` both count in `n_nonconv`; only the former has
    /// no pose and therefore no statistics row.
    pub fn add(&mut self, cam_true: &CamPose, res: Result<ResectionResult, ResectError>) -> Result<(), PlaceError> {
        let r = match res {
            Ok(r) => r,
            Err(ResectError::Linalg(LinalgError::NoConvergence)) => {
                self.n_nonconv += 1;
                return Ok(());
            }
            Err(e) => return Err(resect_err(e)),
        };
        self.n_nonconv += usize::from(!r.converged);
        let e = [r.cam.c[0] - cam_true.c[0], r.cam.c[1] - cam_true.c[1], r.cam.c[2] - cam_true.c[2]];
        let cov_pos = top3(&r.cov);
        self.e.push(e);
        self.sigma.push(trace3(&r.cov).sqrt());
        self.d2.push(mahalanobis2(&e, &cov_pos)?);
        Ok(())
    }
}

fn opts(sigma_px: f64, nm: NoiseModel) -> ResectOptions {
    ResectOptions { x0: None, sigma_px, noise_model: nm, loss: HUBER, dof: CamDof::Yaw }
}

/// Python `_resect_stats`: pairs `cams_true[i]` with `frames[i]`; an empty frame is skipped (`n_skip`).
pub fn resect_stats(
    cams_true: &[CamPose],
    frames: &[Vec<Observation>],
    markers: &BTreeMap<i64, MarkerPose>,
    spec: &PlacementSpec,
    sigma_px: f64,
) -> Result<Stats, PlaceError> {
    let mut st = Stats::default();
    for (cam_true, obs) in cams_true.iter().zip(frames) {
        if obs.is_empty() {
            st.n_skip += 1;
            continue;
        }
        let res = resect(obs, markers, spec.a, &spec.cam_model, 0.0, 0.0, &opts(sigma_px, spec.noise_model));
        st.add(cam_true, res)?;
    }
    Ok(st)
}

// ---------------------------------------------------------------- setup

struct Setup {
    spec: PlacementSpec,
    tb: Tables,
    sel: Vec<usize>,
    survey_ids: Vec<usize>,
    key: (usize, f64, f64),
    mks_true: Vec<MarkerPose>,
    mk_dict: BTreeMap<i64, MarkerPose>,
    cams_test: Vec<CamPose>,
    test_kh: Vec<(usize, usize)>,
}

fn setup(budget: &Budget, sigma_ts: f64) -> Result<Setup, PlaceError> {
    let spec = demo_spec(10);
    let tb = precompute(&spec)?;
    let res = optimize_ift(&spec, sigma_ts, budget)?;
    let key = erploc_place::placement::key3(&res.stats, &spec);
    let mks_all = markers_from_candidates(&spec.candidates);
    let mks_true: Vec<MarkerPose> = mks_all.into_iter().filter(|m| res.sel.contains(&(m.id as usize))).collect();
    let mk_dict: BTreeMap<i64, MarkerPose> = mks_true.iter().map(|m| (m.id, *m)).collect();
    let mut cams_test = Vec::new();
    let mut test_kh = Vec::new();
    for (k, c) in tb.cells.iter().enumerate() {
        for (h_idx, &h) in spec.ws.headings.iter().enumerate() {
            cams_test.push(CamPose::euler(*c, h, 0.0, 0.0));
            test_kh.push((k, h_idx));
        }
    }
    Ok(Setup {
        spec,
        tb,
        sel: res.sel,
        survey_ids: res.survey_ids,
        key: (key.n_infeasible, key.penalty_sum, key.neg_margin),
        mks_true,
        mk_dict,
        cams_test,
        test_kh,
    })
}

fn synth_cfg(spec: &PlacementSpec, sigma_px: f64, seed: u64) -> SynthConfig {
    SynthConfig { cam: spec.cam_model, a: spec.a, sigma_px, noise_model: spec.noise_model, seed }
}

// ---------------------------------------------------------------- E2E-2 helpers

fn hstack(blocks: &[DMat]) -> DMat {
    let rows = blocks[0].rows();
    let cols: usize = blocks.iter().map(|b| b.cols()).sum();
    let mut m = DMat::zeros(rows, cols);
    let mut c0 = 0;
    for b in blocks {
        for j in 0..b.cols() {
            m.set_column(c0 + j, &b.column(j));
        }
        c0 += b.cols();
    }
    m
}

/// Python `_build_map`: `n_ba` frames (cell by `integers`, yaw uniform, roll = pitch = 0), render, noise, survey, `bundle_adjust`.
/// Returns the BA result and the continued RNG.
fn build_map(
    s: &Setup,
    survey_ids: &[usize],
    seed: u64,
    sigma_ts: f64,
    n_ba: usize,
) -> Result<(erploc_est::ba::BaResult, Pcg64), PlaceError> {
    let spec = &s.spec;
    let mut rng = Pcg64::new(seed);
    let k = s.tb.cells.len();
    let idx_k: Vec<usize> = (0..n_ba).map(|_| rng.integers(0, k)).collect();
    let yaws: Vec<f64> = (0..n_ba).map(|_| rng.uniform(0.0, 2.0 * PI)).collect();
    let cams_ba: Vec<CamPose> = idx_k.iter().zip(&yaws).map(|(&ki, &yw)| CamPose::euler(s.tb.cells[ki], yw, 0.0, 0.0)).collect();
    let cfg_geom = synth_cfg(spec, spec.sigma_px, seed);
    let mask = render_mask(&cams_ba, &s.mks_true, spec.a, &spec.cam_model, &spec.obstacles, Some(&spec.vp));
    let frames0 = render(&cams_ba, &s.mks_true, &cfg_geom, Some(&mask));
    let frames_ba = add_noise(&frames0, &cfg_geom, &mut rng);
    let ids: Vec<i64> = survey_ids.iter().map(|&i| i as i64).collect();
    let surveyed = survey(&s.mks_true, &ids, sigma_ts, &mut rng);
    let problem = BaProblem {
        frames: frames_ba,
        rp: vec![[0.0, 0.0]; n_ba],
        surveyed,
        a: spec.a,
        cam_model: spec.cam_model,
        sigma_px: spec.sigma_px,
        noise_model: spec.noise_model,
        cam_dof: CamDof::Yaw,
    };
    let res = bundle_adjust(&problem, None, HUBER, 50).map_err(|e| site_err(format!("bundle_adjust: {e}")))?;
    Ok((res, rng))
}

/// Outputs of `_propagate_frame`: (e, sigma_place, sigma_full, d2_place, d2_full, converged).
type Prop = ([f64; 3], f64, f64, f64, f64, bool);

/// Python `_propagate_frame`. `Ok(None)` when the resection itself failed with `NoConvergence` (counted by the caller).
fn propagate_frame(
    cam_true: &CamPose,
    obs: &[Observation],
    state_markers: &BTreeMap<i64, MarkerPose>,
    layout: &Layout,
    cov_map: &DMat,
    spec: &PlacementSpec,
) -> Result<Option<Prop>, PlaceError> {
    let r = match resect(obs, state_markers, spec.a, &spec.cam_model, 0.0, 0.0, &opts(spec.sigma_px, spec.noise_model)) {
        Ok(r) => r,
        Err(ResectError::Linalg(LinalgError::NoConvergence)) => return Ok(None),
        Err(e) => return Err(resect_err(e)),
    };
    let mut ids: Vec<i64> = obs.iter().map(|o| o.id).filter(|i| state_markers.contains_key(i)).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut i_cc = DMat::zeros(4, 4);
    let mut icm_blocks = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    for mid in &ids {
        let mi = info_marker(cam_true, &state_markers[mid], spec.a, &spec.cam_model, spec.sigma_px, spec.noise_model, None, &[Axis::Z]);
        i_cc = i_cc.add(&mi.icc);
        icm_blocks.push(mi.icm);
        cols.extend(layout.mk_cols(*mid));
    }
    let i_cm = hstack(&icm_blocks);
    let n = cov_map.cols();
    let cov_mm = DMat::from_fn(cols.len(), cols.len(), |a, b| cov_map.as_slice()[cols[a] * n + cols[b]]);
    let i_cc_inv = lu::inv(&i_cc)?;
    let cov_full = i_cc_inv.add(&i_cc_inv.matmul(&i_cm).matmul(&cov_mm).matmul(&i_cm.transpose()).matmul(&i_cc_inv));
    if !cov_full.as_slice().iter().all(|v| v.is_finite()) {
        return Err(site_err("_propagate_frame: Cov_full is not finite"));
    }
    let e = [r.cam.c[0] - cam_true.c[0], r.cam.c[1] - cam_true.c[1], r.cam.c[2] - cam_true.c[2]];
    let cov_place = top3(&r.cov);
    let cov_full_pos = top3(&cov_full);
    let sigma_place = trace3(&r.cov).sqrt();
    let sigma_full = trace3(&cov_full).sqrt();
    let d2_place = mahalanobis2(&e, &cov_place)?;
    let d2_full = mahalanobis2(&e, &cov_full_pos)?;
    Ok(Some((e, sigma_place, sigma_full, d2_place, d2_full, r.converged)))
}

// ---------------------------------------------------------------- main

/// `run_self_checks`: E2E-1, E2E-1b, E2E-2, E2E-3, E2E-4 and the run-time line (six lines). `budget` is the wall-clock budget of the
/// IFT selection (Python default `tbudget = 120 s`).
pub fn run_e2e(c: &E2eConsts, budget: &Budget) -> Result<Vec<CheckLine>, PlaceError> {
    run_e2e_info(c, budget).map(|(l, _)| l)
}

/// [`run_e2e`] plus the selection, key and per-stage timings.
pub fn run_e2e_info(c: &E2eConsts, budget: &Budget) -> Result<(Vec<CheckLine>, E2eInfo), PlaceError> {
    let t0 = Instant::now();
    let mut info = E2eInfo::default();
    let s = setup(budget, c.sigma_ts)?;
    info.t_ift = t0.elapsed().as_secs_f64();
    info.sel = s.sel.clone();
    info.survey_ids = s.survey_ids.clone();
    info.key = s.key;
    let spec = &s.spec;
    let n_test = s.cams_test.len();
    let cfg_geom = synth_cfg(spec, spec.sigma_px, c.seed);
    let mask0 = render_mask(&s.cams_test, &s.mks_true, spec.a, &spec.cam_model, &spec.obstacles, Some(&spec.vp));
    let frames0 = render(&s.cams_test, &s.mks_true, &cfg_geom, Some(&mask0));
    let cfg03 = synth_cfg(spec, spec.sigma_px, c.seed);
    if c.n_test_ba > n_test {
        return Err(site_err(format!("E2E-2: n_test_ba {} > number of test frames {n_test}", c.n_test_ba)));
    }
    let mut out = Vec::new();

    // --- E2E-1: resection against the true map, two noise realisations (e2e.py:E2E-1)
    let t = Instant::now();
    let frames_a = add_noise(&frames0, &cfg03, &mut Pcg64::new(seed_at(c, 1)?));
    let frames_b = add_noise(&frames0, &cfg03, &mut Pcg64::new(seed_at(c, 2)?));
    let sa = resect_stats(&s.cams_test, &frames_a, &s.mk_dict, spec, spec.sigma_px)?;
    let sb = resect_stats(&s.cams_test, &frames_b, &s.mk_dict, spec, spec.sigma_px)?;
    let n_nonconv1 = sa.n_nonconv + sb.n_nonconv;
    let skip1 = sa.n_skip + sb.n_skip;
    let e1: Vec<f64> = sa.e_norms().into_iter().chain(sb.e_norms()).collect();
    let sig1: Vec<f64> = sa.sigma.iter().chain(&sb.sigma).copied().collect();
    let d21: Vec<f64> = sa.d2.iter().chain(&sb.d2).copied().collect();
    let rms_e1 = rms(&e1);
    let mean_d2_3_1 = mean(&d21) / 3.0;
    let ratio1 = rms_e1 / rms(&sig1);
    let frac3 = e1.iter().zip(&sig1).filter(|(e, s)| **e <= 3.0 * **s).count() as f64 / e1.len() as f64;
    let max_e1 = e1.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // e2e.py: ok1 = (0.85 <= mean_d2_3_1 <= 1.15) and (0.9 <= ratio1 <= 1.1)
    let ok1 = (0.85..=1.15).contains(&mean_d2_3_1) && (0.9..=1.1).contains(&ratio1);
    out.push(CheckLine {
        name: "E2E-1 resection with the true map achieves placement sigma_pos".into(),
        passed: ok1,
        measured: format!(
            "mean(d^2)/3={mean_d2_3_1:.4}, RMS(|e|)/RMS(sigma_pos)={ratio1:.4}, within-3sigma fraction={frac3:.4}, max|e|={max_e1:.4} m \
             ({} runs, {skip1} missing, LM non-converged {n_nonconv1})",
            2 * n_test
        ),
        criterion: "mean(d^2)/3 in [0.85, 1.15] and RMS(|e|)/RMS(sigma_pos) in [0.9, 1.1]".into(),
    });
    info.t_e1 = t.elapsed().as_secs_f64();

    // --- E2E-1b: placement-table sigma vs resection covariance, noise-free (e2e.py:E2E-1b)
    let mut max_rel = 0.0_f64;
    let mut n_mismatch = 0usize;
    let mut n_empty = 0usize;
    for (i, &(k, h_idx)) in s.test_kh.iter().enumerate() {
        let vis_set: std::collections::BTreeSet<i64> = s.sel.iter().filter(|&&j| s.tb.vis_at(k, h_idx, j)).map(|&j| j as i64).collect();
        let obs_set: std::collections::BTreeSet<i64> = frames0[i].iter().map(|o| o.id).collect();
        if vis_set != obs_set {
            n_mismatch += 1;
            continue;
        }
        if obs_set.is_empty() {
            n_empty += 1;
            continue;
        }
        let mut i_place = DMat::zeros(4, 4);
        for &j in &vis_set {
            i_place = i_place.add(&DMat::from_rows(4, 4, s.tb.info_at(k, j as usize)));
        }
        let sigma_place = sigma_pos(&i_place, SIGMA_RTOL)?;
        let r = resect(&frames0[i], &s.mk_dict, spec.a, &spec.cam_model, 0.0, 0.0, &opts(spec.sigma_px, spec.noise_model))
            .map_err(resect_err)?;
        let sigma_resect = trace3(&r.cov).sqrt();
        let rel = (sigma_place - sigma_resect).abs() / sigma_resect;
        max_rel = if rel > max_rel || rel.is_nan() { rel } else { max_rel };
    }
    // e2e.py: ok1b = (max_rel < 1e-6) and (n_mismatch == 0)
    let ok1b = max_rel < 1e-6 && n_mismatch == 0;
    out.push(CheckLine {
        name: "E2E-1b placement-table sigma equals resection covariance".into(),
        passed: ok1b,
        measured: format!(
            "max relative difference={max_rel:.2e}, visible-set mismatches={n_mismatch}, no observation={n_empty} (all {} frames)",
            s.test_kh.len()
        ),
        criterion: "max relative difference < 1e-6 and 0 visible-set mismatches".into(),
    });

    // --- E2E-2: self-localisation with a BA map (e2e.py:E2E-2)
    let t = Instant::now();
    let (mut e2, mut sp2, mut sf2, mut d2p2, mut d2f2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut n_skip2, mut n_nonconv2) = (0usize, 0usize);
    for rlz in 0..c.n_map_realizations {
        let (result, mut rng) = build_map(&s, &s.survey_ids, seed_at(c, 100 + rlz as u64)?, c.sigma_ts, c.n_ba_frames)?;
        let layout = Layout::new(&result.state, CamDof::Yaw);
        let cov_map = lu::inv(&result.info)?;
        if !cov_map.as_slice().iter().all(|v| v.is_finite()) {
            return Err(site_err(format!("E2E-2: Cov_map is not finite (rlz={rlz})")));
        }
        let idx_test = rng.choice(n_test, c.n_test_ba);
        for &i in &idx_test {
            let obs = add_noise(&frames0[i..=i], &cfg03, &mut rng).remove(0);
            let obs_known: Vec<Observation> = obs.into_iter().filter(|o| result.state.markers.contains_key(&o.id)).collect();
            if obs_known.is_empty() {
                n_skip2 += 1;
                continue;
            }
            match propagate_frame(&s.cams_test[i], &obs_known, &result.state.markers, &layout, &cov_map, spec)? {
                None => n_nonconv2 += 1,
                Some((e, sp, sf, d2p, d2f, conv)) => {
                    n_nonconv2 += usize::from(!conv);
                    e2.push(norm3(&e));
                    sp2.push(sp);
                    sf2.push(sf);
                    d2p2.push(d2p);
                    d2f2.push(d2f);
                }
            }
        }
    }
    let rms_e2 = rms(&e2);
    let rms_sf2 = rms(&sf2);
    let mean_d2_full_3 = mean(&d2f2) / 3.0;
    let mean_d2_place_3 = mean(&d2p2) / 3.0;
    let ratio2 = rms_e2 / rms_sf2;
    let infl: Vec<f64> = sf2.iter().zip(&sp2).map(|(f, p)| f / p).collect();
    let inflation = median(&infl);
    let med_sp = median(&sp2) * 1e3;
    let med_sf = median(&sf2) * 1e3;
    let smap: Vec<f64> = sf2.iter().zip(&sp2).map(|(f, p)| (f * f - p * p).max(0.0).sqrt()).collect();
    let med_smap = median(&smap) * 1e3;
    // e2e.py: ok2 = (0.7 <= mean_d2_full_3 <= 1.4) and (0.8 <= ratio2 <= 1.25)
    let ok2 = (0.7..=1.4).contains(&mean_d2_full_3) && (0.8..=1.25).contains(&ratio2);
    out.push(CheckLine {
        name: "E2E-2 self-localisation with the BA map".into(),
        passed: ok2,
        measured: format!(
            "mean(d^2_full)/3={mean_d2_full_3:.4}, RMS(|e|)/RMS(sigma_full)={ratio2:.4}, placement-only mean(d^2)/3={mean_d2_place_3:.4}, \
             inflation median(sigma_full/sigma_place)={inflation:.4}, median sigma_place={med_sp:.2} mm / sigma_full={med_sf:.2} mm / map part={med_smap:.2} mm \
             ({} realisations x {} frames, {n_skip2} missing, LM non-converged {n_nonconv2})",
            c.n_map_realizations, c.n_test_ba
        ),
        criterion: "mean(d^2_full)/3 in [0.7, 1.4] and RMS(|e|)/RMS(sigma_full) in [0.8, 1.25]".into(),
    });
    info.t_e2 = t.elapsed().as_secs_f64();

    // --- E2E-3: leave-one-marker-out (e2e.py:E2E-3)
    let t = Instant::now();
    let mut rng3 = Pcg64::new(seed_at(c, 3000)?);
    let frames_c = add_noise(&frames0, &cfg03, &mut rng3);
    // Python `int(round(LOO_FRAC * n_test))`: round half to even
    let n_loo = (c.loo_frac * n_test as f64).round_ties_even() as usize;
    if n_loo > n_test {
        return Err(site_err(format!("E2E-3: n_loo {n_loo} > number of test frames {n_test}")));
    }
    let half_idx = rng3.choice(n_test, n_loo);
    let (mut e3, mut sig3) = (Vec::new(), Vec::new());
    let (mut n_skip3, mut n_single3, mut n_nonconv3) = (0usize, 0usize, 0usize);
    for &m in &s.sel {
        let map_loo: BTreeMap<i64, MarkerPose> = s.mk_dict.iter().filter(|(id, _)| **id != m as i64).map(|(i, mk)| (*i, *mk)).collect();
        for &i in &half_idx {
            let obs = &frames_c[i];
            let known: std::collections::BTreeSet<i64> = obs.iter().map(|o| o.id).filter(|id| map_loo.contains_key(id)).collect();
            if known.is_empty() {
                n_skip3 += 1;
                continue;
            }
            n_single3 += usize::from(known.len() == 1);
            let mut st = Stats::default();
            let res = resect(obs, &map_loo, spec.a, &spec.cam_model, 0.0, 0.0, &opts(spec.sigma_px, spec.noise_model));
            st.add(&s.cams_test[i], res)?;
            n_nonconv3 += st.n_nonconv;
            if let (Some(e), Some(sg)) = (st.e.first(), st.sigma.first()) {
                e3.push(norm3(e));
                sig3.push(*sg);
            }
        }
    }
    let frac_ok3 = e3.iter().filter(|&&e| e <= c.tau_loo).count() as f64 / e3.len() as f64;
    let ratio3 = rms(&e3) / rms(&sig3);
    info.e3_frac = frac_ok3;
    info.e3_ratio = ratio3;
    let max_e3 = e3.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // e2e.py: ok3 = (frac_ok3 >= 0.95) and (0.8 <= ratio3 <= 1.25)
    let ok3 = frac_ok3 >= 0.95 && (0.8..=1.25).contains(&ratio3);
    out.push(CheckLine {
        name: "E2E-3 tolerance to one missing marker".into(),
        passed: ok3,
        measured: format!(
            "fraction |e|<={:.1}m={frac_ok3:.4}, RMS(|e|)/RMS(sigma_pos)={ratio3:.4}, max|e|={max_e3:.4} m, single-marker frames={n_single3}, \
             LM non-converged={n_nonconv3} ({} markers x {n_loo} frames, {n_skip3} missing)",
            c.tau_loo,
            s.sel.len()
        ),
        criterion: "fraction >= 0.95 and RMS(|e|)/RMS(sigma_pos) in [0.8, 1.25]".into(),
    });
    info.t_e3 = t.elapsed().as_secs_f64();

    // --- E2E-4: noise scaling, sigma_px = 1.0 vs 0.3 (e2e.py:E2E-4)
    let t = Instant::now();
    let cfg10 = synth_cfg(spec, 1.0, seed_at(c, 4000)?);
    let frames_d = add_noise(&frames0, &cfg10, &mut Pcg64::new(seed_at(c, 4000)?));
    let s4 = resect_stats(&s.cams_test, &frames_d, &s.mk_dict, spec, 1.0)?;
    let rms_e4 = rms(&s4.e_norms());
    let ratio4 = rms_e4 / rms_e1;
    // e2e.py: ok4 = 3.0 <= ratio4 <= 3.7
    let ok4 = (3.0..=3.7).contains(&ratio4);
    out.push(CheckLine {
        name: "E2E-4 noise scaling".into(),
        passed: ok4,
        measured: format!(
            "RMS(|e|)_sigma_px=1.0={rms_e4:.4} m, RMS(|e|)_sigma_px=0.3={rms_e1:.4} m, ratio={ratio4:.4} ({}/{n_test} missing)",
            s4.n_skip
        ),
        criterion: "RMS(|e|)_1.0 / RMS(|e|)_0.3 in [3.0, 3.7]".into(),
    });
    info.t_e4 = t.elapsed().as_secs_f64();

    // --- run time (e2e.py: elapsed < 90.0)
    let elapsed = t0.elapsed().as_secs_f64();
    info.t_total = elapsed;
    out.push(CheckLine {
        name: "e2e run time".into(),
        passed: elapsed < RUNTIME_LIMIT_S,
        measured: format!("{elapsed:.1} s"),
        criterion: "< 90 s".into(),
    });
    Ok((out, info))
}

/// Table of the lines (name | measured | criterion | verdict) followed by the summary.
pub fn format_table(lines: &[CheckLine]) -> String {
    let mut s = String::new();
    for l in lines {
        s.push_str(&format!("{} | {} | {} | {}\n", l.name, l.measured, l.criterion, if l.passed { "pass" } else { "FAIL" }));
    }
    let np = lines.iter().filter(|l| l.passed).count();
    s.push_str(&format!("{np}/{} pass\n", lines.len()));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mahalanobis_constant_error() {
        // e = (0.01, 0.02, 0.02), cov = diag(0.01^2, 0.02^2, 0.02^2) -> d2 = 3 -> d2/3 = 1
        let cov = DMat::from_rows(3, 3, &[1e-4, 0.0, 0.0, 0.0, 4e-4, 0.0, 0.0, 0.0, 4e-4]);
        let d2 = mahalanobis2(&[0.01, 0.02, 0.02], &cov).unwrap();
        assert!((d2 / 3.0 - 1.0).abs() < 1e-12, "{d2}");
        // synthetic constant error 0.03 per axis, cov = 0.03^2 I -> mean(d^2)/3 = 1
        let cov2 = DMat::identity(3).scale(9e-4);
        let d = mahalanobis2(&[0.03, 0.03, 0.03], &cov2).unwrap();
        assert!((mean(&[d, d]) / 3.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stat_helpers_match_definitions() {
        assert_eq!(rms(&[3.0, 4.0]), (12.5_f64).sqrt());
        assert_eq!(median(&[1.0, 5.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 4.0, 2.0, 3.0]), 2.5);
        // 3-sigma fraction counts `<=`
        let e = [0.03, 0.0301, 0.01];
        let sg = [0.01, 0.01, 0.01];
        let f = e.iter().zip(&sg).filter(|(e, s)| **e <= 3.0 * **s).count() as f64 / 3.0;
        assert!((f - 2.0 / 3.0).abs() < 1e-15);
        // equality counts: 0.5 <= 3 * (0.5 / 3) holds exactly in binary floating point
        assert!([0.5_f64].iter().zip(&[0.5_f64 / 3.0]).all(|(e, s)| *e <= 3.0 * *s));
        // ratio as in Python: rms(e)/rms(sigma)
        assert!((rms(&[0.02, 0.02]) / rms(&[0.01, 0.01]) - 2.0).abs() < 1e-12);
    }

    #[test]
    #[cfg_attr(miri, ignore = "runs the full IFT/BA pipeline setup, over 1 h under Miri; covered by the debug and release suites")]
    fn skip_and_nonconv_are_counted_not_dropped() {
        let s = setup(&Budget::Iterations(30), SIGMA_TS).unwrap();
        let spec = &s.spec;
        // empty frame -> n_skip
        let cams = vec![s.cams_test[0]];
        let st = resect_stats(&cams, &[vec![]], &s.mk_dict, spec, spec.sigma_px).unwrap();
        assert_eq!((st.n_skip, st.n_nonconv, st.e.len()), (1, 0, 0));
        // Err(NoConvergence) -> n_nonconv (no row, but counted)
        let mut st = Stats::default();
        st.add(&cams[0], Err(ResectError::Linalg(LinalgError::NoConvergence))).unwrap();
        assert_eq!((st.n_skip, st.n_nonconv, st.e.len()), (0, 1, 0));
        // another error aborts
        assert!(Stats::default().add(&cams[0], Err(ResectError::NoKnownMarkers)).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "runs the full IFT/BA pipeline setup, over 1 h under Miri; covered by the debug and release suites")]
    fn nonconverged_row_is_kept_and_counted() {
        let s = setup(&Budget::Iterations(30), SIGMA_TS).unwrap();
        let spec = &s.spec;
        let frames = render(&s.cams_test[..1], &s.mks_true, &synth_cfg(spec, spec.sigma_px, 1), None);
        let mut r = resect(&frames[0], &s.mk_dict, spec.a, &spec.cam_model, 0.0, 0.0, &opts(spec.sigma_px, spec.noise_model)).unwrap();
        r.converged = false;
        let mut st = Stats::default();
        st.add(&s.cams_test[0], Ok(r)).unwrap();
        assert_eq!((st.n_nonconv, st.e.len(), st.sigma.len(), st.d2.len()), (1, 1, 1, 1));
    }

    #[test]
    #[cfg_attr(miri, ignore = "runs the full IFT/BA pipeline setup, over 1 h under Miri; covered by the debug and release suites")]
    fn oversized_sample_counts_are_errors() {
        let c = E2eConsts { n_test_ba: 10_000, ..E2eConsts::default() };
        assert!(run_e2e(&c, &Budget::Iterations(30)).is_err());
        let c = E2eConsts { loo_frac: 2.0, n_map_realizations: 1, n_test_ba: 4, ..E2eConsts::default() };
        assert!(run_e2e(&c, &Budget::Iterations(30)).is_err());
    }

    /// Same code path as the release run, tiny: 1 map, 4 BA test frames, 5% leave-one-out frames. Shapes / no panic only.
    #[test]
    #[cfg_attr(miri, ignore = "runs the full IFT/BA pipeline setup, over 1 h under Miri; covered by the debug and release suites")]
    fn tiny_run_has_six_lines() {
        let c = E2eConsts { n_map_realizations: 1, n_test_ba: 4, loo_frac: 0.05, ..E2eConsts::default() };
        let lines = run_e2e(&c, &Budget::Wall(PY_IFT_BUDGET_S)).unwrap();
        assert_eq!(lines.len(), 6);
        assert!(lines[0].name.starts_with("E2E-1 "));
        assert!(lines[1].name.starts_with("E2E-1b"));
        assert!(lines[5].measured.ends_with(" s"));
        // the exactness check does not depend on the sample sizes
        assert!(lines[1].passed, "{}", lines[1].measured);
    }
}
