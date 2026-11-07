//! Deterministic self-checks of `bundle_adjust` (Python `ba.run_self_checks`: A-1, A-1b, A-3/A-3b, A6-1, A6-3,
//! check 6 "init vs true", runtime). Release only:
//! `cargo test --release -p erploc-est --test ba_checks -- --nocapture`.

use erploc_core::erp::{wrap_pi, ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::rot::log_so3;
use erploc_core::V3;
use erploc_est::ba::{bundle_adjust, normal_equations, residuals, BaResult, Layout};
use erploc_est::ba_init::{initialize, naive_bfs_initialize};
use erploc_est::ba_types::{BaProblem, BaState};
use erploc_est::lm::Loss;
use erploc_est::resection::CamDof;
use erploc_est::synth::{add_noise, make_markers, make_trajectory, render, survey, MarkerLayout, SynthConfig, TrajectoryKind};
use erploc_linalg::eig::eigvalsh;
use erploc_linalg::lu;
use erploc_linalg::rng::Pcg64;
use erploc_linalg::DMat;
use erploc_testkit::{mean, median};
use std::collections::BTreeMap;
use std::time::Instant;

const A_CHK: f64 = 0.16;
const ROOM: [f64; 4] = [-4.0, 4.0, -3.0, 3.0];
const Z_RANGE: [f64; 2] = [0.8, 1.8];
const SIGMA_PX: f64 = 0.3;
const SIGMA_TS: f64 = 0.005;
const SURVEY_IDS: [i64; 3] = [0, 4, 8];
const SEED: u64 = 20260911;

fn cam_model() -> ErpCamera {
    ErpCamera::new(6528, 3264, 1)
}

fn cfg() -> SynthConfig {
    SynthConfig { cam: cam_model(), a: A_CHK, sigma_px: SIGMA_PX, noise_model: NoiseModel::Tangent, seed: SEED }
}

/// Walls 12 markers, random_walk `n_frames` at z 1.2, the same rng; returns the rng for further draws.
fn scene(seed: u64, n_frames: usize) -> (Vec<MarkerPose>, Vec<CamPose>, Pcg64) {
    let mut rng = Pcg64::new(seed);
    let mks = make_markers(MarkerLayout::Walls, 12, &ROOM, Z_RANGE, &mut rng);
    let cams = make_trajectory(TrajectoryKind::RandomWalk, n_frames, &ROOM, 1.2, &mut rng);
    (mks, cams, rng)
}

fn problem_of(
    frames: Vec<Vec<erploc_est::synth::Observation>>,
    rp: Vec<[f64; 2]>,
    surveyed: BTreeMap<i64, (V3, f64)>,
    dof: CamDof,
) -> BaProblem {
    BaProblem { frames, rp, surveyed, a: A_CHK, cam_model: cam_model(), sigma_px: SIGMA_PX, noise_model: NoiseModel::Tangent, cam_dof: dof }
}

fn true_survey(mks: &[MarkerPose], ids: &[i64]) -> BTreeMap<i64, (V3, f64)> {
    ids.iter().map(|&i| (i, (mks.iter().find(|m| m.id == i).unwrap().p, SIGMA_TS))).collect()
}

/// NaN-propagating max (`f64::max` drops NaN).
fn nan_max(it: impl IntoIterator<Item = f64>) -> f64 {
    it.into_iter().fold(0.0, |m, x| if m.is_nan() || x.is_nan() { f64::NAN } else { m.max(x) })
}

fn norm3(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn dist(a: &V3, b: &V3) -> f64 {
    norm3(&[a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// (marker position, marker orientation, camera position, camera yaw, camera orientation) errors vs truth.
fn errors(res: &BaResult, mks: &[MarkerPose], cams: &[CamPose]) -> (f64, f64, f64, f64, f64) {
    let by_id: BTreeMap<i64, &MarkerPose> = mks.iter().map(|m| (m.id, m)).collect();
    let dp_mk = nan_max(res.state.markers.iter().map(|(id, m)| dist(&m.p, &by_id[id].p)));
    let dr_mk = nan_max(res.state.markers.iter().map(|(id, m)| norm3(&log_so3(&m.r.transpose().mul(&by_id[id].r)))));
    let pairs = || cams.iter().zip(&res.state.cams).filter_map(|(t, e)| e.as_ref().map(|e| (t, e)));
    let dp_cam = nan_max(pairs().map(|(t, e)| dist(&e.c, &t.c)));
    let dyaw = nan_max(pairs().map(|(t, e)| wrap_pi(e.yaw - t.yaw).abs()));
    let dr_cam = nan_max(pairs().map(|(t, e)| norm3(&log_so3(&e.r().mul(&t.r().transpose())))));
    (dp_mk, dr_mk, dp_cam, dyaw, dr_cam)
}

fn a1_result() -> (BaResult, Vec<MarkerPose>, Vec<CamPose>) {
    let (mks, cams, _) = scene(SEED, 60);
    let frames = render(&cams, &mks, &cfg(), None);
    let p = problem_of(frames, vec![[0.0, 0.0]; 60], true_survey(&mks, &SURVEY_IDS), CamDof::Yaw);
    (bundle_adjust(&p, None, Loss::Linear, 50).unwrap(), mks, cams)
}

fn a6_scene() -> (BaProblem, Vec<MarkerPose>, Vec<CamPose>) {
    let (roll, pitch) = (6f64.to_radians(), (-4f64).to_radians());
    let (mks, lvl, _) = scene(SEED, 25);
    let cams: Vec<CamPose> = lvl.iter().map(|c| CamPose::euler(c.c, c.yaw, roll, pitch)).collect();
    let frames = render(&cams, &mks, &cfg(), None);
    let p = problem_of(frames, vec![[roll, pitch]; 25], true_survey(&mks, &SURVEY_IDS), CamDof::Full);
    (p, mks, cams)
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a1_zero_noise() {
    let t = Instant::now();
    let (res, mks, cams) = a1_result();
    let (dp_mk, dr_mk, dp_cam, dyaw, _) = errors(&res, &mks, &cams);
    println!("A-1: marker pos {dp_mk:.2e} m, marker rot {dr_mk:.2e} rad, cam pos {dp_cam:.2e} m, yaw {dyaw:.2e} rad ({:.2?})", t.elapsed());
    assert!(dp_mk < 1e-6 && dr_mk < 1e-7 && dp_cam < 1e-6 && dyaw < 1e-8);
    // A-3 / A-3b (reuse of the A-1 result)
    println!("A-3: null_dim_noprior = {}, A-3b: eig_min_ratio = {:.3e}", res.gauge.null_dim_noprior, res.gauge.eig_min_ratio);
    assert_eq!(res.gauge.null_dim_noprior, 4);
    assert!(res.gauge.eig_min_ratio > 1e-12);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a1b_roll_pitch() {
    let (roll, pitch) = (5f64.to_radians(), (-3f64).to_radians());
    let (mks, lvl, _) = scene(SEED, 20);
    let cams: Vec<CamPose> = lvl.iter().map(|c| CamPose::euler(c.c, c.yaw, roll, pitch)).collect();
    let frames = render(&cams, &mks, &cfg(), None);
    let p = problem_of(frames, vec![[roll, pitch]; 20], true_survey(&mks, &SURVEY_IDS), CamDof::Yaw);
    let res = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
    let (dp_mk, _, dp_cam, dyaw, _) = errors(&res, &mks, &cams);
    println!("A-1b: marker pos {dp_mk:.2e} m, cam pos {dp_cam:.2e} m, yaw {dyaw:.2e} rad");
    assert!(dp_mk < 1e-6 && dp_cam < 1e-6 && dyaw < 1e-8);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a3_gauge() {
    // separate run of the A-1 scene so the test stands alone
    let (res, _, _) = a1_result();
    println!("A-3: null_dim_noprior = {}, eig_min_ratio = {:.3e}", res.gauge.null_dim_noprior, res.gauge.eig_min_ratio);
    assert_eq!(res.gauge.null_dim_noprior, 4);
    assert!(res.gauge.eig_min_ratio > 1e-12);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a61_zero_noise_full_and_a63_gauge() {
    let t = Instant::now();
    let (p, mks, cams) = a6_scene();
    let res = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
    let (dp_mk, dr_mk, dp_cam, _, dr_cam) = errors(&res, &mks, &cams);
    println!(
        "A6-1: marker pos {dp_mk:.2e} m, marker rot {dr_mk:.2e} rad, cam pos {dp_cam:.2e} m, cam rot {dr_cam:.2e} rad ({:.2?})",
        t.elapsed()
    );
    assert!(dp_mk < 1e-6 && dr_mk < 1e-6 && dp_cam < 1e-6 && dr_cam < 1e-6);

    // A6-3
    println!("A6-3: null_dim_noprior = {}, A6-3b: eig_min_ratio = {:.3e}", res.gauge.null_dim_noprior, res.gauge.eig_min_ratio);
    assert_eq!(res.gauge.null_dim_noprior, 6);
    assert!(res.gauge.eig_min_ratio > 1e-9);

    // collinearity report (no thresholds)
    let pos = |id: i64| mks.iter().find(|m| m.id == id).unwrap().p;
    let compactness = |t: [i64; 3]| {
        let (p0, p1, p2) = (pos(t[0]), pos(t[1]), pos(t[2]));
        let (a, b, c) = (dist(&p1, &p0), dist(&p2, &p1), dist(&p0, &p2));
        let s = 0.5 * (a + b + c);
        let area = (s * (s - a) * (s - b) * (s - c)).max(0.0).sqrt();
        let m = a.max(b).max(c);
        if m > 1e-9 {
            2.0 * area / (m * m)
        } else {
            0.0
        }
    };
    let ids: Vec<i64> = res.state.markers.keys().copied().collect();
    let mut worst: Option<([i64; 3], f64)> = None;
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            for k in j + 1..ids.len() {
                let tri = [ids[i], ids[j], ids[k]];
                let comp = compactness(tri);
                // strict `<`: the FIRST minimum in combinations order wins
                if worst.map_or(true, |(_, w)| comp < w) {
                    // is_none_or needs Rust 1.82 (MSRV 1.80)
                    worst = Some((tri, comp));
                }
            }
        }
    }
    let (worst_tri, worst_comp) = worst.unwrap();
    let ratio_of = |sv: &[i64]| {
        let pr = BaProblem { surveyed: true_survey(&mks, sv), ..p.clone() };
        let layout = Layout::new(&res.state, CamDof::Full);
        let r = residuals(&res.state, &pr, &layout);
        let (a, _) = normal_equations(&res.state, &pr, &layout, &r, None);
        let w = eigvalsh(&a.add(&a.transpose()).scale(0.5)).unwrap();
        w[0] / w[w.len() - 1]
    };
    println!(
        "A6-3 report: spread {:?} compactness={:.3} eig_min_ratio={:.3e} / most collinear {:?} compactness={:.3} eig_min_ratio={:.3e}",
        SURVEY_IDS,
        compactness(SURVEY_IDS),
        ratio_of(&SURVEY_IDS),
        worst_tri,
        worst_comp,
        ratio_of(&worst_tri)
    );
}

fn cost_of(state: &BaState, problem: &BaProblem) -> f64 {
    let r = residuals(state, problem, &Layout::new(state, problem.cam_dof));
    let c = 0.5 * r.iter().map(|v| v * v).sum::<f64>();
    assert!(c.is_finite());
    c
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn init_vs_true_same_solution() {
    let t = Instant::now();
    let (mks, cams, mut rng) = scene(0, 60);
    let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut rng);
    let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng);
    let p = problem_of(frames, vec![[0.0, 0.0]; 60], surveyed, CamDof::Yaw);

    let r_init = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
    let cams_true = (0..60).map(|f| if p.frames[f].is_empty() { None } else { Some(cams[f]) }).collect();
    let x0_true = BaState { cams: cams_true, markers: mks.iter().map(|m| (m.id, *m)).collect() };
    let r_true = bundle_adjust(&p, Some(x0_true), Loss::Linear, 50).unwrap();
    let (c_init, c_true) = (cost_of(&r_init.state, &p), cost_of(&r_true.state, &p));
    let rel = (c_init - c_true).abs() / c_true.abs().max(1e-300);

    let x0_init0 = initialize(&p, 0).unwrap();
    let r_init0 = bundle_adjust(&p, Some(x0_init0), Loss::Linear, 50).unwrap();
    let c_init0 = cost_of(&r_init0.state, &p);
    println!(
        "init vs true: cost(init)={c_init:.6e} cost(true)={c_true:.6e} rel diff={rel:.2e} (refine_rounds=0: cost={c_init0:.6e}) ({:.2?})",
        t.elapsed()
    );
    assert!(rel < 1e-6);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn runtime_report() {
    let (mks, cams, mut rng) = scene(SEED + 1000, 40);
    let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut rng);
    let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng);
    let p = problem_of(frames, vec![[0.0, 0.0]; 40], surveyed, CamDof::Yaw);
    let t = Instant::now();
    let res = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
    println!(
        "runtime: one 40-frame x 12-marker 4-dof BA = {:.3?} (spec reference 0.5 s), converged={}, rms_px={:.4}",
        t.elapsed(),
        res.converged,
        res.rms_px
    );
}

// ---------------------------------------------------------------------------------------------------------
// Statistical checks (Python `ba_init.run_self_checks` check 3 / 6-dof report / order, `ba.run_self_checks` A-2, A6-2)
// ---------------------------------------------------------------------------------------------------------

/// ba_init's scene: walls 12 markers, circle `n_frames` at z 1.2, the same rng.
fn scene_circle(seed: u64, n_frames: usize) -> (Vec<MarkerPose>, Vec<CamPose>, Pcg64) {
    let mut rng = Pcg64::new(seed);
    let mks = make_markers(MarkerLayout::Walls, 12, &ROOM, Z_RANGE, &mut rng);
    let cams = make_trajectory(TrajectoryKind::Circle, n_frames, &ROOM, 1.2, &mut rng);
    (mks, cams, rng)
}

fn finite_nan_max(it: impl IntoIterator<Item = f64>) -> f64 {
    nan_max(it)
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ba_init_noisy() {
    let (mks, cams, mut rng) = scene_circle(SEED + 1, 40);
    let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut rng);
    let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng);
    let p = problem_of(frames, vec![[0.0, 0.0]; 40], surveyed, CamDof::Yaw);
    let st = initialize(&p, 1).unwrap();
    let by_id: BTreeMap<i64, &MarkerPose> = mks.iter().map(|m| (m.id, m)).collect();
    let dp = nan_max(st.markers.iter().map(|(id, m)| dist(&m.p, &by_id[id].p)));
    let dyaw = finite_nan_max(cams.iter().zip(&st.cams).filter_map(|(t, e)| e.as_ref().map(|e| wrap_pi(e.yaw - t.yaw).abs())));
    let dyaw_deg = dyaw.to_degrees();
    println!("ba_init noisy: marker pos {dp:.4} m, yaw {dyaw_deg:.4} deg");
    assert!(dp.is_finite() && dyaw_deg.is_finite());
    assert!(dp < 0.05 && dyaw_deg < 0.5);
}

/// One `_measure` of Python: (median marker pos err, median cam pos err, median rotation err [deg]).
fn measure(full: bool, s: u64) -> (f64, f64, f64) {
    let (mks, cams, _) = scene_circle(SEED + 200 + s, 40);
    let mut nrng = Pcg64::new(SEED + 300 + s);
    // Python: survey draws from the scene rng (rng_s), which is untouched by the noise rng
    let (_, _, mut rng_s) = scene_circle(SEED + 200 + s, 40);
    let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut nrng);
    let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng_s);
    let p = problem_of(frames, vec![[0.0, 0.0]; 40], surveyed, if full { CamDof::Full } else { CamDof::Yaw });
    let st = initialize(&p, 1).unwrap();
    let by_id: BTreeMap<i64, &MarkerPose> = mks.iter().map(|m| (m.id, m)).collect();
    let dp_mk: Vec<f64> = st.markers.iter().map(|(id, m)| dist(&m.p, &by_id[id].p)).collect();
    let pairs = || cams.iter().zip(&st.cams).filter_map(|(t, e)| e.as_ref().map(|e| (t, e)));
    let dp_cam: Vec<f64> = pairs().map(|(t, e)| dist(&e.c, &t.c)).collect();
    let drot: Vec<f64> = if full {
        pairs().map(|(t, e)| norm3(&log_so3(&e.r().mul(&t.r().transpose()))).to_degrees()).collect()
    } else {
        pairs().map(|(t, e)| wrap_pi(e.yaw - t.yaw).abs().to_degrees()).collect()
    };
    (median(&dp_mk), median(&dp_cam), median(&drot))
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ba_init_full_report() {
    for (name, full) in [("Yaw", false), ("Full", true)] {
        let m: Vec<(f64, f64, f64)> = (0..5).map(|s| measure(full, s)).collect();
        for (s, v) in m.iter().enumerate() {
            println!("ba_init report {name} seed {s}: marker pos med {:.4} m, cam pos med {:.4} m, rot med {:.4} deg", v.0, v.1, v.2);
        }
        let rng_of = |f: fn(&(f64, f64, f64)) -> f64| {
            let v: Vec<f64> = m.iter().map(f).collect();
            (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        };
        let (a, b, c) = (rng_of(|x| x.0), rng_of(|x| x.1), rng_of(|x| x.2));
        println!(
            "ba_init report {name}: marker pos {:.4}-{:.4} m, cam pos {:.4}-{:.4} m, rot {:.4}-{:.4} deg",
            a.0, a.1, b.0, b.1, c.0, c.1
        );
        assert!(m.iter().all(|v| v.0.is_finite() && v.1.is_finite() && v.2.is_finite()));
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ba_init_order_vs_naive() {
    let (mut naive, mut support) = (Vec::new(), Vec::new());
    let mut naive_failed = 0;
    for s in 0..5u64 {
        let (mks, cams, mut rng_s) = scene_circle(SEED + 200 + s, 40);
        let mut nrng = Pcg64::new(SEED + 300 + s);
        let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut nrng);
        let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng_s);
        let p = problem_of(frames, vec![[0.0, 0.0]; 40], surveyed, CamDof::Full);
        let by_id: BTreeMap<i64, &MarkerPose> = mks.iter().map(|m| (m.id, m)).collect();
        let meds = |st: &BaState| {
            let dp_mk: Vec<f64> = st.markers.iter().map(|(id, m)| dist(&m.p, &by_id[id].p)).collect();
            let dp_cam: Vec<f64> = cams.iter().zip(&st.cams).filter_map(|(t, e)| e.as_ref().map(|e| dist(&e.c, &t.c))).collect();
            (median(&dp_mk), median(&dp_cam))
        };
        support.push(meds(&initialize(&p, 0).unwrap()));
        match naive_bfs_initialize(&p) {
            Ok(st) => naive.push(meds(&st)),
            Err(e) => {
                println!("naive BFS seed {s} failed: {e:?}");
                naive_failed += 1;
            }
        }
    }
    let med_of = |v: &[(f64, f64)], k: usize| {
        if v.is_empty() {
            f64::NAN
        } else {
            median(&v.iter().map(|x| if k == 0 { x.0 } else { x.1 }).collect::<Vec<_>>())
        }
    };
    let (nm, nc, sm, sc) = (med_of(&naive, 0), med_of(&naive, 1), med_of(&support, 0), med_of(&support, 1));
    println!("order: support-first marker pos med {sm:.4} m, cam pos med {sc:.4} m / naive BFS marker pos med {nm:.4} m, cam pos med {nc:.4} m (failed {naive_failed}/5)");
    assert!(sm.is_finite() && nm.is_finite());
    assert_eq!(naive_failed, 0);
    assert!(nm >= sm - 1e-12);
}

/// Accumulates e/sigma and e' Cov^-1 e for the unsurveyed markers.
fn accumulate(res: &BaResult, mks: &[MarkerPose], e_over_sigma: &mut Vec<f64>, maha: &mut Vec<f64>) {
    for m in mks {
        if SURVEY_IDS.contains(&m.id) {
            continue;
        }
        let p_hat = res.state.markers[&m.id].p;
        let e = [p_hat[0] - m.p[0], p_hat[1] - m.p[1], p_hat[2] - m.p[2]];
        let cov6 = &res.cov_markers[&m.id];
        let cov = DMat::from_fn(3, 3, |i, j| cov6[(i, j)]);
        for k in 0..3 {
            e_over_sigma.push(e[k] / cov[(k, k)].sqrt());
        }
        let x = lu::solve(&cov, &e).unwrap();
        maha.push(e[0] * x[0] + e[1] * x[1] + e[2] * x[2]);
    }
}

fn report_normalised(tag: &str, e_over_sigma: &[f64], maha: &[f64]) {
    let rms = (e_over_sigma.iter().map(|v| v * v).sum::<f64>() / e_over_sigma.len() as f64).sqrt();
    let chi2 = mean(maha) / 3.0;
    println!("{tag}: RMS(e/sigma) = {rms:.4} ({} samples) [0.9, 1.1]; mean(e' Cov^-1 e)/3 = {chi2:.4} [0.8, 1.25]", e_over_sigma.len());
    assert!(rms.is_finite() && chi2.is_finite());
    assert!((0.9..=1.1).contains(&rms), "{tag}: RMS(e/sigma) = {rms}");
    assert!((0.8..=1.25).contains(&chi2), "{tag}: chi2 = {chi2}");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a2_marker_normalised_error() {
    let t = Instant::now();
    let (mut eos, mut maha) = (Vec::new(), Vec::new());
    for i in 0..20u64 {
        let (mks, cams, mut rng) = scene(SEED + 1000 + i, 40);
        let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut rng);
        let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng);
        let p = problem_of(frames, vec![[0.0, 0.0]; 40], surveyed, CamDof::Yaw);
        let res = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
        accumulate(&res, &mks, &mut eos, &mut maha);
    }
    report_normalised("A-2/A-2b", &eos, &maha);
    println!("A-2 wall time {:.2?}", t.elapsed());
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a62_full_normalised_error() {
    let t = Instant::now();
    let (mut eos, mut maha) = (Vec::new(), Vec::new());
    for i in 0..10u64 {
        let (mks, lvl, mut rng) = scene(SEED + 3000 + i, 40);
        let roll = rng.uniform(-8f64.to_radians(), 8f64.to_radians());
        let pitch = rng.uniform(-8f64.to_radians(), 8f64.to_radians());
        let cams: Vec<CamPose> = lvl.iter().map(|c| CamPose::euler(c.c, c.yaw, roll, pitch)).collect();
        let frames = add_noise(&render(&cams, &mks, &cfg(), None), &cfg(), &mut rng);
        let surveyed = survey(&mks, &SURVEY_IDS, SIGMA_TS, &mut rng);
        let p = problem_of(frames, vec![[roll, pitch]; 40], surveyed, CamDof::Full);
        let res = bundle_adjust(&p, None, Loss::Linear, 50).unwrap();
        accumulate(&res, &mks, &mut eos, &mut maha);
    }
    report_normalised("A6-2/A6-2b", &eos, &maha);
    println!("A6-2 wall time {:.2?}", t.elapsed());
}
