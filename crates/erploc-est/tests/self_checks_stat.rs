//! Statistical (Monte-Carlo) self-checks of erploc-est, ported from `erploc/fisher.py` and `erploc/resection.py`.
//! Release only: run with `cargo test --release -p erploc-est --test self_checks_stat -- --nocapture`.
//! Seeds, trial counts, thresholds and the order of random draws follow the Python bodies line by line.

use erploc_core::erp::{lamphi_to_bearing, residual_lamphi, ErpCamera, NoiseModel};
use erploc_core::jacobian::{jac_marker, predict_marker, Axis, CamPose};
use erploc_core::markers::{facing_pose, MarkerPose};
use erploc_core::rot::{log_so3, mat3_vec};
use erploc_core::{V2, V3};
use erploc_est::fisher::{crlb, dvec, info_camera, sigma_pos, SIGMA_RTOL};
use erploc_est::lm::{levenberg_marquardt, vec_retract, LmOptions, LmResult};
use erploc_est::resection::{resect, CamDof, ResectOptions, ResectionResult};
use erploc_est::synth::{add_noise, make_markers, render, MarkerLayout, Observation, SynthConfig};
use erploc_linalg::rng::Pcg64;
use erploc_linalg::DMat;
use erploc_testkit::{mean, median, std_ddof1};
use std::collections::BTreeMap;

const A_CHK: f64 = 0.16;
const SIGMA_PX_CHK: f64 = 0.3;
const C_TRUE: V3 = [0.0, 0.0, 1.2];
const YAW_TRUE: f64 = 0.3;
const N_TRIALS: usize = 4000;
const N_TRIALS_6: usize = 1500;
const N_SCENES_46B: usize = 400;
const N_TRIALS_CHK: usize = 200;
const DOF4: [Axis; 1] = [Axis::Z];
const DOF6: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

type SceneRow = (f64, f64, f64, f64, f64, f64);
// 1.0472 is copied verbatim from fisher.SCENE (60 deg rounded), not pi/3
#[allow(clippy::approx_constant)]
const SCENE: [SceneRow; 4] = [
    (3.0, 0.0, 0.05, 20.0, 0.0, 0.0),
    (2.5, 2.2, -0.10, 15.0, 1.0, 0.4),
    (4.0, -2.0, 0.02, 25.0, 2.5, -0.7),
    (2.8, 1.0, 1.0472, 10.0, 0.5, 1.2),
];
const SCENE_POLE: [SceneRow; 4] = [
    (2.6, 0.4, 1.2217, 15.0, 0.0, 0.0),
    (3.0, 1.9, 1.2217, 15.0, 1.0, 0.4),
    (3.4, -2.88, 1.2217, 15.0, 2.5, -0.7),
    (2.8, -1.38, 1.2217, 15.0, 0.5, 1.2),
];
const SCENE_EXTRA2: [SceneRow; 2] = [(3.5, -1.0, 0.3, 30.0, -1.0, 0.2), (2.2, 2.8, -0.4, 18.0, 1.8, -0.5)];

fn cam_model() -> ErpCamera {
    ErpCamera::new(6528, 3264, 1)
}

/// NaN-propagating max (`f64::max` drops NaN).
fn nan_max(it: impl IntoIterator<Item = f64>) -> f64 {
    it.into_iter().fold(0.0, |m, x| if m.is_nan() || x.is_nan() { f64::NAN } else { m.max(x) })
}

fn norm3(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn sub3(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn diag_sqrt(m: &DMat) -> Vec<f64> {
    (0..m.rows()).map(|i| m[(i, i)].sqrt()).collect()
}

fn bearing(al: f64, ph: f64) -> V3 {
    lamphi_to_bearing(&[[al, ph]])[0]
}

fn build_scene(scene: &[SceneRow]) -> (CamPose, Vec<MarkerPose>) {
    let cam = CamPose::euler(C_TRUE, YAW_TRUE, 0.0, 0.0);
    let r = cam.r();
    let mks = scene
        .iter()
        .enumerate()
        .map(|(i, &(d, al, ph, psi_deg, tilt_dir, roll))| {
            let b = bearing(al, ph);
            let w = mat3_vec(&r, &[d * b[0], d * b[1], d * b[2]]);
            let p = [C_TRUE[0] + w[0], C_TRUE[1] + w[1], C_TRUE[2] + w[2]];
            facing_pose(i as i64, &p, &C_TRUE, psi_deg.to_radians(), tilt_dir, roll)
        })
        .collect();
    (cam, mks)
}

fn random_scene(rng: &mut Pcg64, n: usize) -> (CamPose, Vec<MarkerPose>) {
    let cam = CamPose::euler(C_TRUE, rng.uniform(-std::f64::consts::PI, std::f64::consts::PI), 0.0, 0.0);
    let mut mks = Vec::with_capacity(n);
    for i in 0..n {
        let d = rng.uniform(1.5, 6.0);
        let lam = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
        let phi = rng.uniform((-60f64).to_radians(), 80f64.to_radians());
        let b = bearing(lam, phi);
        let p = [C_TRUE[0] + d * b[0], C_TRUE[1] + d * b[1], C_TRUE[2] + d * b[2]];
        let psi = rng.uniform(0.0, 65f64.to_radians());
        let tilt = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
        let roll = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
        mks.push(facing_pose(i as i64, &p, &C_TRUE, psi, tilt, roll));
    }
    (cam, mks)
}

/// Vertical stack of `-(D J_cam)` per marker / sigma, and the stacked residual.
fn fisher_residual(cam: &CamPose, mks: &[MarkerPose], lp_obs: &[[V2; 4]], nm: NoiseModel) -> Vec<f64> {
    let model = cam_model();
    let mut out = Vec::with_capacity(8 * mks.len());
    for (mk, lp) in mks.iter().zip(lp_obs) {
        let (r, _) = residual_lamphi(lp, &predict_marker(cam, mk, A_CHK), &model, nm);
        for v in r {
            out.push(v[0] / SIGMA_PX_CHK);
            out.push(v[1] / SIGMA_PX_CHK);
        }
    }
    out
}

fn fisher_jac(cam: &CamPose, mks: &[MarkerPose], lp_obs: &[[V2; 4]], nm: NoiseModel, dof: &[Axis]) -> DMat {
    let model = cam_model();
    let n = 3 + dof.len();
    let mut j = DMat::zeros(8 * mks.len(), n);
    for (k, (mk, lp)) in mks.iter().zip(lp_obs).enumerate() {
        let (j_cam, _) = jac_marker(cam, mk, A_CHK, dof);
        let dv = dvec(&[lp[0][1], lp[1][1], lp[2][1], lp[3][1]], &model, nm);
        for r in 0..8 {
            for c in 0..n {
                j[(8 * k + r, c)] = -(dv[r] * j_cam[(r, c)]) / SIGMA_PX_CHK;
            }
        }
    }
    j
}

fn resect4_fisher(lp_obs: &[[V2; 4]], mks: &[MarkerPose], x0: &[f64; 4], nm: NoiseModel) -> LmResult<Vec<f64>> {
    let cam_of = |x: &Vec<f64>| CamPose::euler([x[0], x[1], x[2]], x[3], 0.0, 0.0);
    levenberg_marquardt(
        &mut |x: &Vec<f64>| fisher_residual(&cam_of(x), mks, lp_obs, nm),
        &mut |x: &Vec<f64>| fisher_jac(&cam_of(x), mks, lp_obs, nm, &DOF4),
        x0.to_vec(),
        &vec_retract,
        &LmOptions::default(),
    )
    .expect("resect4: LM failed")
}

fn resect6_fisher(lp_obs: &[[V2; 4]], mks: &[MarkerPose], cam0: CamPose, nm: NoiseModel) -> LmResult<CamPose> {
    levenberg_marquardt(
        &mut |cam: &CamPose| fisher_residual(cam, mks, lp_obs, nm),
        &mut |cam: &CamPose| fisher_jac(cam, mks, lp_obs, nm, &DOF6),
        cam0,
        &|cam: &CamPose, dx: &[f64]| cam.retract(&[dx[0], dx[1], dx[2]], &[dx[3], dx[4], dx[5]]),
        &LmOptions::default(),
    )
    .expect("resect6: LM failed")
}

/// Tangent-plane isotropic noise on the true lam/phi, drawn in Python's order (lam x4, then phi x4, per marker).
fn noisy_lp(lp_true: &[[V2; 4]], rng: &mut Pcg64) -> Vec<[V2; 4]> {
    let sig_ang = SIGMA_PX_CHK * cam_model().s();
    lp_true
        .iter()
        .map(|lp| {
            let mut o = *lp;
            for k in 0..4 {
                o[k][0] = lp[k][0] + sig_ang * rng.normal() / lp[k][1].cos();
            }
            for k in 0..4 {
                o[k][1] = lp[k][1] + sig_ang * rng.normal();
            }
            o
        })
        .collect()
}

fn mc_ratio(scene: &[SceneRow], est_model: NoiseModel, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let (cam, mks) = build_scene(scene);
    let lp_true: Vec<[V2; 4]> = mks.iter().map(|m| predict_marker(&cam, m, A_CHK)).collect();
    let i = info_camera(&cam, &mks, A_CHK, &cam_model(), SIGMA_PX_CHK, est_model, &DOF4);
    let sig_th = diag_sqrt(&crlb(&i).expect("crlb"));
    let x_true = [C_TRUE[0], C_TRUE[1], C_TRUE[2], YAW_TRUE];
    let mut rng = Pcg64::new(seed);
    let mut est: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(N_TRIALS)).collect();
    for _ in 0..N_TRIALS {
        let obs = noisy_lp(&lp_true, &mut rng);
        let x = resect4_fisher(&obs, &mks, &x_true, est_model).x;
        for c in 0..4 {
            est[c].push(x[c]);
        }
    }
    (sig_th, est.iter().map(|e| std_ddof1(e)).collect())
}

fn mc_ratio6(scene: &[SceneRow], nm: NoiseModel, seed: u64, n_trials: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let (cam, mks) = build_scene(scene);
    let lp_true: Vec<[V2; 4]> = mks.iter().map(|m| predict_marker(&cam, m, A_CHK)).collect();
    let i = info_camera(&cam, &mks, A_CHK, &cam_model(), SIGMA_PX_CHK, nm, &DOF6);
    let sig_th = diag_sqrt(&crlb(&i).expect("crlb"));
    let mut rng = Pcg64::new(seed);
    let mut est: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(n_trials)).collect();
    let mut n_nonconv = 0;
    let rt = cam.r().transpose();
    for _ in 0..n_trials {
        let obs = noisy_lp(&lp_true, &mut rng);
        let r = resect6_fisher(&obs, &mks, CamPose::euler(cam.c, cam.yaw, 0.0, 0.0), nm);
        if !r.converged {
            n_nonconv += 1;
        }
        let th = log_so3(&r.x.r().mul(&rt));
        for k in 0..3 {
            est[k].push(r.x.c[k]);
            est[3 + k].push(th[k]);
        }
    }
    (sig_th, est.iter().map(|e| std_ddof1(e)).collect(), n_nonconv)
}

fn sp(i: &DMat) -> f64 {
    sigma_pos(i, SIGMA_RTOL).unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------- fisher.py

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn s02_crlb_4dof() {
    let (sig_th, sig_mc) = mc_ratio(&SCENE, NoiseModel::Tangent, 20260911);
    let ratio: Vec<f64> = sig_mc.iter().zip(&sig_th).map(|(a, b)| a / b).collect();
    let dev = nan_max(ratio.iter().map(|r| (r - 1.0).abs()));
    println!("Fisher vs Monte Carlo, {} trials: ratio {:?}, max|ratio-1| {:.4} (< 0.05)", N_TRIALS, ratio, dev);
    assert!(dev.is_finite());
    assert!(dev < 0.05);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn s02_negative_control() {
    let (sig_th, sig_mc) = mc_ratio(&SCENE_POLE, NoiseModel::Pixel, 20260911);
    let ratio: Vec<f64> = sig_mc.iter().zip(&sig_th).map(|(a, b)| a / b).collect();
    let dev = nan_max(ratio.iter().map(|r| (r - 1.0).abs()));
    println!("Fisher vs Monte Carlo, negative control: ratio {:?}, max|ratio-1| {:.3} (> 0.05)", ratio, dev);
    assert!(dev.is_finite());
    assert!(dev > 0.05);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a62_crlb_6dof() {
    let (sig_th, sig_mc, nonconv) = mc_ratio6(&SCENE, NoiseModel::Tangent, 20260912, N_TRIALS_6);
    let ratio: Vec<f64> = sig_mc.iter().zip(&sig_th).map(|(a, b)| a / b).collect();
    let dev = nan_max(ratio.iter().map(|r| (r - 1.0).abs()));
    println!("A6-2 {} trials: ratio {:?}, max|ratio-1| {:.4} (< 0.1), nonconverged {}", N_TRIALS_6, ratio, dev, nonconv);
    assert!(dev.is_finite());
    assert!(dev < 0.1);
    assert_eq!(nonconv, 0);
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a64_ratio_report() {
    let all: Vec<SceneRow> = SCENE.iter().chain(SCENE_EXTRA2.iter()).copied().collect();
    let (cam, mks) = build_scene(&all);
    let m = cam_model();
    let mut ratios = Vec::new();
    for n in 1..=6 {
        let sub = &mks[..n];
        let s4 = sp(&info_camera(&cam, sub, A_CHK, &m, SIGMA_PX_CHK, NoiseModel::Tangent, &DOF4));
        let s6 = sp(&info_camera(&cam, sub, A_CHK, &m, SIGMA_PX_CHK, NoiseModel::Tangent, &DOF6));
        ratios.push(if s4.is_finite() && s6.is_finite() && s4 > 0.0 { s6 / s4 } else { f64::NAN });
    }
    let txt: Vec<String> = ratios.iter().enumerate().map(|(i, r)| format!("n={}:{:.3}", i + 1, r)).collect();
    println!("A6-4 sigma_pos 6dof/4dof: {}", txt.join(", "));
    assert!(ratios.iter().all(|r| r.is_finite()));
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a64b_random_scenes_report() {
    let m = cam_model();
    let nm = NoiseModel::Tangent;
    let mut rng = Pcg64::new(20260912);
    let mut med = Vec::new();
    let mut med_sh = Vec::new();
    for n in 1..=6usize {
        let (mut rr, mut rsh) = (Vec::new(), Vec::new());
        for _ in 0..N_SCENES_46B {
            let (cam, mks) = random_scene(&mut rng, n);
            let s4 = sp(&info_camera(&cam, &mks, A_CHK, &m, SIGMA_PX_CHK, nm, &DOF4));
            let s6 = sp(&info_camera(&cam, &mks, A_CHK, &m, SIGMA_PX_CHK, nm, &DOF6));
            if s4.is_finite() && s6.is_finite() && s4 > 0.0 {
                rr.push(s6 / s4);
            }
            if n >= 2 {
                let s4p = sp(&info_camera(&cam, &mks[..n - 1], A_CHK, &m, SIGMA_PX_CHK, nm, &DOF4));
                if s4p.is_finite() && s6.is_finite() && s4p > 0.0 {
                    rsh.push(s6 / s4p);
                }
            }
        }
        println!("A6-4b n={}: kept {} / {} ratios, {} pair ratios", n, rr.len(), N_SCENES_46B, rsh.len());
        med.push(if rr.is_empty() { f64::NAN } else { median(&rr) });
        if n >= 2 {
            med_sh.push(if rsh.is_empty() { f64::NAN } else { median(&rsh) });
        }
    }
    let t1: Vec<String> = med.iter().enumerate().map(|(i, r)| format!("n={}:{:.2}", i + 1, r)).collect();
    let t2: Vec<String> = med_sh.iter().enumerate().map(|(i, r)| format!("n={}:{:.3}", i + 2, r)).collect();
    println!("A6-4b median sigma6/sigma4: {}", t1.join(", "));
    println!("A6-4b median sigma6(n)/sigma4(n-1): {}", t2.join(", "));
    assert!(med.iter().all(|r| r.is_finite()));
    assert!(med_sh.iter().all(|r| r.is_finite()));
}

// ------------------------------------------------------------- resection.py

fn cfg() -> SynthConfig {
    SynthConfig { cam: cam_model(), a: A_CHK, sigma_px: SIGMA_PX_CHK, noise_model: NoiseModel::Tangent, seed: 0 }
}

fn base_by_id() -> (Vec<MarkerPose>, BTreeMap<i64, MarkerPose>) {
    let mut rng = Pcg64::new(20260911);
    let mks = make_markers(MarkerLayout::Walls, 6, &[-4.0, 4.0, -3.0, 3.0], [1.0, 1.8], &mut rng);
    let by_id = mks.iter().map(|m| (m.id, *m)).collect();
    (mks, by_id)
}

fn ropts(x0: Option<CamPose>, dof: CamDof) -> ResectOptions {
    ResectOptions { x0, sigma_px: SIGMA_PX_CHK, noise_model: NoiseModel::Tangent, loss: erploc_est::lm::Loss::Linear, dof }
}

fn cam_true6() -> CamPose {
    CamPose::euler([0.4, -0.3, 1.2], 0.7, 4f64.to_radians(), (-2f64).to_radians())
}

fn pos_err(res: &ResectionResult, truth: &CamPose) -> f64 {
    norm3(&sub3(&res.cam.c, &truth.c))
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn resection_noisy_rms_and_cov() {
    let (mks, by_id) = base_by_id();
    let cam_true = CamPose::euler([0.4, -0.3, 1.2], 0.7, 0.0, 0.0);
    let obs = render(&[cam_true], &mks, &cfg(), None).remove(0);
    let m = cam_model();
    let res0 = resect(&obs, &by_id, A_CHK, &m, 0.0, 0.0, &ropts(Some(cam_true), CamDof::Yaw)).expect("resect noise-free");
    let sig_th = diag_sqrt(&res0.cov);
    let rms_pred = SIGMA_PX_CHK * (1.0 - 4.0 / (8.0 * 6.0f64)).sqrt();
    let mut rms_list = Vec::new();
    let mut est: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for t in 0..N_TRIALS_CHK {
        let mut rng = Pcg64::new(1000 + t as u64);
        let noisy = add_noise(std::slice::from_ref(&obs), &cfg(), &mut rng).remove(0);
        let r = resect(&noisy, &by_id, A_CHK, &m, 0.0, 0.0, &ropts(Some(cam_true), CamDof::Yaw)).expect("resect noisy");
        rms_list.push(r.rms_px);
        for (k, v) in [r.cam.c[0], r.cam.c[1], r.cam.c[2], r.cam.yaw].into_iter().enumerate() {
            est[k].push(v);
        }
    }
    let rms_mean = mean(&rms_list);
    let dev_rms = (rms_mean / rms_pred - 1.0).abs();
    let ratio: Vec<f64> = est.iter().zip(&sig_th).map(|(e, s)| std_ddof1(e) / s).collect();
    println!("resection noisy: mean rms_px {:.4} (pred {:.4}), dev {:.4} (< 0.2)", rms_mean, rms_pred, dev_rms);
    println!("resection noisy: sig/CRLB ratio {:?} (each in (0.8, 1.2))", ratio);
    assert!(dev_rms.is_finite() && ratio.iter().all(|r| r.is_finite()));
    assert!(dev_rms < 0.2);
    assert!(ratio.iter().all(|&r| r > 0.8 && r < 1.2));
}

fn err6_vec(r: &ResectionResult, truth: &CamPose) -> Vec<f64> {
    let dth = log_so3(&r.cam.r().mul(&truth.r().transpose()));
    let dc = sub3(&r.cam.c, &truth.c);
    vec![dc[0], dc[1], dc[2], dth[0], dth[1], dth[2]]
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a62_resection_6dof_rms() {
    let (mks, by_id) = base_by_id();
    let truth = cam_true6();
    let obs6 = render(&[truth], &mks, &cfg(), None).remove(0);
    let m = cam_model();
    let o6 = ropts(Some(truth), CamDof::Full);
    let res0 = resect(&obs6, &by_id, A_CHK, &m, 0.0, 0.0, &o6).expect("resect6 noise-free");
    let sig_th = diag_sqrt(&res0.cov);
    let mut acc = 0.0;
    let mut cnt = 0usize;
    for t in 0..N_TRIALS_CHK {
        let mut rng = Pcg64::new(2026091200 + t as u64);
        let noisy = add_noise(std::slice::from_ref(&obs6), &cfg(), &mut rng).remove(0);
        let rt = resect(&noisy, &by_id, A_CHK, &m, 0.0, 0.0, &o6).expect("resect6 noisy");
        for (e, s) in err6_vec(&rt, &truth).iter().zip(&sig_th) {
            let z = e / s;
            acc += z * z;
            cnt += 1;
        }
    }
    let rms6 = (acc / cnt as f64).sqrt();
    println!("A6-2 resection RMS(e/sigma) {:.4} in [0.9, 1.1], sig_th {:?}", rms6, sig_th);
    assert!(rms6.is_finite());
    assert!((0.9..=1.1).contains(&rms6));
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a64_resection_report() {
    let (mks, by_id) = base_by_id();
    let truth = cam_true6();
    let obs6 = render(&[truth], &mks, &cfg(), None).remove(0);
    let m = cam_model();
    let o6 = ropts(Some(truth), CamDof::Full);
    let o4 = ropts(Some(truth), CamDof::Yaw);
    let (roll, pitch) = (truth.roll, truth.pitch);
    let res6_0 = resect(&obs6, &by_id, A_CHK, &m, 0.0, 0.0, &o6).expect("resect6 noise-free");
    let res4_0 = resect(&obs6, &by_id, A_CHK, &m, roll, pitch, &o4).expect("resect4 noise-free");
    let sp4 = sp(&res4_0.info);
    let sp6 = sp(&res6_0.info);
    let (mut s4, mut s6) = (0.0, 0.0);
    for t in 0..N_TRIALS_CHK {
        let mut rng = Pcg64::new(2026091200 + t as u64);
        let noisy = add_noise(std::slice::from_ref(&obs6), &cfg(), &mut rng).remove(0);
        let r4 = resect(&noisy, &by_id, A_CHK, &m, roll, pitch, &o4).expect("resect4 noisy");
        let r6 = resect(&noisy, &by_id, A_CHK, &m, 0.0, 0.0, &o6).expect("resect6 noisy");
        let (d4, d6) = (pos_err(&r4, &truth), pos_err(&r6, &truth));
        s4 += d4 * d4;
        s6 += d6 * d6;
    }
    let rms4 = (s4 / N_TRIALS_CHK as f64).sqrt();
    let rms6 = (s6 / N_TRIALS_CHK as f64).sqrt();
    println!(
        "A6-4 sigma_pos ratio(6/4)={:.3} (4dof {:.2e} m, 6dof {:.2e} m); position RMS ratio(6/4)={:.3} (4dof {:.2e} m, 6dof {:.2e} m)",
        sp6 / sp4,
        sp4,
        sp6,
        rms6 / rms4,
        rms4,
        rms6
    );
    assert!(sp4.is_finite() && sp6.is_finite() && rms4.is_finite() && rms6.is_finite());
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a65_single_marker_report() {
    let (mks, by_id) = base_by_id();
    let truth = cam_true6();
    let obs6 = render(&[truth], &mks, &cfg(), None).remove(0);
    let m = cam_model();
    let mut ids: Vec<i64> = obs6.iter().map(|o| o.id).collect();
    ids.sort();
    ids.dedup();
    let one_id = ids[0];
    let obs_one: Vec<Observation> = obs6.iter().filter(|o| o.id == one_id).cloned().collect();
    let two_ids = &ids[..2];
    let obs_two: Vec<Observation> = obs6.iter().filter(|o| two_ids.contains(&o.id)).cloned().collect();
    let by_one: BTreeMap<i64, MarkerPose> = [(one_id, by_id[&one_id])].into_iter().collect();
    let by_two: BTreeMap<i64, MarkerPose> = two_ids.iter().map(|i| (*i, by_id[i])).collect();
    let o6 = ropts(None, CamDof::Full);
    let n = 150usize;
    let (mut dp_one, mut dp_two) = (Vec::new(), Vec::new());
    let (mut err_one, mut err_two) = (0usize, 0usize);
    for t in 0..n {
        let mut rng = Pcg64::new(2026091300 + t as u64);
        let noisy_one = add_noise(std::slice::from_ref(&obs_one), &cfg(), &mut rng).remove(0);
        match resect(&noisy_one, &by_one, A_CHK, &m, 0.0, 0.0, &o6) {
            Ok(r) => dp_one.push(pos_err(&r, &truth)),
            Err(e) => {
                println!("A6-5 trial {} 1-marker resect Err: {:?}", t, e);
                err_one += 1;
                dp_one.push(f64::INFINITY);
            }
        }
        let mut rng2 = Pcg64::new(2026091300 + t as u64);
        let noisy_two = add_noise(std::slice::from_ref(&obs_two), &cfg(), &mut rng2).remove(0);
        match resect(&noisy_two, &by_two, A_CHK, &m, 0.0, 0.0, &o6) {
            Ok(r) => dp_two.push(pos_err(&r, &truth)),
            Err(e) => {
                println!("A6-5 trial {} 2-marker resect Err: {:?}", t, e);
                err_two += 1;
                dp_two.push(f64::INFINITY);
            }
        }
    }
    let ok_one = dp_one.iter().filter(|&&d| d < 0.1).count();
    let ok_two = dp_two.iter().filter(|&&d| d < 0.1).count();
    let max_of = |v: &[f64]| nan_max(v.iter().cloned());
    println!(
        "A6-5 1 marker: dp<0.1m {}/{} median {:.3} m max {:.3} m (Err {}) / 2 markers: dp<0.1m {}/{} median {:.4} m max {:.4} m (Err {})",
        ok_one,
        n,
        median(&dp_one),
        max_of(&dp_one),
        err_one,
        ok_two,
        n,
        median(&dp_two),
        max_of(&dp_two),
        err_two
    );
    let frac_two = ok_two as f64 / n as f64;
    assert!(frac_two >= 0.99, "2-marker cold start fraction {}", frac_two);
}
