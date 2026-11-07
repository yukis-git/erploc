//! Differential tests against the Python `erploc` package (fixtures/est.txt, tools/export_est_fixtures.py).
//! Tolerances: deterministic geometry 1e-12, Fisher/Schur 1e-12..1e-10, estimates 1e-9, covariance 1e-8 (see each test).
//! Every exported record must be read: each group ends with `finish`, which compares the records read with the records exported.
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::{Axis, CamPose};
use erploc_core::markers::MarkerPose;
use erploc_core::{DMat, Mat, M3};
use erploc_est::fisher;
use erploc_est::lm::{levenberg_marquardt, vec_retract, LmOptions, Loss};
use erploc_est::resection::{resect, resect_init, CamDof, ResectOptions};
use erploc_est::synth::{self, MarkerLayout, Observation, SynthConfig, TrajectoryKind};
use erploc_testkit::fixture::{self, Fixture};
use erploc_testkit::Replay;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Tolerance factor: 1 on the reference platform (macOS aarch64, where the fixtures were recorded), 100 elsewhere. A libm that
/// differs in the last ulp moves the stopping point of the Levenberg-Marquardt fits and so every estimate and covariance by a
/// factor of that order (checked with a libm shim that moves 1 % and 5 % of the results by one ulp: `ERPLOC_TEST_LOOSE=1`).
fn lf() -> f64 {
    if erploc_testkit::strict() {
        1.0
    } else {
        100.0
    }
}

/// Worst `|a-b| / max(1,|b|)` over entries; NaN propagates.
fn err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).fold(0.0, |m, (x, y)| nmax(m, (x - y).abs() / y.abs().max(1.0)))
}

/// Worst `|a-b|` relative to the largest entry of `b`; NaN propagates.
fn mat_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let scale = b.iter().fold(0.0f64, |m, y| m.max(y.abs())).max(f64::MIN_POSITIVE);
    a.iter().zip(b).fold(0.0, |m, (x, y)| nmax(m, (x - y).abs() / scale))
}

/// Worst error per quantity (`what`), with the case that produced it.
struct Worst(BTreeMap<String, (f64, String)>);
impl Worst {
    fn new() -> Self {
        Worst(BTreeMap::new())
    }
    fn check(&mut self, case: &str, what: &str, e: f64, tol: f64) {
        assert!(e <= tol, "{case} {what}: error {e:e} > {tol:e}");
        let w = self.0.entry(what.to_string()).or_insert((0.0, String::new()));
        if e >= w.0 {
            *w = (e, case.to_string());
        }
    }
    fn report(&self, label: &str, n: usize) {
        for (what, (e, case)) in &self.0 {
            eprintln!("{label}: {n} cases, {what}: worst error {e:e} ({case})");
        }
    }
}

/// Fixture wrapper that remembers which records were read.
struct Fx {
    f: Fixture,
    read: RefCell<BTreeSet<(String, String)>>,
}
impl Fx {
    fn new() -> Self {
        Fx { f: fixture::load("est.txt"), read: RefCell::new(BTreeSet::new()) }
    }
    fn m(&self, tag: &str, name: &str) -> &DMat {
        let d = self.f.get(tag, name).unwrap_or_else(|| panic!("missing {tag} {name}"));
        self.read.borrow_mut().insert((tag.to_string(), name.to_string()));
        d
    }
    fn v(&self, tag: &str, name: &str) -> Vec<f64> {
        self.m(tag, name).as_slice().to_vec()
    }
    fn s(&self, tag: &str, name: &str) -> f64 {
        let d = self.m(tag, name);
        assert_eq!((d.rows(), d.cols()), (1, 1), "{tag} {name} is not a scalar");
        d.as_slice()[0]
    }
    fn names(&self, prefix: &str) -> Vec<String> {
        self.f.names.iter().filter(|n| n.starts_with(prefix)).cloned().collect()
    }
    /// n_read == n_records for the records whose case name starts with one of `prefixes`.
    fn finish(&self, label: &str, prefixes: &[&str]) {
        let all: Vec<_> = self.f.recs.keys().filter(|(_, n)| prefixes.iter().any(|p| n.starts_with(p))).collect();
        let read = self.read.borrow();
        let n_read = all.iter().filter(|k| read.contains(**k)).count();
        assert_eq!(n_read, all.len(), "{label}: read {n_read} of {} records", all.len());
        eprintln!("{label}: n_read == n_records == {n_read}");
    }
}

fn cam_model() -> ErpCamera {
    ErpCamera::new(6528, 3264, 1)
}
fn m3(r: &[f64]) -> M3 {
    assert_eq!(r.len(), 9);
    Mat([[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
}
fn f9(a: &M3) -> Vec<f64> {
    a.0.iter().flatten().copied().collect()
}
fn row(d: &DMat, i: usize) -> Vec<f64> {
    d.as_slice()[i * d.cols()..(i + 1) * d.cols()].to_vec()
}
fn cam_of(r: &[f64]) -> CamPose {
    assert_eq!(r.len(), 6);
    CamPose::euler([r[0], r[1], r[2]], r[3], r[4], r[5])
}
fn mks_of(d: &DMat) -> Vec<MarkerPose> {
    assert_eq!(d.cols(), 13);
    (0..d.rows())
        .map(|i| {
            let r = row(d, i);
            MarkerPose { id: r[0] as i64, p: [r[1], r[2], r[3]], r: m3(&r[4..13]) }
        })
        .collect()
}
fn by_id(mks: &[MarkerPose]) -> BTreeMap<i64, MarkerPose> {
    mks.iter().map(|m| (m.id, *m)).collect()
}
fn frames_of(nf: &DMat, d: &DMat) -> Vec<Vec<Observation>> {
    assert_eq!(d.cols(), 10);
    let mut out = vec![Vec::new(); nf.as_slice()[0] as usize];
    for i in 0..d.rows() {
        let r = row(d, i);
        let uv = [[r[2], r[3]], [r[4], r[5]], [r[6], r[7]], [r[8], r[9]]];
        out[r[0] as usize].push(Observation::new(r[1] as i64, uv));
    }
    out
}
/// Set `corner_mask` / `corner_sigma_scale` from a `cm` record (rows `f id mask x4 scale x4`, same order as the `obs` rows).
fn apply_cm(frames: &mut [Vec<Observation>], cm: &DMat) {
    assert_eq!(cm.cols(), 10);
    let n: usize = frames.iter().map(|f| f.len()).sum();
    assert_eq!(cm.rows(), n, "cm rows != observations");
    let mut i = 0;
    for (f, fr) in frames.iter_mut().enumerate() {
        for ob in fr.iter_mut() {
            let r = row(cm, i);
            assert_eq!((r[0] as usize, r[1] as i64), (f, ob.id), "cm row {i} does not match its observation");
            for k in 0..4 {
                assert!(r[2 + k] == 0.0 || r[2 + k] == 1.0);
                ob.corner_mask[k] = r[2 + k] == 1.0;
                ob.corner_sigma_scale[k] = r[6 + k];
            }
            i += 1;
        }
    }
}
/// Observations of a case: `nf` + `obs`, plus `cm` when the case has masks (ms_ cases).
fn case_frames(fx: &Fx, nm: &str, tag: &str) -> Vec<Vec<Observation>> {
    let mut frames = frames_of(fx.m("nf", nm), fx.m(tag, nm));
    if nm.starts_with("ms_") {
        apply_cm(&mut frames, fx.m("cm", nm));
    }
    frames
}

fn obs_flat(frames: &[Vec<Observation>]) -> Vec<f64> {
    frames.iter().flatten().flat_map(|o| o.uv.iter().flatten().copied()).collect()
}
fn obs_ids(frames: &[Vec<Observation>]) -> Vec<(usize, i64)> {
    frames.iter().enumerate().flat_map(|(f, fr)| fr.iter().map(move |o| (f, o.id))).collect()
}
fn dof_axes(n: f64) -> &'static [Axis] {
    if n == 1.0 {
        &[Axis::Z]
    } else {
        &[Axis::X, Axis::Y, Axis::Z]
    }
}
fn nm_of(pixel: f64) -> NoiseModel {
    if pixel == 1.0 {
        NoiseModel::Pixel
    } else {
        NoiseModel::Tangent
    }
}

#[test]
fn lm_matches_python() {
    let fx = Fx::new();
    let mut wx = Worst::new();
    let mut wc = Worst::new();
    let names = fx.names("lm_");
    for nm in &names {
        let opt = fx.v("opt", nm);
        let opts = LmOptions {
            loss: if opt[1] == 1.0 { Loss::Huber { delta: opt[2] } } else { Loss::Linear },
            max_iter: opt[0] as usize,
            ..LmOptions::default()
        };
        let x0 = fx.v("x0", nm);
        let res = if nm == "lm_lin" {
            let (x, y) = (fx.m("X", nm).clone(), fx.v("y", nm));
            let res = levenberg_marquardt(
                &mut |p: &Vec<f64>| x.matvec(p).iter().zip(&y).map(|(a, b)| a - b).collect(),
                &mut |_: &Vec<f64>| x.clone(),
                x0,
                &|p, d| vec_retract(p, d),
                &opts,
            );
            res.unwrap()
        } else if nm.starts_with("lm_rosen") {
            levenberg_marquardt(
                &mut |p: &Vec<f64>| vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]],
                &mut |p: &Vec<f64>| DMat::from_rows(2, 2, &[-20.0 * p[0], 10.0, -1.0, 0.0]),
                x0,
                &|p, d| vec_retract(p, d),
                &opts,
            )
            .unwrap()
        } else if nm.starts_with("lm_exp") {
            let (t, y) = (fx.v("t", nm), fx.v("y", nm));
            levenberg_marquardt(
                &mut |p: &Vec<f64>| t.iter().zip(&y).map(|(ti, yi)| p[0] * (p[1] * ti).exp() - yi).collect(),
                &mut |p: &Vec<f64>| {
                    DMat::from_fn(t.len(), 2, |i, j| {
                        let ex = (p[1] * t[i]).exp();
                        if j == 0 {
                            ex
                        } else {
                            p[0] * t[i] * ex
                        }
                    })
                },
                x0,
                &|p, d| vec_retract(p, d),
                &opts,
            )
            .unwrap()
        } else if nm.starts_with("lm_hub_") {
            let (x, y, sig) = (fx.m("X", nm).clone(), fx.v("y", nm), fx.s("sig", nm));
            let js = DMat::from_fn(x.rows(), x.cols(), |i, j| x.as_slice()[i * x.cols() + j] / sig);
            levenberg_marquardt(
                &mut |p: &Vec<f64>| x.matvec(p).iter().zip(&y).map(|(a, b)| (a - b) / sig).collect(),
                &mut |_: &Vec<f64>| js.clone(),
                x0,
                &|p, d| vec_retract(p, d),
                &opts,
            )
            .unwrap()
        } else {
            assert_eq!(nm, "lm_retract");
            let (x, y, s) = (fx.m("X", nm).clone(), fx.v("y", nm), fx.m("S", nm).clone());
            let xs = x.matmul(&s);
            levenberg_marquardt(
                &mut |p: &Vec<f64>| x.matvec(p).iter().zip(&y).map(|(a, b)| a - b).collect(),
                &mut |_: &Vec<f64>| xs.clone(),
                x0,
                &|p, d| p.iter().zip(s.matvec(d)).map(|(a, b)| a + b).collect(),
                &opts,
            )
            .unwrap()
        };
        wx.check(nm, "x", err(&res.x, &fx.v("x", nm)), 1e-9);
        // cost relative to the Python cost, floored at 1e-6 so near-zero costs (noise-free fits, ~1e-26) are compared absolutely
        let c = fx.s("cost", nm);
        wc.check(nm, "cost", (res.cost - c).abs() / c.abs().max(1e-6), 1e-9);
        assert_eq!(res.iters as f64, fx.s("iters", nm), "{nm}: iters");
        assert_eq!(res.converged, fx.s("conv", nm) == 1.0, "{nm}: converged");
    }
    wx.report("lm x", names.len());
    wc.report("lm cost", names.len());
    fx.finish("lm", &["lm_"]);
}

#[test]
fn fisher_matches_python() {
    let fx = Fx::new();
    let cm = cam_model();
    let mut w = Worst::new();
    let mut n = 0;
    for k in 0..6 {
        let nm = format!("fi_im_{k}");
        let cam = cam_of(&fx.v("cam", &nm));
        let mk = mks_of(fx.m("mk", &nm))[0];
        let par = fx.v("par", &nm);
        let phi: Option<[f64; 4]> = if par[3] == 1.0 {
            let p = fx.v("phi", &nm);
            Some([p[0], p[1], p[2], p[3]])
        } else {
            None
        };
        let r = fisher::info_marker(&cam, &mk, par[0], &cm, par[1], nm_of(par[2]), phi.as_ref(), dof_axes(par[4]));
        w.check(&nm, "Icc", mat_rel(r.icc.as_slice(), fx.m("Icc", &nm).as_slice()), 1e-12);
        w.check(&nm, "Icm", mat_rel(r.icm.as_slice(), fx.m("Icm", &nm).as_slice()), 1e-12);
        w.check(&nm, "Imm", mat_rel(r.imm.as_slice(), fx.m("Imm", &nm).as_slice()), 1e-12);
        n += 1;
    }
    w.report("fisher info_marker", n);

    let nm = "fi_scene";
    let cam = cam_of(&fx.v("cam", nm));
    let mks = mks_of(fx.m("mk", nm));
    let par = fx.v("par", nm);
    let mut ws = Worst::new();
    for (tag, sp, cv, dof) in [("I4", "sp4", "cov4", dof_axes(1.0)), ("I6", "sp6", "cov6", dof_axes(3.0))] {
        let i = fisher::info_camera(&cam, &mks, par[0], &cm, par[1], NoiseModel::Tangent, dof);
        ws.check(nm, tag, mat_rel(i.as_slice(), fx.m(tag, nm).as_slice()), 1e-12);
        let s = fisher::sigma_pos(&i, fisher::SIGMA_RTOL).unwrap();
        ws.check(nm, sp, err(&[s], &[fx.s(sp, nm)]), 1e-12);
        let cov = fisher::crlb(&i).unwrap();
        ws.check(nm, cv, mat_rel(cov.as_slice(), fx.m(cv, nm).as_slice()), 1e-10);
    }
    ws.report("fisher scene", 1);

    let mut wsc = Worst::new();
    let d = |t: &str, n: &str| fx.m(t, n).clone();
    let s = fisher::schur_markers(
        &[d("Icc1", "fi_schur"), d("Icc2", "fi_schur")],
        &[d("Icm1", "fi_schur"), d("Icm2", "fi_schur")],
        &d("Imm", "fi_schur"),
    )
    .unwrap();
    wsc.check("fi_schur", "S", mat_rel(s.as_slice(), fx.m("S", "fi_schur").as_slice()), 1e-10);
    let s = fisher::schur_markers(&[d("Icc", "fi_schur_pinv")], &[d("Icm", "fi_schur_pinv")], &d("Imm", "fi_schur_pinv")).unwrap();
    wsc.check("fi_schur_pinv", "S", mat_rel(s.as_slice(), fx.m("S", "fi_schur_pinv").as_slice()), 1e-9);
    wsc.report("fisher schur", 2);
    let (dim, basis) = fisher::nullspace(fx.m("M", "fi_null"), fisher::NULL_RTOL).unwrap();
    assert_eq!(dim as f64, fx.s("dim", "fi_null"), "fi_null dim");
    assert_eq!(basis.cols(), dim);
    let resid = fx.m("M", "fi_null").matmul(&basis);
    let e = resid.max_abs() / fx.m("M", "fi_null").max_abs();
    assert!(e < 1e-8, "fi_null: M * basis not ~ 0 ({e:e})");

    // per-corner masks: info_marker with per-corner masks / scales (ms_fi_im_*): 2-valid, 0-valid (exactly zero), mask-only, scale-only
    let mut wm = Worst::new();
    let ms_names = fx.names("ms_fi_im_");
    assert_eq!(ms_names.len(), 10);
    for nm in &ms_names {
        let cam = cam_of(&fx.v("cam", nm));
        let mk = mks_of(fx.m("mk", nm))[0];
        let par = fx.v("par", nm);
        let phi: Option<[f64; 4]> = if par[3] == 1.0 {
            let p = fx.v("phi", nm);
            Some([p[0], p[1], p[2], p[3]])
        } else {
            None
        };
        let (mm, sc) = (fx.v("mask", nm), fx.v("scale", nm));
        let mask = [mm[0] == 1.0, mm[1] == 1.0, mm[2] == 1.0, mm[3] == 1.0];
        let scale = [sc[0], sc[1], sc[2], sc[3]];
        let r = fisher::info_marker_masked(&cam, &mk, par[0], &cm, par[1], nm_of(par[2]), phi.as_ref(), dof_axes(par[4]), &mask, &scale);
        for (tag, got) in [("Icc", &r.icc), ("Icm", &r.icm), ("Imm", &r.imm)] {
            wm.check(nm, tag, mat_rel(got.as_slice(), fx.m(tag, nm).as_slice()), 1e-12);
        }
        if mask == [false; 4] {
            assert!(
                r.icc.as_slice().iter().chain(r.icm.as_slice()).chain(r.imm.as_slice()).all(|&v| v == 0.0),
                "{nm}: 0-valid info must be exactly zero"
            );
        }
    }
    wm.report("fisher masked info_marker", ms_names.len());
    fx.finish("fisher", &["fi_", "ms_fi_"]);
}

#[test]
fn synth_matches_python() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut n = 0;
    for (kind, lay) in [("walls", MarkerLayout::Walls), ("ring", MarkerLayout::Ring), ("random", MarkerLayout::Random)] {
        let nm = format!("sy_markers_{kind}");
        let room = fx.v("room", &nm);
        let zr = fx.v("zr", &nm);
        let mut rp = Replay::from_record(fx.m("draws", &nm));
        let got = synth::make_markers(lay, fx.s("n", &nm) as usize, &[room[0], room[1], room[2], room[3]], [zr[0], zr[1]], &mut rp);
        rp.assert_exhausted();
        let exp = mks_of(fx.m("mk", &nm));
        assert_eq!(got.len(), exp.len());
        let ids: Vec<i64> = got.iter().map(|m| m.id).collect();
        assert_eq!(ids, exp.iter().map(|m| m.id).collect::<Vec<_>>(), "{nm}: ids");
        let g: Vec<f64> = got.iter().flat_map(|m| m.p).collect();
        let e: Vec<f64> = exp.iter().flat_map(|m| m.p).collect();
        w.check(&nm, "p", err(&g, &e), 1e-12);
        let g: Vec<f64> = got.iter().flat_map(|m| f9(&m.r)).collect();
        let e: Vec<f64> = exp.iter().flat_map(|m| f9(&m.r)).collect();
        w.check(&nm, "R", err(&g, &e), 1e-12);
        n += 1;
    }
    for (kind, k) in
        [("circle", TrajectoryKind::Circle), ("random_walk", TrajectoryKind::RandomWalk), ("waypoints", TrajectoryKind::Waypoints)]
    {
        let nm = format!("sy_traj_{kind}");
        let room = fx.v("room", &nm);
        let par = fx.v("par", &nm);
        let mut rp = Replay::from_record(fx.m("draws", &nm));
        let got = synth::make_trajectory(k, par[0] as usize, &[room[0], room[1], room[2], room[3]], par[1], &mut rp);
        rp.assert_exhausted();
        let ec = fx.m("c", &nm);
        let ey = fx.v("yaw", &nm);
        assert_eq!(got.len(), ey.len(), "{nm}: length");
        let gc: Vec<f64> = got.iter().flat_map(|c| c.c).collect();
        w.check(&nm, "c", err(&gc, ec.as_slice()), 1e-12);
        let gy: Vec<f64> = got.iter().map(|c| c.yaw).collect();
        w.check(&nm, "yaw", err(&gy, &ey), 1e-12);
        n += 1;
    }
    w.report("synth markers/trajectories", n);

    let mut w = Worst::new();
    let cfg = |a: f64, sg: f64, nm: NoiseModel| SynthConfig { cam: cam_model(), a, sigma_px: sg, noise_model: nm, seed: 0 };
    let nm = "sy_render";
    let cams: Vec<CamPose> = {
        let d = fx.m("cam", nm);
        (0..d.rows()).map(|i| cam_of(&row(d, i))).collect()
    };
    let mks = mks_of(fx.m("mk", nm));
    let got = synth::render(&cams, &mks, &cfg(fx.v("par", nm)[0], 0.3, NoiseModel::Tangent), None);
    let exp = frames_of(fx.m("nf", nm), fx.m("out", nm));
    assert_eq!(got.len(), exp.len());
    assert_eq!(obs_ids(&got), obs_ids(&exp), "{nm}: ids");
    w.check(nm, "uv", err(&obs_flat(&got), &obs_flat(&exp)), 1e-12);
    for kind in ["tangent", "pixel"] {
        let nm = format!("sy_noise_{kind}");
        let par = fx.v("par", &nm);
        let input = frames_of(fx.m("nf", &nm), fx.m("obs", &nm));
        let mut rp = Replay::from_record(fx.m("draws", &nm));
        let got = synth::add_noise(&input, &cfg(par[0], par[1], nm_of(par[2])), &mut rp);
        rp.assert_exhausted();
        let exp: Vec<Vec<Observation>> = {
            let d = fx.m("out", &nm);
            let mut o = vec![Vec::new(); input.len()];
            for i in 0..d.rows() {
                let r = row(d, i);
                o[r[0] as usize].push(Observation::new(r[1] as i64, [[r[2], r[3]], [r[4], r[5]], [r[6], r[7]], [r[8], r[9]]]));
            }
            o
        };
        assert_eq!(obs_ids(&got), obs_ids(&exp), "{nm}: ids");
        w.check(&nm, "uv", err(&obs_flat(&got), &obs_flat(&exp)), 1e-12);
    }
    let mut wn = Worst::new();
    for kind in ["tangent", "pixel"] {
        let nm = format!("ms_noise_{kind}");
        let par = fx.v("par", &nm);
        let input = case_frames(&fx, &nm, "obs");
        let mut rp = Replay::from_record(fx.m("draws", &nm));
        let got = synth::add_noise(&input, &cfg(par[0], par[1], nm_of(par[2])), &mut rp);
        rp.assert_exhausted();
        let exp = frames_of(fx.m("nf", &nm), fx.m("out", &nm));
        assert_eq!(obs_ids(&got), obs_ids(&exp), "{nm}: ids");
        for (g, i) in got.iter().flatten().zip(input.iter().flatten()) {
            assert_eq!(g.corner_mask, i.corner_mask, "{nm}: mask passes through");
            assert_eq!(g.corner_sigma_scale, i.corner_sigma_scale, "{nm}: scale passes through");
        }
        assert!(input.iter().flatten().any(|o| o.corner_sigma_scale.iter().any(|&s| s != 1.0)), "{nm}: scales must differ from 1");
        wn.check(&nm, "uv", err(&obs_flat(&got), &obs_flat(&exp)), 1e-12);
    }
    wn.report("synth masked noise", 2);
    let nm = "sy_survey";
    let mks = mks_of(fx.m("mk", nm));
    let ids: Vec<i64> = fx.v("ids", nm).iter().map(|&v| v as i64).collect();
    let mut rp = Replay::from_record(fx.m("draws", nm));
    let got = synth::survey(&mks, &ids, fx.v("par", nm)[0], &mut rp);
    rp.assert_exhausted();
    let exp = fx.m("p", nm);
    assert_eq!(got.len(), exp.rows(), "{nm}: number of ids");
    for i in 0..exp.rows() {
        let r = row(exp, i);
        let (p, s) = got.get(&(r[0] as i64)).unwrap_or_else(|| panic!("{nm}: id {} missing", r[0]));
        w.check(nm, "p", err(p, &r[1..4]), 1e-12);
        w.check(nm, "sigma", err(&[*s], &[r[4]]), 1e-12);
    }
    w.report("synth render/noise/survey", 4);
    fx.finish("synth", &["sy_", "ms_noise_"]);
}

fn dof_of(name: &str) -> CamDof {
    if name.starts_with("rs_init_4") || name.starts_with("rs_4") || name.starts_with("ms_init_4") || name.starts_with("ms_rs_4") {
        CamDof::Yaw
    } else {
        CamDof::Full
    }
}

#[test]
fn resection_matches_python() {
    let fx = Fx::new();
    let cm = cam_model();
    let (mut wi4, mut wi6) = (Worst::new(), Worst::new());
    let mut wr = Worst::new();
    // masked cases (ms_init_*, ms_rs_*) get their own worst-error tables, with the same tolerances
    let (mut mi4, mut mi6) = (Worst::new(), Worst::new());
    let mut mr = Worst::new();
    let (mut ni, mut nr) = (0, 0);
    let (mut nmi, mut nmr) = (0, 0);
    let mut names = fx.names("rs_");
    names.extend(fx.names("ms_init_"));
    names.extend(fx.names("ms_rs_"));
    for nm in names {
        let masked = nm.starts_with("ms_");
        let is_init = nm.starts_with("rs_init_") || nm.starts_with("ms_init_");
        let dof = dof_of(&nm);
        let mks = mks_of(fx.m("mk", &nm));
        let map = by_id(&mks);
        let obs = case_frames(&fx, &nm, "obs").remove(0);
        let par = fx.v("par", &nm);
        let (a, sg, roll, pitch) = (par[0], par[1], par[4], par[5]);
        let x0 = if par[6] == 1.0 { Some(cam_of(&fx.v("x0", &nm))) } else { None };
        // resect_init 6-dof composes planar_pnp (Jacobi SVD vs LAPACK), whose tolerance is 1e-11, through c = p - R t;
        // the 4-dof init has no PnP and keeps the spec's 1e-12
        let (tol_pose, w) = match (is_init, dof == CamDof::Full, masked) {
            (true, true, false) => (1e-11, &mut wi6),
            (true, false, false) => (1e-12, &mut wi4),
            (true, true, true) => (1e-11, &mut mi6),
            (true, false, true) => (1e-12, &mut mi4),
            (false, _, false) => (1e-9, &mut wr),
            (false, _, true) => (1e-9, &mut mr),
        };
        let tol_pose = tol_pose * lf();
        let cam = if is_init {
            if masked {
                nmi += 1;
            } else {
                ni += 1;
            }
            resect_init(&obs, &map, a, &cm, roll, pitch, dof).unwrap_or_else(|e| panic!("{nm}: {e:?}"))
        } else {
            if masked {
                nmr += 1;
            } else {
                nr += 1;
            }
            let opts = ResectOptions {
                x0,
                sigma_px: sg,
                noise_model: nm_of(par[2]),
                loss: if par[3] == 1.0 { Loss::Huber { delta: 1.5 } } else { Loss::Linear },
                dof,
            };
            let res = resect(&obs, &map, a, &cm, roll, pitch, &opts).unwrap_or_else(|e| panic!("{nm}: {e:?}"));
            w.check(&nm, "cov", mat_rel(res.cov.as_slice(), fx.m("cov", &nm).as_slice()), 1e-8 * lf());
            w.check(&nm, "info", mat_rel(res.info.as_slice(), fx.m("info", &nm).as_slice()), 1e-8 * lf());
            // rms_px is ~1e-13 for noise-free scenes, so it is compared relative to max(1, |value|)
            w.check(&nm, "rms_px", err(&[res.rms_px], &[fx.s("rms", &nm)]), 1e-9 * lf());
            assert_eq!(res.n_markers as f64, fx.s("nmk", &nm), "{nm}: n_markers");
            assert_eq!(res.converged, fx.s("conv", &nm) == 1.0, "{nm}: converged");
            res.cam
        };
        w.check(&nm, "c", err(&cam.c, &fx.v("c", &nm)), tol_pose);
        match dof {
            CamDof::Yaw => w.check(&nm, "yaw", err(&[cam.yaw], &[fx.s("yaw", &nm)]), tol_pose),
            CamDof::Full => w.check(&nm, "R", err(&f9(&cam.r()), &fx.v("R", &nm)), tol_pose),
        }
    }
    wi4.report("resect_init 4dof", ni / 2);
    wi6.report("resect_init 6dof", ni / 2);
    wr.report("resect", nr);
    mi4.report("masked resect_init 4dof", nmi / 2);
    mi6.report("masked resect_init 6dof", nmi / 2);
    mr.report("masked resect", nmr);
    fx.finish("resection", &["rs_", "ms_init_", "ms_rs_"]);
}

// ---------------------------------------------------------------- ba_init / bundle_adjust
use erploc_est::ba::{bundle_adjust, BaResult};
use erploc_est::ba_init::initialize;
use erploc_est::ba_types::{BaError, BaProblem, BaState};

fn wrap_pi(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    x - t * (x / t).round()
}

fn ba_problem(fx: &Fx, nm: &str) -> (BaProblem, Loss) {
    let frames = case_frames(fx, nm, "obs");
    let rp_d = fx.m("rp", nm);
    let rp: Vec<[f64; 2]> = (0..rp_d.rows())
        .map(|i| {
            let r = row(rp_d, i);
            [r[0], r[1]]
        })
        .collect();
    let sv = fx.m("sv", nm);
    let surveyed: BTreeMap<i64, ([f64; 3], f64)> = (0..sv.rows())
        .map(|i| {
            let r = row(sv, i);
            (r[0] as i64, ([r[1], r[2], r[3]], r[4]))
        })
        .collect();
    let par = fx.v("par", nm);
    let problem = BaProblem {
        frames,
        rp,
        surveyed,
        a: par[0],
        cam_model: cam_model(),
        sigma_px: par[1],
        noise_model: nm_of(par[2]),
        cam_dof: if par[3] == 1.0 { CamDof::Full } else { CamDof::Yaw },
    };
    let loss = if par[4] == 1.0 { Loss::Huber { delta: par[5] } } else { Loss::Linear };
    (problem, loss)
}

/// Worst camera (c, yaw or R) and marker (p, R) difference between a state and the exported `cams` / `mk` records.
fn check_state(w: &mut Worst, fx: &Fx, nm: &str, st: &BaState, dof: CamDof, tol: f64) {
    let tol = tol * lf();
    let cams = fx.m("cams", nm);
    assert_eq!(st.cams.len(), cams.rows(), "{nm}: number of frames");
    for (f, cam) in st.cams.iter().enumerate() {
        let r = row(cams, f);
        assert_eq!(cam.is_some(), r[0] == 1.0, "{nm}: frame {f} presence");
        if let Some(cam) = cam {
            w.check(nm, "cam c", err(&cam.c, &r[1..4]), tol);
            w.check(nm, "cam R", err(&f9(&cam.r()), &r[5..14]), tol);
            if dof == CamDof::Yaw {
                w.check(nm, "cam yaw", err(&[wrap_pi(cam.yaw - r[4])], &[0.0]), tol);
            }
        }
    }
    let mk = mks_of(fx.m("mk", nm));
    assert_eq!(st.markers.len(), mk.len(), "{nm}: number of markers");
    for e in &mk {
        let g = st.markers.get(&e.id).unwrap_or_else(|| panic!("{nm}: marker {} missing", e.id));
        w.check(nm, "marker p", err(&g.p, &e.p), tol);
        // bi_pixel_*: the marker rotation is the weakest direction of the LM fit stopped at ftol/xtol 1e-12; measured worst 3.3e-10
        // (debug) but 1.15e-8 (release), while cam c / R / yaw and marker p stay <= 7e-10
        // (off the reference platform: bi_noisy marker R to about 1e-7)
        let tol_r = if nm.starts_with("bi_pixel_") { 2e-8 * lf() } else { tol };
        w.check(nm, "marker R", err(&f9(&g.r), &f9(&e.r)), tol_r);
    }
}

#[test]
fn ba_init_matches_python() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut wm = Worst::new();
    let (mut n, mut nm_) = (0, 0);
    let mut names = fx.names("bi_");
    names.extend(fx.names("ms_bi_"));
    names.extend(fx.names("ms_thr_"));
    for nm in names {
        let (problem, _) = ba_problem(&fx, &nm);
        let masked = nm.starts_with("ms_");
        if nm.contains("bi_errs") {
            let code = fx.s("code", &nm);
            let e = initialize(&problem, 1).expect_err(&nm);
            match (code as i64, &e) {
                (0, BaError::NoSeedFrame) | (1, BaError::Disconnected { .. }) => {}
                _ => panic!("{nm}: python error code {code}, rust {e:?}"),
            }
        } else {
            let st = initialize(&problem, 1).unwrap_or_else(|e| panic!("{nm}: {e:?}"));
            // Full DoF: every marker fit / resection is an LM stopped at its tolerance, so differences of the planar_pnp
            // start (1e-11) surface at ~1e-9; Python itself moves by 2e-10 under a 1e-12 px input perturbation
            // ms_thr_*: the at-threshold frame is resected from a single marker (3 corners = 6 rows for 4 unknowns, or 8 corners for
            // 6 unknowns): worse conditioned, measured worst 2.5e-9 (cam c) against 1e-9 elsewhere
            let tol = if nm.starts_with("ms_thr_") {
                1e-8
            } else if problem.cam_dof == CamDof::Full {
                2e-8
            } else {
                1e-9
            };
            check_state(if masked { &mut wm } else { &mut w }, &fx, &nm, &st, problem.cam_dof, tol);
        }
        if masked {
            nm_ += 1;
        } else {
            n += 1;
        }
    }
    w.report("ba_init", n);
    wm.report("masked ba_init", nm_);
    fx.finish("ba_init", &["bi_", "ms_bi_", "ms_thr_"]);
}

fn cov_cams_flat(r: &BaResult, width: usize) -> Vec<f64> {
    r.cov_cams
        .iter()
        .flat_map(|c| match c {
            Some(m) => m.as_slice().to_vec(),
            None => vec![0.0; width * width],
        })
        .collect()
}

#[test]
fn bundle_adjust_matches_python() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut n = 0;
    let mut names = fx.names("ba_");
    names.extend(fx.names("ms_ba_"));
    for nm in names {
        let (problem, loss) = ba_problem(&fx, &nm);
        let res = bundle_adjust(&problem, None, loss, 50).unwrap_or_else(|e| panic!("{nm}: {e:?}"));
        check_state(&mut w, &fx, &nm, &res.state, problem.cam_dof, 1e-9);
        let width = if problem.cam_dof == CamDof::Full { 6 } else { 4 };
        // covariance of a marker: 6x6 per id, compared as one matrix relative to its largest entry
        let cm = fx.m("covm", &nm);
        assert_eq!(res.cov_markers.len(), cm.rows(), "{nm}: number of cov_markers");
        for i in 0..cm.rows() {
            let r = row(cm, i);
            let g = res.cov_markers.get(&(r[0] as i64)).unwrap_or_else(|| panic!("{nm}: cov id {}", r[0]));
            w.check(&nm, "cov_markers", mat_rel(g.as_slice(), &r[1..]), 1e-8 * lf());
        }
        // None <=> NaN rows in Python (exported as zeros; the exporter asserts the NaN pattern equals the None pattern)
        let cc = fx.m("covc", &nm);
        for (f, c) in res.cov_cams.iter().enumerate() {
            assert_eq!(c.is_some(), res.state.cams[f].is_some(), "{nm}: cov_cams presence {f}");
        }
        w.check(&nm, "cov_cams", mat_rel(&cov_cams_flat(&res, width), cc.as_slice()), 1e-8 * lf());
        w.check(&nm, "rms_px", err(&[res.rms_px], &[fx.s("rms", &nm)]), 1e-9 * lf());
        let g = fx.v("gauge", &nm);
        assert_eq!(res.gauge.null_dim_noprior as f64, g[0], "{nm}: null_dim_noprior");
        w.check(&nm, "eig_min_ratio", (res.gauge.eig_min_ratio - g[1]).abs(), 1e-12 * lf());
        assert_eq!(res.converged, fx.s("conv", &nm) == 1.0, "{nm}: converged");
        n += 1;
    }
    w.report("bundle_adjust", n);
    fx.finish("bundle_adjust", &["ba_", "ms_ba_"]);
}
