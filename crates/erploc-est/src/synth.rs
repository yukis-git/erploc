//! Synthetic data generation (layouts, trajectories, rendering, noise, survey). Port of `erploc/synth.py`.
//!
//! The order of random draws is identical to numpy's, so a replay source can reproduce Python exactly.
//! Python's string `kind` / `noise_model` arguments are enums, so the `ValueError` cases are unrepresentable.
//! Python's `obstacles` / `vp` visibility path is replaced by the optional `mask` argument of [`render`].
//! `Observation` carries the per-corner mask and sigma scale. Python's `render(partial=True, detect=...)` (per-corner
//! occlusion via `visibility.occluded`) has no analogue here, because `render` takes a precomputed per-marker `mask`
//! instead of obstacles; the per-corner occlusion render lives in `erploc-place` (`partial::render_partial`).

use erploc_core::erp::{lamphi_to_px, px_to_lamphi, wrap_pi, ErpCamera, NoiseModel};
use erploc_core::jacobian::{predict_marker, CamPose};
use erploc_core::markers::{upright_pose, MarkerPose};
use erploc_core::{V2, V3};
use erploc_linalg::rng::RandomSource;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Incidence-angle limit \[deg\] of the unobstructed visibility cull (front side only).
pub const PSI_MAX_RENDER_DEG: f64 = 85.0;

/// Same shape as the detector output: 4 corners in ERP pixels, with the per-corner validity mask and sigma
/// scale of the per-corner interface (`corner_valid` / `corner_sigma_scale`).
///
/// `corner_mask[k] == false` means corner k was not observed: the estimators never read `uv[k]` (it may be NaN) and
/// build no residual rows for it. `corner_sigma_scale[k]` multiplies sigma of corner k (must be finite and > 0,
/// checked by the estimators, which panic otherwise: Python raises `ValueError`). Defaults: all valid, all 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub id: i64,
    pub uv: [V2; 4],
    pub corner_mask: [bool; 4],
    pub corner_sigma_scale: [f64; 4],
}

impl Observation {
    /// All corners valid, all scales 1 (Python `Observation(id, uv)` with the two optional fields `None`).
    pub fn new(id: i64, uv: [V2; 4]) -> Self {
        Observation { id, uv, corner_mask: [true; 4], corner_sigma_scale: [1.0; 4] }
    }
}

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub cam: ErpCamera,
    pub a: f64,
    pub sigma_px: f64,
    pub noise_model: NoiseModel,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerLayout {
    Walls,
    Ring,
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryKind {
    Circle,
    RandomWalk,
    Waypoints,
}

/// (x_min, x_max, y_min, y_max).
pub type Room = [f64; 4];

/// All markers are upright (horizontal normal). Heights are uniform in `z_range`.
/// Walls: perimeter-equispaced, normals inward. Ring: radius 0.45 min(Lx, Ly), normals to the centre.
/// Random: uniform in the room with uniform azimuth.
pub fn make_markers(kind: MarkerLayout, n: usize, room: &Room, z_range: [f64; 2], rng: &mut dyn RandomSource) -> Vec<MarkerPose> {
    let [x0, x1, y0, y1] = *room;
    let (lx, ly) = (x1 - x0, y1 - y0);
    let mut out = Vec::with_capacity(n);
    match kind {
        MarkerLayout::Walls => {
            let per = 2.0 * (lx + ly);
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64 * per;
                let (px, py, az) = if t < lx {
                    (x0 + t, y0, PI / 2.0)
                } else if t < lx + ly {
                    (x1, y0 + (t - lx), PI)
                } else if t < 2.0 * lx + ly {
                    (x1 - (t - lx - ly), y1, -PI / 2.0)
                } else {
                    (x0, y1 - (t - 2.0 * lx - ly), 0.0)
                };
                let z = rng.uniform(z_range[0], z_range[1]);
                out.push(upright_pose(i as i64, &[px, py, z], az));
            }
        }
        MarkerLayout::Ring => {
            let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
            let rad = 0.45 * lx.min(ly);
            for i in 0..n {
                let th = 2.0 * PI * i as f64 / n as f64;
                let z = rng.uniform(z_range[0], z_range[1]);
                out.push(upright_pose(i as i64, &[cx + rad * th.cos(), cy + rad * th.sin(), z], th + PI));
            }
        }
        MarkerLayout::Random => {
            for i in 0..n {
                let x = rng.uniform(x0, x1);
                let y = rng.uniform(y0, y1);
                let z = rng.uniform(z_range[0], z_range[1]);
                let az = rng.uniform(-PI, PI);
                out.push(upright_pose(i as i64, &[x, y, z], az));
            }
        }
    }
    out
}

/// roll = pitch = 0, height `z_cam`. Yaw is the direction of travel (tangent for the circle).
pub fn make_trajectory(kind: TrajectoryKind, n_frames: usize, room: &Room, z_cam: f64, rng: &mut dyn RandomSource) -> Vec<CamPose> {
    let [x0, x1, y0, y1] = *room;
    let (cx, cy) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
    let mut cams = Vec::with_capacity(n_frames);
    match kind {
        TrajectoryKind::Circle => {
            let rad = 0.3 * (x1 - x0).min(y1 - y0);
            for i in 0..n_frames {
                let th = 2.0 * PI * i as f64 / n_frames as f64;
                let c = [cx + rad * th.cos(), cy + rad * th.sin(), z_cam];
                cams.push(CamPose::euler(c, wrap_pi(th + PI / 2.0), 0.0, 0.0));
            }
        }
        TrajectoryKind::RandomWalk => {
            let step = 0.15;
            let mut pos = [cx, cy];
            let mut h = rng.uniform(-PI, PI);
            for _ in 0..n_frames {
                for _try in 0..8 {
                    let nxt = [pos[0] + step * h.cos(), pos[1] + step * h.sin()];
                    if nxt[0] < x0 || nxt[0] > x1 {
                        h = PI - h;
                        continue;
                    }
                    if nxt[1] < y0 || nxt[1] > y1 {
                        h = -h;
                        continue;
                    }
                    break;
                }
                h = wrap_pi(h);
                cams.push(CamPose::euler([pos[0], pos[1], z_cam], h, 0.0, 0.0));
                pos = [pos[0] + step * h.cos(), pos[1] + step * h.sin()];
                h = wrap_pi(h + 0.3 * rng.normal());
            }
        }
        TrajectoryKind::Waypoints => {
            let mut w = [[0.0; 2]; 4];
            for p in w.iter_mut() {
                let x = rng.uniform(x0, x1);
                let y = rng.uniform(y0, y1);
                *p = [x, y];
            }
            let mut per_seg = [n_frames / 3; 3];
            per_seg[2] += n_frames - per_seg.iter().sum::<usize>();
            for s in 0..3 {
                let (a, b) = (w[s], w[s + 1]);
                let az = (b[1] - a[1]).atan2(b[0] - a[0]);
                for k in 0..per_seg[s] {
                    let f = k as f64 / per_seg[s].max(1) as f64;
                    let p = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
                    cams.push(CamPose::euler([p[0], p[1], z_cam], az, 0.0, 0.0));
                }
            }
        }
    }
    cams
}

/// Noise-free observations. `mask[f][j]` (if given) replaces the default visibility cull.
/// lam goes through `wrap_pi` before conversion, so u lies in [-0.5, W-0.5).
/// Panics if a given mask does not cover every (frame, marker) pair.
pub fn render(cams: &[CamPose], markers: &[MarkerPose], cfg: &SynthConfig, mask: Option<&[Vec<bool>]>) -> Vec<Vec<Observation>> {
    let cos_max = PSI_MAX_RENDER_DEG.to_radians().cos();
    let mut frames = Vec::with_capacity(cams.len());
    for (f, cam) in cams.iter().enumerate() {
        let mut obs = Vec::new();
        for (j, mk) in markers.iter().enumerate() {
            match mask {
                Some(m) => {
                    if !m[f][j] {
                        continue;
                    }
                }
                None => {
                    let u = [cam.c[0] - mk.p[0], cam.c[1] - mk.p[1], cam.c[2] - mk.p[2]];
                    let dist = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    if dist < 1e-9 {
                        continue;
                    }
                    let ndot = mk.r.0[0][2] * u[0] + mk.r.0[1][2] * u[1] + mk.r.0[2][2] * u[2];
                    if ndot / dist <= cos_max {
                        continue;
                    }
                }
            }
            let lp = predict_marker(cam, mk, cfg.a);
            let lp: Vec<V2> = lp.iter().map(|p| [wrap_pi(p[0]), p[1]]).collect();
            let px = lamphi_to_px(&lp, &cfg.cam);
            obs.push(Observation::new(mk.id, [px[0], px[1], px[2], px[3]]));
        }
        frames.push(obs);
    }
    frames
}

/// New frames (input untouched). Tangent: isotropic `sigma_px * s` \[rad\] on the tangent plane
/// (dlam = eps / cos phi). Pixel: `sigma_px` \[px\] on u and v. The noise of corner k is multiplied by
/// `corner_sigma_scale[k]` (a scale of 1 is bit-identical to the unscaled operation); all 4 corners draw noise
/// whatever the mask (the draw order is fixed), and mask and scale are passed through. A masked corner's uv is
/// perturbed like any other (a NaN stays NaN).
pub fn add_noise(frames: &[Vec<Observation>], cfg: &SynthConfig, rng: &mut dyn RandomSource) -> Vec<Vec<Observation>> {
    let sig_ang = cfg.sigma_px * cfg.cam.s();
    frames
        .iter()
        .map(|fr| {
            fr.iter()
                .map(|ob| {
                    let uv = match cfg.noise_model {
                        NoiseModel::Tangent => {
                            let lp = px_to_lamphi(&ob.uv, &cfg.cam);
                            let mut lam = [0.0; 4];
                            let mut phi = [0.0; 4];
                            for k in 0..4 {
                                lam[k] = lp[k][0] + sig_ang * (rng.normal() * ob.corner_sigma_scale[k]) / lp[k][1].cos();
                            }
                            for k in 0..4 {
                                phi[k] = lp[k][1] + sig_ang * (rng.normal() * ob.corner_sigma_scale[k]);
                            }
                            let nl: Vec<V2> = (0..4).map(|k| [wrap_pi(lam[k]), phi[k]]).collect();
                            let px = lamphi_to_px(&nl, &cfg.cam);
                            [px[0], px[1], px[2], px[3]]
                        }
                        NoiseModel::Pixel => {
                            let mut uv = ob.uv;
                            for (p, &sc) in uv.iter_mut().zip(&ob.corner_sigma_scale) {
                                p[0] += cfg.sigma_px * (rng.normal() * sc);
                                p[1] += cfg.sigma_px * (rng.normal() * sc);
                            }
                            uv
                        }
                    };
                    Observation { id: ob.id, uv, corner_mask: ob.corner_mask, corner_sigma_scale: ob.corner_sigma_scale }
                })
                .collect()
        })
        .collect()
}

/// Total-station survey values: position plus N(0, sigma_ts^2). Panics if an id is missing.
pub fn survey(markers: &[MarkerPose], ids: &[i64], sigma_ts: f64, rng: &mut dyn RandomSource) -> BTreeMap<i64, (V3, f64)> {
    let mut out = BTreeMap::new();
    for &i in ids {
        // last match wins, like Python's {mk.id: mk} dict
        let mk = markers.iter().rfind(|m| m.id == i).unwrap_or_else(|| panic!("survey: marker id {i} not found"));
        let mut p = mk.p;
        for v in p.iter_mut() {
            *v += sigma_ts * rng.normal();
        }
        out.insert(i, (p, sigma_ts));
    }
    out
}

#[cfg(test)]
mod tests {
    /// `max` that propagates NaN like numpy (`f64::max` drops it, which would let a NaN result pass a `<` check).
    fn nmax(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    }

    use super::*;
    use erploc_core::erp::{lamphi_to_bearing, residual_lamphi};
    use erploc_core::markers::corners_world;
    use erploc_linalg::rng::Pcg64;
    use erploc_testkit::{mean, std_ddof1};

    const ROOM: Room = [-3.0, 3.0, -2.5, 2.5];
    const ZR: [f64; 2] = [1.0, 2.0];

    fn cfg(nm: NoiseModel, seed: u64) -> SynthConfig {
        SynthConfig { cam: ErpCamera::new(6528, 3264, 1), a: 0.16, sigma_px: 0.3, noise_model: nm, seed }
    }

    #[test]
    fn render_roundtrip_bearing() {
        let c = cfg(NoiseModel::Tangent, 5);
        let mut rng = Pcg64::new(c.seed);
        let mks = make_markers(MarkerLayout::Walls, 16, &ROOM, ZR, &mut rng);
        let cams = make_trajectory(TrajectoryKind::Circle, 12, &ROOM, 1.2, &mut rng);
        let frames = render(&cams, &mks, &c, None);
        let mut worst: f64 = 0.0;
        let mut n = 0;
        for (cam, fr) in cams.iter().zip(&frames) {
            let r = cam.r();
            for ob in fr {
                n += 1;
                let mk = mks.iter().find(|m| m.id == ob.id).unwrap();
                let xw = corners_world(&mk.p, &mk.r, c.a);
                let d2 = lamphi_to_bearing(&px_to_lamphi(&ob.uv, &c.cam));
                for k in 0..4 {
                    let v = [xw[k][0] - cam.c[0], xw[k][1] - cam.c[1], xw[k][2] - cam.c[2]];
                    let mut d = [0.0; 3];
                    for (j, dj) in d.iter_mut().enumerate() {
                        *dj = v[0] * r.0[0][j] + v[1] * r.0[1][j] + v[2] * r.0[2][j];
                    }
                    let nr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let e = (0..3).map(|j| (d[j] / nr - d2[k][j]).powi(2)).sum::<f64>().sqrt();
                    worst = nmax(worst, e);
                }
            }
        }
        assert!(n > 0);
        println!("roundtrip worst {worst:.3e} over {n} obs");
        assert!(worst < 1e-12, "{worst}");
    }

    fn noisy_setup() -> (Vec<Vec<Observation>>, usize, Pcg64) {
        let c = cfg(NoiseModel::Tangent, 5);
        let mut rng = Pcg64::new(11);
        let bm = make_markers(MarkerLayout::Walls, 50, &ROOM, ZR, &mut rng);
        let bc = make_trajectory(TrajectoryKind::Circle, 100, &ROOM, 1.2, &mut rng);
        let f0 = render(&bc, &bm, &c, None);
        let n_corner = 4 * f0.iter().map(|f| f.len()).sum::<usize>();
        (f0, n_corner, rng)
    }

    #[test]
    fn noise_tangent_sigma() {
        let c = cfg(NoiseModel::Tangent, 5);
        let (f0, n_corner, mut rng) = noisy_setup();
        let fn_ = add_noise(&f0, &c, &mut rng);
        let (mut r0, mut r1) = (vec![], vec![]);
        for (fa, fb) in f0.iter().zip(&fn_) {
            for (o0, o1) in fa.iter().zip(fb) {
                let (r, _) = residual_lamphi(&px_to_lamphi(&o1.uv, &c.cam), &px_to_lamphi(&o0.uv, &c.cam), &c.cam, NoiseModel::Tangent);
                for p in r {
                    r0.push(p[0]);
                    r1.push(p[1]);
                }
            }
        }
        let dev = nmax((std_ddof1(&r0) / c.sigma_px - 1.0).abs(), (std_ddof1(&r1) / c.sigma_px - 1.0).abs());
        println!("tangent dev {:.4} %, corners {n_corner}", 100.0 * dev);
        assert!(dev < 0.03 && n_corner >= 20000);
    }

    #[test]
    fn noise_pixel_sigma() {
        let c = cfg(NoiseModel::Pixel, 5);
        let (f0, _, _) = noisy_setup();
        let fp = add_noise(&f0, &c, &mut Pcg64::new(12));
        let (mut d0, mut d1) = (vec![], vec![]);
        for (fa, fb) in f0.iter().zip(&fp) {
            for (o0, o1) in fa.iter().zip(fb) {
                for k in 0..4 {
                    d0.push(o1.uv[k][0] - o0.uv[k][0]);
                    d1.push(o1.uv[k][1] - o0.uv[k][1]);
                }
            }
        }
        let dev = nmax((std_ddof1(&d0) / c.sigma_px - 1.0).abs(), (std_ddof1(&d1) / c.sigma_px - 1.0).abs());
        println!("pixel dev {:.4} %", 100.0 * dev);
        assert!(dev < 0.03);
    }

    #[test]
    fn survey_gaussian() {
        let mut rng0 = Pcg64::new(5);
        let mks = make_markers(MarkerLayout::Walls, 16, &ROOM, ZR, &mut rng0);
        let mut rng = Pcg64::new(3);
        let mut e = [vec![], vec![], vec![]];
        for _ in 0..4000 {
            let s = survey(&mks, &[0, 1, 2], 0.005, &mut rng);
            for i in 0..3 {
                for (ax, ev) in e.iter_mut().enumerate() {
                    ev.push(s[&(i as i64)].0[ax] - mks[i].p[ax]);
                }
            }
        }
        let n = e[0].len() as f64;
        let dev = e.iter().map(|v| (std_ddof1(v) / 0.005 - 1.0).abs()).fold(0.0, nmax);
        let mx = e.iter().map(|v| mean(v).abs()).fold(0.0, nmax);
        println!("survey sigma dev {:.3} %, max|mean| {mx:.3e} (limit {:.3e})", 100.0 * dev, 3.0 * 0.005 / n.sqrt());
        assert!(dev < 0.03 && mx < 3.0 * 0.005 / n.sqrt());
    }

    #[test]
    fn make_markers_geometry() {
        for kind in [MarkerLayout::Walls, MarkerLayout::Ring, MarkerLayout::Random] {
            let ms = make_markers(kind, 12, &ROOM, ZR, &mut Pcg64::new(1));
            assert_eq!(ms.len(), 12);
            for m in &ms {
                assert!(m.p[0] >= ROOM[0] - 1e-9 && m.p[0] <= ROOM[1] + 1e-9, "{kind:?}");
                assert!(m.p[1] >= ROOM[2] - 1e-9 && m.p[1] <= ROOM[3] + 1e-9, "{kind:?}");
                assert!(m.p[2] >= ZR[0] && m.p[2] <= ZR[1], "{kind:?}");
                let mut worst: f64 = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        let s: f64 = (0..3).map(|k| m.r.0[k][i] * m.r.0[k][j]).sum();
                        worst = nmax(worst, (s - if i == j { 1.0 } else { 0.0 }).abs());
                    }
                }
                assert!(worst < 1e-12, "{kind:?} {worst}");
                assert!(m.r.0[2][2].abs() < 1e-15, "{kind:?}");
                if kind != MarkerLayout::Random {
                    let inward = m.r.0[0][2] * (0.0 - m.p[0]) + m.r.0[1][2] * (0.0 - m.p[1]);
                    assert!(inward > 0.0, "{kind:?} id {}", m.id);
                }
            }
        }
    }

    #[test]
    fn make_trajectory_geometry() {
        let cs = make_trajectory(TrajectoryKind::Circle, 30, &ROOM, 1.2, &mut Pcg64::new(2));
        let e_c = cs.iter().map(|c| (c.yaw.cos() * c.c[0] + c.yaw.sin() * c.c[1]).abs()).fold(0.0, nmax);
        println!("circle tangent {e_c:.2e}");
        assert!(e_c < 1e-12);
        for kind in [TrajectoryKind::RandomWalk, TrajectoryKind::Waypoints] {
            let cs = make_trajectory(kind, 30, &ROOM, 1.2, &mut Pcg64::new(2));
            assert_eq!(cs.len(), 30);
            for c in &cs {
                assert!(c.c[0] >= ROOM[0] - 1e-9 && c.c[0] <= ROOM[1] + 1e-9, "{kind:?}");
                assert!(c.c[1] >= ROOM[2] - 1e-9 && c.c[1] <= ROOM[3] + 1e-9, "{kind:?}");
            }
            let mut e_m: f64 = 0.0;
            for i in 0..cs.len() - 1 {
                if kind == TrajectoryKind::Waypoints && wrap_pi(cs[i + 1].yaw - cs[i].yaw).abs() > 1e-12 {
                    continue;
                }
                let dv = [cs[i + 1].c[0] - cs[i].c[0], cs[i + 1].c[1] - cs[i].c[1]];
                if dv[0].hypot(dv[1]) > 1e-9 {
                    e_m = nmax(e_m, wrap_pi(dv[1].atan2(dv[0]) - cs[i].yaw).abs());
                }
            }
            println!("{kind:?} heading err {e_m:.2e}");
            assert!(e_m < 1e-12, "{kind:?} {e_m}");
        }
    }

    /// The noise of corner k scales with `corner_sigma_scale[k]`; unit scales and the default `new` are the same operation.
    #[test]
    fn noise_scales_per_corner() {
        for nm in [NoiseModel::Tangent, NoiseModel::Pixel] {
            let c = cfg(nm, 5);
            let (f0, _, _) = noisy_setup();
            let base = add_noise(&f0, &c, &mut Pcg64::new(7));
            let mut scaled = f0.clone();
            for ob in scaled.iter_mut().flatten() {
                ob.corner_sigma_scale = [1.0, 2.0, 0.5, 3.0];
                ob.corner_mask = [true, false, true, true];
            }
            let got = add_noise(&scaled, &c, &mut Pcg64::new(7));
            let (mut worst, mut n) = (0.0f64, 0);
            for ((a, b), o) in base.iter().flatten().zip(got.iter().flatten()).zip(f0.iter().flatten()) {
                assert_eq!(b.corner_mask, [true, false, true, true]);
                assert_eq!(b.corner_sigma_scale, [1.0, 2.0, 0.5, 3.0]);
                for k in 0..4 {
                    let (da, db) = (a.uv[k][1] - o.uv[k][1], b.uv[k][1] - o.uv[k][1]);
                    if da.abs() > 1e-9 {
                        worst = nmax(worst, (db / da - b.corner_sigma_scale[k]).abs());
                        n += 1;
                    }
                }
            }
            println!("{nm:?} per-corner scale ratio dev {worst:.2e} over {n}");
            assert!(n > 1000 && worst < 1e-6, "{worst}");
        }
    }
}
