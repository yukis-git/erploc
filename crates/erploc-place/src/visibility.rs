//! Visibility (port of `erploc/visibility.py`): occlusion by 2.5D obstacles, incidence angle, apparent size and
//! body self-occlusion sectors. Occlusion is by default judged on 13 face samples (`samples = true`); `samples = false`
//! is the legacy single centre ray.
//!
//! NaN/finite contract: inputs are finite. `occluded` uses Python's builtin `min`/`max` semantics (the second
//! argument wins only on a strict comparison), so NaN behaves as in Python. The `geom_mask` bounding-box prefilter does use `f64::min/max` where Python has
//! `np.minimum/np.maximum`; that cannot matter, because a non-finite cell or marker already has `base = false` (NaN psi).
#![allow(clippy::too_many_arguments)] // signatures mirror the Python call shapes
use crate::npcompat::{col_max, col_min, np_mod, TWO_PI};
use crate::scene::{segment_inside_intervals, Obstacle};
use erploc_core::erp::ErpCamera;
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::rot::{mat3_vec, rot_z};
use erploc_core::V3;
use erploc_est::synth::PSI_MAX_RENDER_DEG;
use erploc_linalg::{dot, norm};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sector {
    pub lam_min: f64,
    pub lam_max: f64,
    pub phi_min: f64,
    pub phi_max: f64,
}

/// `eps` is a fraction of the sight-line length (dimensionless), unlike the metre offsets of `candidates_from_walls`.
#[derive(Clone, Debug)]
pub struct VisParams {
    pub psi_max: f64,
    pub px_min: f64,
    pub sectors: Vec<Sector>,
    pub eps: f64,
}

impl VisParams {
    /// No sectors, `eps = 1e-3`.
    pub fn new(psi_max: f64, px_min: f64) -> Self {
        VisParams { psi_max, px_min, sectors: Vec::new(), eps: 1e-3 }
    }
}

/// Sight-line cut-off in front of a sample point \[m\] (same values as the Python stand-occlusion helpers).
pub const FRONT_GAP_WALL: f64 = 0.07;
pub const FRONT_GAP_STAND: f64 = 0.02;

fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Is the sight line `c -> m` blocked by an obstacle? The sight line is cut `gap` metres before `m` when `gap > 0`.
/// The XY inside-intervals are clipped to `[eps, t_end]`; z(t) is linear, so the minimum over an interval is at an end.
pub fn occluded(c: &V3, m: &V3, obstacles: &[Obstacle], eps: f64, gap: f64) -> bool {
    let dz = m[2] - c[2];
    let mut t_end = 1.0 - eps;
    if gap > 0.0 {
        // Python: min(t_end, 1 - gap / max(norm, 1e-12)); builtin max/min pick the 2nd arg only on strict <, >.
        let d = norm(&sub(m, c));
        let d = if 1e-12 > d { 1e-12 } else { d };
        let x = 1.0 - gap / d;
        if x < t_end {
            t_end = x;
        }
    }
    for ob in obstacles {
        for (t0, t1) in segment_inside_intervals([c[0], c[1]], [m[0], m[1]], &ob.poly) {
            let t0 = if eps > t0 { eps } else { t0 };
            let t1 = if t_end < t1 { t_end } else { t1 };
            if t1 <= t0 {
                continue;
            }
            let (za, zb) = (c[2] + t0 * dz, c[2] + t1 * dz);
            let zmin = if zb < za { zb } else { za };
            if zmin < ob.height {
                return true;
            }
        }
    }
    false
}

/// 13 face points: the 4 tag corners `(-h,-h), (h,-h), (h,h), (-h,h)` (h = A/2) in the marker's first two axes, then the
/// 3x3 grid `{-A/3, 0, A/3}` with `b` outer and `a` inner. Point = `(p + a*col0) + b*col1`.
pub fn face_samples(mk: &MarkerPose, a: f64) -> [V3; 13] {
    let h = a / 2.0;
    let t = a / 3.0;
    let mut ab: Vec<(f64, f64)> = vec![(-h, -h), (h, -h), (h, h), (-h, h)];
    for b in [-t, 0.0, t] {
        for a_ in [-t, 0.0, t] {
            ab.push((a_, b));
        }
    }
    let r = &mk.r.0;
    let mut out = [[0.0; 3]; 13];
    for (k, &(a_, b)) in ab.iter().enumerate() {
        for i in 0..3 {
            out[k][i] = (mk.p[i] + a_ * r[i][0]) + b * r[i][1];
        }
    }
    out
}

/// True if any of the 13 sight lines is blocked.
pub fn occluded_samples(c: &V3, mk: &MarkerPose, a: f64, obstacles: &[Obstacle], eps: f64, gap: f64) -> bool {
    face_samples(mk, a).iter().any(|p| occluded(c, p, obstacles, eps, gap))
}

fn normal(mk: &MarkerPose) -> V3 {
    [mk.r.0[0][2], mk.r.0[1][2], mk.r.0[2][2]]
}

/// `acos(clip(n . (c - p)/|c - p|, -1, 1))`; `psi >= pi/2` is the back side. NaN stays NaN.
pub fn incidence_angle(c: &V3, mk: &MarkerPose) -> f64 {
    let v = sub(c, &mk.p);
    let d = norm(&v);
    (dot(&normal(mk), &v) / d).clamp(-1.0, 1.0).acos()
}

/// Apparent short side \[px\] = `A cos(psi) / (D s)`; negative on the back side.
pub fn apparent_short_side_px(c: &V3, mk: &MarkerPose, a: f64, s: f64) -> f64 {
    let v = sub(c, &mk.p);
    let d = norm(&v);
    let cospsi = dot(&normal(mk), &v) / d;
    a * cospsi / (d * s)
}

/// Longitude membership with wrap-around (counter-clockwise from `lam_min` to `lam_max`).
pub fn lam_in(lam: f64, lam_min: f64, lam_max: f64) -> bool {
    let width = np_mod(lam_max - lam_min, TWO_PI);
    let off = np_mod(lam - lam_min, TWO_PI);
    lam_in_core(off, width)
}

/// The pure comparison of `lam_in` (factored for the Kani proof).
pub(crate) fn lam_in_core(off: f64, width: f64) -> bool {
    off <= width
}

/// Does the body-frame unit direction fall in any self-occlusion sector? Empty list: never.
pub fn in_sector(d_body: &V3, sectors: &[Sector]) -> bool {
    if sectors.is_empty() {
        return false;
    }
    let lam = d_body[1].atan2(d_body[0]);
    let phi = d_body[2].atan2(d_body[0].hypot(d_body[1]));
    sectors.iter().any(|s| s.phi_min <= phi && phi <= s.phi_max && lam_in(lam, s.lam_min, s.lam_max))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Back,
    Incidence,
    Size,
    Occluded,
    SelfOcclusion,
    Ok,
}

/// Single visibility decision. Order: back, incidence, size, occlusion, self-occlusion.
/// `samples = false` is the legacy centre ray (gap 0).
pub fn visible(
    c: &V3,
    heading: f64,
    mk: &MarkerPose,
    a: f64,
    cam: &ErpCamera,
    obstacles: &[Obstacle],
    vp: &VisParams,
    samples: bool,
    gap: f64,
) -> (bool, Reason) {
    let psi = incidence_angle(c, mk);
    if psi >= PI / 2.0 {
        return (false, Reason::Back);
    }
    if psi > vp.psi_max {
        return (false, Reason::Incidence);
    }
    if apparent_short_side_px(c, mk, a, cam.s()) < vp.px_min {
        return (false, Reason::Size);
    }
    let blocked = if samples { occluded_samples(c, mk, a, obstacles, vp.eps, gap) } else { occluded(c, &mk.p, obstacles, vp.eps, 0.0) };
    if blocked {
        return (false, Reason::Occluded);
    }
    let v = sub(&mk.p, c);
    let d = norm(&v);
    let u = [v[0] / d, v[1] / d, v[2] / d];
    let d_body = mat3_vec(&rot_z(-heading), &u);
    if in_sector(&d_body, &vp.sectors) {
        return (false, Reason::SelfOcclusion);
    }
    (true, Reason::Ok)
}

/// `np.broadcast_to(scalar, (n,))`.
pub fn gaps_uniform(g: f64, n: usize) -> Vec<f64> {
    vec![g; n]
}

fn gap_vec(gaps: Option<&[f64]>, n: usize) -> Vec<f64> {
    match gaps {
        None => gaps_uniform(FRONT_GAP_WALL, n),
        Some(g) => {
            assert_eq!(g.len(), n, "gaps must have one entry per marker");
            g.to_vec()
        }
    }
}

/// Heading-independent part: `base[k*n + j]` and `d_world[k*n + j]` (camera -> marker unit vector).
pub struct Geom {
    pub k: usize,
    pub n: usize,
    pub base: Vec<bool>,
    pub d_world: Vec<V3>,
}

/// Prefilter-free core shared with the test-only brute force: `prefilter = false` tests every pair still in `base`.
fn geom_mask_impl(
    cells: &[V3],
    markers: &[MarkerPose],
    a: f64,
    cam: &ErpCamera,
    obstacles: &[Obstacle],
    vp: &VisParams,
    samples: bool,
    gaps: Option<&[f64]>,
    prefilter: bool,
) -> Geom {
    let (k, n) = (cells.len(), markers.len());
    let gap = gap_vec(gaps, n);
    let s = cam.s();
    let mut base = vec![false; k * n];
    let mut d_world = vec![[0.0; 3]; k * n];
    for (ki, c) in cells.iter().enumerate() {
        for (j, mk) in markers.iter().enumerate() {
            let v = sub(c, &mk.p);
            let d = norm(&v);
            let cospsi = dot(&v, &normal(mk)) / d;
            let psi = cospsi.clamp(-1.0, 1.0).acos();
            let px = a * cospsi / (d * s);
            base[ki * n + j] = psi < PI / 2.0 && psi <= vp.psi_max && px >= vp.px_min;
            d_world[ki * n + j] = [-v[0] / d, -v[1] / d, -v[2] / d];
        }
    }
    for ob in obstacles {
        let lo = [col_min(&ob.poly, 0), col_min(&ob.poly, 1)];
        let hi = [col_max(&ob.poly, 0), col_max(&ob.poly, 1)];
        // np.nonzero(base) order (k-major), taken before this obstacle's updates.
        let pairs: Vec<(usize, usize)> = (0..k * n).filter(|&i| base[i]).map(|i| (i / n, i % n)).collect();
        for (ki, j) in pairs {
            let (c, p) = (&cells[ki], &markers[j].p);
            if prefilter {
                // The face points lie within 0.75 A of the centre: widen the sight-line XY box by A.
                let w = if samples { a } else { 0.0 };
                let mut near = true;
                for i in 0..2 {
                    let seg_lo = c[i].min(p[i]) - w;
                    let seg_hi = c[i].max(p[i]) + w;
                    if seg_hi < lo[i] || seg_lo > hi[i] {
                        near = false;
                    }
                }
                if !near {
                    continue;
                }
            }
            if !base[ki * n + j] {
                continue;
            }
            let one = std::slice::from_ref(ob);
            let hit = if samples {
                face_samples(&markers[j], a).iter().any(|q| occluded(c, q, one, vp.eps, gap[j]))
            } else {
                occluded(c, p, one, vp.eps, 0.0)
            };
            if hit {
                base[ki * n + j] = false;
            }
        }
    }
    Geom { k, n, base, d_world }
}

/// Heading-independent visibility. `gaps = None` means `FRONT_GAP_WALL` for every marker; otherwise one per marker.
pub fn geom_mask(
    cells: &[V3],
    markers: &[MarkerPose],
    a: f64,
    cam: &ErpCamera,
    obstacles: &[Obstacle],
    vp: &VisParams,
    samples: bool,
    gaps: Option<&[f64]>,
) -> Geom {
    geom_mask_impl(cells, markers, a, cam, obstacles, vp, samples, gaps, true)
}

fn sector_blocked(d_world: &V3, heading: f64, sectors: &[Sector]) -> bool {
    // numpy `d_world @ rot_z(-h).T`: d_body[i] = sum_c d_world[c] * Rz(-h)[i][c], i.e. `mat3_vec`.
    in_sector(&mat3_vec(&rot_z(-heading), d_world), sectors)
}

/// (K, Hd, N) flat `[(k*hd + h)*n + j]`.
pub fn visible_batch(
    cells: &[V3],
    headings: &[f64],
    markers: &[MarkerPose],
    a: f64,
    cam: &ErpCamera,
    obstacles: &[Obstacle],
    vp: &VisParams,
    samples: bool,
    gaps: Option<&[f64]>,
) -> Vec<bool> {
    let g = geom_mask(cells, markers, a, cam, obstacles, vp, samples, gaps);
    let (k, n, hd) = (g.k, g.n, headings.len());
    let mut out = vec![false; k * hd * n];
    for ki in 0..k {
        for (h, &ang) in headings.iter().enumerate() {
            for j in 0..n {
                let i = ki * n + j;
                out[(ki * hd + h) * n + j] = g.base[i] && !sector_blocked(&g.d_world[i], ang, &vp.sectors);
            }
        }
    }
    out
}

/// (F, N) flat `[f*n + j]`; frame `f` uses `headings[f]`.
pub fn visible_pairs(
    cells: &[V3],
    headings: &[f64],
    markers: &[MarkerPose],
    a: f64,
    cam: &ErpCamera,
    obstacles: &[Obstacle],
    vp: &VisParams,
    samples: bool,
    gaps: Option<&[f64]>,
) -> Vec<bool> {
    assert_eq!(cells.len(), headings.len(), "visible_pairs: one heading per frame");
    let g = geom_mask(cells, markers, a, cam, obstacles, vp, samples, gaps);
    let n = g.n;
    let mut out = vec![false; g.k * n];
    for (f, &ang) in headings.iter().enumerate() {
        for j in 0..n {
            let i = f * n + j;
            out[i] = g.base[i] && !sector_blocked(&g.d_world[i], ang, &vp.sectors);
        }
    }
    out
}

/// Mask for `erploc_est::synth::render(.., Some(&mask))`, built like Python `synth.render(obstacles=..)`: samples = true,
/// default wall gap, headings = camera yaw, cells = camera positions. `vp = None` is `psi_max = 85 deg`, `px_min = 0`,
/// no sectors. Note: with no obstacles and no `vp` Python's `render` uses its plain cull (`psi <= 85 deg` is kept
/// there, `ndot/dist <= cos_max` drops exactly 85 deg), whereas this mask path keeps exactly 85 deg (`psi <= psi_max`).
/// A camera exactly at a marker gives a NaN angle and is dropped (as in Python).
pub fn render_mask(
    cams: &[CamPose],
    markers: &[MarkerPose],
    a: f64,
    cam_model: &ErpCamera,
    obstacles: &[Obstacle],
    vp: Option<&VisParams>,
) -> Vec<Vec<bool>> {
    let default = VisParams::new(PSI_MAX_RENDER_DEG.to_radians(), 0.0);
    let vp = vp.unwrap_or(&default);
    let cells: Vec<V3> = cams.iter().map(|c| c.c).collect();
    let headings: Vec<f64> = cams.iter().map(|c| c.yaw).collect();
    let flat = visible_pairs(&cells, &headings, markers, a, cam_model, obstacles, vp, true, None);
    let n = markers.len();
    (0..cams.len()).map(|f| flat[f * n..(f + 1) * n].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_core::markers::upright_pose;
    use erploc_linalg::rng::Pcg64;

    const A: f64 = 0.16;

    fn cam() -> ErpCamera {
        ErpCamera::new(6528, 3264, 1)
    }
    fn vp70() -> VisParams {
        VisParams::new(70f64.to_radians(), 12.0)
    }
    fn rect(x0: f64, y0: f64, x1: f64, y1: f64, h: f64) -> Obstacle {
        Obstacle { poly: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]], height: h }
    }
    fn vis(c: &V3, hd: f64, mk: &MarkerPose, obs: &[Obstacle], vp: &VisParams) -> (bool, Reason) {
        visible(c, hd, mk, A, &cam(), obs, vp, true, FRONT_GAP_WALL)
    }

    #[test]
    fn b2_wall_cases() {
        let mk = upright_pose(0, &[4.0, 0.0, 1.5], PI);
        let c = [0.0, 0.0, 1.2];
        let vp = vp70();
        assert_eq!(vis(&c, 0.0, &mk, &[rect(1.9, -1., 2.1, 1., 2.0)], &vp), (false, Reason::Occluded));
        assert_eq!(vis(&c, 0.0, &mk, &[rect(1.9, -1., 2.1, 1., 1.0)], &vp), (true, Reason::Ok));
        assert_eq!(vis(&c, 0.0, &mk, &[rect(1.9, 2., 2.1, 4., 2.0)], &vp), (true, Reason::Ok));
    }

    #[test]
    fn b2_psi_boundary() {
        let vp = vp70();
        let mk = upright_pose(1, &[0.0, 0.0, 1.5], 0.0);
        for (dpsi, want) in [(-1e-4, (true, Reason::Ok)), (1e-4, (false, Reason::Incidence))] {
            let psi = vp.psi_max + dpsi;
            let cc = [mk.p[0] + 3.0 * psi.cos(), mk.p[1] + 3.0 * psi.sin(), mk.p[2]];
            assert_eq!(vis(&cc, 0.0, &mk, &[], &vp), want);
        }
    }

    #[test]
    fn b2_px_boundary() {
        let vp = vp70();
        let mk = upright_pose(1, &[0.0, 0.0, 1.5], 0.0);
        let s = cam().s();
        for (dpx, want, dd) in [(0.01, (true, Reason::Ok), 13.841), (-0.01, (false, Reason::Size), 13.864)] {
            let d = A / ((vp.px_min + dpx) * s);
            assert!((d - dd).abs() < 5e-4, "D = {d}");
            assert_eq!(vis(&[mk.p[0] + d, mk.p[1], mk.p[2]], 0.0, &mk, &[], &vp), want);
        }
    }

    #[test]
    fn b2_back_face() {
        let mk = upright_pose(1, &[0.0, 0.0, 1.5], 0.0);
        assert_eq!(vis(&[-3.0, 0.0, 1.5], 0.0, &mk, &[], &vp70()), (false, Reason::Back));
    }

    fn vp_s() -> VisParams {
        VisParams {
            sectors: vec![Sector {
                lam_min: -30f64.to_radians(),
                lam_max: 30f64.to_radians(),
                phi_min: -90f64.to_radians(),
                phi_max: 90f64.to_radians(),
            }],
            ..vp70()
        }
    }

    #[test]
    fn b2_sector_heading90() {
        let c0 = [0.0, 0.0, 1.2];
        let mk_y = upright_pose(2, &[0.0, 3.0, 1.5], -PI / 2.0);
        let mk_x = upright_pose(3, &[3.0, 0.0, 1.5], PI);
        assert_eq!(vis(&c0, PI / 2.0, &mk_y, &[], &vp_s()), (false, Reason::SelfOcclusion));
        assert_eq!(vis(&c0, PI / 2.0, &mk_x, &[], &vp_s()), (true, Reason::Ok));
    }

    #[test]
    fn b2_sector_wraps_pi() {
        let vp = VisParams {
            sectors: vec![Sector { lam_min: 150f64.to_radians(), lam_max: -150f64.to_radians(), phi_min: -PI / 2.0, phi_max: PI / 2.0 }],
            ..vp70()
        };
        let c0 = [0.0, 0.0, 1.2];
        let mk_px = upright_pose(20, &[3.0, 0.0, 1.5], PI);
        let mk_nx = upright_pose(21, &[-3.0, 0.0, 1.5], 0.0);
        assert_eq!(vis(&c0, 0.0, &mk_px, &[], &vp), (true, Reason::Ok));
        assert_eq!(vis(&c0, 0.0, &mk_nx, &[], &vp), (false, Reason::SelfOcclusion));
    }

    fn batch_scene() -> (Vec<V3>, Vec<f64>, Vec<MarkerPose>, Vec<Obstacle>) {
        let c = vec![[1.0, 1.0, 1.2], [4.0, 1.0, 1.2], [4.0, 3.0, 1.2], [6.5, 5.0, 1.2], [2.0, 5.0, 1.2]];
        let hd = vec![0.0, PI / 2.0, PI];
        let mks = vec![
            upright_pose(10, &[0.05, 3.0, 1.5], 0.0),
            upright_pose(11, &[7.95, 3.0, 1.5], PI),
            upright_pose(12, &[4.0, 0.05, 1.5], PI / 2.0),
            upright_pose(13, &[4.0, 5.95, 1.5], -PI / 2.0),
        ];
        (c, hd, mks, vec![rect(3., 2.5, 5., 3.5, 1.5)])
    }

    #[test]
    fn b2_batch_equals_loop() {
        let (c, hd, mks, obs) = batch_scene();
        let vp = vp_s();
        let m = visible_batch(&c, &hd, &mks, A, &cam(), &obs, &vp, true, None);
        assert_eq!(m.len(), 60);
        let mut mismatch = 0;
        for k in 0..c.len() {
            for h in 0..hd.len() {
                for j in 0..mks.len() {
                    let r = visible(&c[k], hd[h], &mks[j], A, &cam(), &obs, &vp, true, FRONT_GAP_WALL).0;
                    mismatch += (m[(k * hd.len() + h) * mks.len() + j] != r) as usize;
                }
            }
        }
        let t = m.iter().filter(|&&x| x).count();
        println!("mismatch {mismatch} / 60, True {t}");
        assert_eq!(mismatch, 0);
        assert_eq!(t, 31);
    }

    #[test]
    fn b2_pairs_equals_loop() {
        let (c, hd, mks, obs) = batch_scene();
        let vp = vp_s();
        let m = visible_pairs(&c[..3], &hd, &mks, A, &cam(), &obs, &vp, true, None);
        let mut mismatch = 0;
        for f in 0..3 {
            for j in 0..4 {
                let r = visible(&c[f], hd[f], &mks[j], A, &cam(), &obs, &vp, true, FRONT_GAP_WALL).0;
                mismatch += (m[f * 4 + j] != r) as usize;
            }
        }
        assert_eq!(mismatch, 0);
    }

    #[test]
    fn samples_false_is_centre_ray() {
        let mk = upright_pose(0, &[4.0, 0.0, 1.5], PI);
        let c = [0.0, 0.0, 1.2];
        let vp = vp70();
        let leg = |obs: &[Obstacle]| visible(&c, 0.0, &mk, A, &cam(), obs, &vp, false, 0.0);
        assert_eq!(leg(&[rect(1.9, -1., 2.1, 1., 2.0)]), (false, Reason::Occluded));
        assert_eq!(leg(&[rect(1.9, -1., 2.1, 1., 1.0)]), (true, Reason::Ok));
        assert_eq!(leg(&[rect(1.9, 2., 2.1, 4., 2.0)]), (true, Reason::Ok));
        // Thin pole on the sight line to the (+h,+h) corner (y = -0.04 at x = 2), clear of the centre ray (y = 0).
        let pole = [rect(1.95, -0.06, 2.05, -0.02, 2.0)];
        assert_eq!(visible(&c, 0.0, &mk, A, &cam(), &pole, &vp, true, FRONT_GAP_WALL), (false, Reason::Occluded));
        assert_eq!(visible(&c, 0.0, &mk, A, &cam(), &pole, &vp, false, FRONT_GAP_WALL), (true, Reason::Ok));
    }

    #[test]
    fn face_samples_layout() {
        let mk = upright_pose(0, &[4.0, 1.0, 1.5], PI);
        let s = face_samples(&mk, A);
        let (h, t) = (A / 2.0, A / 3.0);
        // col0 = (0,-1,0), col1 = (0,0,1) for this pose.
        let want_ab = [(-h, -h), (h, -h), (h, h), (-h, h)];
        for (k, &(a, b)) in want_ab.iter().enumerate() {
            assert!((s[k][1] - (1.0 - a)).abs() < 1e-15 && (s[k][2] - (1.5 + b)).abs() < 1e-15 && s[k][0] == 4.0);
        }
        let mut k = 4;
        for b in [-t, 0.0, t] {
            for a in [-t, 0.0, t] {
                assert!((s[k][1] - (1.0 - a)).abs() < 1e-15 && (s[k][2] - (1.5 + b)).abs() < 1e-15, "k = {k}");
                k += 1;
            }
        }
        for q in &s {
            assert!(norm(&sub(q, &mk.p)) <= 0.75 * A);
        }
    }

    #[test]
    fn gap_stand_vs_wall() {
        let m = [4.0, 0.0, 1.5];
        let c = [0.0, 0.0, 1.2];
        let pole = [rect(3.94, -1.0, 3.96, 1.0, 2.0)];
        assert!(occluded(&c, &m, &pole, 1e-3, FRONT_GAP_STAND));
        assert!(!occluded(&c, &m, &pole, 1e-3, FRONT_GAP_WALL));
        let mks = [upright_pose(0, &m, PI), upright_pose(1, &m, PI)];
        let vp = vp70();
        let g = [FRONT_GAP_STAND, FRONT_GAP_WALL];
        let r = visible_batch(&[c], &[0.0], &mks, A, &cam(), &pole, &vp, true, Some(&g));
        assert_eq!(r, vec![false, true]);
        let r = visible_batch(&[c], &[0.0], &mks, A, &cam(), &pole, &vp, true, None);
        assert_eq!(r, vec![true, true]);
    }

    fn rand_scene(rng: &mut Pcg64) -> (Vec<V3>, Vec<MarkerPose>, Vec<Obstacle>) {
        let cells: Vec<V3> = (0..20).map(|_| [rng.uniform(0.0, 8.0), rng.uniform(0.0, 6.0), 1.2]).collect();
        let markers: Vec<MarkerPose> = (0..10)
            .map(|i| upright_pose(i, &[rng.uniform(0.0, 8.0), rng.uniform(0.0, 6.0), rng.uniform(1.0, 2.0)], rng.uniform(-PI, PI)))
            .collect();
        let obs: Vec<Obstacle> = (0..4)
            .map(|_| {
                let (x, y) = (rng.uniform(0.5, 7.0), rng.uniform(0.5, 5.0));
                rect(x, y, x + rng.uniform(0.05, 1.0), y + rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.5))
            })
            .collect();
        (cells, markers, obs)
    }

    #[test]
    fn gap_zero_limit() {
        let mut rng = Pcg64::new(7);
        let mut hits = 0;
        for _ in 0..300 {
            let (cells, markers, obs) = rand_scene(&mut rng);
            let (c, m) = (&cells[0], &markers[0].p);
            let (eps, dz) = (1e-3, m[2] - c[2]);
            let mut want = false;
            for ob in &obs {
                for (t0, t1) in segment_inside_intervals([c[0], c[1]], [m[0], m[1]], &ob.poly) {
                    let (t0, t1) = (t0.max(eps), t1.min(1.0 - eps));
                    if t1 > t0 && (c[2] + t0 * dz).min(c[2] + t1 * dz) < ob.height {
                        want = true;
                    }
                }
            }
            hits += want as usize;
            assert_eq!(occluded(c, m, &obs, eps, 0.0), want);
        }
        assert!(hits > 10 && hits < 290, "hits = {hits}");
    }

    #[test]
    fn prefilter_conservative() {
        let mut rng = Pcg64::new(11);
        let vp = VisParams::new(85f64.to_radians(), 0.0);
        let (mut blocked, mut total) = (0, 0);
        for samples in [true, false] {
            for _ in 0..10 {
                let (cells, markers, obs) = rand_scene(&mut rng);
                let gaps: Vec<f64> = (0..markers.len()).map(|j| if j % 2 == 0 { FRONT_GAP_STAND } else { FRONT_GAP_WALL }).collect();
                let a = geom_mask(&cells, &markers, A, &cam(), &obs, &vp, samples, Some(&gaps));
                let b = geom_mask_impl(&cells, &markers, A, &cam(), &obs, &vp, samples, Some(&gaps), false);
                assert_eq!(a.base, b.base);
                let none = geom_mask(&cells, &markers, A, &cam(), &[], &vp, samples, Some(&gaps));
                blocked += none.base.iter().zip(&a.base).filter(|(x, y)| **x && !**y).count();
                total += a.base.len();
            }
        }
        assert_eq!(total, 4000);
        assert!(blocked > 50, "obstacles must matter: {blocked}");
    }

    #[test]
    fn render_mask_default_params() {
        let mk = upright_pose(0, &[0.0, 0.0, 1.5], 0.0);
        let cfg = cam();
        let at = |deg: f64| {
            let psi = deg.to_radians();
            CamPose::euler([3.0 * psi.cos(), 3.0 * psi.sin(), 1.5], 0.0, 0.0, 0.0)
        };
        let cams = [at(84.0), at(86.0), CamPose::euler([0.0, 0.0, 1.5], 0.0, 0.0, 0.0)];
        let m = render_mask(&cams, &[mk], A, &cfg, &[], None);
        assert_eq!(m, vec![vec![true], vec![false], vec![false]]);
        // Defaults: psi_max = 85 deg, px_min = 0, no sectors (a sector-free, size-free mask).
        let vp = VisParams::new(85f64.to_radians(), 0.0);
        assert_eq!(render_mask(&cams, &[mk], A, &cfg, &[], Some(&vp)), m);
        assert!(vp.sectors.is_empty() && vp.eps == 1e-3);
        // A camera 20 km away: apparent size far below any px_min, but default px_min = 0 keeps it.
        let far = [CamPose::euler([2.0e4, 0.0, 1.5], 0.0, 0.0, 0.0)];
        assert_eq!(render_mask(&far, &[mk], A, &cfg, &[], None), vec![vec![true]]);
        let strict = VisParams::new(85f64.to_radians(), 12.0);
        assert_eq!(render_mask(&far, &[mk], A, &cfg, &[], Some(&strict)), vec![vec![false]]);
    }

    #[test]
    fn render_mask_obstacles_and_render_no_skip() {
        use erploc_core::erp::NoiseModel;
        use erploc_est::synth::{render, SynthConfig};
        let mk = upright_pose(0, &[4.0, 0.0, 1.5], PI);
        let c = CamPose::euler([0.0, 0.0, 1.2], 0.0, 0.0, 0.0);
        let wall = [rect(1.9, -1., 2.1, 1., 2.0)];
        assert_eq!(render_mask(&[c], &[mk], A, &cam(), &wall, None), vec![vec![false]]);
        assert_eq!(render_mask(&[c], &[mk], A, &cam(), &[], None), vec![vec![true]]);
        // On the mask path `render` has no `dist < 1e-9` skip: a camera at the marker is rendered if the mask says so
        // (the visibility mask itself drops it: NaN angle, same as Python).
        let cfg = SynthConfig { cam: cam(), a: A, sigma_px: 0.3, noise_model: NoiseModel::Tangent, seed: 0 };
        let at_marker = CamPose::euler([4.0, 0.0, 1.5], 0.0, 0.0, 0.0);
        let frames = render(&[at_marker], &[mk], &cfg, Some(&[vec![true]]));
        assert_eq!(frames[0].len(), 1);
        let frames = render(&[at_marker], &[mk], &cfg, None);
        assert!(frames[0].is_empty());
    }

    // Python reference values below were computed with the Python reference (`erploc/visibility.py`).

    #[test]
    fn prefilter_corner_ray_outside_centre_bbox() {
        // Pole y in [-0.06,-0.02] lies outside the centre-ray box (y = 0) but blocks the (+h,+h) corner ray.
        let mk = upright_pose(0, &[4.0, 0.0, 1.5], PI);
        let pole = [rect(1.95, -0.06, 2.05, -0.02, 2.0)];
        let vp = vp70();
        let c = [[0.0, 0.0, 1.2]];
        let a = geom_mask(&c, &[mk], A, &cam(), &pole, &vp, true, None);
        let b = geom_mask_impl(&c, &[mk], A, &cam(), &pole, &vp, true, None, false);
        assert_eq!(a.base, vec![false]);
        assert_eq!(a.base, b.base);
        // Centre-ray mode: plain box, not blocked, and prefilter-on equals prefilter-off.
        let a = geom_mask(&c, &[mk], A, &cam(), &pole, &vp, false, None);
        let b = geom_mask_impl(&c, &[mk], A, &cam(), &pole, &vp, false, None, false);
        assert_eq!(a.base, vec![true]);
        assert_eq!(a.base, b.base);
    }

    #[test]
    fn eps_clipping() {
        // Python: occluded -> False for both.
        let (c, m) = ([0.0, 0.0, 1.2], [4.0, 0.0, 1.5]);
        // Interval [0, 0.0005] is entirely inside [0, eps): dropped by max(t0, eps).
        assert!(!occluded(&c, &m, &[rect(-1., -1., 0.002, 1., 2.0)], 1e-3, 0.0));
        // Interval [0.9995, 1] is entirely inside (1 - eps, 1]: dropped by t_end = 1 - eps.
        assert!(!occluded(&c, &m, &[rect(3.998, -1., 5., 1., 2.0)], 1e-3, 0.0));
        // With a smaller eps both are seen.
        assert!(occluded(&c, &m, &[rect(-1., -1., 0.002, 1., 2.0)], 1e-5, 0.0));
        assert!(occluded(&c, &m, &[rect(3.998, -1., 5., 1., 2.0)], 1e-5, 0.0));
    }

    #[test]
    fn equality_boundaries() {
        // z equality: Python False / True.
        let (c, m) = ([0.0, 0.0, 1.5], [4.0, 0.0, 1.5]);
        assert!(!occluded(&c, &m, &[rect(1., -1., 2., 1., 1.5)], 1e-3, 0.0));
        assert!(occluded(&c, &m, &[rect(1., -1., 2., 1., 1.5000001)], 1e-3, 0.0));
        // psi == pi/2 exactly: Python (False, 'back') and base False even with psi_max = pi, px_min = 0.
        let mk = upright_pose(0, &[0.0, 0.0, 1.5], 0.0);
        let vp = VisParams::new(PI, 0.0);
        let cc = [0.0, 3.0, 1.5];
        assert_eq!(incidence_angle(&cc, &mk), PI / 2.0);
        assert_eq!(vis(&cc, 0.0, &mk, &[], &vp), (false, Reason::Back));
        assert_eq!(geom_mask(&[cc], &[mk], A, &cam(), &[], &vp, true, None).base, vec![false]);
        // px == px_min exactly: Python (True, 'ok') and visible_pairs True.
        let cc = [3.0, 1.0, 1.5];
        let px = apparent_short_side_px(&cc, &mk, A, cam().s());
        let vq = VisParams::new(PI, px);
        assert_eq!(vis(&cc, 0.0, &mk, &[], &vq), (true, Reason::Ok));
        assert_eq!(visible_pairs(&[cc], &[0.0], &[mk], A, &cam(), &[], &vq, true, None), vec![true]);
    }

    #[test]
    fn lam_in_edges() {
        // Python _lam_in: [True, True, False, False] and wrap [True, True, True].
        assert_eq!([0.5, 1.5, 1.5 + 1e-9, 0.5 - 1e-9].map(|l| lam_in(l, 0.5, 1.5)), [true, true, false, false]);
        let (a, b) = (150f64.to_radians(), -150f64.to_radians());
        assert_eq!([a, b, 180f64.to_radians()].map(|l| lam_in(l, a, b)), [true, true, true]);
    }

    #[test]
    fn lam_in_full_width_is_degenerate() {
        // Python _lam_in([-pi, 0, pi, 1], -pi, pi) == [True, False, True, False]: width np_mod(2*pi, 2*pi) = 0.
        let p = std::f64::consts::PI;
        assert_eq!([-p, 0.0, p, 1.0].map(|l| lam_in(l, -p, p)), [true, false, true, false]);
    }

    #[test]
    fn lam_in_one_ulp_finding() {
        // FINDING (Python _lam_in identical): lam = pi, lam_max = -pi, lam_min = 1.1317005146388275 is false by one ulp
        // (offset 2.009892138950965 vs width 2.0098921389509656).
        assert!(!lam_in(std::f64::consts::PI, 1.1317005146388275, -std::f64::consts::PI));
    }

    #[test]
    fn lam_in_random_sectors() {
        use erploc_linalg::rng::Pcg64;
        let p = std::f64::consts::PI;
        let mut rng = Pcg64::new(7);
        let draw = |r: &mut Pcg64| {
            let u = r.uniform01();
            if u < 0.2 {
                [-p, p, 0.0][(u * 15.0) as usize % 3]
            } else {
                (r.uniform01() * 2.0 - 1.0) * p
            }
        };
        let (mut n_in, mut n_out) = (0usize, 0usize);
        let mut n_ulp: Vec<f64> = Vec::new();
        for _ in 0..200_000 {
            let (l, a, b) = (draw(&mut rng), draw(&mut rng), draw(&mut rng));
            if (b - a).abs() >= TWO_PI {
                continue;
            }
            let inside = |l: f64| if a <= b { a <= l && l <= b } else { l >= a || l <= b };
            if inside(l) {
                if (l, b) == (p, -p) {
                    // FINDING class (see lam_in_one_ulp_finding): the wrap end l = pi, lam_max = -pi can miss by an ulp
                    // for many lam_min (78 of 200k draws, not just the triple above); counted, not asserted.
                    if !lam_in(l, a, b) {
                        n_ulp.push(a);
                    }
                    continue;
                }
                assert!(lam_in(l, a, b), "{l} {a} {b}");
                n_in += 1;
            } else if !inside(l + TWO_PI) && !inside(l - TWO_PI) {
                // outside on the circle too (l = -pi is the same direction as b = pi)
                assert!(!lam_in(l, a, b), "{l} {a} {b}");
                n_out += 1;
            }
        }
        println!("lam_in ulp misses (l = pi, lam_max = -pi): {} lam_min values", n_ulp.len());
        assert!(n_in > 1000 && n_out > 1000, "{n_in} {n_out}");
    }
}
