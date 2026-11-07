//! `render(partial=True, detect=...)` of `erploc/synth.py` (`_render_partial`): observations with a PER-CORNER occlusion
//! test instead of the face-sample test of the unobstructed visibility, for the partial-occlusion consumers (corner masks and sigma
//! scales in the estimators of `erploc-est`).
use crate::scene::Obstacle;
use crate::visibility::{occluded, visible_pairs, VisParams, FRONT_GAP_WALL};
use erploc_core::erp::{lamphi_to_px, wrap_pi};
use erploc_core::jacobian::{predict_marker, CamPose};
use erploc_core::markers::{corners_world, MarkerPose};
use erploc_core::{V2, V3};
use erploc_est::synth::{Observation, SynthConfig, PSI_MAX_RENDER_DEG};

/// `detect(cam, marker, corner_clear) -> (keep, corner_sigma_scale)`: may drop the marker and set the per-corner sigma scale.
pub type DetectFn<'a> = &'a dyn Fn(&CamPose, &MarkerPose, [bool; 4]) -> (bool, [f64; 4]);

/// For every `(frame, marker)` pair that satisfies the geometric conditions (incidence angle, apparent size, self-occlusion sectors,
/// NO obstacles) every corner is tested against `obstacles` with the sight line `camera -> corner` (the wall gap and `vp.eps` of the
/// face-sample rule); a corner is `clear` when it passes. Markers with fewer than `min_valid` clear corners are dropped; with
/// `min_valid = 0` and no clear corner the marker centre must still be visible. `uv` are the geometric projections of all four
/// corners (the invalid ones are present but never read). `vp = None` is `psi_max = 85 deg`, `px_min = 0`.
pub fn render_partial(
    cams: &[CamPose],
    markers: &[MarkerPose],
    cfg: &SynthConfig,
    obstacles: &[Obstacle],
    vp: Option<&VisParams>,
    min_valid: usize,
    detect: Option<DetectFn>,
) -> Vec<Vec<Observation>> {
    let default = VisParams::new(PSI_MAX_RENDER_DEG.to_radians(), 0.0);
    let vp_use = vp.unwrap_or(&default);
    let cells: Vec<V3> = cams.iter().map(|c| c.c).collect();
    let headings: Vec<f64> = cams.iter().map(|c| c.yaw).collect();
    let n = markers.len();
    let geom = if cams.is_empty() || markers.is_empty() {
        vec![false; cams.len() * n]
    } else {
        visible_pairs(&cells, &headings, markers, cfg.a, &cfg.cam, &[], vp_use, true, None)
    };
    let mut frames = Vec::with_capacity(cams.len());
    for (f, cam) in cams.iter().enumerate() {
        let mut obs = Vec::new();
        for (j, mk) in markers.iter().enumerate() {
            if !geom[f * n + j] {
                continue;
            }
            let x = corners_world(&mk.p, &mk.r, cfg.a);
            let clear: [bool; 4] = [0, 1, 2, 3].map(|k| !occluded(&cells[f], &x[k], obstacles, vp_use.eps, FRONT_GAP_WALL));
            let n_clear = clear.iter().filter(|&&c| c).count();
            if n_clear < min_valid {
                continue;
            }
            if n_clear == 0 && occluded(&cells[f], &mk.p, obstacles, vp_use.eps, FRONT_GAP_WALL) {
                continue;
            }
            let mut scale = [1.0; 4];
            if let Some(d) = detect {
                let (keep, s) = d(cam, mk, clear);
                if !keep {
                    continue;
                }
                scale = s;
            }
            let lp = predict_marker(cam, mk, cfg.a);
            let lp: Vec<V2> = lp.iter().map(|p| [wrap_pi(p[0]), p[1]]).collect();
            let px = lamphi_to_px(&lp, &cfg.cam);
            obs.push(Observation { id: mk.id, uv: [px[0], px[1], px[2], px[3]], corner_mask: clear, corner_sigma_scale: scale });
        }
        frames.push(obs);
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_core::erp::{ErpCamera, NoiseModel};
    use erploc_core::markers::upright_pose;
    use erploc_est::lm::Loss;
    use erploc_est::resection::{resect, CamDof, ResectOptions};
    use erploc_est::synth::render;
    use std::collections::BTreeMap;

    const A: f64 = 0.16;

    fn cfg() -> SynthConfig {
        SynthConfig { cam: ErpCamera::new(6528, 3264, 1), a: A, sigma_px: 0.3, noise_model: NoiseModel::Tangent, seed: 0 }
    }

    /// Square obstacle of side `w` centred on the segment `c -> x` at fraction `t`.
    fn pole_to(c: &V3, x: &V3, t: f64, w: f64, h: f64) -> Obstacle {
        let q = [c[0] + t * (x[0] - c[0]), c[1] + t * (x[1] - c[1])];
        Obstacle {
            poly: vec![
                [q[0] - w / 2.0, q[1] - w / 2.0],
                [q[0] + w / 2.0, q[1] - w / 2.0],
                [q[0] + w / 2.0, q[1] + w / 2.0],
                [q[0] - w / 2.0, q[1] + w / 2.0],
            ],
            height: h,
        }
    }

    /// Full-height pole in front of corner `k`: hides it and the other corner of the same tag column.
    fn full(c: &V3, x: &[V3; 4], k: usize) -> Obstacle {
        pole_to(c, &x[k], 0.5, 0.02, 3.0)
    }

    /// Pole just tall enough to hide the lower corner `k` of its column and not the upper one.
    fn short(c: &V3, x: &[V3; 4], k: usize) -> Obstacle {
        let t = 0.5;
        pole_to(c, &x[k], t, 0.02, c[2] + t * (x[k][2] - c[2]) + t * A / 2.0)
    }

    /// Marker on the wall x = 3 facing the camera at the origin side; camera `cam` looks at it.
    fn setup() -> (CamPose, MarkerPose, [V3; 4]) {
        let mk = upright_pose(0, &[3.0, 0.2, 1.6], std::f64::consts::PI);
        let cam = CamPose::euler([0.0, 0.0, 1.2], 0.05, 0.0, 0.0);
        let x = corners_world(&mk.p, &mk.r, A);
        (cam, mk, x)
    }

    fn lowest(x: &[V3; 4]) -> usize {
        (0..4).min_by(|&a, &b| x[a][2].partial_cmp(&x[b][2]).unwrap()).unwrap()
    }

    #[test]
    fn no_obstacles_all_corners_clear_and_equal_to_render() {
        let (cam, mk, _) = setup();
        let fr = render_partial(&[cam], &[mk], &cfg(), &[], None, 4, None);
        let plain = render(&[cam], &[mk], &cfg(), None);
        assert_eq!(fr.len(), 1);
        assert_eq!(fr[0].len(), 1);
        assert_eq!(fr[0][0].corner_mask, [true; 4]);
        assert_eq!(fr[0][0].corner_sigma_scale, [1.0; 4]);
        assert_eq!(fr[0][0].uv, plain[0][0].uv);
        assert_eq!(fr[0][0].id, 0);
    }

    #[test]
    fn one_corner_masked_uv_finite() {
        let (cam, mk, x) = setup();
        let k = lowest(&x);
        let obs = [short(&cam.c, &x, k)];
        let plain = render(&[cam], &[mk], &cfg(), None);
        let fr = render_partial(&[cam], &[mk], &cfg(), &obs, None, 2, None);
        let o = &fr[0][0];
        let mut want = [true; 4];
        want[k] = false;
        assert_eq!(o.corner_mask, want);
        assert!(o.uv.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
        assert_eq!(o.uv, plain[0][0].uv, "invalid corners keep their geometric projection");
    }

    #[test]
    fn min_valid_drops_and_zero_clear_needs_the_centre() {
        let (cam, mk, x) = setup();
        let k = lowest(&x);
        let one = [short(&cam.c, &x, k)];
        // a full-height pole in front of corner 0 hides its column (2 corners)
        let col = [full(&cam.c, &x, 0)];
        let run = |obs: &[Obstacle], mv| render_partial(&[cam], &[mk], &cfg(), obs, None, mv, None)[0].len();
        assert_eq!((run(&one, 3), run(&one, 4)), (1, 0));
        let nclear = render_partial(&[cam], &[mk], &cfg(), &col, None, 0, None)[0][0].corner_mask.iter().filter(|&&c| c).count();
        assert_eq!(nclear, 2);
        assert_eq!((run(&col, 2), run(&col, 3)), (1, 0));
        // both columns hidden but the centre visible: kept only through the n_clear == 0 branch (min_valid 0)
        let both = [full(&cam.c, &x, 0), full(&cam.c, &x, 1)];
        let o = render_partial(&[cam], &[mk], &cfg(), &both, None, 0, None);
        assert_eq!(o[0].len(), 1);
        assert_eq!(o[0][0].corner_mask, [false; 4]);
        assert_eq!(run(&both, 1), 0);
        // a pole on the centre line hides everything including the centre: dropped even at min_valid 0
        let mut all = both.to_vec();
        all.push(pole_to(&cam.c, &mk.p, 0.5, 0.02, 3.0));
        assert_eq!(run(&all, 0), 0);
    }

    #[test]
    fn detect_can_drop_and_scale() {
        let (cam, mk, x) = setup();
        let k = lowest(&x);
        let obs = [short(&cam.c, &x, k)];
        let seen = std::cell::RefCell::new(Vec::new());
        let keep3 = |_: &CamPose, m: &MarkerPose, clear: [bool; 4]| {
            seen.borrow_mut().push((m.id, clear));
            (clear.iter().filter(|&&c| c).count() >= 3, clear.map(|c| if c { 2.0 } else { 1.0 }))
        };
        let fr = render_partial(&[cam], &[mk], &cfg(), &obs, None, 0, Some(&keep3));
        let mut want = [true; 4];
        want[k] = false;
        assert_eq!(seen.borrow().as_slice(), &[(0, want)]);
        assert_eq!(fr[0][0].corner_mask, want);
        assert_eq!(fr[0][0].corner_sigma_scale, want.map(|c| if c { 2.0 } else { 1.0 }));
        // dropped by the detector
        let none = |_: &CamPose, _: &MarkerPose, _: [bool; 4]| (false, [1.0; 4]);
        assert!(render_partial(&[cam], &[mk], &cfg(), &obs, None, 0, Some(&none))[0].is_empty());
    }

    #[test]
    fn geometric_conditions_apply() {
        // the back of the marker and a marker outside psi_max are not rendered, whatever the obstacles
        let (cam, _, _) = setup();
        let back = upright_pose(0, &[3.0, 0.2, 1.6], 0.0);
        assert!(render_partial(&[cam], &[back], &cfg(), &[], None, 0, None)[0].is_empty());
        let grazing = upright_pose(1, &[3.0, 0.2, 1.6], std::f64::consts::PI - 80f64.to_radians());
        let vp = VisParams::new(70f64.to_radians(), 0.0);
        assert!(render_partial(&[cam], &[grazing], &cfg(), &[], Some(&vp), 0, None)[0].is_empty());
        assert_eq!(render_partial(&[cam], &[grazing], &cfg(), &[], None, 0, None)[0].len(), 1);
        assert!(render_partial(&[], &[], &cfg(), &[], None, 0, None).is_empty());
    }

    /// Consumer check: noise-free observations with a masked corner recover the pose through the 4-dof resection with masked rows.
    #[test]
    fn partial_render_resect_roundtrip() {
        let mks: Vec<MarkerPose> = [
            (3.0, 0.2, std::f64::consts::PI),
            (-3.0, -0.5, 0.0),
            (0.4, 2.8, -std::f64::consts::FRAC_PI_2),
            (-0.6, -2.9, std::f64::consts::FRAC_PI_2),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, y, az))| upright_pose(i as i64, &[x, y, 1.4 + 0.1 * i as f64], az))
        .collect();
        let truth = CamPose::euler([0.3, -0.2, 1.2], 0.4, 0.0, 0.0);
        // a short pole hides the lowest corner of marker 0 and a column pole two corners of marker 2
        let x0 = corners_world(&mks[0].p, &mks[0].r, A);
        let x2 = corners_world(&mks[2].p, &mks[2].r, A);
        let obs_list = [short(&truth.c, &x0, lowest(&x0)), full(&truth.c, &x2, 0)];
        let mut frames = render_partial(&[truth], &mks, &cfg(), &obs_list, None, 2, None);
        let mut obs = frames.remove(0);
        let masked: usize = obs.iter().map(|o| o.corner_mask.iter().filter(|&&c| !c).count()).sum();
        assert!(masked >= 3, "expected masked corners, got {masked}");
        assert_eq!(obs.len(), 4);
        // the estimator must not read the invalid corners
        for o in obs.iter_mut() {
            for k in 0..4 {
                if !o.corner_mask[k] {
                    o.uv[k] = [f64::NAN, f64::NAN];
                }
            }
        }
        let known: BTreeMap<i64, MarkerPose> = mks.iter().map(|m| (m.id, *m)).collect();
        let opts = ResectOptions { x0: None, sigma_px: 0.3, noise_model: NoiseModel::Tangent, loss: Loss::Linear, dof: CamDof::Yaw };
        let res = resect(&obs, &known, A, &cfg().cam, 0.0, 0.0, &opts).unwrap();
        let dp = (0..3).map(|i| (res.cam.c[i] - truth.c[i]).powi(2)).sum::<f64>().sqrt();
        let dy = erploc_core::erp::wrap_pi(res.cam.yaw - truth.yaw).abs();
        eprintln!("resect with {masked} masked corners: pos err {dp:.3e}, yaw err {dy:.3e}");
        assert!(dp <= 1e-9 && dy <= 1e-9, "{dp:e} {dy:e}");
    }
}
