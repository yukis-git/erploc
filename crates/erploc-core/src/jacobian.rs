//! Observation model h = (lam, phi)(R^T (X - c)), analytic Jacobians, numerical Jacobian.
//! Port of `erploc/jacobian.py`.
use crate::erp::{bearing_to_lamphi, d_lamphi_d_bearing, wrap_pi};
use crate::markers::{corners_local, corners_world, MarkerPose};
use crate::rot::{exp_so3, mat3_vec, rot_zyx, skew, yaw_of};
use crate::{DMat, M3, V2, V3};

/// Camera pose: Euler (roll/pitch known, yaw unknown) or a full rotation matrix (`r_mat`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamPose {
    pub c: V3,
    pub yaw: f64,
    pub roll: f64,
    pub pitch: f64,
    pub r_mat: Option<M3>,
}

impl CamPose {
    /// Euler pose, `R = Rz(yaw) Ry(pitch) Rx(roll)`.
    pub fn euler(c: V3, yaw: f64, roll: f64, pitch: f64) -> Self {
        CamPose { c, yaw, roll, pitch, r_mat: None }
    }

    pub fn r(&self) -> M3 {
        match self.r_mat {
            Some(r) => r,
            None => rot_zyx(self.yaw, self.pitch, self.roll),
        }
    }

    pub fn from_r(c: V3, r: M3) -> Self {
        CamPose { c, yaw: yaw_of(&r), roll: 0.0, pitch: 0.0, r_mat: Some(r) }
    }

    /// Left perturbation: `c + dc`, `exp([dtheta]x) R`.
    pub fn retract(&self, dc: &V3, dtheta: &V3) -> Self {
        let r_new = exp_so3(dtheta).mul(&self.r());
        CamPose::from_r([self.c[0] + dc[0], self.c[1] + dc[1], self.c[2] + dc[2]], r_new)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// `d_i = R^T (X_i - c)` (unnormalised), via an explicit `R`.
fn predict_raw(r: &M3, c: &V3, x: &[V3]) -> (Vec<V2>, Vec<V3>) {
    let rt = r.transpose();
    let d: Vec<V3> = x.iter().map(|xi| mat3_vec(&rt, &[xi[0] - c[0], xi[1] - c[1], xi[2] - c[2]])).collect();
    (bearing_to_lamphi(&d), d)
}

/// `(lam_phi, d)` with `d = R^T (X - c)`, unnormalised.
pub fn project(cam: &CamPose, x: &[V3]) -> (Vec<V2>, Vec<V3>) {
    predict_raw(&cam.r(), &cam.c, x)
}

pub fn predict_marker(cam: &CamPose, mk: &MarkerPose, a: f64) -> [V2; 4] {
    let (lp, _) = project(cam, &corners_world(&mk.p, &mk.r, a));
    [lp[0], lp[1], lp[2], lp[3]]
}

/// rad-unit `dh/dtheta`: `J_cam` is 8x(3+dof.len()), `J_mk` is 8x6. Rows interleaved (lam_k, phi_k).
/// Camera: left perturbation; marker: right perturbation.
pub fn jac_marker(cam: &CamPose, mk: &MarkerPose, a: f64, dof: &[Axis]) -> (DMat, DMat) {
    let r = cam.r();
    let c = cam.c;
    let x = corners_world(&mk.p, &mk.r, a);
    let q = corners_local(a);
    let (_, d) = predict_raw(&r, &c, &x);
    let nd = dof.len();
    let mut j_cam = DMat::zeros(8, 3 + nd);
    let mut j_mk = DMat::zeros(8, 6);
    let rt = r.transpose();
    let neg_rt = rt.scale(-1.0);
    let rm_neg_rt = rt.mul(&mk.r).scale(-1.0);
    for k in 0..4 {
        let bl = d_lamphi_d_bearing(&d[k]);
        let xc = [x[k][0] - c[0], x[k][1] - c[1], x[k][2] - c[2]];
        let rt_sk = rt.mul(&skew(&xc));
        let mk_sk = rm_neg_rt.mul(&skew(&q[k]));
        for (i, bli) in bl.iter().enumerate() {
            let row = 2 * k + i;
            for j in 0..3 {
                let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
                for (m, b) in bli.iter().enumerate() {
                    s0 += b * neg_rt.0[m][j];
                    s1 += b * rt.0[m][j];
                    s2 += b * mk_sk.0[m][j];
                }
                j_cam[(row, j)] = s0;
                j_mk[(row, j)] = s1;
                j_mk[(row, 3 + j)] = s2;
            }
            for (jj, ax) in dof.iter().enumerate() {
                let col = ax.index();
                let mut s = 0.0;
                for (m, b) in bli.iter().enumerate() {
                    s += b * rt_sk.0[m][col];
                }
                j_cam[(row, 3 + jj)] = s;
            }
        }
    }
    (j_cam, j_mk)
}

/// Central-difference Jacobian. Output entries listed in `wrap_rows` of the difference pass through `wrap_pi`.
pub fn numerical_jac(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], h: f64, wrap_rows: &[usize]) -> DMat {
    let f0 = f(x);
    assert!(wrap_rows.iter().all(|&i| i < f0.len()), "wrap_rows index out of range (output has {} rows)", f0.len());
    let mut jm = DMat::zeros(f0.len(), x.len());
    for j in 0..x.len() {
        let mut xp = x.to_vec();
        xp[j] += h;
        let mut xm = x.to_vec();
        xm[j] -= h;
        let (fp, fm) = (f(&xp), f(&xm));
        for i in 0..f0.len() {
            let mut df = fp[i] - fm[i];
            if wrap_rows.contains(&i) {
                df = wrap_pi(df);
            }
            jm[(i, j)] = df / (2.0 * h);
        }
    }
    jm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erp::lamphi_to_bearing;
    use crate::markers::facing_pose;
    use crate::rot::rot_zyx;
    use erploc_linalg::rng::Pcg64;
    use std::f64::consts::PI;

    const A_CHK: f64 = 0.16;
    const FAMILIES: [&str; 4] = ["seam", "pole", "tilt", "rollpitch"];

    fn sign(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    fn make_config(g: &mut Pcg64, family: &str) -> (CamPose, MarkerPose) {
        let (roll, pitch) = if family == "rollpitch" {
            let lim = 10f64.to_radians();
            let r = g.uniform(-lim, lim);
            let p = g.uniform(-lim, lim);
            (r, p)
        } else {
            (0.0, 0.0)
        };
        let c = [g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0)];
        let cam = CamPose::euler(c, g.uniform(-PI, PI), roll, pitch);
        let (lam, phi, dd, psi);
        match family {
            "seam" => {
                lam = sign(g.uniform(-1.0, 1.0)) * (PI - g.uniform(0.0, 0.04));
                phi = g.uniform(-0.6, 0.6);
                dd = g.uniform(1.0, 2.0);
                psi = g.uniform(0.0, 50f64.to_radians());
            }
            "pole" => {
                phi = sign(g.uniform(-1.0, 1.0)) * g.uniform(80.0, 82.0).to_radians();
                lam = g.uniform(-PI, PI);
                dd = g.uniform(3.0, 6.0);
                psi = g.uniform(0.0, 50f64.to_radians());
            }
            "tilt" => {
                lam = g.uniform(-PI, PI);
                let l = 55f64.to_radians();
                phi = g.uniform(-l, l);
                dd = g.uniform(1.5, 5.0);
                psi = g.uniform(60f64.to_radians(), 75f64.to_radians());
            }
            _ => {
                lam = g.uniform(-PI, PI);
                let l = 60f64.to_radians();
                phi = g.uniform(-l, l);
                dd = g.uniform(1.5, 5.0);
                psi = g.uniform(0.0, 60f64.to_radians());
            }
        }
        let d = lamphi_to_bearing(&[[lam, phi]])[0];
        let rd = mat3_vec(&cam.r(), &[dd * d[0], dd * d[1], dd * d[2]]);
        let p = [cam.c[0] + rd[0], cam.c[1] + rd[1], cam.c[2] + rd[2]];
        let td = g.uniform(-PI, PI);
        let roll_m = g.uniform(-PI, PI);
        (cam, facing_pose(0, &p, &cam.c, psi, td, roll_m))
    }

    fn configs() -> Vec<(&'static str, Vec<(CamPose, MarkerPose)>)> {
        let mut g = Pcg64::new(20260911);
        FAMILIES.iter().map(|&f| (f, (0..50).map(|_| make_config(&mut g, f)).collect())).collect()
    }

    fn jac_error(cam: &CamPose, mk: &MarkerPose, dof: &[Axis], h: f64) -> f64 {
        let (jc, jm) = jac_marker(cam, mk, A_CHK, dof);
        let nd = dof.len();
        let (r0, c0, p0, rm0) = (cam.r(), cam.c, mk.p, mk.r);
        let f = |x: &[f64]| -> Vec<f64> {
            let mut dth = [0.0; 3];
            for (i, ax) in dof.iter().enumerate() {
                dth[ax.index()] = x[3 + i];
            }
            let p = [p0[0] + x[3 + nd], p0[1] + x[4 + nd], p0[2] + x[5 + nd]];
            let xs = corners_world(&p, &rm0.mul(&exp_so3(&[x[6 + nd], x[7 + nd], x[8 + nd]])), A_CHK);
            let c = [c0[0] + x[0], c0[1] + x[1], c0[2] + x[2]];
            let (lp, _) = predict_raw(&exp_so3(&dth).mul(&r0), &c, &xs);
            lp.iter().flat_map(|v| [v[0], v[1]]).collect()
        };
        let jn = numerical_jac(&f, &vec![0.0; 9 + nd], h, &[0, 2, 4, 6]);
        let mut worst = 0.0f64;
        for i in 0..8 {
            for j in 0..(3 + nd) {
                worst = worst.max((jc[(i, j)] - jn[(i, j)]).abs());
            }
            for j in 0..6 {
                worst = worst.max((jm[(i, j)] - jn[(i, 3 + nd + j)]).abs());
            }
        }
        worst
    }

    const XYZ: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn worst_over(cfgs: &[(&str, Vec<(CamPose, MarkerPose)>)], fams: &[&str], dof: &[Axis], h: f64) -> f64 {
        let mut w = 0.0f64;
        for (f, v) in cfgs {
            if fams.contains(f) {
                for (cam, mk) in v {
                    w = w.max(jac_error(cam, mk, dof, h));
                }
            }
        }
        w
    }

    #[test]
    fn s0_1a_dof_z() {
        let e = worst_over(&configs(), &FAMILIES, &[Axis::Z], 1e-5);
        eprintln!("S0-1a max {e:.3e}");
        assert!(e < 1e-8, "{e}");
    }

    #[test]
    fn s0_1b_dof_xyz() {
        let e = worst_over(&configs(), &["seam", "tilt", "rollpitch"], &XYZ, 1e-5);
        eprintln!("S0-1b max {e:.3e}");
        assert!(e < 1e-8, "{e}");
    }

    #[test]
    fn s0_1c_pole_h1e5() {
        let e = worst_over(&configs(), &["pole"], &XYZ, 1e-5);
        eprintln!("S0-1c max {e:.3e}");
        assert!(e < 1e-7, "{e}");
    }

    #[test]
    fn s0_1d_pole_h3e6() {
        let cfgs = configs();
        let e5 = worst_over(&cfgs, &["pole"], &XYZ, 1e-5);
        let e6 = worst_over(&cfgs, &["pole"], &XYZ, 3e-6);
        eprintln!("S0-1d max {e6:.3e}, improvement {:.2}x", e5 / e6);
        assert!(e6 < 1e-8, "{e6}");
        assert!(e5 / e6 > 8.0, "{}", e5 / e6);
    }

    fn sample() -> (CamPose, MarkerPose) {
        let cam = CamPose::euler([0.3, -0.2, 1.2], 0.4, 0.05, -0.03);
        let mk = facing_pose(0, &[2.0, 1.0, 1.4], &cam.c, 0.3, 0.7, -0.2);
        (cam, mk)
    }

    #[test]
    fn cam_r_and_predict() {
        let (cam, mk) = sample();
        let r = cam.r().sub(&rot_zyx(0.4, -0.03, 0.05));
        assert!(r.to_dmat().max_abs() < 1e-15);
        let pm = predict_marker(&cam, &mk, A_CHK);
        let (lp, _) = project(&cam, &corners_world(&mk.p, &mk.r, A_CHK));
        for k in 0..4 {
            assert!((pm[k][0] - lp[k][0]).abs() < 1e-15 && (pm[k][1] - lp[k][1]).abs() < 1e-15);
        }
    }

    #[test]
    fn dof_subset_columns() {
        let (cam, mk) = sample();
        let (jx, _) = jac_marker(&cam, &mk, A_CHK, &XYZ);
        let mut worst = 0.0f64;
        let subsets: [&[Axis]; 5] = [&[Axis::X], &[Axis::Y], &[Axis::X, Axis::Y], &[Axis::Y, Axis::Z], &[Axis::Z, Axis::X]];
        for dof in subsets {
            let (js, _) = jac_marker(&cam, &mk, A_CHK, dof);
            for i in 0..8 {
                for j in 0..3 {
                    worst = worst.max((js[(i, j)] - jx[(i, j)]).abs());
                }
                for (jj, ax) in dof.iter().enumerate() {
                    worst = worst.max((js[(i, 3 + jj)] - jx[(i, 3 + ax.index())]).abs());
                }
            }
        }
        eprintln!("dof subset worst {worst:.3e}");
        assert!(worst < 1e-14, "{worst}");
    }

    #[test]
    fn yaw_column_equals_d_dyaw() {
        let (cam, mk) = sample();
        let (j, _) = jac_marker(&cam, &mk, A_CHK, &[Axis::Z]);
        let h = 1e-6;
        let at = |dy: f64| {
            let c = CamPose::euler(cam.c, cam.yaw + dy, cam.roll, cam.pitch);
            predict_marker(&c, &mk, A_CHK)
        };
        let (pp, pm) = (at(h), at(-h));
        let mut worst = 0.0f64;
        for k in 0..4 {
            let dl = wrap_pi(pp[k][0] - pm[k][0]) / (2.0 * h);
            let dp = (pp[k][1] - pm[k][1]) / (2.0 * h);
            worst = worst.max((j[(2 * k, 3)] - dl).abs()).max((j[(2 * k + 1, 3)] - dp).abs());
        }
        eprintln!("yaw column worst {worst:.3e}");
        assert!(worst < 1e-8, "{worst}");
    }

    #[test]
    fn retract_left() {
        let (cam, _) = sample();
        let dth = [0.1, -0.2, 0.3];
        let r = cam.retract(&[0.0; 3], &dth);
        let e = r.r().sub(&exp_so3(&dth).mul(&cam.r())).to_dmat().max_abs();
        assert!(e < 1e-15, "{e}");
        let r2 = cam.retract(&[1.0, 2.0, 3.0], &[0.0; 3]);
        assert_eq!(r2.c, [1.3, 1.8, 4.2]);
    }

    #[test]
    #[should_panic(expected = "wrap_rows index out of range")]
    fn numerical_jac_rejects_bad_wrap_row() {
        let f = |x: &[f64]| vec![x[0], x[0] * 2.0];
        let _ = numerical_jac(&f, &[0.1], 1e-5, &[2]);
    }

    #[test]
    fn numerical_jac_checks() {
        let f = |z: &[f64]| vec![z[0].powi(3), z[1].sin()];
        let j = numerical_jac(&f, &[1.3, 0.7], 1e-5, &[]);
        let e = (j[(0, 0)] - 3.0 * 1.3 * 1.3).abs().max(j[(0, 1)].abs()).max(j[(1, 0)].abs()).max((j[(1, 1)] - 0.7f64.cos()).abs());
        assert!(e < 1e-8, "{e}");
        let g = |z: &[f64]| vec![wrap_pi(PI - 1e-9 + z[0])];
        let j = numerical_jac(&g, &[0.0], 1e-5, &[0]);
        assert!((j[(0, 0)] - 1.0).abs() < 1e-9);
    }
}
