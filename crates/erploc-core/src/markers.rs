//! Marker pose, 4 corner points in 3D, planar PnP. Port of `erploc/markers.py`.
use crate::rot::{mat3_vec, skew};
use crate::{LinalgError, Mat, M3, V3};
use erploc_linalg::svd::svd;
use erploc_linalg::{cross, dot, norm, DMat};

/// TL, TR, BR, BL (output order of the detector eqar).
pub const CORNER_SIGNS: [[f64; 2]; 4] = [[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]];

/// Columns of `r`: tag x = right, y = up, z = n = normal (viewing side).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarkerPose {
    pub id: i64,
    pub p: V3,
    pub r: M3,
}

fn cols(x: &V3, y: &V3, z: &V3) -> M3 {
    Mat([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])
}

fn col(m: &M3, j: usize) -> V3 {
    [m.0[0][j], m.0[1][j], m.0[2][j]]
}

/// `q_k = (A/2)(s_x, s_y, 0)` in tag-plane coordinates.
pub fn corners_local(a: f64) -> [V3; 4] {
    let mut q = [[0.0; 3]; 4];
    for (k, s) in CORNER_SIGNS.iter().enumerate() {
        q[k][0] = (a / 2.0) * s[0];
        q[k][1] = (a / 2.0) * s[1];
    }
    q
}

/// `X_k = p + R q_k`.
pub fn corners_world(p: &V3, r: &M3, a: f64) -> [V3; 4] {
    let q = corners_local(a);
    let mut x = [[0.0; 3]; 4];
    for k in 0..4 {
        let rq = mat3_vec(r, &q[k]);
        x[k] = [p[0] + rq[0], p[1] + rq[1], p[2] + rq[2]];
    }
    x
}

/// `(dX/dp = I, dX/dω = −R [q_k]×)`; right perturbation `R_m <- R_m exp([dω]×)`.
pub fn d_corners(r: &M3, a: f64) -> ([M3; 4], [M3; 4]) {
    let q = corners_local(a);
    let mut dp = [M3::identity(); 4];
    let mut dw = [M3::zeros(); 4];
    for k in 0..4 {
        dp[k] = M3::identity();
        dw[k] = r.mul(&skew(&q[k])).scale(-1.0);
    }
    (dp, dw)
}

/// Upright tag: n = (cos az, sin az, 0), y = up = e_z, x = y × n (right-handed).
pub fn upright_pose(id: i64, p: &V3, azimuth: f64) -> MarkerPose {
    let n = [azimuth.cos(), azimuth.sin(), 0.0];
    let y = [0.0, 0.0, 1.0];
    let x = cross(&y, &n);
    MarkerPose { id, p: *p, r: cols(&x, &y, &n) }
}

fn normalized(v: &V3) -> V3 {
    let n = norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Orthonormal basis (e1, e2) orthogonal to d.
fn tangent_basis(d: &V3) -> (V3, V3) {
    let d = normalized(d);
    let mut i = 0;
    for j in 1..3 {
        if d[j].abs() < d[i].abs() {
            i = j;
        }
    }
    let mut a = [0.0; 3];
    a[i] = 1.0;
    let e1 = normalized(&cross(&d, &a));
    let e2 = cross(&d, &e1);
    (e1, e2)
}

/// DLT `d_k × (H x_k) = 0`, `x_k = (X_k, Y_k, 1)`; returns `(t, R)` of the marker in the camera frame.
/// Sign: `Σ d_k·(H x_k) > 0`; scale: `(|h1|+|h2|)/2 = 1`; R = polar factor of `[h1 h2 h1×h2]` with det +1.
pub fn planar_pnp(d: &[V3; 4], a: f64) -> Result<(V3, M3), LinalgError> {
    let q = corners_local(a);
    let mut m = DMat::zeros(8, 9);
    for k in 0..4 {
        let xk = [q[k][0], q[k][1], 1.0];
        let (e1, e2) = tangent_basis(&d[k]);
        for (row, e) in [(2 * k, e1), (2 * k + 1, e2)] {
            for i in 0..3 {
                for j in 0..3 {
                    m[(row, 3 * i + j)] = e[i] * xk[j];
                }
            }
        }
    }
    let f = svd(&m)?;
    let mut h = Mat::<3, 3>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            h.0[i][j] = f.vt[(8, 3 * i + j)];
        }
    }
    let mut acc = 0.0;
    for k in 0..4 {
        acc += dot(&d[k], &mat3_vec(&h, &[q[k][0], q[k][1], 1.0]));
    }
    if acc < 0.0 {
        h = h.scale(-1.0);
    }
    let scale = 0.5 * (norm(&col(&h, 0)) + norm(&col(&h, 1)));
    h = h.scale(1.0 / scale);
    let (h1, h2, h3) = (col(&h, 0), col(&h, 1), col(&h, 2));
    let g = cols(&h1, &h2, &cross(&h1, &h2));
    let f2 = svd(&g.to_dmat())?;
    let to_m3 = |x: &DMat| Mat::<3, 3>::from_dmat(x).expect("3x3");
    let mut u = to_m3(&f2.u);
    let vt = to_m3(&f2.vt);
    let mut r = u.mul(&vt);
    if erploc_linalg::lu::det(&r.to_dmat()) < 0.0 {
        for row in u.0.iter_mut() {
            row[2] = -row[2];
        }
        r = u.mul(&vt);
    }
    Ok((h3, r))
}

/// Marker at `p` facing `target`, tilted by incidence angle `psi` towards `tilt_dir` (angle in the plane
/// orthogonal to the target direction), then rolled in-plane by `roll`.
pub fn facing_pose(mid: i64, p: &V3, target: &V3, psi: f64, tilt_dir: f64, roll: f64) -> MarkerPose {
    let u = normalized(&[target[0] - p[0], target[1] - p[1], target[2] - p[2]]);
    let a0 = if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let e1 = normalized(&cross(&u, &a0));
    let e2 = cross(&u, &e1);
    let a: V3 = std::array::from_fn(|i| tilt_dir.cos() * e1[i] + tilt_dir.sin() * e2[i]);
    let n = normalized(&std::array::from_fn(|i| psi.cos() * u[i] + psi.sin() * a[i]));
    let rf = if n[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let x = normalized(&cross(&rf, &n));
    let y = cross(&n, &x);
    let xr: V3 = std::array::from_fn(|i| roll.cos() * x[i] + roll.sin() * y[i]);
    let yr: V3 = std::array::from_fn(|i| -roll.sin() * x[i] + roll.cos() * y[i]);
    MarkerPose { id: mid, p: *p, r: cols(&xr, &yr, &n) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rot::{exp_so3, log_so3};
    use erploc_linalg::rng::Pcg64;
    use std::f64::consts::PI;

    const A: f64 = 0.16;

    fn rand_rot(g: &mut Pcg64) -> M3 {
        exp_so3(&[g.normal(), g.normal(), g.normal()])
    }

    #[test]
    fn corner_side_length() {
        let q = corners_local(A);
        for k in 0..4 {
            let d = q[(k + 1) % 4];
            let s = norm(&[d[0] - q[k][0], d[1] - q[k][1], d[2] - q[k][2]]);
            assert!((s - A).abs() < 1e-15, "{s}");
        }
    }

    #[test]
    fn corners_world_is_p_plus_rq() {
        let mut g = Pcg64::new(0);
        let mut worst = 0.0f64;
        for _ in 0..50 {
            let p = [g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0)];
            let r = rand_rot(&mut g);
            let x = corners_world(&p, &r, A);
            let q = corners_local(A);
            for k in 0..4 {
                let rq = mat3_vec(&r, &q[k]);
                for i in 0..3 {
                    worst = worst.max((x[k][i] - (p[i] + rq[i])).abs());
                }
            }
        }
        println!("corners_world worst {worst:.3e}");
        assert!(worst < 1e-15);
    }

    #[test]
    fn corner_order_tl_tr_br_bl() {
        assert_eq!(corners_local(0.16), [[-0.08, 0.08, 0.0], [0.08, 0.08, 0.0], [0.08, -0.08, 0.0], [-0.08, -0.08, 0.0]]);
    }

    #[test]
    fn d_corners_numeric() {
        let mut g = Pcg64::new(1);
        let h = 1e-6;
        let mut worst = 0.0f64;
        for _ in 0..30 {
            let p = [g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0)];
            let r = rand_rot(&mut g);
            let (dp, dw) = d_corners(&r, A);
            for m in &dp {
                assert_eq!(*m, M3::identity());
            }
            for j in 0..3 {
                let mut e = [0.0; 3];
                e[j] = h;
                let em = [-e[0], -e[1], -e[2]];
                let xp = corners_world(&p, &r.mul(&exp_so3(&e)), A);
                let xm = corners_world(&p, &r.mul(&exp_so3(&em)), A);
                for k in 0..4 {
                    for i in 0..3 {
                        let num = (xp[k][i] - xm[k][i]) / (2.0 * h);
                        worst = worst.max((num - dw[k].0[i][j]).abs());
                    }
                }
            }
        }
        eprintln!("d_corners worst {worst:.3e}");
        assert!(worst < 1e-9, "{worst}");
    }

    #[test]
    fn upright_is_right_handed() {
        let mut g = Pcg64::new(7);
        let mut worst = 0.0f64;
        for _ in 0..50 {
            let th = g.uniform(-std::f64::consts::PI, std::f64::consts::PI);
            let mk = upright_pose(3, &[1.0, 2.0, 1.5], th);
            let (x, y, n) = (col(&mk.r, 0), col(&mk.r, 1), col(&mk.r, 2));
            let c = cross(&x, &y);
            for j in 0..3 {
                worst = worst.max((c[j] - n[j]).abs());
            }
            worst = worst.max((y[0]).abs()).max(y[1].abs()).max((y[2] - 1.0).abs());
            worst = worst.max((n[0] - th.cos()).abs()).max((n[1] - th.sin()).abs()).max(n[2].abs());
            let rtr = mk.r.transpose().mul(&mk.r).sub(&M3::identity());
            worst = worst.max(rtr.to_dmat().max_abs());
            worst = worst.max((erploc_linalg::lu::det(&mk.r.to_dmat()) - 1.0).abs());
        }
        eprintln!("upright worst {worst:.3e}");
        assert!(worst < 1e-15, "{worst}");
    }

    #[test]
    fn pnp_noise_free() {
        let mut g = Pcg64::new(0);
        let (mut er, mut et) = (0.0f64, 0.0f64);
        for _ in 0..200 {
            let c = [g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0), g.uniform(-3.0, 3.0)];
            let rcam = rand_rot(&mut g);
            let u = normalized(&[g.normal(), g.normal(), g.normal()]);
            let dist = g.uniform(1.0, 6.0);
            let t_true: V3 = [dist * u[0], dist * u[1], dist * u[2]];
            let psi = g.uniform(0.0, 70f64.to_radians());
            let tilt = g.uniform(-PI, PI);
            let roll = g.uniform(-PI, PI);
            // marker in the camera frame, facing the camera at the origin
            let mk = facing_pose(0, &t_true, &[0.0; 3], psi, tilt, roll);
            let pw = mat3_vec(&rcam, &t_true);
            let pw: V3 = [c[0] + pw[0], c[1] + pw[1], c[2] + pw[2]];
            let xs = corners_world(&pw, &rcam.mul(&mk.r), A);
            let rt = rcam.transpose();
            let mut d = [[0.0; 3]; 4];
            for k in 0..4 {
                let v = mat3_vec(&rt, &[xs[k][0] - c[0], xs[k][1] - c[1], xs[k][2] - c[2]]);
                d[k] = normalized(&v);
            }
            let (t, r) = planar_pnp(&d, A).unwrap();
            et = et.max(norm(&[t[0] - t_true[0], t[1] - t_true[1], t[2] - t_true[2]]));
            er = er.max(norm(&log_so3(&mk.r.transpose().mul(&r))));
        }
        eprintln!("pnp worst R {er:.3e} rad, t {et:.3e} m");
        assert!(er < 1e-9 && et < 1e-9, "R {er} t {et}");
    }
}
