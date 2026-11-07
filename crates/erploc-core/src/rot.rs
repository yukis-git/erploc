//! Rotation utilities (world = right-handed, z up). Port of `erploc/rot.py`.
use crate::{Mat, M3, V3};
use erploc_linalg::norm;

pub fn rot_x(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    Mat([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
}

pub fn rot_y(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    Mat([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
}

pub fn rot_z(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    Mat([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
}

/// `R_z(yaw) R_y(pitch) R_x(roll)`.
pub fn rot_zyx(yaw: f64, pitch: f64, roll: f64) -> M3 {
    rot_z(yaw).mul(&rot_y(pitch)).mul(&rot_x(roll))
}

pub fn skew(v: &V3) -> M3 {
    Mat([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
}

/// Rodrigues; theta < 1e-8 uses `I + K + 0.5 K^2`.
pub fn exp_so3(w: &V3) -> M3 {
    let th = norm(w);
    let k = skew(w);
    let k2 = k.mul(&k);
    let i = M3::identity();
    if th < 1e-8 {
        return i.add(&k).add(&k2.scale(0.5));
    }
    i.add(&k.scale(th.sin() / th)).add(&k2.scale((1.0 - th.cos()) / (th * th)))
}

/// Rotation matrix to rotation vector (|w| <= pi) via Shepperd's method and `2 atan2(|q_v|, q_w)`.
///
/// For a non-finite input, or a finite matrix that is not a rotation (e.g. `1 + trace` with a negative radicand), the output is
/// non-finite (NaN); the result is only meaningful for `R` in SO(3).
pub fn log_so3(r: &M3) -> V3 {
    let m = &r.0;
    let tr = m[0][0] + m[1][1] + m[2][2];
    let mut qv = [0.0; 3];
    let mut qw;
    if tr > 0.0 {
        let t = (1.0 + tr).sqrt();
        qw = 0.5 * t;
        let f = 0.5 / t;
        qv = [(m[2][1] - m[1][2]) * f, (m[0][2] - m[2][0]) * f, (m[1][0] - m[0][1]) * f];
    } else {
        let mut i = 0;
        if m[1][1] > m[i][i] {
            i = 1;
        }
        if m[2][2] > m[i][i] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let t = (1.0 + m[i][i] - m[j][j] - m[k][k]).sqrt();
        let f = 0.5 / t;
        qv[i] = 0.5 * t;
        qv[j] = (m[j][i] + m[i][j]) * f;
        qv[k] = (m[k][i] + m[i][k]) * f;
        qw = (m[k][j] - m[j][k]) * f;
    }
    if qw < 0.0 {
        qw = -qw;
        qv = [-qv[0], -qv[1], -qv[2]];
    }
    let nv = norm(&qv);
    if nv < 1e-12 {
        if qw > 0.0 {
            return [2.0 * qv[0] / qw, 2.0 * qv[1] / qw, 2.0 * qv[2] / qw];
        }
        return [0.0; 3];
    }
    let s = 2.0 * nv.atan2(qw) / nv;
    [s * qv[0], s * qv[1], s * qv[2]]
}

pub fn yaw_of(r: &M3) -> f64 {
    r.0[1][0].atan2(r.0[0][0])
}

pub fn mat3_vec(r: &M3, v: &V3) -> V3 {
    let m = &r.0;
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erp::wrap_pi;
    use erploc_linalg::rng::Pcg64;
    use std::f64::consts::PI;

    fn eye_err(r: &M3) -> f64 {
        let e = r.transpose().mul(r).sub(&M3::identity());
        e.0.iter().flatten().fold(0.0f64, |a, x| a.max(x.abs()))
    }
    fn det(r: &M3) -> f64 {
        let m = &r.0;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
    fn unit(rng: &mut Pcg64) -> V3 {
        let a = [rng.normal(), rng.normal(), rng.normal()];
        let n = norm(&a);
        [a[0] / n, a[1] / n, a[2] / n]
    }
    fn mdiff(a: &M3, b: &M3) -> f64 {
        a.sub(b).0.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn rot_orthonormal() {
        let mut rng = Pcg64::new(0);
        let mut worst = 0.0f64;
        for _ in 0..200 {
            let r = rot_zyx(rng.uniform(-PI, PI), rng.uniform(-PI, PI), rng.uniform(-PI, PI));
            worst = worst.max(eye_err(&r)).max((det(&r) - 1.0).abs());
        }
        for f in [rot_x, rot_y, rot_z] {
            for _ in 0..50 {
                let r = f(rng.uniform(-PI, PI));
                worst = worst.max(eye_err(&r)).max((det(&r) - 1.0).abs());
            }
        }
        println!("rot orthonormal worst {worst:.3e}");
        assert!(worst < 1e-14);
    }

    #[test]
    fn exp_log_roundtrip() {
        let mut rng = Pcg64::new(0);
        let mut worst = 0.0f64;
        for _ in 0..1000 {
            let u = unit(&mut rng);
            let th = rng.uniform(1e-10, PI - 1e-9);
            let w = [u[0] * th, u[1] * th, u[2] * th];
            let w2 = log_so3(&exp_so3(&w));
            for i in 0..3 {
                worst = worst.max((w2[i] - w[i]).abs());
            }
        }
        println!("exp/log worst {worst:.3e}");
        assert!(worst < 1e-12);
    }

    #[test]
    fn log_near_pi() {
        let mut rng = Pcg64::new(1);
        let (mut worst, mut worst_mat) = (0.0f64, 0.0f64);
        // same sampling and comparison as Python rot.py: theta = pi - U(0, 1e-7), vectors up to sign
        for _ in 0..100 {
            let u = unit(&mut rng);
            let th = PI - rng.uniform(0.0, 1e-7);
            let w = [u[0] * th, u[1] * th, u[2] * th];
            let r = exp_so3(&w);
            let w2 = log_so3(&r);
            let (mut dm, mut dp) = (0.0f64, 0.0f64);
            for i in 0..3 {
                dm = dm.max((w2[i] - w[i]).abs());
                dp = dp.max((w2[i] + w[i]).abs());
            }
            worst = worst.max(dm.min(dp));
            worst_mat = worst_mat.max(mdiff(&exp_so3(&w2), &r));
        }
        println!("near pi matrix round trip worst {worst_mat:.3e}");
        assert!(worst_mat < 1e-9);
        println!("near pi worst {worst:.3e}");
        assert!(worst < 1e-9);
    }

    #[test]
    fn exp_so3_properties() {
        let mut rng = Pcg64::new(2);
        let mut worst = 0.0f64;
        for _ in 0..200 {
            let w = [rng.normal() * 0.7, rng.normal() * 0.7, rng.normal() * 0.7];
            let r = exp_so3(&w);
            worst = worst.max(eye_err(&r)).max((det(&r) - 1.0).abs());
        }
        assert!(worst < 1e-14, "{worst:e}");
        let mut e = 0.0f64;
        for _ in 0..50 {
            let a = rng.uniform(-3.0, 3.0);
            e = e.max(mdiff(&exp_so3(&[0.0, 0.0, a]), &rot_z(a)));
        }
        println!("exp_so3 orth {worst:.3e}, exp(a ez)=rot_z {e:.3e}");
        assert!(e < 1e-14);
        // skew(v) w = v x w; rot_z(a) e_x
        let mut es = 0.0f64;
        let mut ex = 0.0f64;
        for _ in 0..50 {
            let v = [rng.normal(), rng.normal(), rng.normal()];
            let w = [rng.normal(), rng.normal(), rng.normal()];
            let c = erploc_linalg::cross(&v, &w);
            let s = mat3_vec(&skew(&v), &w);
            for i in 0..3 {
                es = es.max((s[i] - c[i]).abs());
            }
            let a = rng.uniform(-PI, PI);
            let p = mat3_vec(&rot_z(a), &[1.0, 0.0, 0.0]);
            ex = ex.max((p[0] - a.cos()).abs()).max((p[1] - a.sin()).abs()).max(p[2].abs());
        }
        println!("skew {es:.3e}, rot_z e_x {ex:.3e}");
        assert!(es < 1e-15 && ex < 1e-15);
    }

    #[test]
    fn yaw_of_rot_zyx() {
        let mut rng = Pcg64::new(3);
        let mut e = 0.0f64;
        let mut e2 = 0.0f64;
        for _ in 0..200 {
            let y = rng.uniform(-PI, PI);
            let p = rng.uniform(-PI / 2.0 + 0.05, PI / 2.0 - 0.05);
            let r = rng.uniform(-PI, PI);
            e = e.max(wrap_pi(yaw_of(&rot_zyx(y, p, r)) - wrap_pi(y)).abs());
            e2 = e2.max(mdiff(&rot_zyx(y, p, r), &rot_z(y).mul(&rot_y(p)).mul(&rot_x(r))));
        }
        println!("yaw_of worst {e:.3e}");
        assert!(e < 1e-12 && e2 < 1e-15);
    }
}
