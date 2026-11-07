//! SO(3)/SE(3) Lie-algebra utilities. Port of `erploc/lie.py`.
//!
//! Camera increments are LEFT perturbations in the world frame: `R <- exp([dtheta]x) R`, `c <- c + dc`.
//! `xi = (rho, theta)` (translation first); `t = J_l(theta) rho`. `split_retract` is the paper's
//! decoupled retraction, which differs from `exp_se3` at O(|xi|^2).
use crate::rot::{exp_so3, log_so3, mat3_vec, skew};
use crate::{Mat, M3, V3};

/// Taylor-switch threshold for the `jl` coefficients.
const TH_SMALL: f64 = 0.1;

pub fn hat(w: &V3) -> M3 {
    skew(w)
}

/// so(3) element to vector; averages the antisymmetric parts.
pub fn vee(w: &M3) -> V3 {
    let m = &w.0;
    [0.5 * (m[2][1] - m[1][2]), 0.5 * (m[0][2] - m[2][0]), 0.5 * (m[1][0] - m[0][1])]
}

fn norm3(w: &V3) -> f64 {
    erploc_linalg::norm(w)
}

/// `J_l(w) = I + a [w]x + b [w]x^2`.
pub fn jl_so3(w: &V3) -> M3 {
    let th = norm3(w);
    let k = hat(w);
    let k2 = k.mul(&k);
    let (a, b);
    if th < 1e-12 {
        a = 0.5;
        b = 1.0 / 6.0;
    } else {
        a = 2.0 * ((th / 2.0).sin() / th).powi(2);
        if th < TH_SMALL {
            let t2 = th * th;
            b = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362880.0;
        } else {
            b = (th - th.sin()) / th.powi(3);
        }
    }
    M3::identity().add(&k.scale(a)).add(&k2.scale(b))
}

/// `J_l(w)^-1 = I - 0.5 [w]x + c [w]x^2`.
pub fn jl_inv_so3(w: &V3) -> M3 {
    let th = norm3(w);
    let k = hat(w);
    let k2 = k.mul(&k);
    let c = if th < 1e-12 {
        1.0 / 12.0
    } else if th < TH_SMALL {
        let t2 = th * th;
        1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0 + t2 * t2 * t2 / 1209600.0
    } else {
        1.0 / (th * th) - (th / 2.0).cos() / (2.0 * th * (th / 2.0).sin())
    };
    M3::identity().sub(&k.scale(0.5)).add(&k2.scale(c))
}

pub fn adjoint_so3(r: &M3) -> M3 {
    *r
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3 {
    pub r: M3,
    pub t: V3,
}

impl Se3 {
    pub fn identity() -> Self {
        Se3 { r: M3::identity(), t: [0.0; 3] }
    }
    pub fn inv(&self) -> Self {
        let rt = self.r.transpose();
        let t = mat3_vec(&rt, &self.t);
        Se3 { r: rt, t: [-t[0], -t[1], -t[2]] }
    }
    /// `self * o`.
    pub fn compose(&self, o: &Se3) -> Se3 {
        let rt = mat3_vec(&self.r, &o.t);
        Se3 { r: self.r.mul(&o.r), t: [rt[0] + self.t[0], rt[1] + self.t[1], rt[2] + self.t[2]] }
    }
    pub fn to_mat4(&self) -> Mat<4, 4> {
        let mut m = Mat::<4, 4>::identity();
        for i in 0..3 {
            for j in 0..3 {
                m.0[i][j] = self.r.0[i][j];
            }
            m.0[i][3] = self.t[i];
        }
        m
    }
}

/// `xi = (rho, theta)`; `t = J_l(theta) rho`.
pub fn exp_se3(xi: &[f64; 6]) -> Se3 {
    let rho = [xi[0], xi[1], xi[2]];
    let theta = [xi[3], xi[4], xi[5]];
    Se3 { r: exp_so3(&theta), t: mat3_vec(&jl_so3(&theta), &rho) }
}

/// `rho = J_l(theta)^-1 t`.
pub fn log_se3(t: &Se3) -> [f64; 6] {
    let theta = log_so3(&t.r);
    let rho = mat3_vec(&jl_inv_so3(&theta), &t.t);
    [rho[0], rho[1], rho[2], theta[0], theta[1], theta[2]]
}

/// `[[R, [t]x R], [0, R]]`.
pub fn adjoint_se3(t: &Se3) -> Mat<6, 6> {
    let mut ad = Mat::<6, 6>::zeros();
    let txr = hat(&t.t).mul(&t.r);
    for i in 0..3 {
        for j in 0..3 {
            ad.0[i][j] = t.r.0[i][j];
            ad.0[i][j + 3] = txr.0[i][j];
            ad.0[i + 3][j + 3] = t.r.0[i][j];
        }
    }
    ad
}

/// `T exp(xi)`.
pub fn boxplus(t: &Se3, xi: &[f64; 6]) -> Se3 {
    t.compose(&exp_se3(xi))
}

/// `log(T2^-1 T1)`.
pub fn boxminus(t1: &Se3, t2: &Se3) -> [f64; 6] {
    log_se3(&t2.inv().compose(t1))
}

/// Decoupled retraction: `(exp([dtheta]x) R, c + dc)`.
pub fn split_retract(r: &M3, c: &V3, dc: &V3, dtheta: &V3) -> (M3, V3) {
    (exp_so3(dtheta).mul(r), [c[0] + dc[0], c[1] + dc[1], c[2] + dc[2]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_linalg::rng::Pcg64;
    use std::f64::consts::PI;

    fn unit(rng: &mut Pcg64) -> V3 {
        let a = [rng.normal(), rng.normal(), rng.normal()];
        let n = norm3(&a);
        [a[0] / n, a[1] / n, a[2] / n]
    }
    /// Direction random, |theta| log-uniform in [1e-8, pi - 1e-6].
    fn sample_theta(rng: &mut Pcg64) -> V3 {
        let u = unit(rng);
        let th = rng.uniform(1e-8f64.ln(), (PI - 1e-6).ln()).exp();
        [u[0] * th, u[1] * th, u[2] * th]
    }
    fn xi6(rho: V3, th: V3) -> [f64; 6] {
        [rho[0], rho[1], rho[2], th[0], th[1], th[2]]
    }
    fn mx<const R: usize, const C: usize>(a: &Mat<R, C>, b: &Mat<R, C>) -> f64 {
        a.sub(b).0.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()))
    }
    fn se3_diff(a: &Se3, b: &Se3) -> f64 {
        mx(&a.to_mat4(), &b.to_mat4())
    }
    fn vmax(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    }
    fn rand_se3(rng: &mut Pcg64, st: f64, sr: f64) -> Se3 {
        let rho = [rng.normal() * st, rng.normal() * st, rng.normal() * st];
        let th = [rng.normal() * sr, rng.normal() * sr, rng.normal() * sr];
        exp_se3(&xi6(rho, th))
    }

    #[test]
    fn l1_se3_roundtrip() {
        let mut rng = Pcg64::new(20260912);
        let mut worst = 0.0f64;
        for _ in 0..200 {
            let th = sample_theta(&mut rng);
            let s = rng.uniform(0.01, 5.0);
            let rho = [rng.normal() * s, rng.normal() * s, rng.normal() * s];
            let xi = xi6(rho, th);
            worst = worst.max(vmax(&log_se3(&exp_se3(&xi)), &xi));
        }
        println!("L-1 worst {worst:.3e}");
        assert!(worst < 1e-9);
    }

    #[test]
    fn l1b_jl_numeric() {
        let mut rng = Pcg64::new(20260912);
        let mut worst = 0.0f64;
        for _ in 0..100 {
            let u = unit(&mut rng);
            let th = rng.uniform(1e-3, PI - 0.05);
            let w = [u[0] * th, u[1] * th, u[2] * th];
            let r0 = exp_so3(&w);
            let jl = jl_so3(&w);
            let h = 1e-6;
            for j in 0..3 {
                let (mut wp, mut wm) = (w, w);
                wp[j] += h;
                wm[j] -= h;
                let dr = exp_so3(&wp).sub(&exp_so3(&wm)).scale(1.0 / (2.0 * h));
                let col = vee(&dr.mul(&r0.transpose()));
                for (i, c) in col.iter().enumerate() {
                    worst = worst.max((c - jl.0[i][j]).abs());
                }
            }
        }
        println!("L-1b worst {worst:.3e}");
        assert!(worst < 1e-7);
    }

    #[test]
    fn l1c_jl_inv() {
        let mut rng = Pcg64::new(20260912);
        let mut worst = 0.0f64;
        for _ in 0..200 {
            let w = sample_theta(&mut rng);
            worst = worst.max(mx(&jl_inv_so3(&w).mul(&jl_so3(&w)), &M3::identity()));
        }
        println!("L-1c worst {worst:.3e}");
        assert!(worst < 1e-12);
    }

    #[test]
    fn l2_adjoint() {
        let mut rng = Pcg64::new(20260912);
        let mut worst = 0.0f64;
        for _ in 0..50 {
            let t = rand_se3(&mut rng, 2.0, 0.8);
            let rho = [rng.normal() * 0.3, rng.normal() * 0.3, rng.normal() * 0.3];
            let th = [rng.normal() * 0.3, rng.normal() * 0.3, rng.normal() * 0.3];
            let xi = xi6(rho, th);
            let ad = adjoint_se3(&t);
            let mut axi = [0.0; 6];
            for (i, a) in axi.iter_mut().enumerate() {
                *a = (0..6).map(|j| ad.0[i][j] * xi[j]).sum();
            }
            let lhs = exp_se3(&axi);
            let rhs = t.compose(&exp_se3(&xi)).compose(&t.inv());
            worst = worst.max(se3_diff(&lhs, &rhs));
        }
        println!("L-2 worst {worst:.3e}");
        assert!(worst < 1e-12);
    }

    #[test]
    fn l3_split_vs_exp() {
        let mut rng = Pcg64::new(20260912);
        let dc0 = [rng.normal(), rng.normal(), rng.normal()];
        let dth0 = [rng.normal(), rng.normal(), rng.normal()];
        let split_vs_exp = |sc: f64| {
            let dc = [dc0[0] * sc, dc0[1] * sc, dc0[2] * sc];
            let dth = [dth0[0] * sc, dth0[1] * sc, dth0[2] * sc];
            let (r, c) = split_retract(&M3::identity(), &[0.0; 3], &dc, &dth);
            mx(&Se3 { r, t: c }.to_mat4(), &exp_se3(&xi6(dc, dth)).to_mat4())
        };
        let (e1, e2) = (split_vs_exp(0.2), split_vs_exp(0.1));
        let ratio = e1 / e2;
        println!("L-3 e(0.2)={e1:.3e} e(0.1)={e2:.3e} ratio={ratio:.3}");
        assert!((3.5..=4.5).contains(&ratio));
    }

    #[test]
    fn l4_small_and_pi_angles() {
        let n = [0.3, 0.6, 0.74];
        let nn = norm3(&n);
        let axis = [n[0] / nn, n[1] / nn, n[2] / nn];
        let mut worst = 0.0f64;
        for th in [1e-10, 1e-9, 1e-7, PI - 1e-9, PI - 1e-6, PI - 1e-3] {
            let theta = [axis[0] * th, axis[1] * th, axis[2] * th];
            let xi = xi6([0.5, -0.2, 1.1], theta);
            let t = exp_se3(&xi);
            let jli = jl_inv_so3(&theta);
            assert!(t.to_mat4().0.iter().flatten().all(|x| x.is_finite()));
            assert!(jli.0.iter().flatten().all(|x| x.is_finite()));
            let xi2 = log_se3(&t);
            assert!(xi2.iter().all(|x| x.is_finite()));
            worst = worst.max(vmax(&xi2, &xi));
        }
        println!("L-4 worst {worst:.3e}");
        assert!(worst < 1e-9);
    }

    #[test]
    fn boxplus_boxminus() {
        let mut rng = Pcg64::new(20260912);
        let mut worst = 0.0f64;
        for _ in 0..50 {
            let t = rand_se3(&mut rng, 1.5, 0.6);
            let t2 = rand_se3(&mut rng, 1.5, 0.6);
            let xi = boxminus(&t2, &t);
            worst = worst.max(se3_diff(&boxplus(&t, &xi), &t2));
        }
        println!("boxplus worst {worst:.3e}");
        assert!(worst < 1e-9);
    }
}
