//! ERP pixel <-> (lam, phi) <-> bearing conversions and residuals.
//!
//! Port of `erploc/erp.py`. Python's `noise_model` string is the enum [`NoiseModel`], so the
//! `ValueError` for an invalid value has no Rust counterpart (it is unrepresentable).
use crate::{V2, V3};
use std::f64::consts::{PI, TAU};

/// ERP camera. Integer coordinates are pixel centres. `s = 2pi/W = pi/H` [rad/px].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErpCamera {
    w: usize,
    h: usize,
    lam_sign: i32,
}

impl ErpCamera {
    /// Panics unless `h > 0`, `w == 2h` (no overflow) and `lam_sign` is +1 or -1.
    pub fn new(w: usize, h: usize, lam_sign: i32) -> Self {
        assert!(h > 0 && h.checked_mul(2) == Some(w), "ERP requires W = 2H (got W={w}, H={h})");
        assert!(lam_sign == 1 || lam_sign == -1, "lam_sign must be +1 or -1 (got {lam_sign})");
        Self { w, h, lam_sign }
    }
    pub fn w(&self) -> usize {
        self.w
    }
    pub fn h(&self) -> usize {
        self.h
    }
    pub fn lam_sign(&self) -> i32 {
        self.lam_sign
    }
    pub fn s(&self) -> f64 {
        TAU / self.w as f64
    }
    pub fn u0(&self) -> f64 {
        (self.w as f64 - 1.0) / 2.0
    }
    pub fn v0(&self) -> f64 {
        (self.h as f64 - 1.0) / 2.0
    }
}

/// lam = lam_sign * s * (u - u0), phi = -s * (v - v0).
pub fn px_to_lamphi(uv: &[V2], cam: &ErpCamera) -> Vec<V2> {
    let (s, u0, v0, sg) = (cam.s(), cam.u0(), cam.v0(), cam.lam_sign() as f64);
    uv.iter().map(|p| [sg * s * (p[0] - u0), -s * (p[1] - v0)]).collect()
}

/// Exact inverse of [`px_to_lamphi`]; does not wrap.
pub fn lamphi_to_px(lp: &[V2], cam: &ErpCamera) -> Vec<V2> {
    let (s, u0, v0, sg) = (cam.s(), cam.u0(), cam.v0(), cam.lam_sign() as f64);
    lp.iter().map(|p| [sg * p[0] / s + u0, -p[1] / s + v0]).collect()
}

/// d = (cos phi cos lam, cos phi sin lam, sin phi).
pub fn lamphi_to_bearing(lp: &[V2]) -> Vec<V3> {
    lp.iter()
        .map(|p| {
            let cp = p[1].cos();
            [cp * p[0].cos(), cp * p[0].sin(), p[1].sin()]
        })
        .collect()
}

/// lam = atan2(dy, dx), phi = atan2(dz, hypot(dx, dy)). `d` need not be normalised.
pub fn bearing_to_lamphi(d: &[V3]) -> Vec<V2> {
    d.iter().map(|v| [v[1].atan2(v[0]), v[2].atan2(v[0].hypot(v[1]))]).collect()
}

/// Jacobian of (lam, phi) w.r.t. an unnormalised bearing. No pole guard (rho = 0 gives non-finite values).
pub fn d_lamphi_d_bearing(d: &V3) -> [[f64; 3]; 2] {
    let (dx, dy, dz) = (d[0], d[1], d[2]);
    let rho2 = dx * dx + dy * dy;
    let rho = rho2.sqrt();
    let r2 = rho2 + dz * dz;
    [[-dy / rho2, dx / rho2, 0.0], [-dx * dz / (r2 * rho), -dy * dz / (r2 * rho), rho / r2]]
}

/// Folds to [-pi, pi) with floor-mod; `wrap_pi(PI) == -PI`.
///
/// For x one ulp below -pi, `x + pi` is a tiny negative number and `rem_euclid` rounds `r + 2pi` up to exactly
/// 2pi, which would give +pi (outside the half-open range); that single case is mapped to the largest double
/// below pi (found by Kani).
pub fn wrap_pi(x: f64) -> f64 {
    let y = (x + PI).rem_euclid(TAU) - PI;
    if y >= PI {
        f64::from_bits(PI.to_bits() - 1)
    } else {
        y
    }
}

/// Wrapped longitude difference used by [`residual_lamphi`].
pub(crate) fn dlam(obs_lam: f64, pred_lam: f64) -> f64 {
    wrap_pi(obs_lam - pred_lam)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseModel {
    /// Isotropic on the tangent plane: r = (cos phi_obs * wrap(dlam), dphi) / s.
    Tangent,
    /// Isotropic in ERP pixels: r = (wrap(dlam), dphi) / s.
    Pixel,
}

/// Residual \[px\] and residual-scale diagonal D. cos uses phi_obs, so the Jacobian is exact.
/// Panics on length mismatch.
pub fn residual_lamphi(obs: &[V2], pred: &[V2], cam: &ErpCamera, nm: NoiseModel) -> (Vec<V2>, Vec<V2>) {
    assert_eq!(obs.len(), pred.len(), "residual_lamphi: length mismatch");
    let s = cam.s();
    let mut rs = Vec::with_capacity(obs.len());
    let mut ds = Vec::with_capacity(obs.len());
    for (o, p) in obs.iter().zip(pred) {
        let dlam = dlam(o[0], p[0]);
        let dphi = o[1] - p[1];
        let d = match nm {
            NoiseModel::Tangent => [o[1].cos() / s, 1.0 / s],
            NoiseModel::Pixel => [1.0 / s, 1.0 / s],
        };
        rs.push([d[0] * dlam, d[1] * dphi]);
        ds.push(d);
    }
    (rs, ds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_linalg::rng::Pcg64;
    use erploc_testkit::max_abs_diff;

    fn cam() -> ErpCamera {
        ErpCamera::new(6528, 3264, 1)
    }
    fn flat2(v: &[V2]) -> Vec<f64> {
        v.iter().flatten().copied().collect()
    }

    #[test]
    fn s_equals_pi_over_h() {
        let c = cam();
        let e = (c.s() - PI / c.h() as f64).abs();
        println!("s err {e:.3e}");
        assert!(e < 1e-18);
    }

    fn sample_px(rng: &mut Pcg64, n: usize) -> Vec<V2> {
        let u: Vec<f64> = (0..n).map(|_| rng.uniform(-0.5, 6527.5)).collect();
        let v: Vec<f64> = (0..n).map(|_| rng.uniform(-0.5, 3263.5)).collect();
        u.iter().zip(&v).map(|(&a, &b)| [a, b]).collect()
    }

    #[test]
    fn px_lamphi_roundtrip() {
        let mut rng = Pcg64::new(0);
        let uv = sample_px(&mut rng, 4000);
        for sg in [1, -1] {
            let c = ErpCamera::new(6528, 3264, sg);
            let e = max_abs_diff(&flat2(&lamphi_to_px(&px_to_lamphi(&uv, &c), &c)), &flat2(&uv));
            println!("px roundtrip sign {sg}: {e:.3e}");
            assert!(e < 1e-9);
        }
    }

    #[test]
    fn lamphi_bearing_roundtrip() {
        let mut rng = Pcg64::new(0);
        let uv = sample_px(&mut rng, 4000);
        let lp = px_to_lamphi(&uv, &cam());
        let e = max_abs_diff(&flat2(&bearing_to_lamphi(&lamphi_to_bearing(&lp))), &flat2(&lp));
        println!("bearing roundtrip {e:.3e}");
        assert!(e < 1e-12);
    }

    fn num_jac(d: &V3) -> [[f64; 3]; 2] {
        let h = 1e-6;
        let mut j = [[0.0; 3]; 2];
        for k in 0..3 {
            let (mut dp, mut dm) = (*d, *d);
            dp[k] += h;
            dm[k] -= h;
            let a = bearing_to_lamphi(&[dp])[0];
            let b = bearing_to_lamphi(&[dm])[0];
            j[0][k] = wrap_pi(a[0] - b[0]) / (2.0 * h);
            j[1][k] = (a[1] - b[1]) / (2.0 * h);
        }
        j
    }

    #[test]
    fn d_lamphi_matches_numeric() {
        let mut rng = Pcg64::new(2);
        let (mut n, mut worst) = (0, 0.0f64);
        while n < 200 {
            let v = [rng.normal(), rng.normal(), rng.normal()];
            if v[0].hypot(v[1]) < 0.2 {
                continue;
            }
            let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let sc = rng.uniform(0.5, 5.0) / nrm;
            let v = [v[0] * sc, v[1] * sc, v[2] * sc];
            let (a, b) = (d_lamphi_d_bearing(&v), num_jac(&v));
            worst = worst.max(max_abs_diff(&a.concat(), &b.concat()));
            n += 1;
        }
        println!("d_lamphi worst {worst:.3e}");
        assert!(worst < 1e-6);
    }

    #[test]
    fn wrap_boundaries() {
        assert_eq!(wrap_pi(PI), -PI);
        assert_eq!(wrap_pi(-PI), -PI);
        assert_eq!(wrap_pi(0.0), 0.0);
        // one ulp below -pi used to wrap to +pi (outside [-pi, pi)); found by Kani
        let below = f64::from_bits((-PI).to_bits() + 1);
        assert!(below < -PI);
        let y = wrap_pi(below);
        assert!((-PI..PI).contains(&y) && (y - PI).abs() < 1e-15, "{y:e}");
        assert!((wrap_pi(1.5 * PI) + PI / 2.0).abs() < 1e-15);
        assert!((wrap_pi(-1.5 * PI) - PI / 2.0).abs() < 1e-15);
        let mut rng = Pcg64::new(3);
        let xs: Vec<f64> = (0..2000).map(|_| rng.uniform(-PI, PI)).collect();
        let ks: Vec<i32> = (0..2000).map(|_| (rng.uniform01() * 7.0) as i32 - 3).collect();
        let worst = xs.iter().zip(&ks).map(|(&x, &k)| (wrap_pi(x + TAU * k as f64) - wrap_pi(x)).abs()).fold(0.0f64, f64::max);
        println!("wrap periodicity worst {worst:.3e}");
        assert!(worst < 1e-12);
        for _ in 0..2 {
            for _ in 0..5000 {
                let y = wrap_pi(rng.uniform(-50.0, 50.0));
                assert!((-PI..PI).contains(&y));
            }
        }
    }

    #[test]
    fn seam_residual() {
        let c = cam();
        let (r, _) = residual_lamphi(&[[PI - 1e-3, 0.4]], &[[-PI + 1e-3, 0.4]], &c, NoiseModel::Tangent);
        let exp = 0.4f64.cos() * (-2e-3) / c.s();
        println!("seam diff {:.3e}", (r[0][0] - exp).abs());
        assert!((r[0][0] - exp).abs() < 1e-9);
    }

    #[test]
    fn d_values() {
        let c = cam();
        let o = [[PI - 1e-3, 0.4]];
        let p = [[-PI + 1e-3, 0.4]];
        let (_, d) = residual_lamphi(&o, &p, &c, NoiseModel::Tangent);
        assert!((d[0][0] - 0.4f64.cos() / c.s()).abs() < 1e-12 && (d[0][1] - 1.0 / c.s()).abs() < 1e-12);
        let (_, d) = residual_lamphi(&o, &p, &c, NoiseModel::Pixel);
        assert!((d[0][0] - 1.0 / c.s()).abs() < 1e-12 && (d[0][1] - 1.0 / c.s()).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn bad_camera_panics() {
        ErpCamera::new(100, 60, 1);
    }
}
