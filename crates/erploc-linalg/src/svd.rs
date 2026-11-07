//! SVD by one-sided Jacobi rotations (Hestenes). Full U (m×m) and V (n×n) like numpy `svd(full_matrices=True)`;
//! singular values descending. Signs of singular vectors may differ from LAPACK (callers must be sign-invariant).
//! The input is pre-scaled by an exact power of two near its max |a_ij|; non-finite input is `Err(NoConvergence)`.
//! Left vectors of singular values ≤ tiny = max(m,n)·eps·‖A‖_F are completion vectors; `pinv` (and `lstsq`) use `u`
//! only above tiny and A·v_k/s_k in (cutoff, tiny].

use crate::{pow2_scale, DMat, LinalgError};

pub const PINV_RCOND: f64 = 1e-15;
const MAX_SWEEPS: usize = 100;

#[derive(Clone, Debug)]
pub struct Svd {
    pub u: DMat,
    pub s: Vec<f64>,
    pub vt: DMat,
}

/// Extend orthonormal columns `q` (m×r) to an orthonormal basis (m×m): Gram–Schmidt (twice) on e_0, e_1, … in order.
pub fn complete_orthonormal(q: &DMat) -> DMat {
    let m = q.rows();
    let mut cols: Vec<Vec<f64>> = (0..q.cols()).map(|j| q.column(j)).collect();
    let mut k = 0;
    while cols.len() < m && k < m {
        let mut v = vec![0.0; m];
        v[k] = 1.0;
        for _ in 0..2 {
            for c in &cols {
                let d: f64 = c.iter().zip(&v).map(|(a, b)| a * b).sum();
                for i in 0..m {
                    v[i] -= d * c[i];
                }
            }
        }
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nv > 1e-8 {
            cols.push(v.iter().map(|x| x / nv).collect());
        }
        k += 1;
    }
    DMat::from_fn(m, m, |i, j| cols[j][i])
}

/// m ≥ n case: returns (U = A·V with orthogonal columns above the floor, s (n), V n×n, floor), unsorted.
/// floor = (max(m,n)·eps)²·‖A‖_F²: columns with ‖u_j‖² ≤ floor are never rotated.
fn jacobi_tall(a: &DMat) -> Result<(DMat, Vec<f64>, DMat, f64), LinalgError> {
    let (m, n) = (a.rows(), a.cols());
    let mut u = a.clone();
    let mut v = DMat::identity(n);

    // Compute Frobenius norm and thresholds
    let mut fro2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let x = u[(i, j)];
            fro2 += x * x;
        }
    }
    let floor = ((m.max(n) as f64) * f64::EPSILON).powi(2) * fro2;
    if n <= 1 {
        let s: Vec<f64> = (0..n).map(|j| (0..m).map(|i| u[(i, j)] * u[(i, j)]).sum::<f64>().sqrt()).collect();
        return Ok((u, s, v, floor));
    }
    let tol = (m as f64).sqrt() * f64::EPSILON;

    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    alpha += u[(i, p)] * u[(i, p)];
                    beta += u[(i, q)] * u[(i, q)];
                    gamma += u[(i, p)] * u[(i, q)];
                }
                // Skip if columns are negligible or orthogonal within tolerance
                if alpha <= floor || beta <= floor || gamma.abs() <= tol * (alpha.sqrt() * beta.sqrt()) {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (up, uq) = (u[(i, p)], u[(i, q)]);
                    u[(i, p)] = c * up - s * uq;
                    u[(i, q)] = s * up + c * uq;
                }
                for i in 0..n {
                    let (vp, vq) = (v[(i, p)], v[(i, q)]);
                    v[(i, p)] = c * vp - s * vq;
                    v[(i, q)] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence);
    }
    let s: Vec<f64> = (0..n).map(|j| (0..m).map(|i| u[(i, j)] * u[(i, j)]).sum::<f64>().sqrt()).collect();
    Ok((u, s, v, floor))
}

pub fn svd(a: &DMat) -> Result<Svd, LinalgError> {
    svd_with_tiny(a).map(|(f, _)| f)
}

/// `svd` plus `tiny = max(m,n)·eps·‖A‖_F` in the units of `a` (columns of `u` for s ≤ tiny are completion vectors).
pub(crate) fn svd_with_tiny(a: &DMat) -> Result<(Svd, f64), LinalgError> {
    if a.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(LinalgError::NoConvergence);
    }
    let scale = pow2_scale(a.max_abs());
    let (mut f, tiny) =
        if scale == 1.0 { svd_unscaled(a)? } else { svd_unscaled(&DMat::from_fn(a.rows(), a.cols(), |i, j| a[(i, j)] / scale))? };
    for x in &mut f.s {
        *x *= scale;
    }
    Ok((f, tiny * scale))
}

fn svd_unscaled(a: &DMat) -> Result<(Svd, f64), LinalgError> {
    let (m, n) = (a.rows(), a.cols());
    if m < n {
        let (f, tiny) = svd_unscaled(&a.transpose())?;
        return Ok((Svd { u: f.vt.transpose(), s: f.s, vt: f.u.transpose() }, tiny));
    }
    let (uw, s, v, floor) = jacobi_tall(a)?;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| s[j].total_cmp(&s[i]));
    let s_sorted: Vec<f64> = idx.iter().map(|&i| s[i]).collect();
    // tiny = max(m,n)·eps·‖A‖_F = sqrt(floor) (sqrt is monotone, so s_j > tiny ⇔ ‖u_j‖² > floor bit-exactly: every
    // kept column was tested against every other kept column in the final rotation-free sweep)
    let tiny = floor.sqrt();
    // normalised left vectors for the non-negligible singular values, then complete to m×m
    let mut keep: Vec<Vec<f64>> = Vec::new();
    for &j in &idx {
        if s[j] > tiny {
            keep.push((0..m).map(|i| uw[(i, j)] / s[j]).collect());
        }
    }
    let q = DMat::from_fn(m, keep.len(), |i, j| keep[j][i]);
    let u = complete_orthonormal(&q);
    let vt = DMat::from_fn(n, n, |r, c| v[(c, idx[r])]);
    Ok((Svd { u, s: s_sorted, vt }, tiny))
}

/// Σ_{s_k > rcond·s_max} v_k u_kᵀ / s_k. For s_k > tiny, u_k is the (orthonormal) Jacobi column of `Svd::u`; for
/// s_k in (cutoff, tiny] the left vector of `Svd::u` is a completion vector, so u_k = A·v_k / s_k is used instead.
pub fn pinv(a: &DMat, rcond: f64) -> Result<DMat, LinalgError> {
    if a.rows() < a.cols() {
        // wide: `svd` runs on the transpose, so rows of vt for s ≤ tiny are completion vectors; the tall path has exact v
        return Ok(pinv(&a.transpose(), rcond)?.transpose());
    }
    let (f, tiny) = svd_with_tiny(a)?;
    let (m, n) = (a.rows(), a.cols());
    let cutoff = rcond * f.s.first().copied().unwrap_or(0.0);
    let mut p = DMat::zeros(n, m);
    for (k, &sk) in f.s.iter().enumerate() {
        if sk > cutoff {
            let vk: Vec<f64> = (0..n).map(|i| f.vt[(k, i)]).collect();
            let uk: Vec<f64> = if sk > tiny { f.u.column(k) } else { a.matvec(&vk).iter().map(|x| x / sk).collect() };
            for i in 0..n {
                for j in 0..m {
                    p[(i, j)] += vk[i] * uk[j] / sk;
                }
            }
        }
    }
    Ok(p)
}

/// Minimum-norm least squares with numpy `lstsq(rcond=None)` cutoff eps·max(m,n)·s_max. Panics if `b.len() != m`.
pub fn lstsq(a: &DMat, b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    assert_eq!(b.len(), a.rows(), "lstsq: length mismatch");
    let rcond = f64::EPSILON * (a.rows().max(a.cols()) as f64);
    Ok(pinv(a, rcond)?.matvec(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg64;

    fn diag(m: usize, n: usize, s: &[f64]) -> DMat {
        DMat::from_fn(m, n, |i, j| if i == j && i < s.len() { s[i] } else { 0.0 })
    }

    fn check(a: &DMat, f: &Svd) {
        let (m, n) = (a.rows(), a.cols());
        assert_eq!((f.u.rows(), f.u.cols(), f.vt.rows(), f.vt.cols(), f.s.len()), (m, m, n, n, m.min(n)));
        let rec = f.u.matmul(&diag(m, n, &f.s)).matmul(&f.vt);
        assert!(rec.sub(a).max_abs() <= 1e-13 * (m.max(n) as f64) * a.max_abs().max(1.0), "reconstruction");
        assert!(f.u.transpose().matmul(&f.u).sub(&DMat::identity(m)).max_abs() < 1e-13 * m as f64, "U orth");
        assert!(f.vt.matmul(&f.vt.transpose()).sub(&DMat::identity(n)).max_abs() < 1e-13 * n as f64, "V orth");
        assert!(f.s.windows(2).all(|w| w[0] >= w[1]) && f.s.iter().all(|&x| x >= 0.0), "order");
    }

    #[test]
    fn shapes_random() {
        let mut r = Pcg64::new(6);
        for (m, n) in [(1, 1), (3, 3), (8, 9), (9, 8), (4, 2), (2, 5), (20, 6)] {
            let a = DMat::from_fn(m, n, |_, _| r.uniform(-1.0, 1.0));
            check(&a, &svd(&a).unwrap());
        }
    }

    #[test]
    fn rank_deficient_and_zero() {
        let a = DMat::from_rows(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]);
        let f = svd(&a).unwrap();
        check(&a, &f);
        assert!(f.s[2] < 1e-14);
        let z = DMat::zeros(2, 3);
        let fz = svd(&z).unwrap();
        check(&z, &fz);
        assert_eq!(fz.s, vec![0.0, 0.0]);
    }

    #[test]
    fn null_vector_of_8x9_is_last_row_of_vt() {
        let mut r = Pcg64::new(7);
        let a = DMat::from_fn(8, 9, |_, _| r.uniform(-1.0, 1.0));
        let f = svd(&a).unwrap();
        let v: Vec<f64> = (0..9).map(|j| f.vt[(8, j)]).collect();
        let res = a.matvec(&v).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(res < 1e-13, "A v = {res}");
    }

    #[test]
    fn pinv_and_lstsq() {
        let mut r = Pcg64::new(8);
        let a = DMat::from_fn(6, 3, |_, _| r.uniform(-1.0, 1.0));
        let p = pinv(&a, PINV_RCOND).unwrap();
        assert!(a.matmul(&p).matmul(&a).sub(&a).max_abs() < 1e-13);
        let b: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let x = lstsq(&a, &b).unwrap();
        // normal equations Aᵀ(Ax - b) = 0
        let res: Vec<f64> = a.matvec(&x).iter().zip(&b).map(|(p, q)| p - q).collect();
        let g = a.transpose().matvec(&res);
        assert!(g.iter().all(|v| v.abs() < 1e-12), "{g:?}");
        // rank-deficient: pinv of a rank-1 matrix
        let u = DMat::from_rows(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let pu = pinv(&u, PINV_RCOND).unwrap();
        assert!(u.matmul(&pu).matmul(&u).sub(&u).max_abs() < 1e-13);
    }

    #[test]
    fn completion_is_orthonormal() {
        let q = DMat::from_rows(3, 1, &[0.0, 0.6, 0.8]);
        let c = complete_orthonormal(&q);
        assert!(c.transpose().matmul(&c).sub(&DMat::identity(3)).max_abs() < 1e-15);
        assert_eq!(c.column(0), q.column(0));
    }

    #[test]
    fn rank2_tall_6x4() {
        let mut rb = Pcg64::new(9);
        let b = DMat::from_fn(6, 2, |_, _| rb.uniform(-1.0, 1.0));
        let mut rc = Pcg64::new(10);
        let c = DMat::from_fn(2, 4, |_, _| rc.uniform(-1.0, 1.0));
        let a = b.matmul(&c);
        let f = svd(&a).unwrap();
        check(&a, &f);
        assert!(f.s[2] < 1e-13 * f.s[0]);
    }

    #[test]
    fn rank2_wide_3x6() {
        let mut rb = Pcg64::new(9);
        let b = DMat::from_fn(3, 2, |_, _| rb.uniform(-1.0, 1.0));
        let mut rc = Pcg64::new(10);
        let c = DMat::from_fn(2, 6, |_, _| rc.uniform(-1.0, 1.0));
        let a = b.matmul(&c);
        let f = svd(&a).unwrap();
        check(&a, &f);
        assert!(f.s[2] < 1e-13 * f.s[0]);
    }

    /// Orthogonal n×n from Gram–Schmidt (twice) of a Pcg64 normal matrix.
    fn orth(r: &mut Pcg64, n: usize) -> DMat {
        let g = DMat::from_fn(n, n, |_, _| r.normal());
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for j in 0..n {
            let mut v = g.column(j);
            for _ in 0..2 {
                for c in &cols {
                    let d: f64 = c.iter().zip(&v).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(c).for_each(|(x, y)| *x -= d * y);
                }
            }
            let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            cols.push(v.iter().map(|x| x / nv).collect());
        }
        DMat::from_fn(n, n, |i, j| cols[j][i])
    }

    #[test]
    fn gap_orthonormality() {
        // singular value x in the former gap (sqrt(floor), max(m,n)·eps·s_max]: U must stay orthonormal
        let mut r = Pcg64::new(11);
        let (mut wu, mut wv, mut wr) = (0.0f64, 0.0f64, 0.0f64);
        for (m, n) in [(4, 4), (9, 8), (8, 9)] {
            for x in [1e-16, 3e-16, 1e-15, 2e-15, 3e-15, 1e-14] {
                let (q1, q2) = (orth(&mut r, m), orth(&mut r, n));
                let k = m.min(n);
                let d: Vec<f64> = (0..k).map(|i| if i + 1 == k { x } else { 1.0 }).collect();
                let a = q1.matmul(&diag(m, n, &d)).matmul(&q2.transpose());
                let f = svd(&a).unwrap();
                let eu = f.u.transpose().matmul(&f.u).sub(&DMat::identity(m)).max_abs();
                let ev = f.vt.matmul(&f.vt.transpose()).sub(&DMat::identity(n)).max_abs();
                let er = f.u.matmul(&diag(m, n, &f.s)).matmul(&f.vt).sub(&a).max_abs();
                assert!(eu <= 1e-13 && ev <= 1e-13 && er <= 1e-13, "{m}x{n} x={x:e}: U {eu:e} V {ev:e} rec {er:e}");
                wu = wu.max(eu);
                wv = wv.max(ev);
                wr = wr.max(er);
            }
        }
        println!("gap_orthonormality worst |UtU-I| {wu:e} |VVt-I| {wv:e} rec {wr:e}");
    }

    #[test]
    fn pinv_reference_12x12() {
        // singular values logspace(0, -14.7, 12): the last (≈2e-15) lies in (1e-15·s_max, tiny], so pinv keeps it
        // while `Svd::u` holds a completion vector there (logspace(0, -15.2, 12) puts no value in that interval).
        // Seed 13: the completion vector is -u_11, so a pinv built from `Svd::u` gets that term's sign wrong.
        let n = 12;
        let mut r = Pcg64::new(13);
        let (q1, q2) = (orth(&mut r, n), orth(&mut r, n));
        let sv: Vec<f64> = (0..n).map(|k| 10f64.powf(-14.7 * k as f64 / (n - 1) as f64)).collect();
        let a = q1.matmul(&diag(n, n, &sv)).matmul(&q2.transpose());
        let f = svd(&a).unwrap();
        let tiny = (n as f64) * f64::EPSILON * a.frobenius();
        let cutoff = PINV_RCOND * f.s[0];
        assert!(f.s.iter().filter(|&&x| x > cutoff && x <= tiny).count() >= 1, "no singular value in (cutoff, tiny]");
        // Moore–Penrose identities; at cond ≈ 5e14 the floating-point floor of these products is eps·cond ≈ 0.1
        // (numpy's own pinv gives 1e-3 .. 6e-3 on such matrices), so 1e-12 / 1e-6 are not attainable
        let ident = |p: &DMat| {
            let apa = a.matmul(p).matmul(&a).sub(&a).max_abs() / a.max_abs();
            let pap = p.matmul(&a).matmul(p).sub(p).max_abs() / p.max_abs();
            (apa, pap)
        };
        let (apa, pap) = ident(&pinv(&a, PINV_RCOND).unwrap());
        // the same sum built from the completion vectors of Svd::u (the defect fixed by I1)
        let mut pc = DMat::zeros(n, n);
        for k in (0..n).filter(|&k| f.s[k] > cutoff) {
            for i in 0..n {
                for j in 0..n {
                    pc[(i, j)] += f.vt[(k, i)] * f.u[(j, k)] / f.s[k];
                }
            }
        }
        let (_, pap_c) = ident(&pc);
        println!("pinv_reference_12x12 |APA-A|/|A| {apa:e} |PAP-P|/|P| {pap:e} (completion-vector pinv: {pap_c:e})");
        assert!(pap_c > 1.0, "seed no longer exercises the completion-vector defect: {pap_c:e}");
        assert!(apa <= 0.5 && pap <= 0.5, "APA {apa:e} PAP {pap:e}");
    }

    #[test]
    fn pinv_hybrid_accuracy() {
        // singular values logspace(0, -14.7, 12); e = |A P A - A|_max / |A|_max
        let n = 12;
        let sv: Vec<f64> = (0..n).map(|k| 10f64.powf(-14.7 * k as f64 / (n - 1) as f64)).collect();
        // this test pins accuracy, not the hybrid choice (pure A·v/s measured within a few percent)
        for seed in 11..=15u64 {
            let mut r = Pcg64::new(seed);
            let (q1, q2) = (orth(&mut r, n), orth(&mut r, n));
            let a = q1.matmul(&diag(n, n, &sv)).matmul(&q2.transpose());
            let p = pinv(&a, PINV_RCOND).unwrap();
            let e = a.matmul(&p).matmul(&a).sub(&a).max_abs() / a.max_abs();
            println!("pinv_hybrid_accuracy seed {seed} e {e:e}");
            // bound = eps·cond order (cond ≈ 5e14); numpy measures 1e-3..6e-3 on similar matrices; Jacobi columns are
            // used for s > tiny (orthonormal by the C1 proof), A·v/s for s in (cutoff, tiny]
            // the matrices are numerically rank deficient at this cond: another libm (or the shim of the loose-gate check) moves the
            // error by a large factor, so the bound off the reference platform is 0.5
            let bound = if erploc_testkit::strict() { 0.1 } else { 0.5 };
            assert!(e <= bound, "seed {seed}: e {e:e}");
        }
    }

    #[test]
    fn pinv_wide_equals_transpose() {
        let (m, n) = (12, 20);
        let mut r = Pcg64::new(21);
        let (q1, q2) = (orth(&mut r, m), orth(&mut r, n));
        let sv: Vec<f64> = (0..m).map(|k| 10f64.powf(-14.7 * k as f64 / (m - 1) as f64)).collect();
        let a = q1.matmul(&diag(m, n, &sv)).matmul(&q2.transpose());
        let p = pinv(&a, PINV_RCOND).unwrap();
        let pt = pinv(&a.transpose(), PINV_RCOND).unwrap().transpose();
        assert!(p.sub(&pt).max_abs() <= 1e-12 * p.max_abs(), "{:e}", p.sub(&pt).max_abs());
        let e = a.matmul(&p).matmul(&a).sub(&a).max_abs() / a.max_abs();
        assert!(e <= 0.1, "e {e:e}");
    }

    #[test]
    fn lstsq_wide_min_norm() {
        let mut r = Pcg64::new(22);
        let a = DMat::from_fn(3, 6, |_, _| r.uniform(-1.0, 1.0));
        let b = [1.0, -2.0, 0.5];
        let x = lstsq(&a, &b).unwrap();
        assert!(a.matvec(&x).iter().zip(&b).all(|(p, q)| (p - q).abs() <= 1e-12));
        let xp = pinv(&a, f64::EPSILON * 6.0).unwrap().matvec(&b);
        assert!(x.iter().zip(&xp).all(|(p, q)| (p - q).abs() <= 1e-12));
    }

    #[test]
    fn pinv_gap_exact_sign() {
        // permuted diag(1,…,1,-x) with x in (1e-15, tiny]: no rotations, exact arithmetic; Svd::u completes with
        // +e, the true left vector is -e, so pinv must equal the exact inverse
        let n = 12;
        let x = 2e-15;
        let perm = |i: usize| (5 * i + 3) % n;
        let a = DMat::from_fn(n, n, |i, j| {
            if j == perm(i) {
                if i == n - 1 {
                    -x
                } else {
                    1.0
                }
            } else {
                0.0
            }
        });
        let exact = DMat::from_fn(n, n, |j, i| {
            if j == perm(i) {
                if i == n - 1 {
                    -1.0 / x
                } else {
                    1.0
                }
            } else {
                0.0
            }
        });
        let p = pinv(&a, PINV_RCOND).unwrap();
        assert!(p.sub(&exact).max_abs() <= 1e-15 * exact.max_abs(), "{:e}", p.sub(&exact).max_abs());
    }

    #[test]
    fn svd_extreme_scaling() {
        let mut r = Pcg64::new(13);
        let a = DMat::from_fn(6, 4, |_, _| r.uniform(-1.0, 1.0));
        let s0 = svd(&a).unwrap().s;
        let mut worst = 0.0f64;
        for sc in [1e-160, 1e155] {
            let s = svd(&a.scale(sc)).unwrap().s;
            for (x, y) in s.iter().zip(&s0) {
                let e = (x - y * sc).abs() / (y * sc);
                worst = worst.max(e);
                assert!(e <= 1e-12, "scale {sc}: {x} vs {}", y * sc);
            }
        }
        println!("svd_extreme_scaling worst relative error {worst:e}");
    }

    #[test]
    fn svd_nonfinite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut a = DMat::identity(3);
            a[(1, 2)] = bad;
            assert_eq!(svd(&a).map(|_| ()), Err(LinalgError::NoConvergence));
        }
    }
}
