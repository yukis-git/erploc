//! Symmetric eigen-decomposition by the cyclic Jacobi method (Golub & Van Loan §8.5). Reads the lower triangle
//! (numpy `eigvalsh`/`eigh` default). Values ascending, vectors as columns. Backward stable: eigenvalue errors
//! O(eps·‖A‖); eigenvector signs may differ from LAPACK. The input is pre-scaled by an exact power of two near its
//! max |a_ij| (no overflow/underflow in the norms); a non-finite lower triangle gives all-NaN values and vectors
//! (numpy `eigvalsh` returns nan).

use crate::{pow2_scale, DMat, LinalgError};

pub const MAX_SWEEPS: usize = 100;

#[derive(Clone, Debug)]
pub struct SymEig {
    pub values: Vec<f64>,
    pub vectors: DMat,
}

pub fn eigh(a: &DMat) -> Result<SymEig, LinalgError> {
    eigh_with_sweeps(a, MAX_SWEEPS)
}

pub fn eigvalsh(a: &DMat) -> Result<Vec<f64>, LinalgError> {
    eigh(a).map(|e| e.values)
}

fn off_norm(m: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            s += m[i * n + j] * m[i * n + j];
        }
    }
    s.sqrt()
}

/// Panics if `a` is not square.
pub fn eigh_with_sweeps(a: &DMat, max_sweeps: usize) -> Result<SymEig, LinalgError> {
    assert!(a.rows() == a.cols(), "eigh: matrix must be square, got {}x{}", a.rows(), a.cols());
    let n = a.rows();
    let sym = a.sym_from_lower();
    if sym.as_slice().iter().any(|x| !x.is_finite()) {
        return Ok(SymEig { values: vec![f64::NAN; n], vectors: DMat::from_fn(n, n, |_, _| f64::NAN) });
    }
    let scale = pow2_scale(sym.max_abs());
    let mut m: Vec<f64> = sym.as_slice().iter().map(|x| x / scale).collect();
    let mut v = DMat::identity(n);
    let fro = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut converged = n <= 1;
    for _ in 0..max_sweeps {
        let off = off_norm(&m, n);
        if off <= f64::EPSILON * fro {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let tau = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = if tau.abs() > 1e150 { 0.5 / tau } else { tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt()) };
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    if !converged {
        let off = off_norm(&m, n);
        if off > f64::EPSILON * fro {
            return Err(LinalgError::NoConvergence);
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
    let values = idx.iter().map(|&i| m[i * n + i] * scale).collect();
    let vectors = DMat::from_fn(n, n, |r, c| v[(r, idx[c])]);
    Ok(SymEig { values, vectors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg64;

    fn check(a: &DMat, e: &SymEig, tol: f64) {
        let n = a.rows();
        let v = &e.vectors;
        let lam = DMat::from_fn(n, n, |i, j| if i == j { e.values[i] } else { 0.0 });
        let s = a.sym_from_lower();
        let r = s.matmul(v).sub(&v.matmul(&lam)).max_abs();
        assert!(r <= tol * s.max_abs().max(1.0), "AV-VL {r}");
        let o = v.transpose().matmul(v).sub(&DMat::identity(n)).max_abs();
        assert!(o <= 1e-13 * n as f64, "orthogonality {o}");
        assert!(e.values.windows(2).all(|w| w[0] <= w[1]), "not ascending");
    }

    #[test]
    fn random_symmetric() {
        let mut r = Pcg64::new(3);
        for n in [1, 2, 3, 4, 6, 10, 50] {
            let b = DMat::from_fn(n, n, |_, _| r.uniform(-1.0, 1.0));
            let a = b.add(&b.transpose());
            check(&a, &eigh(&a).unwrap(), 1e-13 * n as f64);
        }
    }

    #[test]
    fn diagonal_repeated_and_lower_only() {
        let a = DMat::from_rows(3, 3, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0]);
        assert_eq!(eigvalsh(&a).unwrap(), vec![-1.0, 2.0, 2.0]);
        // upper triangle is ignored (numpy UPLO='L')
        let b = DMat::from_rows(2, 2, &[1.0, 1e9, 0.0, 3.0]);
        assert_eq!(eigvalsh(&b).unwrap(), vec![1.0, 3.0]);
    }

    #[test]
    fn ill_conditioned_spd_absolute_accuracy() {
        // Q diag(1e-10, 1, 1e4, 1e8) Qᵀ: for a general (non-graded) matrix the backward-stable bound is
        // |Δλ| ≤ c·eps·‖A‖ (Jacobi's extra relative accuracy needs scaled diagonal dominance, not assumed here)
        let mut r = Pcg64::new(4);
        let g = DMat::from_fn(4, 4, |_, _| r.normal());

        // Gram–Schmidt orthonormalisation: columns in order, normalise each
        let mut q = g.clone();
        for j in 0..q.cols() {
            // Subtract projections onto previous orthonormal columns
            for i in 0..j {
                let mut dot = 0.0;
                for k in 0..q.rows() {
                    dot += q[(k, i)] * g[(k, j)];
                }
                for k in 0..q.rows() {
                    q[(k, j)] -= dot * q[(k, i)];
                }
            }
            // Normalize column j
            let mut norm = 0.0;
            for k in 0..q.rows() {
                norm += q[(k, j)] * q[(k, j)];
            }
            norm = norm.sqrt();
            for k in 0..q.rows() {
                q[(k, j)] /= norm;
            }
        }

        let d = DMat::from_fn(4, 4, |i, j| if i == j { [1e-10, 1.0, 1e4, 1e8][i] } else { 0.0 });
        let a = q.matmul(&d).matmul(&q.transpose());
        let w = eigvalsh(&a).unwrap();
        assert!((w[3] / 1e8 - 1.0).abs() < 1e-12);
        assert!((w[0] - 1e-10).abs() < 100.0 * f64::EPSILON * 1e8, "small eig {w:?}");
    }

    #[test]
    fn sweep_limit_reports_no_convergence() {
        let mut r = Pcg64::new(5);
        let b = DMat::from_fn(8, 8, |_, _| r.uniform(-1.0, 1.0));
        let a = b.add(&b.transpose());
        assert_eq!(eigh_with_sweeps(&a, 0).map(|_| ()), Err(LinalgError::NoConvergence));
    }

    fn sym3() -> DMat {
        DMat::from_rows(3, 3, &[4.0, 1.0, -2.0, 1.0, 3.0, 0.5, -2.0, 0.5, 1.0])
    }

    #[test]
    fn eig_extreme_scaling() {
        let a = sym3();
        let w0 = eigvalsh(&a).unwrap();
        let mut worst = 0.0f64;
        for sc in [1e-170, 1e-300, 1e160, 1e300] {
            let w = eigvalsh(&a.scale(sc)).unwrap();
            for (x, y) in w.iter().zip(&w0) {
                let e = (x - y * sc).abs() / (y * sc).abs();
                worst = worst.max(e);
                assert!(e <= 1e-12, "scale {sc}: {x} vs {}", y * sc);
            }
        }
        println!("eig_extreme_scaling worst relative error {worst:e}");
    }

    #[test]
    fn eig_nonfinite() {
        let mut a = sym3();
        a[(2, 0)] = f64::INFINITY;
        let e = eigh(&a).unwrap();
        assert!(e.values.iter().all(|x| x.is_nan()) && e.vectors.as_slice().iter().all(|x| x.is_nan()));
        let mut b = sym3();
        b[(1, 1)] = f64::NAN;
        let e = eigh(&b).unwrap();
        assert!(e.values.iter().all(|x| x.is_nan()) && e.vectors.as_slice().iter().all(|x| x.is_nan()));
    }

    #[test]
    #[should_panic(expected = "must be square")]
    fn eigh_non_square_panics() {
        let _ = eigh(&DMat::zeros(2, 3));
    }
}
