//! Cholesky A = L Lᵀ from the lower triangle of A. Fails (NotPositiveDefinite) when a pivot is not > 0
//! (this also rejects NaN).

use crate::{DMat, LinalgError};

/// Panics if `a` is not square.
pub fn cholesky(a: &DMat) -> Result<DMat, LinalgError> {
    assert!(a.rows() == a.cols(), "cholesky: matrix must be square, got {}x{}", a.rows(), a.cols());
    let n = a.rows();
    let mut l = DMat::zeros(n, n);
    for j in 0..n {
        let mut s = a[(j, j)];
        for k in 0..j {
            s -= l[(j, k)] * l[(j, k)];
        }
        if s <= 0.0 || s.is_nan() {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let d = s.sqrt();
        l[(j, j)] = d;
        for i in (j + 1)..n {
            let mut t = a[(i, j)];
            for k in 0..j {
                t -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = t / d;
        }
    }
    Ok(l)
}

/// Solve L Lᵀ x = b. Panics if `b.len() != l.rows()`.
pub fn cho_solve(l: &DMat, b: &[f64]) -> Vec<f64> {
    let n = l.rows();
    assert_eq!(b.len(), n, "cho_solve: length mismatch");
    let mut y = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            y[i] -= l[(i, k)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            y[i] -= l[(k, i)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg64;

    #[test]
    fn factor_and_solve_spd() {
        let mut r = Pcg64::new(2);
        for n in [1, 3, 6, 30] {
            let b = DMat::from_fn(n, n, |_, _| r.uniform(-1.0, 1.0));
            let a = b.transpose().matmul(&b).add(&DMat::identity(n).scale(n as f64));
            let l = cholesky(&a).unwrap();
            let e = l.matmul(&l.transpose()).sub(&a).max_abs();
            assert!(e < 1e-12 * a.max_abs(), "n={n} err={e}");
            let rhs: Vec<f64> = (0..n).map(|i| i as f64 - 1.0).collect();
            let x = cho_solve(&l, &rhs);
            let res = a.matvec(&x).iter().zip(&rhs).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
            assert!(res < 1e-11, "n={n} res={res}");
        }
    }

    #[test]
    fn not_positive_definite() {
        let a = DMat::from_rows(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert_eq!(cholesky(&a), Err(LinalgError::NotPositiveDefinite));
        let nan = DMat::from_rows(1, 1, &[f64::NAN]);
        assert_eq!(cholesky(&nan), Err(LinalgError::NotPositiveDefinite));
    }
}
