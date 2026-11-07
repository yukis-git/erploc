//! LU with partial pivoting (LAPACK getrf semantics: the pivot is the first entry of maximal |a| in the column;
//! factorisation fails only on an exactly zero pivot, like numpy's LinAlgError).

use crate::{DMat, LinalgError};

#[derive(Clone, Debug)]
pub struct Lu {
    n: usize,
    lu: Vec<f64>,
    perm: Vec<usize>,
    sign: f64,
}

/// Panics if `a` is not square.
pub fn lu(a: &DMat) -> Result<Lu, LinalgError> {
    assert!(a.rows() == a.cols(), "lu: matrix must be square, got {}x{}", a.rows(), a.cols());
    let n = a.rows();
    let mut m = a.as_slice().to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;
    for k in 0..n {
        let mut p = k;
        let mut best = m[k * n + k].abs();
        for i in (k + 1)..n {
            let v = m[i * n + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if m[p * n + k] == 0.0 {
            return Err(LinalgError::Singular);
        }
        if p != k {
            for j in 0..n {
                m.swap(k * n + j, p * n + j);
            }
            perm.swap(k, p);
            sign = -sign;
        }
        let piv = m[k * n + k];
        for i in (k + 1)..n {
            let f = m[i * n + k] / piv;
            m[i * n + k] = f;
            for j in (k + 1)..n {
                m[i * n + j] -= f * m[k * n + j];
            }
        }
    }
    Ok(Lu { n, lu: m, perm, sign })
}

impl Lu {
    /// Panics if `b.len() != n`.
    #[allow(clippy::needless_range_loop)]
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        assert_eq!(b.len(), n, "Lu::solve: length mismatch");
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let mut s = x[i];
            for j in 0..i {
                s -= self.lu[i * n + j] * x[j];
            }
            x[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..n {
                s -= self.lu[i * n + j] * x[j];
            }
            x[i] = s / self.lu[i * n + i];
        }
        x
    }

    /// Panics if `b.rows() != n`.
    pub fn solve_mat(&self, b: &DMat) -> DMat {
        assert_eq!(b.rows(), self.n, "Lu::solve_mat: dimension mismatch");
        let mut out = DMat::zeros(b.rows(), b.cols());
        for j in 0..b.cols() {
            out.set_column(j, &self.solve(&b.column(j)));
        }
        out
    }

    pub fn det(&self) -> f64 {
        (0..self.n).fold(self.sign, |d, i| d * self.lu[i * self.n + i])
    }

    /// Diagonal of U.
    pub fn pivots(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.lu[i * self.n + i]).collect()
    }
}

/// Panics if `a` is not square or `b.len() != a.rows()`.
pub fn solve(a: &DMat, b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    assert!(b.len() == a.rows(), "solve: rhs length {} != rows {}", b.len(), a.rows());
    Ok(lu(a)?.solve(b))
}

/// Panics if `a` is not square or `b.rows() != a.rows()`.
pub fn solve_mat(a: &DMat, b: &DMat) -> Result<DMat, LinalgError> {
    assert!(b.rows() == a.rows(), "solve_mat: rhs rows {} != rows {}", b.rows(), a.rows());
    Ok(lu(a)?.solve_mat(b))
}

/// Panics if `a` is not square.
pub fn inv(a: &DMat) -> Result<DMat, LinalgError> {
    let f = lu(a)?;
    Ok(f.solve_mat(&DMat::identity(a.rows())))
}

/// numpy semantics: 0.0 for an exactly singular matrix. Panics if not square.
pub fn det(a: &DMat) -> f64 {
    match lu(a) {
        Ok(f) => f.det(),
        Err(LinalgError::Singular) => 0.0,
        Err(e) => panic!("det: {e:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg64;

    fn rand_mat(r: &mut Pcg64, n: usize) -> DMat {
        // diagonally dominant: well conditioned
        DMat::from_fn(n, n, |i, j| r.uniform(-1.0, 1.0) + if i == j { n as f64 } else { 0.0 })
    }

    #[test]
    fn solve_and_inverse_residuals() {
        let mut r = Pcg64::new(1);
        for n in [1, 2, 3, 4, 6, 10, 40] {
            let a = rand_mat(&mut r, n);
            let b: Vec<f64> = (0..n).map(|_| r.uniform(-1.0, 1.0)).collect();
            let x = solve(&a, &b).unwrap();
            let res = a.matvec(&x).iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
            assert!(res < 1e-12, "n={n} res={res}");
            let ai = inv(&a).unwrap();
            let e = a.matmul(&ai).sub(&DMat::identity(n)).max_abs();
            assert!(e < 1e-12, "n={n} inv err={e}");
        }
    }

    #[test]
    fn det_values() {
        let a = DMat::from_rows(3, 3, &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
        assert!((det(&a) - 36.0).abs() < 1e-12);
        let p = DMat::from_rows(2, 2, &[0.0, 1.0, 1.0, 0.0]); // one row swap
        assert!((det(&p) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn exactly_singular_is_error_and_det_zero() {
        let a = DMat::from_rows(3, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);
        assert_eq!(solve(&a, &[1.0, 2.0, 3.0]), Err(LinalgError::Singular));
        assert_eq!(inv(&a), Err(LinalgError::Singular));
        assert_eq!(det(&a), 0.0);
    }

    #[test]
    #[should_panic(expected = "must be square")]
    fn inv_non_square_panics() {
        let _ = inv(&DMat::zeros(2, 3));
    }

    #[test]
    #[should_panic(expected = "rhs length")]
    fn solve_length_mismatch_panics() {
        let _ = solve(&DMat::identity(2), &[1.0]);
    }

    #[test]
    fn partial_pivoting_picks_first_max() {
        let a = DMat::from_rows(2, 2, &[1e-20, 1.0, 1.0, 1.0]);
        let x = solve(&a, &[1.0, 2.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lu_propagates_nan() {
        let a = DMat::from_rows(2, 2, &[1.0, f64::INFINITY, 0.0, 1.0]);
        // no pivot is exactly zero, so the factorisation succeeds; the row update 1 - 0·inf makes U[1][1] = NaN
        // (no skip of zero multipliers, numpy parity), and back substitution carries the NaN into the solution
        let f = lu(&a).expect("factorisation of [[1,inf],[0,1]] must succeed");
        let x = f.solve(&[1.0, 1.0]);
        assert!(x.iter().any(|v| v.is_nan()), "NaN expected in solution, got {x:?}");
    }
}
