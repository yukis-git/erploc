//! Fixed-size row-major matrix and 3-vector helpers (used by the hot small-matrix code of the other crates).

use crate::DMat;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat<const R: usize, const C: usize>(pub [[f64; C]; R]);

pub type V3 = [f64; 3];

impl<const R: usize, const C: usize> Mat<R, C> {
    pub fn zeros() -> Self {
        Mat([[0.0; C]; R])
    }

    pub fn transpose(&self) -> Mat<C, R> {
        let mut t = Mat::<C, R>::zeros();
        for i in 0..R {
            for j in 0..C {
                t.0[j][i] = self.0[i][j];
            }
        }
        t
    }

    pub fn mul<const K: usize>(&self, other: &Mat<C, K>) -> Mat<R, K> {
        let mut out = Mat::<R, K>::zeros();
        for i in 0..R {
            for k in 0..C {
                for j in 0..K {
                    out.0[i][j] += self.0[i][k] * other.0[k][j];
                }
            }
        }
        out
    }

    pub fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        for i in 0..R {
            for j in 0..C {
                out.0[i][j] += o.0[i][j];
            }
        }
        out
    }

    pub fn sub(&self, o: &Self) -> Self {
        let mut out = *self;
        for i in 0..R {
            for j in 0..C {
                out.0[i][j] -= o.0[i][j];
            }
        }
        out
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut out = *self;
        for row in out.0.iter_mut() {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
        out
    }

    pub fn to_dmat(&self) -> DMat {
        DMat::from_fn(R, C, |i, j| self.0[i][j])
    }

    /// `None` if the shape differs.
    pub fn from_dmat(m: &DMat) -> Option<Self> {
        if m.rows() != R || m.cols() != C {
            return None;
        }
        let mut out = Self::zeros();
        for i in 0..R {
            for j in 0..C {
                out.0[i][j] = m[(i, j)];
            }
        }
        Some(out)
    }
}

impl<const N: usize> Mat<N, N> {
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..N {
            m.0[i][i] = 1.0;
        }
        m
    }
}

pub fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(a: &V3) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_ops_match_dmat() {
        let a = Mat::<2, 3>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let p = a.mul(&a.transpose());
        assert_eq!(p.0, [[14.0, 32.0], [32.0, 77.0]]);
        assert_eq!(Mat::<2, 3>::from_dmat(&a.to_dmat()), Some(a));
        assert_eq!(Mat::<3, 3>::from_dmat(&a.to_dmat()), None);
        assert_eq!(Mat::<3, 3>::identity().mul(&Mat::<3, 3>::identity()), Mat::<3, 3>::identity());
    }

    #[test]
    fn vec3_helpers() {
        assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(norm(&[3.0, 4.0, 12.0]), 13.0);
    }
}
