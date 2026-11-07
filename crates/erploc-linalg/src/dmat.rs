//! Dynamic row-major matrix. Dimension mismatches are programming errors and panic (documented per method).

use core::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct DMat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        DMat { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Panics if `v.len() != rows * cols`.
    pub fn from_rows(rows: usize, cols: usize, v: &[f64]) -> Self {
        assert_eq!(v.len(), rows * cols, "from_rows: length mismatch");
        DMat { rows, cols, data: v.to_vec() }
    }

    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        DMat { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |i, j| self[(j, i)])
    }

    /// Panics if `self.cols != other.rows`.
    pub fn matmul(&self, other: &DMat) -> Self {
        assert_eq!(self.cols, other.rows, "matmul: dimension mismatch");
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += a * other.data[k * other.cols + j];
                }
            }
        }
        out
    }

    /// Panics if `v.len() != self.cols`.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols, "matvec: dimension mismatch");
        (0..self.rows).map(|i| (0..self.cols).map(|j| self.data[i * self.cols + j] * v[j]).sum()).collect()
    }

    /// Panics on shape mismatch.
    pub fn add(&self, other: &DMat) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "add: shape mismatch");
        DMat { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect() }
    }

    /// Panics on shape mismatch.
    pub fn sub(&self, other: &DMat) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "sub: shape mismatch");
        DMat { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect() }
    }

    pub fn scale(&self, s: f64) -> Self {
        DMat { rows: self.rows, cols: self.cols, data: self.data.iter().map(|a| a * s).collect() }
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |m, a| m.max(a.abs()))
    }

    pub fn frobenius(&self) -> f64 {
        self.data.iter().map(|a| a * a).sum::<f64>().sqrt()
    }

    /// Symmetric matrix built from the lower triangle (numpy `eigvalsh` default `UPLO='L'`). Panics if not square.
    pub fn sym_from_lower(&self) -> Self {
        assert_eq!(self.rows, self.cols, "sym_from_lower: not square");
        Self::from_fn(self.rows, self.cols, |i, j| if i >= j { self[(i, j)] } else { self[(j, i)] })
    }

    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }

    /// Panics if `v.len() != rows`.
    pub fn set_column(&mut self, j: usize, v: &[f64]) {
        assert_eq!(v.len(), self.rows, "set_column: length mismatch");
        for i in 0..self.rows {
            self[(i, j)] = v[i];
        }
    }
}

impl Index<(usize, usize)> for DMat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        assert!(i < self.rows && j < self.cols, "DMat index out of range");
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for DMat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        assert!(i < self.rows && j < self.cols, "DMat index out of range");
        &mut self.data[i * self.cols + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_transpose_identity() {
        let a = DMat::from_rows(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let at = a.transpose();
        assert_eq!((at.rows(), at.cols()), (3, 2));
        assert_eq!(at[(2, 1)], 6.0);
        let p = a.matmul(&at);
        assert_eq!(p.as_slice(), &[14.0, 32.0, 32.0, 77.0]);
        assert_eq!(a.matmul(&DMat::identity(3)), a);
        assert_eq!(a.matvec(&[1.0, 0.0, -1.0]), vec![-2.0, -2.0]);
    }

    #[test]
    fn sym_from_lower_uses_lower_triangle() {
        let a = DMat::from_rows(2, 2, &[1.0, 99.0, 2.0, 3.0]);
        assert_eq!(a.sym_from_lower().as_slice(), &[1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn norms_and_columns() {
        let mut a = DMat::from_rows(2, 2, &[3.0, -4.0, 0.0, 0.0]);
        assert_eq!(a.max_abs(), 4.0);
        assert_eq!(a.frobenius(), 5.0);
        assert_eq!(a.column(1), vec![-4.0, 0.0]);
        a.set_column(0, &[7.0, 8.0]);
        assert_eq!(a.as_slice(), &[7.0, -4.0, 8.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn matmul_dimension_mismatch_panics() {
        let _ = DMat::zeros(2, 3).matmul(&DMat::zeros(2, 3));
    }

    #[test]
    fn matmul_propagates_nan() {
        let result = DMat::from_rows(1, 1, &[0.0]).matmul(&DMat::from_rows(1, 1, &[f64::INFINITY]));
        assert!(result[(0, 0)].is_nan());
    }
}
