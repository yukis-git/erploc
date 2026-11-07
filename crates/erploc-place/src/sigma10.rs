//! `sigma10` of `erploc/ift.py`: `sqrt(tr(inv(I)[0:3,0:3]))` of a symmetric 4x4 matrix given as its 10 upper-triangle
//! entries, with the 3-way rule of the Lean bounds (SitePlan/Cond.lean).
//!
//! The camera information matrix is regular for `erploc` (`fisher::sigma_pos`) iff `lambda_min > RTOL * lambda_max`. With
//! `T = tr(I) tr(I^-1) = sum_ij l_j / l_i`: `T < 1/RTOL` implies regular and `T >= n^2/RTOL` implies singular (n = 4); the
//! thresholds carry a factor-2 margin for rounding. Entries whose `T` falls in between (or is NaN, or whose Schur complement is not
//! positive) are sent to the exact rule `sigma_pos` (eigenvalue test + LU inverse), exactly as the Python does.
//!
//! The arithmetic follows the numpy expression order of `ift.sigma10` operation by operation (no `mul_add`: numpy's elementwise
//! operations do not fuse), and `np.minimum/np.maximum` are the NaN-propagating `nmin/nmax`.
use crate::npcompat::{nmax, nmin};
use crate::placement::{cap as cap_value, PlaceError};
use crate::site::IU;
use erploc_est::fisher::{sigma_pos, SIGMA_RTOL};
use erploc_linalg::DMat;

/// `RTOL` of `erploc.fisher.sigma_pos`.
pub const RTOL: f64 = 1e-12;
/// Sure regular below (Lean bound `1/RTOL`, factor-2 margin for rounding).
pub const T_OK: f64 = 0.5 / RTOL;
/// Sure singular above (Lean bound `n^2/RTOL` with n = 4, factor-2 margin).
pub const T_BAD: f64 = 2.0 * 16.0 / RTOL;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    SureOk,
    SureBad,
    Ambiguous,
}

/// Branch decision from the three scalars the Python code compares: `pos = ok & (det3 > 0) & (trI > 0)`, `trI`, `T = tr_full * trI`.
/// `sure_ok = pos & (T < T_OK)`; `sure_bad = (trI <= 0) | (pos & (T >= T_BAD))`; otherwise ambiguous (a NaN `T` is ambiguous).
/// Float comparisons only.
pub fn classify_branch(pos: bool, tr_i: f64, t: f64) -> Branch {
    let sure_ok = pos && t < T_OK;
    let sure_bad = tr_i <= 0.0 || (pos && t >= T_BAD);
    if sure_ok {
        Branch::SureOk
    } else if sure_bad {
        Branch::SureBad
    } else {
        Branch::Ambiguous
    }
}

/// Value of the sure-regular branch: `min(sqrt(max(tr3, 0)), cap)` with numpy's NaN-propagating `maximum/minimum` (Kani).
pub(crate) fn finish(tr3: f64, cap: f64) -> f64 {
    nmin(nmax(tr3, 0.0).sqrt(), cap)
}

/// `AMB_COUNT`: `[entries sent to the exact rule, entries evaluated]`.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Sigma10Counts {
    pub amb: usize,
    pub total: usize,
}

impl std::ops::AddAssign for Sigma10Counts {
    fn add_assign(&mut self, o: Self) {
        self.amb += o.amb;
        self.total += o.total;
    }
}

/// The 4x4 symmetric matrix of the 10 upper-triangle entries (`IU` order).
pub fn mat4_from_iu(a: &[f64; 10]) -> DMat {
    let mut m = [0.0; 16];
    for (t, &(i, j)) in IU.iter().enumerate() {
        m[i * 4 + j] = a[t];
        m[j * 4 + i] = a[t];
    }
    DMat::from_rows(4, 4, &m)
}

/// One symmetric 4x4 given as its 10 upper-triangle entries (`IU` order); capped sigma (inf/NaN -> `cap`).
/// An `Err` is the exact rule's (`sigma_pos`) failure on an ambiguous entry (e.g. a matrix passing the eigenvalue test that
/// is exactly singular in the LU inverse).
/// "A NaN `tr3` never reaches SureOk" is argued, not Kani-proved: `t = tr_full * tr_i` is then NaN and `NaN < T_OK` is false
/// (Kani proves `classify_branch` on symbolic inputs and `finish(NaN) = NaN`, not the float chain that produces `t`).
pub fn sigma10_one(a: &[f64; 10], cap: f64, cnt: &mut Sigma10Counts) -> Result<f64, PlaceError> {
    let [a00, a01, a02, a03, a11, a12, a13, a22, a23, a33] = *a;
    let ok = a33 > 0.0;
    let ri = 1.0 / if ok { a33 } else { 1.0 };
    let m00 = a00 - a03 * a03 * ri;
    let m01 = a01 - a03 * a13 * ri;
    let m02 = a02 - a03 * a23 * ri;
    let m11 = a11 - a13 * a13 * ri;
    let m12 = a12 - a13 * a23 * ri;
    let m22 = a22 - a23 * a23 * ri;
    let c00 = m11 * m22 - m12 * m12;
    let c11 = m00 * m22 - m02 * m02;
    let c22 = m00 * m11 - m01 * m01;
    let c01 = m02 * m12 - m01 * m22;
    let c02 = m01 * m12 - m02 * m11;
    let c12 = m01 * m02 - m00 * m12;
    let det3 = m00 * c00 + m01 * c01 + m02 * c02;
    let tr_i = a00 + a11 + a22 + a33;
    let pos = ok && det3 > 0.0 && tr_i > 0.0;
    let d3 = if pos { det3 } else { 1.0 };
    let qaq = (a03 * a03 * c00 + a13 * a13 * c11 + a23 * a23 * c22 + 2.0 * (a03 * a13 * c01 + a03 * a23 * c02 + a13 * a23 * c12)) / d3;
    let tr3 = (c00 + c11 + c22) / d3;
    let tr_full = tr3 + ri + qaq * ri * ri;
    let t = tr_full * tr_i;
    cnt.total += 1;
    match classify_branch(pos, tr_i, t) {
        Branch::SureOk => Ok(finish(tr3, cap)),
        Branch::SureBad => Ok(cap),
        Branch::Ambiguous => {
            cnt.amb += 1;
            Ok(cap_value(sigma_pos(&mat4_from_iu(a), SIGMA_RTOL)?, cap))
        }
    }
}

/// `sigma10` of `n` rows (`a` is `n*10`, row-major).
pub fn sigma10(a: &[f64], cap: f64, cnt: &mut Sigma10Counts) -> Result<Vec<f64>, PlaceError> {
    assert_eq!(a.len() % 10, 0, "sigma10: input length must be a multiple of 10");
    a.chunks_exact(10).map(|r| sigma10_one(r.try_into().expect("chunk of 10"), cap, cnt)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAP: f64 = 10.0;

    fn diag(d: [f64; 4]) -> [f64; 10] {
        [d[0], 0.0, 0.0, 0.0, d[1], 0.0, 0.0, d[2], 0.0, d[3]]
    }

    #[test]
    fn rtol_is_the_fisher_one() {
        assert_eq!(RTOL, SIGMA_RTOL);
        assert_eq!(T_OK, 5e11);
        assert_eq!(T_BAD, 3.2e13);
    }

    #[test]
    fn identity_matrix() {
        let mut c = Sigma10Counts::default();
        let s = sigma10_one(&diag([1.0; 4]), CAP, &mut c).unwrap();
        assert!((s - 3.0f64.sqrt()).abs() < 1e-15);
        assert_eq!(c, Sigma10Counts { amb: 0, total: 1 });
    }

    #[test]
    fn diag_known() {
        // sigma = sqrt(1/d0 + 1/d1 + 1/d2), independent of the yaw entry d3
        let mut c = Sigma10Counts::default();
        let s = sigma10_one(&diag([4.0, 2.0, 0.5, 7.0]), CAP, &mut c).unwrap();
        assert!((s - (0.25f64 + 0.5 + 2.0).sqrt()).abs() < 1e-14, "{s}");
        assert_eq!(c.amb, 0);
    }

    #[test]
    fn zero_matrix() {
        let mut c = Sigma10Counts::default();
        assert_eq!(sigma10_one(&[0.0; 10], CAP, &mut c).unwrap(), CAP);
        assert_eq!(c, Sigma10Counts { amb: 0, total: 1 });
    }

    #[test]
    fn rank_deficient_is_cap() {
        let mut c = Sigma10Counts::default();
        // rank 1: v v^T
        let v = [1.0, 2.0, -1.0, 0.5];
        let mut a = [0.0; 10];
        for (t, &(i, j)) in IU.iter().enumerate() {
            a[t] = v[i] * v[j];
        }
        assert_eq!(sigma10_one(&a, CAP, &mut c).unwrap(), CAP);
        // zero in the first diagonal entry
        assert_eq!(sigma10_one(&diag([0.0, 1.0, 1.0, 1.0]), CAP, &mut c).unwrap(), CAP);
        // a33 = 0
        assert_eq!(sigma10_one(&diag([1.0, 1.0, 1.0, 0.0]), CAP, &mut c).unwrap(), CAP);
        assert_eq!(c.total, 3);
    }

    #[test]
    fn nan_input_goes_exact_rule_and_caps() {
        let mut c = Sigma10Counts::default();
        let mut a = diag([1.0; 4]);
        a[1] = f64::NAN;
        assert_eq!(sigma10_one(&a, CAP, &mut c).unwrap(), CAP);
        assert_eq!(c, Sigma10Counts { amb: 1, total: 1 });
        // DELIBERATE DIFFERENCE from Python: an all-NaN row makes numpy's `eigvalsh` raise `LinAlgError` ("Eigenvalues did not
        // converge", verified with numpy 2.0.2 ), whereas the Rust exact rule returns the cap. A single NaN
        // entry gives the cap in both. Non-finite rows need non-finite tables and do not occur with finite input.
        assert_eq!(sigma10_one(&[f64::NAN; 10], CAP, &mut c).unwrap(), CAP);
        assert_eq!(c.amb, 2);
    }

    #[test]
    fn branch_table() {
        use Branch::*;
        let nan = f64::NAN;
        // (pos, trI, T) -> branch
        assert_eq!(classify_branch(true, 1.0, 1.0), SureOk);
        assert_eq!(classify_branch(true, 1.0, T_OK), Ambiguous);
        assert_eq!(classify_branch(true, 1.0, f64::from_bits(T_OK.to_bits() - 1)), SureOk);
        assert_eq!(classify_branch(true, 1.0, T_BAD), SureBad);
        assert_eq!(classify_branch(true, 1.0, f64::from_bits(T_BAD.to_bits() - 1)), Ambiguous);
        assert_eq!(classify_branch(true, 1.0, 1e12), Ambiguous);
        assert_eq!(classify_branch(true, 1.0, f64::INFINITY), SureBad);
        assert_eq!(classify_branch(true, 1.0, nan), Ambiguous);
        assert_eq!(classify_branch(false, 1.0, 1.0), Ambiguous);
        assert_eq!(classify_branch(false, 1.0, nan), Ambiguous);
        assert_eq!(classify_branch(false, 0.0, 1.0), SureBad);
        assert_eq!(classify_branch(false, -1.0, nan), SureBad);
        assert_eq!(classify_branch(false, nan, 1.0), Ambiguous);
        assert_eq!(classify_branch(true, nan, nan), Ambiguous);
        assert_eq!(classify_branch(true, nan, 1.0), SureOk);
        // pos implies trI > 0, so these two are not reachable from sigma10_one; sure_ok is tested first, as np.where does
        assert_eq!(classify_branch(true, 0.0, 1.0), SureOk);
    }

    #[test]
    fn ambiguous_calls_sigma_pos() {
        // diag(1, 1, 1, e): T = (3 + e)(3 + 1/e) ~ 3/e; e = 1e-12 gives T ~ 3e12, between T_OK and T_BAD.
        let a = diag([1.0, 1.0, 1.0, 1e-12]);
        let mut c = Sigma10Counts::default();
        let s = sigma10_one(&a, CAP, &mut c).unwrap();
        assert_eq!(c.amb, 1);
        // exact rule: lambda_min = 1e-12 > 1e-12 * 1 is false -> singular -> inf -> cap
        assert_eq!(s, CAP);
        // slightly better conditioned (e = 1e-11): T ~ 3e11 < T_OK is sure regular; 2e-12 * 3 = 1.5e12 ambiguous and regular
        let a = diag([1.0, 1.0, 1.0, 5e-12]);
        let mut c = Sigma10Counts::default();
        let s = sigma10_one(&a, CAP, &mut c).unwrap();
        assert_eq!(c.amb, 1, "T ~ 6e11 is above T_OK");
        assert!((s - 3.0f64.sqrt()).abs() < 1e-12, "{s}");
    }

    /// A matrix that passes the ambiguity test but is exactly singular in `inv` should give `Err(PlaceError::Linalg)`:
    /// No such matrix exists for finite input: the exact rule inverts only if `lambda_min > 1e-12 lambda_max`, i.e. the matrix is
    /// positive definite and well conditioned, where LU has no zero pivot; non-finite input gives all-NaN eigenvalues and the
    /// rule answers `inf` (capped). Probed with subnormal, huge, infinite and mixed-exponent matrices (all `Ok`). What is
    /// tested instead: exactly singular matrices (integer rank-1..3, exact in floating point) are answered by the cap, in the sure
    /// branch or through the exact rule, never by an error; the `?` path is typed `PlaceError::Linalg`.
    #[test]
    fn exact_singular_matrices_cap_without_error() {
        let mut c = Sigma10Counts::default();
        for v in [[1.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [3.0, 0.0, -1.0, 2.0]] {
            let mut a = [0.0; 10];
            for (t, &(i, j)) in IU.iter().enumerate() {
                a[t] = v[i] * v[j];
            }
            assert_eq!(sigma10_one(&a, CAP, &mut c).unwrap(), CAP);
        }
        // rank 2 and 3 with integer entries
        let g = [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 1.0, 3.0], [2.0, 0.0, 1.0, 1.0]];
        let mut a = [0.0; 10];
        for (t, &(i, j)) in IU.iter().enumerate() {
            a[t] = g.iter().map(|r| r[i] * r[j]).sum();
        }
        assert_eq!(sigma10_one(&a, CAP, &mut c).unwrap(), CAP);
        let e: PlaceError = erploc_linalg::LinalgError::Singular.into();
        assert!(matches!(e, PlaceError::Linalg(_)));
        assert_eq!(c.total, 4);
    }

    #[test]
    fn counts() {
        let mut c = Sigma10Counts::default();
        let mut rows = Vec::new();
        rows.extend_from_slice(&diag([1.0; 4]));
        rows.extend_from_slice(&diag([1.0, 1.0, 1.0, 5e-12])); // ambiguous
        rows.extend_from_slice(&[0.0; 10]);
        rows.extend_from_slice(&diag([1.0, 1.0, 1.0, 1e-12])); // ambiguous
        let s = sigma10(&rows, CAP, &mut c).unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!(c, Sigma10Counts { amb: 2, total: 4 });
        assert_eq!(s[2], CAP);
    }
}
