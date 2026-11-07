//! Kani proof harnesses (`cargo kani -p erploc-linalg`). Floats are symbolic but bounded and finite, which keeps
//! CBMC tractable. Properties: absence of panics and out-of-range indexing, and simple structural postconditions.
//! Numerical accuracy is NOT claimed here; it is covered by unit and differential tests (see verify/README.md).

use crate::{chol, eig, lu, rng, DMat, LinalgError};

fn any_bounded() -> f64 {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e3);
    x
}

fn any_mat(r: usize, c: usize) -> DMat {
    DMat::from_fn(r, c, |_, _| any_bounded())
}

fn check_matmul_shape(r: usize, k: usize, c: usize) {
    let a = DMat::zeros(r, k);
    let b = DMat::zeros(k, c);
    let p = a.matmul(&b);
    assert!(p.rows() == r && p.cols() == c);
    let t = a.transpose();
    assert!(t.rows() == k && t.cols() == r);
}

/// matmul/transpose never panic and give the right shape. Symbolic dimensions (even <= 2) and a symbolic
/// choice among shapes both exceed the 6 GB memory gate (symbolic Vec sizes), so the bound is narrowed to
/// concrete shapes: (2,3,2) here, (0,0,0) and (3,3,3) in the two companion harnesses.
#[kani::proof]
#[kani::unwind(5)]
fn matmul_shapes() {
    check_matmul_shape(2, 3, 2);
}

#[kani::proof]
#[kani::unwind(5)]
fn matmul_shapes_empty() {
    check_matmul_shape(0, 0, 0);
}

#[kani::proof]
#[kani::unwind(5)]
fn matmul_shapes_3x3() {
    check_matmul_shape(3, 3, 3);
}

/// LU of any bounded 2×2 never panics; on success every pivot is nonzero. (`solve` on a symbolic factor
/// exceeds the 6 GB gate; its length/no-panic property is checked in `lu_2x2_solve_len` on a concrete matrix.)
#[kani::proof]
#[kani::unwind(4)]
fn lu_2x2_total() {
    let a = any_mat(2, 2);
    match lu::lu(&a) {
        Ok(f) => {
            assert!(f.pivots().iter().all(|p| *p != 0.0));
        }
        Err(e) => assert!(e == LinalgError::Singular),
    }
}

/// `solve` on a fixed 2×2 factor with a symbolic bounded right-hand side never panics and returns 2 values.
#[kani::proof]
#[kani::unwind(4)]
fn lu_2x2_solve_len() {
    let a = DMat::from_rows(2, 2, &[2.0, 1.0, 1.0, 3.0]);
    let f = lu::lu(&a).unwrap();
    let x = f.solve(&[any_bounded(), any_bounded()]);
    assert!(x.len() == 2);
}

/// Cholesky of a bounded 2×2 never panics; on success the diagonal of L is > 0.
#[kani::proof]
#[kani::unwind(4)]
fn cholesky_2x2_total() {
    let a = any_mat(2, 2);
    match chol::cholesky(&a) {
        Ok(l) => assert!(l[(0, 0)] > 0.0 && l[(1, 1)] > 0.0),
        Err(e) => assert!(e == LinalgError::NotPositiveDefinite),
    }
}

/// Jacobi eigen with a 1-sweep budget on a bounded symmetric 2×2 never panics; on success the values
/// are ascending.
#[kani::proof]
#[kani::unwind(6)]
fn eigh_2x2_sorted() {
    let a = any_bounded();
    let b = any_bounded();
    let d = any_bounded();
    let m = DMat::from_rows(2, 2, &[a, b, b, d]);
    if let Ok(e) = eig::eigh_with_sweeps(&m, 1) {
        assert!(e.values.len() == 2);
        assert!(e.values[0] <= e.values[1]);
    }
}

/// the RNG's unit conversion maps every u64 into [0, 1).
#[kani::proof]
fn unit_interval() {
    let x: u64 = kani::any();
    let u = rng::u64_to_unit(x);
    assert!((0.0..1.0).contains(&u));
}
