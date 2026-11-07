//! Kani proof harnesses (`cargo kani -p erploc-core`). Floats are symbolic but bounded and finite.
//! Numerical accuracy is NOT claimed here; it is covered by unit and differential tests (see verify/README.md).

use crate::erp::{dlam, wrap_pi, ErpCamera};
use std::f64::consts::PI;

/// for finite |x| <= 1e6, `wrap_pi(x)` is finite and in [-pi, pi).
#[kani::proof]
fn wrap_pi_range() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    let y = wrap_pi(x);
    assert!(y.is_finite());
    assert!(y >= -PI && y < PI);
}

/// valid arguments never panic and the fields are stored.
#[kani::proof]
fn erp_camera_new_valid() {
    let h: usize = kani::any();
    kani::assume(h > 0 && h <= 5_000);
    let w = 2 * h;
    let ls: i32 = kani::any();
    kani::assume(ls == 1 || ls == -1);
    let c = ErpCamera::new(w, h, ls);
    assert!(c.w() == w && c.h() == h && c.lam_sign() == ls);
}

/// any invalid argument combination panics.
#[kani::proof]
#[kani::should_panic]
fn erp_camera_new_invalid() {
    let w: usize = kani::any();
    let h: usize = kani::any();
    kani::assume(w <= 10_000 && h <= 10_000); // 2 * h cannot overflow
    let ls: i32 = kani::any();
    kani::assume(!(h > 0 && w == 2 * h && (ls == 1 || ls == -1)));
    let _ = ErpCamera::new(w, h, ls);
}

/// the wrapped longitude difference of two in-range longitudes lies in [-pi, pi).
#[kani::proof]
fn residual_wrap_bound() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= -PI && a < PI && b >= -PI && b < PI);
    let d = dlam(a, b);
    assert!(d.is_finite() && d >= -PI && d < PI);
}
