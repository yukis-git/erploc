//! Per-corner validity masks and sigma scales of an observation. Port of `erploc/corners.py` (the parts the estimators use).
//!
//! Rules: a corner with `corner_mask[k] == false` is never used (no residual rows, `uv[k]` never read, may be NaN). The
//! 8-row layout of one marker is `(lam0, phi0, lam1, phi1, ...)`: corner k owns rows `2k` and `2k + 1`. The rows of the valid
//! corners are kept in corner order and whitened by `1 / (sigma * scale[k])`. An observation with no valid corner is absent
//! everywhere (callers filter with [`n_valid`]).
//!
//! `corner_sigma_scale` must be finite and > 0 for all four corners, masked ones included: [`row_select`] panics otherwise
//! (Python raises `ValueError`; a bad scale is a caller bug, not a data-dependent estimator failure).

use crate::synth::Observation;
use erploc_core::V2;

/// Number of valid corners (0..=4).
pub fn n_valid(ob: &Observation) -> usize {
    ob.corner_mask.iter().filter(|&&m| m).count()
}

/// Float-free row map: the source rows (in the 8-row layout) kept for `mask`, as `(idx, n)`; only `idx[..n]` is
/// meaningful (the rest is 0). Valid corner number `p` (0-based among the valid ones) contributes `idx[2p] = 2k` and
/// `idx[2p + 1] = 2k + 1`, so `n = 2 * count(valid)`, `idx[..n]` is strictly increasing and every entry is below 8.
pub fn valid_rows(mask: &[bool; 4]) -> ([usize; 8], usize) {
    let mut idx = [0usize; 8];
    let mut n = 0;
    for (k, &m) in mask.iter().enumerate() {
        if m {
            idx[n] = 2 * k;
            idx[n + 1] = 2 * k + 1;
            n += 2;
        }
    }
    (idx, n)
}

/// Rows kept for one observation and their weights `1 / scale` (Python `row_select`): `idx[..n]` indexes the 8-row layout,
/// `w[..n]` is `1 / scale[corner of that row]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowSel {
    pub idx: [usize; 8],
    pub w: [f64; 8],
    pub n: usize,
}

/// Python `row_select_ms`. Panics unless all four scales are finite and > 0.
pub fn row_select_ms(mask: &[bool; 4], scale: &[f64; 4]) -> RowSel {
    assert!(scale.iter().all(|s| s.is_finite() && *s > 0.0), "corner_sigma_scale must be finite and > 0: {scale:?}");
    let (idx, n) = valid_rows(mask);
    let mut w = [0.0; 8];
    for j in 0..n {
        w[j] = 1.0 / scale[idx[j] / 2];
    }
    RowSel { idx, w, n }
}

/// Python `row_select`.
pub fn row_select(ob: &Observation) -> RowSel {
    row_select_ms(&ob.corner_mask, &ob.corner_sigma_scale)
}

/// `uv` with the masked corners replaced by 0 (never reads a NaN or huge value of a masked corner).
pub fn safe_uv(ob: &Observation) -> [V2; 4] {
    let mut uv = ob.uv;
    for (p, &m) in uv.iter_mut().zip(&ob.corner_mask) {
        if !m {
            *p = [0.0, 0.0];
        }
    }
    uv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ob(mask: [bool; 4], scale: [f64; 4]) -> Observation {
        let mut o = Observation::new(1, [[f64::NAN, f64::NAN]; 4]);
        o.corner_mask = mask;
        o.corner_sigma_scale = scale;
        o
    }

    #[test]
    fn row_select_matches_python_self_check() {
        let o = ob([true, false, true, true], [1.0, 1.0, 2.0, 1.0]);
        let s = row_select(&o);
        assert_eq!(&s.idx[..s.n], &[0, 1, 4, 5, 6, 7]);
        assert_eq!(&s.w[..s.n], &[1.0, 1.0, 0.5, 0.5, 1.0, 1.0]);
        assert_eq!(n_valid(&o), 3);
        let all = row_select(&Observation::new(1, [[0.0; 2]; 4]));
        assert_eq!(&all.idx[..all.n], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(all.w.iter().all(|&w| w == 1.0));
        let none = row_select(&ob([false; 4], [1.0; 4]));
        assert_eq!(none.n, 0);
    }

    #[test]
    fn safe_uv_zeroes_masked_only() {
        let mut o = ob([true, false, true, false], [1.0; 4]);
        o.uv[0] = [10.0, 20.0];
        o.uv[2] = [30.0, 40.0];
        assert_eq!(safe_uv(&o), [[10.0, 20.0], [0.0, 0.0], [30.0, 40.0], [0.0, 0.0]]);
        let full = Observation::new(2, [[1.0, 2.0]; 4]);
        assert_eq!(safe_uv(&full), full.uv);
    }

    #[test]
    #[should_panic(expected = "corner_sigma_scale")]
    fn bad_scale_panics_even_when_masked() {
        let _ = row_select(&ob([true, false, true, true], [1.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "corner_sigma_scale")]
    fn nan_scale_panics() {
        let _ = row_select(&ob([true; 4], [1.0, f64::NAN, 1.0, 1.0]));
    }
}
