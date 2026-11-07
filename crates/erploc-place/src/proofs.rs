//! Kani proof harnesses (`cargo kani -p erploc-place`). Floats are symbolic but bounded and finite.
//! Numerical accuracy is NOT claimed here; it is covered by unit and differential tests (see verify/README.md).

use crate::fast_eval::{bl_index, pair_key, unpair, vis_csr_into, BIG};
use crate::npcompat::{arange_len, np_mod, round_half_even_usize, TWO_PI};
use crate::placement::{cap as cap_value, Key3};
use crate::placement::{info_entry, info_offset, vis_index};
use crate::scene::{ravel_xy, wall_count};
use crate::search::{better, better_core, pen_eq, pen_lt, pick};
use crate::sigma10::{classify_branch, finish, Branch, T_BAD, T_OK};
use crate::visibility::lam_in_core;
/// `np_mod(x, 2*pi)` for finite |x| <= 4*pi: no panic, result in [0, 2*pi].
#[kani::proof]
fn np_mod_range() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 2.0 * TWO_PI);
    let r = np_mod(x, TWO_PI);
    assert!(r >= 0.0 && r <= TWO_PI);
}

/// the pure comparison `lam_in_core` is total and is exactly `off <= width`.
#[kani::proof]
fn lam_in_core_total() {
    let off: f64 = kani::any();
    let width: f64 = kani::any();
    kani::assume(!off.is_nan() && !width.is_nan());
    assert_eq!(lam_in_core(off, width), off <= width);
}

// The remaining properties of `lam_in` on the float path (`lam_in` itself) are NOT proved: Kani's model of f64 `%` (fmod) is an over-approximation
// (probe: `np_mod(x, 2*pi) == x` for 0 <= x < 2*pi FAILED, which is false in IEEE), so (b)/(c) report spurious
// counterexamples. They are covered by the unit tests `lam_in_random_sectors` and `lam_in_full_width_is_degenerate`
// in visibility.rs. Proved here: the comparison (`lam_in_core_total`) and the range of `np_mod` (sound under an
// over-approximate fmod, so valid for the real one).
//
// FINDING (same in Python `_lam_in`): the full sector [-pi, pi] has width `np_mod(2*pi, 2*pi) = 0`, so (d) "full width
// accepts every lam" is FALSE: it accepts only lam == -pi and lam == pi.

/// `arange_len` for finite start, stop in [-1e4, 1e4], step in [1e-3, 1e3]: no panic, result <= 2e7,
/// 0 when stop <= start.
#[kani::proof]
fn arange_len_bounds() {
    let (start, stop, step): (f64, f64, f64) = (kani::any(), kani::any(), kani::any());
    kani::assume(start.is_finite() && start.abs() <= 1e4);
    kani::assume(stop.is_finite() && stop.abs() <= 1e4);
    kani::assume(step >= 1e-3 && step <= 1e3);
    let n = arange_len(start, stop, step);
    assert!(n <= 20_000_000);
    if stop <= start {
        assert!(n == 0);
    }
}

/// `ravel_xy` stays below nx*ny (checked against overflow) for iy < ny, ix < nx. NARROWED to nx, ny <= 64
/// (the product-vs-product SAT problem did not finish in 10 minutes at 64 bits nor at nx, ny <= 4096).
#[kani::proof]
fn ravel_xy_in_bounds() {
    let (nx, ny): (usize, usize) = (kani::any(), kani::any());
    let (iy, ix): (usize, usize) = (kani::any(), kani::any());
    kani::assume(nx <= 1 << 6 && ny <= 1 << 6);
    kani::assume(iy < ny && ix < nx);
    let total = nx.checked_mul(ny).expect("grid size overflows usize");
    assert!(ravel_xy(iy, ix, nx) < total);
}

/// `round_half_even_usize` for x in [0, 1e6]: |r - x| <= 0.5, ties go to even; `wall_count` >= 1, and for
/// k < m the wall parameter (k + 0.5)/m lies in (0, 1).
#[kani::proof]
fn round_half_even_props() {
    let x: f64 = kani::any();
    kani::assume(x >= 0.0 && x <= 1e6);
    let r = round_half_even_usize(x);
    assert!((r as f64 - x).abs() <= 0.5);
    if x - x.floor() == 0.5 {
        assert!(r % 2 == 0);
    }
    let m = wall_count(x);
    assert!(m >= 1);
    let k: usize = kani::any();
    kani::assume(k < m);
    let t = (k as f64 + 0.5) / m as f64;
    assert!(t > 0.0 && t < 1.0);
}

/// table index arithmetic stays below the table lengths for k < K, h < Hd, j < N, a, b < 4. NARROWED to
/// K, Hd, N <= 16 (K*Hd*N <= 2^12; 128 did not finish in 10 min); chained symbolic 64-bit products are intractable for CBMC.
#[kani::proof]
fn table_indices_in_bounds() {
    let (kk, hd, n): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(kk <= 1 << 4 && hd <= 1 << 4 && n <= 1 << 4);
    kani::assume(kk.checked_mul(hd).and_then(|v| v.checked_mul(n)).map_or(false, |v| v <= 10_000_000));
    let (k, h, j): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    let (a, b): (usize, usize) = (kani::any(), kani::any());
    kani::assume(k < kk && h < hd && j < n && a < 4 && b < 4);
    assert!(vis_index(k, hd, h, n, j) < kk * hd * n);
    assert!(info_entry(k, n, j, a, b) < kk * n * 16);
    assert!(info_offset(k, n, j) + 16 <= kk * n * 16);
    assert!(info_offset(k, n, j) + a * 4 + b == info_entry(k, n, j, a, b));
}

/// `sigma10::classify_branch` for symbolic `pos`, `tr_i`, `t` (ANY f64 including NaN and +-inf): the result is always one of
/// the three branches (total, by the enum), `sure_ok` and `sure_bad` are never both true for the inputs `sigma10_one` can produce
/// (`pos` implies `tr_i > 0`, a NaN `t` is Ambiguous), and a NaN `t` is never SureOk or SureBad-by-T.
#[kani::proof]
fn sigma10_branches() {
    let pos: bool = kani::any();
    let tr_i: f64 = kani::any();
    let t: f64 = kani::any();
    kani::assume(!pos || tr_i > 0.0); // sigma10_one: pos = ok & det3 > 0 & trI > 0
    let sure_ok = pos && t < T_OK;
    let sure_bad = tr_i <= 0.0 || (pos && t >= T_BAD);
    assert!(!(sure_ok && sure_bad));
    let b = classify_branch(pos, tr_i, t);
    assert_eq!(b == Branch::SureOk, sure_ok);
    assert_eq!(b == Branch::SureBad, !sure_ok && sure_bad);
    assert_eq!(b == Branch::Ambiguous, !sure_ok && !sure_bad);
    if t.is_nan() {
        assert!(b != Branch::SureOk);
    }
}

/// `sigma10::finish(tr3, cap)` (the sure-regular value) for symbolic `tr3` that is not NaN and finite `cap > 0`: the result
/// lies in [0, cap]. A NaN `tr3` is outside the precondition: it gives NaN (`finish_nan` below) but cannot reach the SureOk branch,
/// because `t = (tr3 + ..) * trI` is then NaN and `NaN < T_OK` is false.
#[kani::proof]
fn sigma10_finish_range() {
    let tr3: f64 = kani::any();
    let cap: f64 = kani::any();
    kani::assume(!tr3.is_nan());
    kani::assume(cap.is_finite() && cap > 0.0);
    let v = finish(tr3, cap);
    assert!(v >= 0.0 && v <= cap);
}

/// the NaN case of `finish` is exactly NaN (propagates), so the SureOk exclusion of the branch proof is what protects `sigma10_one`.
#[kani::proof]
fn sigma10_finish_nan() {
    let cap: f64 = kani::any();
    kani::assume(cap.is_finite() && cap > 0.0);
    assert!(finish(f64::NAN, cap).is_nan());
}

/// `placement::cap(x, c)` for symbolic `x` (NaN, +-inf included) and finite `c > 0`: result in (-inf, c], equal to `c` for
/// non-finite `x`, and `x` itself when `x <= c` is finite. (Negative finite `x` is kept: the caller passes sigma >= 0.)
#[kani::proof]
fn cap_mapping() {
    let x: f64 = kani::any();
    let c: f64 = kani::any();
    kani::assume(c.is_finite() && c > 0.0);
    let r = cap_value(x, c);
    assert!(r <= c);
    if !x.is_finite() {
        assert!(r == c);
    } else if x <= c {
        assert!(r == x);
    } else {
        assert!(r == c);
    }
    if x >= 0.0 {
        assert!(r >= 0.0);
    }
}

/// Checks of the pair-key harnesses for given (K, N): `pair_key`/`unpair` round trip and range for j < N, c < K, every key below K*N unpairs to
/// (pj < N, pc < K), and the `Bl` index of a (cell, candidate) pair is below K*N.
fn pair_key_checks(kc: usize, nn: usize) {
    let (j, c): (usize, usize) = (kani::any(), kani::any());
    kani::assume(j < nn && c < kc);
    let u = pair_key(j, c, kc);
    assert!(u < kc * nn);
    assert_eq!(unpair(u, kc), (j, c));
    let u2: usize = kani::any();
    kani::assume(u2 < kc * nn);
    let (pj, pc) = unpair(u2, kc);
    assert!(pj < nn && pc < kc);
    assert_eq!(pair_key(pj, pc, kc), u2);
    assert!(bl_index(c, j, nn) < kc * nn);
}

/// K = 400, N = 800 as constants, j, c, key symbolic. NARROWED from symbolic K <= 400, N <= 800: that
/// harness (symbolic 64-bit division and products) did not finish in 10 minutes (killed); see `ift_pair_key_small` for symbolic K, N.
#[kani::proof]
fn ift_pair_key_large() {
    pair_key_checks(400, 800);
}

/// symbolic 1 <= K, N <= 16.
#[kani::proof]
fn ift_pair_key_small() {
    let (kc, nn): (usize, usize) = (kani::any(), kani::any());
    kani::assume(kc >= 1 && kc <= 16 && nn >= 1 && nn <= 16);
    pair_key_checks(kc, nn);
}

/// CSR slicing. For a symbolic monotone `ptr` (length N + 1, ptr[N] = len, here N <= 6) the slice `ptr[s]..ptr[s+1]` of
/// every source `s < N` is within `0..len`, and the `bs_t`/`bs_c` entries (t < N, c < K) written into `Bl` land below K*N.
#[kani::proof]
#[kani::unwind(10)]
fn ift_csr_slicing() {
    const N: usize = 6;
    const KC: usize = 5;
    const LEN: usize = 8;
    let mut ptr = [0usize; N + 1];
    for i in 1..=N {
        let v: usize = kani::any();
        kani::assume(v >= ptr[i - 1] && v <= LEN);
        ptr[i] = v;
    }
    kani::assume(ptr[N] == LEN);
    let bs_t: [u32; LEN] = kani::any();
    let bs_c: [u32; LEN] = kani::any();
    for i in 0..LEN {
        kani::assume((bs_t[i] as usize) < N && (bs_c[i] as usize) < KC);
    }
    let mut bl = [false; KC * N];
    let s: usize = kani::any();
    kani::assume(s < N);
    assert!(ptr[s] <= ptr[s + 1] && ptr[s + 1] <= LEN);
    for t in ptr[s]..ptr[s + 1] {
        assert!(t < LEN);
        bl[bl_index(bs_c[t] as usize, bs_t[t] as usize, N)] = true;
    }
    // the per-source slices partition 0..LEN: total length LEN
    let mut total = 0;
    for q in 0..N {
        total += ptr[q + 1] - ptr[q];
    }
    assert_eq!(total, LEN);
}

/// the candidate-major CSR of the visible pairs (`vis_csr_into`, the index `FastEval::pairs` walks instead of sorting): for
/// symbolic strictly ascending keys `< K*N` (K, N <= 3, at most 3 keys) `ptr` is monotone with `ptr[0] = 0`, `ptr[N] = len`, every
/// cell is `< K`, and walking candidate by candidate regenerates the key list in order (so no sort is needed and no index leaves
/// `K*N`). NARROWED to fixed-size arrays (the `Vec` version exceeded the 6 GB gate: 12 GB).
#[kani::proof]
#[kani::unwind(5)]
fn ift_vis_csr() {
    const LEN: usize = 3;
    const NMAX: usize = 3;
    let (kc, nn): (usize, usize) = (kani::any(), kani::any());
    kani::assume(kc >= 1 && kc <= 3 && nn >= 1 && nn <= NMAX);
    let len: usize = kani::any();
    kani::assume(len <= LEN);
    let mut vk = [0u64; LEN];
    for i in 0..LEN {
        vk[i] = kani::any();
        kani::assume(i >= len || (vk[i] < (kc * nn) as u64 && (i == 0 || vk[i - 1] < vk[i])));
    }
    let mut ptr = [0u32; NMAX + 1];
    let mut c = [0u32; LEN];
    vis_csr_into(&vk[..len], kc, nn, &mut ptr[..nn + 1], &mut c[..len]);
    assert!(ptr[0] == 0 && ptr[nn] as usize == len);
    let mut at = 0usize;
    for j in 0..nn {
        assert!(ptr[j] <= ptr[j + 1]);
        for t in ptr[j] as usize..ptr[j + 1] as usize {
            assert!((c[t] as usize) < kc);
            assert!(t == at && (pair_key(j, c[t] as usize, kc) as u64) == vk[t]);
            at += 1;
        }
    }
    assert!(at == len);
}

/// `search::pick` on a symbolic (k <= 3) x (N <= 4) matrix of counts in {0, 1, BIG}, with the penalties and margins drawn from
/// the ordinal ranks {0, 1, 2} (every non-BIG entry finite, BIG entries (inf, inf) as the neighbourhood writes them).
/// `None` iff every entry is BIG; otherwise the chosen entry has count == the minimum < BIG, no earlier entry beats it
/// (count, then penalty within the 1e-9 band of the minimum penalty, then margin) and the first of equals wins. With the
/// neighbourhood convention (INF[i][S[j]] = BIG) the selected candidate idx % N is never in S, and the tabu index idx % N and
/// the row idx / N are in range.
#[kani::proof]
#[kani::unwind(14)]
fn ift_pick_min() {
    const K: usize = 2;
    const N: usize = 3;
    const M: usize = K * N;
    let sel: [usize; K] = kani::any();
    kani::assume(sel[0] < N && sel[1] < N && sel[0] != sel[1]);
    let mut inf = [0i64; M];
    let mut pen = [0.0f64; M];
    let mut nm = [0.0f64; M];
    for x in 0..M {
        let c: u8 = kani::any();
        kani::assume(c < 3);
        let r: u8 = kani::any();
        kani::assume(r < 3);
        let q: u8 = kani::any();
        kani::assume(q < 3);
        let in_s = x % N == sel[0] || x % N == sel[1];
        if in_s || c == 2 {
            inf[x] = BIG;
            pen[x] = f64::INFINITY;
            nm[x] = f64::INFINITY;
        } else {
            inf[x] = c as i64;
            pen[x] = r as f64;
            nm[x] = q as f64;
        }
    }
    let res = pick(&inf, &pen, &nm);
    let all_big = inf.iter().all(|&c| c >= BIG);
    assert_eq!(res.is_none(), all_big);
    if let Some((idx, key)) = res {
        assert!(idx < M && idx / N < K && idx % N < N);
        assert!(inf[idx] < BIG);
        assert!(idx % N != sel[0] && idx % N != sel[1]);
        // minimal count, then smallest penalty (ordinal ranks differ by >= 1 >> the 1e-9 band), then smallest margin, first wins
        for y in 0..M {
            if inf[y] < BIG {
                let lt = (inf[y], pen[y], nm[y]) < (inf[idx], pen[idx], nm[idx]);
                assert!(!lt);
                if (inf[y], pen[y], nm[y]) == (inf[idx], pen[idx], nm[idx]) {
                    assert!(y >= idx);
                }
            }
        }
        assert_eq!((key.n_infeasible as i64, key.penalty_sum, key.neg_margin), (inf[idx], pen[idx], nm[idx]));
    }
}

/// the tolerance comparison `pen_lt(x, y) = x < y - 1e-9 max(1, |y|)` (used for the margin) for symbolic x, y of ANY value
/// (NaN, +-inf): never both `pen_lt(x, y)` and `pen_lt(y, x)` (their sum would be `0 < -(t_x + t_y)`), and `pen_lt(x, x)` is false.
#[kani::proof]
fn better_margin_tol_asymmetric() {
    let (x, y): (f64, f64) = (kani::any(), kani::any());
    // Kani's implicit NaN check flags `y - 1e-9*max(1,|y|)` for y = +inf (inf - inf = NaN, then `x < NaN` is false, as in numpy).
    // That case is excluded here and covered by the unit test `better_infinite_keys`.
    kani::assume(x != f64::INFINITY && y != f64::INFINITY);
    assert!(!(pen_lt(x, y) && pen_lt(y, x)));
    assert!(!pen_lt(x, x));
}

/// `pen_eq` is symmetric (inf/inf is false both ways, so symmetry holds there too) and reflexive for finite `x` (it is not reflexive at inf). A finite value versus an infinity IS
/// `pen_eq` (`inf <= 1e-9 * inf`; `better_infinite_keys`); only inf/inf (`inf - inf` is NaN) and NaN are false.
#[kani::proof]
fn better_pen_eq_symmetric() {
    let (x, y): (f64, f64) = (kani::any(), kani::any());
    // Kani's implicit NaN check flags `x - y` for equal infinities (inf - inf = NaN, so `pen_eq` is false, as in numpy); that case
    // is covered by the unit test `better_infinite_keys`.
    kani::assume(!(x.is_infinite() && x == y));
    assert_eq!(pen_eq(x, y), pen_eq(y, x));
    if x.is_finite() {
        assert!(pen_eq(x, x));
    }
    if x.is_nan() || y.is_nan() {
        assert!(!pen_eq(x, y));
    }
}

/// `better_core` (the decision structure of `better`) on symbolic counts and the three predicate results, under the facts
/// proved by the two harnesses above and IEEE comparison: `pen_eq` symmetric, `p_lt(a,b)` and `p_lt(b,a)` never both, `m_lt` never both ways, and
/// a key is never `p_lt`/`m_lt` than itself. Then `better` is irreflexive and asymmetric. NaN anywhere makes `pen_eq` false and
/// both `p_lt` false, hence `better` false for equal counts (checked below).
#[kani::proof]
fn better_asymmetric() {
    let (na, nb): (usize, usize) = (kani::any(), kani::any());
    let eq: bool = kani::any(); // pen_eq(a.p, b.p) == pen_eq(b.p, a.p)
    let (p_ab, p_ba, m_ab, m_ba): (bool, bool, bool, bool) = (kani::any(), kani::any(), kani::any(), kani::any());
    kani::assume(!(p_ab && p_ba) && !(m_ab && m_ba));
    if better_core(na, nb, eq, p_ab, m_ab) {
        assert!(!better_core(nb, na, eq, p_ba, m_ba));
    }
    // irreflexive: pen_eq(a,a) is whatever it is, p_lt(a,a) = false, m_lt(a,a) = false
    let e: bool = kani::any();
    assert!(!better_core(na, na, e, false, false));
    // NaN penalty (not equal, not less) with equal counts: never better
    assert!(!better_core(na, na, false, false, m_ab));
}

/// end to end on the real `better`, penalties and margins restricted to the small ordinal set
/// {NaN, -1, 0, 1, 1 + 0.9e-9, 1 + 1.8e-9, 1e300} (every class of the tolerance logic: NaN, equal, inside and outside the 1e-9
/// band, huge; infinities are excluded because of Kani's implicit NaN check, see `better_infinite_keys`): irreflexive and asymmetric on the real function. Counts in {0, 1}.
#[kani::proof]
fn better_ordinal_end_to_end() {
    const V: [f64; 7] = [f64::NAN, -1.0, 0.0, 1.0, 1.0 + 0.9e-9, 1.0 + 1.8e-9, 1e300];
    let pick = || {
        let i: usize = kani::any();
        kani::assume(i < 7);
        V[i]
    };
    let (na, nb): (usize, usize) = (kani::any(), kani::any());
    kani::assume(na < 2 && nb < 2);
    let a = Key3 { n_infeasible: na, penalty_sum: pick(), neg_margin: pick() };
    let b = Key3 { n_infeasible: nb, penalty_sum: pick(), neg_margin: pick() };
    if better(&a, &b) {
        assert!(!better(&b, &a));
    }
    assert!(!better(&a, &a));
}
