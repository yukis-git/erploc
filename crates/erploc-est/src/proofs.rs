//! Kani proof harnesses (`cargo kani -p erploc-est`). Floats are symbolic but bounded and finite.
//! Numerical accuracy is NOT claimed here; it is covered by unit, statistical and differential tests (see verify/README.md).

use crate::ba::{layout_columns, walk_obs_cells, Blk};
use crate::ba_init::{select_best, PropKey};
use crate::corners::valid_rows;
use crate::lm::{
    advance, cost, dense_normal, drive, huber_sqrt_w, schur_cam_origin, schur_split, vec_retract, LmOptions, LmState, Loss, Solver, Tick,
};
use erploc_linalg::DMat;

/// finite r and delta in [1e-6, 1e6] => the Huber sqrt weight is finite, in (0, 1], equals 1 for |r| <= delta,
/// and the Huber cost of a single residual is non-negative.
/// The delta lower bound keeps delta/|r| above the f64 underflow for every finite r.
#[kani::proof]
fn huber_weight() {
    let r: f64 = kani::any();
    let delta: f64 = kani::any();
    kani::assume(r.is_finite());
    kani::assume(delta >= 1e-6 && delta <= 1e6);
    let w = huber_sqrt_w(r, delta);
    assert!(w.is_finite() && w > 0.0 && w <= 1.0);
    if r.abs() <= delta {
        assert!(w == 1.0);
    }
    assert!(cost(&[r], Loss::Huber { delta }) >= 0.0);
}

/// inductive step of the LM loop controller. For ANY it < max_iter and any tick, the next counter is it + 1 <= max_iter
/// and a stop reports iters <= max_iter. By induction `drive` never reports iters > max_iter, for every max_iter.
#[kani::proof]
fn lm_advance_inductive() {
    let it: usize = kani::any();
    let max_iter: usize = kani::any();
    kani::assume(it < max_iter);
    let t: Tick = kani::any();
    match advance(it, t) {
        Ok(n) => assert!(n == it + 1 && n <= max_iter),
        Err(e) => assert!(e.iters <= max_iter),
    }
}

/// `drive` end to end with symbolic max_iter <= 8, symbolic ticks and injected errors. No tick after a stop, calls <=
/// max_iter, iters <= max_iter, iters == calls (calls - 1 after GradConverged), converged iff GradConverged | Converged.
/// Floats only choose which tick an attempt yields, so this covers every Jacobian, loss and solver. It does not prove
/// that user residual/jac closures return.
#[kani::proof]
#[kani::unwind(10)]
fn lm_drive_any_sequence() {
    let max_iter: usize = kani::any();
    kani::assume(max_iter <= 8);
    let mut calls = 0usize;
    let mut stopped = false;
    let mut last: Option<Tick> = None;
    let out = drive(max_iter, || {
        assert!(!stopped);
        calls += 1;
        assert!(calls <= max_iter);
        let r: Result<Tick, ()> = if kani::any() { Ok(kani::any()) } else { Err(()) };
        match r {
            Ok(t) => {
                last = Some(t);
                if advance(0, t).is_err() {
                    stopped = true;
                }
            }
            Err(()) => stopped = true,
        }
        r
    });
    match out {
        Ok(e) => {
            assert!(e.iters <= max_iter);
            match last {
                Some(Tick::GradConverged) => {
                    assert!(e.iters + 1 == calls && e.converged);
                }
                Some(Tick::Converged) => assert!(e.iters == calls && e.converged),
                Some(Tick::NonFiniteStep) => assert!(e.iters == calls && !e.converged),
                _ => assert!(e.iters == calls && !e.converged && calls == max_iter),
            }
            if last.is_none() {
                assert!(calls == 0 && max_iter == 0);
            }
        }
        Err(()) => assert!(stopped && calls >= 1),
    }
}

/// one LM pass (`LmState::new` + one `tick`) on a 1-unknown, 1-residual problem with symbolic finite
/// residuals (|.| <= 1e3) and the concrete Jacobian [[2]] returns without panic or unbounded loop, and `advance` of the
/// resulting tick from it = 0 stays within max_iter = 1. Combined with the other harnesses this covers one concrete problem end to end.
/// NARROWED: the full `levenberg_marquardt` call with `drive` (as in an earlier harness, 1746 checks, 21.9 s) exceeded the
/// 6 GB gate (12 GB) after the refactor (the drive loop wrapped around a full tick is what blows up); symbolic max_iter <= 3,
/// a symbolic Jacobian, or max_iter = 2 had already exceeded it before the refactor.
#[kani::proof]
#[kani::unwind(5)]
fn lm_float_narrow() {
    let jv: f64 = 2.0;
    let opts = LmOptions { loss: Loss::Linear, max_iter: 1, solver: Solver::Lu, ..Default::default() };
    let mut residual = |_: &Vec<f64>| {
        let v: f64 = kani::any();
        kani::assume(v.is_finite() && v.abs() <= 1e3);
        vec![v]
    };
    let mut jac = |_: &Vec<f64>| DMat::from_rows(1, 1, &[jv]);
    let mut nf = dense_normal(&mut jac, opts.loss);
    let mut st = LmState::new(&mut residual, &mut nf, vec![0.0], opts.loss);
    if let Ok(t) = st.tick(&mut residual, &mut nf, &vec_retract, &opts) {
        match advance(0, t) {
            Ok(n) => assert!(n <= opts.max_iter),
            Err(e) => assert!(e.iters <= opts.max_iter),
        }
    }
}

/// BA layout column arithmetic. Symbolic present/None pattern over 3 frames, 2 markers, camera width 4 or 6. `layout_columns` is the column arithmetic `Layout::new` uses (the BTreeMap id -> range zip stays
/// outside: a symbolic BTreeMap exceeded the 6 GB gate). Asserts no panic, all ranges inside n, and that camera ranges end
/// before the first marker range starts and marker ranges are pairwise disjoint (blocks cannot overlap).
#[kani::proof]
#[kani::unwind(5)]
fn ba_layout_assembly_indices() {
    // narrowed: symbolic Vec capacities exceeded the 6 GB gate, so the sizes are concrete (3 frames, 2 markers) and only
    // the None pattern and the width are symbolic
    let (nf, nm) = (3usize, 2usize);
    let present: [bool; 3] = kani::any();
    let present = &present[..];
    let width: usize = if kani::any() { 4 } else { 6 };
    let (cam, mk, n_cam_cols, n) = layout_columns(present, nm, width);
    let mut n_present = 0;
    for &p in present {
        if p {
            n_present += 1;
        }
    }
    assert!(cam.len() == nf && mk.len() == nm);
    assert!(n_cam_cols == n_present * width && n == n_present * width + 6 * nm);
    for f in 0..nf {
        match &cam[f] {
            Some(r) => assert!(present[f] && r.len() == width && r.end <= n_cam_cols),
            None => assert!(!present[f]),
        }
    }
    for (k, m) in mk.iter().enumerate() {
        assert!(m.len() == 6 && m.start >= n_cam_cols && m.end <= n);
        if k > 0 {
            assert!(mk[k - 1].end == m.start);
        }
    }
    let mut expect = 0;
    for f in 0..nf {
        if let Some(r) = &cam[f] {
            assert!(r.start == expect);
            expect = r.end;
        }
    }
    assert!(expect == n_cam_cols);
    assert!(mk[0].start == n_cam_cols);
}

/// `walk_obs_cells` (the index walk `scatter_obs_block`, hence `normal_equations`, runs; float-free). Symbolic camera
/// width in {4, 6}, marker width 6 and symbolic starts with the camera range before the marker range, inside a symbolic
/// system size n. Every cell is inside the system, belongs to its block (Cc in cam x cam, Cm/Mc in cam x marker and its
/// transpose, Mm in marker x marker, g entries in their own range), local indices stay inside the widths, and the number
/// of visited cells is exactly wc*wc + 2*wc*wm + wm*wm + wc + wm. Camera and marker ranges are disjoint by assumption
/// (proved for the real layout by the layout harness). Narrowed from tracking per-cell hit arrays (exceeded 6 GB).
#[kani::proof]
#[kani::unwind(8)]
fn scatter_block_indices() {
    let wc: usize = if kani::any() { 4 } else { 6 };
    let wm: usize = 6;
    let cs: usize = kani::any();
    let ms: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(n <= 1_000_000 && cs <= n && ms <= n && cs + wc <= ms && ms + wm <= n);
    let (cc, mc) = (cs..cs + wc, ms..ms + wm);
    let mut count = 0usize;
    walk_obs_cells(&cc, &mc, |blk, row, col, p, q| {
        count += 1;
        match blk {
            Blk::Gc => assert!(row < n && p < wc && row == cs + p),
            Blk::Gm => assert!(row < n && p < wm && row == ms + p),
            Blk::Cc => assert!(cc.contains(&row) && cc.contains(&col) && p < wc && q < wc),
            Blk::Cm => assert!(cc.contains(&row) && mc.contains(&col) && p < wc && q < wm),
            Blk::Mc => assert!(mc.contains(&row) && cc.contains(&col) && p < wc && q < wm),
            Blk::Mm => assert!(mc.contains(&row) && mc.contains(&col) && p < wm && q < wm),
        }
        assert!(row < n && col < n);
    });
    assert!(count == wc * wc + 2 * wc * wm + wm * wm + wc + wm);
}

/// index arithmetic of the `Solver::Schur` block slicing (`schur_split`, `schur_cam_origin`, which `schur_solve` uses):
/// for symbolic n, n_cam_cols (a multiple of cam_width) and cam_width in {4, 6}, every camera block lies inside the camera
/// columns and every marker column index n_cam_cols + j (j < nm) is below n; the blocks tile the camera columns exactly.
#[kani::proof]
fn schur_slicing_indices() {
    let n: usize = kani::any();
    let n_cam_cols: usize = kani::any();
    let cam_width: usize = if kani::any() { 4 } else { 6 };
    kani::assume(n <= 1_000_000 && n_cam_cols <= n && n_cam_cols % cam_width == 0);
    let (nc, nm) = schur_split(n, n_cam_cols, cam_width);
    assert!(nc * cam_width == n_cam_cols && n_cam_cols + nm == n);
    let f: usize = kani::any();
    kani::assume(f < nc);
    let o = schur_cam_origin(f, cam_width);
    assert!(o + cam_width <= n_cam_cols);
    assert!(f + 1 == nc || schur_cam_origin(f + 1, cam_width) == o + cam_width);
    let j: usize = kani::any();
    kani::assume(j < nm);
    assert!(n_cam_cols + j < n);
}

fn any_key() -> PropKey {
    PropKey { support: kani::any(), n_obs: kani::any(), kind: kani::any(), neg_index: kani::any() }
}

/// `PropKey` is a total order (antisymmetry, transitivity, totality), distinct (kind, index) pairs give distinct keys,
/// and `select_best` (the selection step `initialize` calls) returns the same candidate for every permutation of up to
/// 4 candidates with distinct (kind, neg_index) (a symbolic sequence of swaps reaches every permutation).
#[kani::proof]
#[kani::unwind(6)]
fn prop_key_total_order() {
    let (a, b, c) = (any_key(), any_key(), any_key());
    assert!(a.cmp(&b) == b.cmp(&a).reverse());
    assert!(a <= b || b <= a);
    if a <= b && b <= c {
        assert!(a <= c);
    }
    assert!((a == b) == (a.cmp(&b) == core::cmp::Ordering::Equal));
    if (a.kind, a.neg_index) != (b.kind, b.neg_index) {
        assert!(a != b);
    }
    // candidates in an array (a symbolic Vec exceeded the 6 GB gate); `len` of them are live
    let len: usize = kani::any();
    kani::assume(len <= 4);
    let mut v: [(PropKey, bool, i64); 4] = [
        (any_key(), kani::any(), kani::any()),
        (any_key(), kani::any(), kani::any()),
        (any_key(), kani::any(), kani::any()),
        (any_key(), kani::any(), kani::any()),
    ];
    for i in 0..len {
        for j in 0..i {
            kani::assume((v[i].0.kind, v[i].0.neg_index) != (v[j].0.kind, v[j].0.neg_index));
        }
    }
    let base = select_best(&v[..len]);
    assert!(base.is_some() == (len > 0));
    for i in 0..len {
        let j: usize = kani::any();
        kani::assume(j >= i && j < len);
        v.swap(i, j);
    }
    let perm = select_best(&v[..len]);
    assert!(base.map(|x| x.0) == perm.map(|x| x.0));
    if let Some((bk, _, _)) = base {
        assert!(v[..len].iter().all(|c| c.0 <= bk));
    }
}

/// the masked row-index map. For up to 3 observations with any validity masks, the compacted residual rows
/// `row0_o + j` (j < n_o, `row0_o` = rows of the earlier observations) are all below the total, pairwise distinct and exactly
/// fill `0..total` (injective and compact); each source row `idx[j]` lies in the 8-row layout, belongs to a valid corner, and
/// the p-th valid corner k owns exactly the entries `idx[2p] = 2k`, `idx[2p + 1] = 2k + 1` (so invalid corners contribute
/// no row and valid ones both of theirs, in corner order). This is the `valid_rows` that `row_select` (hence `resect`,
/// `bundle_adjust`, `ba_init`, `info_marker_masked`) uses. Float weights are not part of the proof.
#[kani::proof]
#[kani::unwind(26)]
fn masked_row_map() {
    let masks: [[bool; 4]; 3] = kani::any();
    let n_obs: usize = kani::any();
    kani::assume(n_obs <= 3);
    let mut seen = [false; 24];
    let mut row0 = 0usize;
    for o in 0..3 {
        if o >= n_obs {
            break;
        }
        let m = masks[o];
        let (idx, n) = valid_rows(&m);
        let n_valid = m.iter().filter(|&&v| v).count();
        assert!(n == 2 * n_valid && n <= 8);
        // each valid corner k, the p-th valid one, owns idx[2p], idx[2p + 1]
        let mut p = 0usize;
        for k in 0..4 {
            if m[k] {
                assert!(2 * p + 1 < n);
                assert!(idx[2 * p] == 2 * k && idx[2 * p + 1] == 2 * k + 1);
                p += 1;
            }
        }
        assert!(p == n_valid);
        for j in 0..n {
            assert!(idx[j] < 8 && m[idx[j] / 2]);
            let g = row0 + j;
            assert!(g < 24);
            assert!(!seen[g], "row assigned twice");
            seen[g] = true;
        }
        row0 += n;
    }
    for r in 0..24 {
        assert!(seen[r] == (r < row0));
    }
}
