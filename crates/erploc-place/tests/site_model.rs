//! The site model on the synthetic site table (`erploc_place::site_synth`): file round trip, recorded counts, the fast evaluator
//! against the from-scratch evaluator, the searches and the zone statistics. No fixture file is read.
use erploc_linalg::rng::Pcg64;
use erploc_place::fast_eval::{FastEval, IftContext};
use erploc_place::occlusion::{blocked_triples, triples_checksum, OccMode};
use erploc_place::search::{ift_det, ift_min_count, Budget};
use erploc_place::site::{near_checksum, near_lists, CandKind, SiteTables, VisMode};
use erploc_place::site_synth::{synthetic_site, synthetic_site_bytes};
use erploc_place::zone::{Crit, Grid};

const MODES: [VisMode; 2] = [VisMode::Centre, VisMode::Samples];

#[test]
fn recorded_counts_equal_recomputed() {
    let s = synthetic_site().unwrap();
    let near = near_lists(&s.cands);
    assert_eq!(near, s.near);
    assert_eq!((near_checksum(&near), near.iter().map(|v| v.len() as u64).sum::<u64>()), (s.refs.fnv_near, s.refs.n_near_total));
    for mode in MODES {
        let m = s.vis[mode.index()].iter().filter(|&&b| b).count() as u64;
        assert_eq!(m, s.refs.n_vis[mode.index()]);
        let om = if mode == VisMode::Centre { OccMode::Centre } else { OccMode::Samples };
        let occ = blocked_triples(&s.cells, &s.cands, &s.kinds, s.spec0.a, &s.vis[mode.index()], om);
        assert_eq!((occ.n_triples as u64, triples_checksum(&occ)), (s.refs.n_triples[mode.index()], s.refs.fnv_triples[mode.index()]));
        // the context builds the same triples
        let mut ctx = IftContext::from_site(&s, mode, Crit::N);
        assert_eq!(ctx.occ().n_triples, occ.n_triples);
    }
    // samples is a subset of centre; stands block something in both modes
    assert!(s.refs.n_vis[1] <= s.refs.n_vis[0] && s.refs.n_triples.iter().all(|&t| t > 0));
}

#[test]
fn bytes_roundtrip_through_a_file() {
    let bytes = synthetic_site_bytes().unwrap();
    let dir = std::env::temp_dir().join(format!("erploc_site_model_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("site.bin");
    std::fs::write(&path, &bytes).unwrap();
    let a = SiteTables::load(&path).unwrap();
    let b = SiteTables::from_bytes(&bytes).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(a.i10, b.i10);
    assert_eq!(a.vis, b.vis);
    assert_eq!(a.refs, b.refs);
}

/// `set_key` / `add_keys` of the fast evaluator against `raw_key` (from scratch), random and clustered selections of sizes 1..10.
#[test]
fn fast_evaluator_equals_from_scratch() {
    let s = synthetic_site().unwrap();
    for mode in MODES {
        let mut ctx = IftContext::from_site(&s, mode, Crit::N);
        let n = ctx.n;
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let mut rng = Pcg64::new(20261016);
        for i in 0..60 {
            let size = 1 + i % 10;
            let mut sel: Vec<usize> = Vec::new();
            while sel.len() < size {
                let j = if i % 4 == 0 && !sel.is_empty() && !s.near[sel[0]].is_empty() {
                    let near = &s.near[sel[0]];
                    near[(rng.next_u64() % near.len() as u64) as usize]
                } else {
                    (rng.next_u64() % n as u64) as usize
                };
                if !sel.contains(&j) {
                    sel.push(j);
                }
            }
            let got = fe.set_key(&sel).unwrap();
            let want = ctx.raw_key(&sel, size).unwrap();
            assert_eq!(got.n_infeasible, want.n_infeasible, "{mode:?} {sel:?}: count");
            assert!((got.penalty_sum - want.penalty_sum).abs() <= 1e-9 * want.penalty_sum.abs().max(1.0), "{mode:?} {sel:?}: penalty");
            assert!((got.neg_margin - want.neg_margin).abs() <= 1e-9 * want.neg_margin.abs().max(1.0), "{mode:?} {sel:?}: margin");
            if size > 1 {
                fe.add_keys(&sel[..size - 1]).unwrap();
            }
        }
    }
}

#[test]
fn deterministic_search_is_consistent_with_from_scratch_key() {
    let s = synthetic_site().unwrap();
    for mode in MODES {
        let mut ctx = IftContext::from_site(&s, mode, Crit::N);
        let mut prev_inf = usize::MAX;
        for k in 3..=8 {
            let r = ift_det(&mut ctx, k, &Budget::Unlimited).unwrap();
            assert_eq!(r.best_sel.len(), k);
            let want = ctx.raw_key(&r.best_sel, k).unwrap();
            assert_eq!(r.best_key.n_infeasible, want.n_infeasible, "{mode:?} k {k}");
            assert!((r.best_key.penalty_sum - want.penalty_sum).abs() <= 1e-9 * want.penalty_sum.abs().max(1.0));
            assert!(r.info.fast_key_ok);
            // more markers never leave more cells infeasible (the search is greedy, so allow one cell of slack)
            assert!(r.best_key.n_infeasible <= prev_inf.saturating_add(1), "{mode:?} k {k}: {} after {prev_inf}", r.best_key.n_infeasible);
            prev_inf = r.best_key.n_infeasible;
        }
    }
}

#[test]
fn min_count_finds_a_feasible_layout() {
    let s = synthetic_site().unwrap();
    let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
    let mc = ift_min_count(&mut ctx, None, &Budget::Unlimited).unwrap();
    assert!(mc.found, "per-k keys {:?}", mc.per_k.iter().map(|p| (p.k, p.key.n_infeasible)).collect::<Vec<_>>());
    assert_eq!(mc.key.n_infeasible, 0);
    assert_eq!(mc.sel.len(), mc.k);
    // the criteria that add constraints cannot need fewer markers
    let mut r3 = ctx.with_crit(Crit::Both);
    let mc3 = ift_min_count(&mut r3, None, &Budget::Unlimited).unwrap();
    if mc3.found {
        assert!(mc3.k >= mc.k, "Both {} < N {}", mc3.k, mc.k);
    }
}

#[test]
fn zone_grid_of_the_site_cells() {
    let s = synthetic_site().unwrap();
    let grid = Grid::new(&s.cells);
    assert_eq!(grid.k, s.cells.len());
    let all = vec![true; s.cells.len()];
    let z = grid.zone(&all);
    assert_eq!(z.zone, s.cells.len());
    assert_eq!((z.outside, z.n_components), (0, 1));
    // the box leaves a hole in the cell grid, but it is not floor
    let none = vec![false; s.cells.len()];
    assert_eq!(grid.zone(&none).zone, 0);
    assert!(s.kinds.contains(&CandKind::Stand));
}
