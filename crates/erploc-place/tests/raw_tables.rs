//! Interface contract: `Tables` built from raw flat arrays (as the site loader imports tables) behaves exactly like the
//! `Tables` of `precompute`; `heading_independent`, `with_n_max`, `cap`, `penalty_of`, `tau_loo_eff` as documented.
use erploc_place::placement::*;
use erploc_place::visibility::Sector;

/// Rebuild `Tables` from plain flat arrays only (cells, vis, info), the site import path.
fn from_raw(tb: &Tables) -> Tables {
    let cells = tb.cells.clone();
    let (k, hd, n) = (tb.k, tb.hd, tb.n);
    let mut vis = Vec::with_capacity(k * hd * n);
    for kk in 0..k {
        for h in 0..hd {
            for j in 0..n {
                vis.push(tb.vis[(kk * hd + h) * n + j]);
            }
        }
    }
    let mut info = Vec::with_capacity(k * n * 16);
    for kk in 0..k {
        for j in 0..n {
            for a in 0..4 {
                for b in 0..4 {
                    info.push(tb.info[((kk * n + j) * 4 + a) * 4 + b]);
                }
            }
        }
    }
    Tables::from_raw(cells, hd, n, vis, info)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn hand_built_tables_equal_precompute() {
    let spec = demo_spec(10);
    let tb = precompute(&spec).unwrap();
    let raw = from_raw(&tb);
    assert_eq!((raw.k, raw.hd, raw.n), (88, 4, 10));
    for k in 0..tb.k {
        for j in 0..tb.n {
            assert_eq!(raw.info_at(k, j), tb.info_at(k, j));
            for h in 0..tb.hd {
                assert_eq!(raw.vis_at(k, h, j), tb.vis_at(k, h, j));
            }
        }
    }
    for sel in [vec![], vec![6], vec![6, 7, 1, 2], vec![6, 5, 1, 0], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] {
        let (a, b) = (evaluate(&tb, &sel, &spec).unwrap(), evaluate(&raw, &sel, &spec).unwrap());
        assert_eq!(bits(&a.sigma), bits(&b.sigma), "{sel:?}");
        assert_eq!(bits(&a.n_vis), bits(&b.n_vis));
        assert_eq!(bits(&a.sigma_loo), bits(&b.sigma_loo));
        assert_eq!(bits(&a.margin), bits(&b.margin));
        assert_eq!(a.feasible, b.feasible);
        let (ka, kb) = (key3(&a, &spec), key3(&b, &spec));
        assert_eq!(ka.n_infeasible, kb.n_infeasible);
        assert_eq!(ka.penalty_sum.to_bits(), kb.penalty_sum.to_bits());
        assert_eq!(ka.neg_margin.to_bits(), kb.neg_margin.to_bits());
        assert_eq!(bits(&penalty_of(&a, &spec)), bits(&penalty_of(&b, &spec)));
    }
    // the search entry points give the same answer on both
    assert_eq!(greedy(&tb, &spec).unwrap().0, greedy(&raw, &spec).unwrap().0);
}

#[test]
fn heading_independence_switch() {
    let spec = demo_spec(10);
    assert!(spec.vp.sectors.is_empty());
    assert!(precompute(&spec).unwrap().heading_independent());
    let mut s2 = demo_spec(10);
    s2.vp.sectors = vec![Sector { lam_min: -0.5, lam_max: 0.5, phi_min: -1.5, phi_max: 1.5 }];
    let tb2 = precompute(&s2).unwrap();
    assert!(!tb2.heading_independent());
    assert!(!from_raw(&tb2).heading_independent());
    // a hand-built table with one heading-dependent entry flips the switch
    let mut raw = from_raw(&precompute(&spec).unwrap());
    let i = (3 * raw.hd + 2) * raw.n + 4;
    raw.vis[i] = !raw.vis[i];
    assert!(!raw.heading_independent());
}

#[test]
fn spec_helpers() {
    let spec = demo_spec(10);
    let s = spec.with_n_max(3);
    assert_eq!(s.n_max, 3);
    assert_eq!(spec.n_max, 10);
    assert_eq!(
        (s.tau, s.tau_loo, s.k_min, s.sigma_cap, s.w_k, s.a, s.sigma_px),
        (spec.tau, spec.tau_loo, spec.k_min, spec.sigma_cap, spec.w_k, spec.a, spec.sigma_px)
    );
    assert_eq!(
        (s.candidates.len(), s.ws.spacing, s.ws.z_cam.clone(), s.ws.headings.clone()),
        (spec.candidates.len(), spec.ws.spacing, spec.ws.z_cam.clone(), spec.ws.headings.clone())
    );
    // tau_loo_eff: tau_loo if set, else tau
    assert_eq!(spec.tau_loo_eff(), 0.5);
    let mut s0 = spec.clone();
    s0.tau_loo = None;
    assert_eq!(s0.tau_loo_eff(), spec.tau);
    // cap: non-finite -> cap, then min with cap
    assert_eq!(cap(f64::NAN, 10.0), 10.0);
    assert_eq!(cap(f64::INFINITY, 10.0), 10.0);
    assert_eq!(cap(f64::NEG_INFINITY, 10.0), 10.0);
    assert_eq!(cap(25.0, 10.0), 10.0);
    assert_eq!(cap(0.25, 10.0), 0.25);
    assert_eq!(cap(-3.0, 10.0), -3.0);
    // penalty_of: max(0, sigma/tau - 1) + max(0, sigma_loo/tau_loo - 1) + w_k*max(0, k_min - n_vis)
    let st = CellStats {
        sigma: vec![0.01, 0.04],
        n_vis: vec![5.0, 1.0],
        sigma_loo: vec![0.1, 1.0],
        feasible: vec![true, false],
        margin: vec![0.5, -1.0],
    };
    let p = penalty_of(&st, &spec);
    assert_eq!(p[0], 0.0);
    assert_eq!(p[1], (0.04 / 0.02 - 1.0) + (1.0 / 0.5 - 1.0) + 1.0 * (2.0 - 1.0));
    let s_w = PlacementSpec { w_k: 3.0, ..spec.clone() };
    assert_eq!(penalty_of(&st, &s_w)[1], (0.04 / 0.02 - 1.0) + (1.0 / 0.5 - 1.0) + 3.0);
    let s_t = PlacementSpec { tau_loo: None, ..spec.clone() };
    assert_eq!(penalty_of(&st, &s_t)[1], (0.04 / 0.02 - 1.0) + (1.0 / 0.02 - 1.0) + 1.0);
}

#[test]
#[should_panic(expected = "info length")]
fn from_raw_rejects_short_info() {
    let tb = precompute(&demo_spec(10)).unwrap();
    // 10 upper-triangle entries per block instead of the full 4x4
    Tables::from_raw(tb.cells.clone(), tb.hd, tb.n, tb.vis.clone(), vec![0.0; tb.k * tb.n * 10]);
}

#[test]
#[should_panic(expected = "vis length")]
fn from_raw_rejects_wrong_hd() {
    let tb = precompute(&demo_spec(10)).unwrap();
    Tables::from_raw(tb.cells.clone(), tb.hd + 1, tb.n, tb.vis.clone(), tb.info.clone());
}
