//! IFT self-checks on the demo room: the evaluator self-test (`erploc.ift.selftest`, Rust's own stream and a replay of
//! Python's recorded draws), the four checks of `erploc.ift.run_self_checks`, and the statistical behaviour of the perturbed search.
use erploc_linalg::rng::Pcg64;
use erploc_place::fast_eval::IftContext;
use erploc_place::placement::{demo_spec, Key3};
use erploc_place::search::{better, run_self_checks_ift, selftest, Budget, Replay, TickClock};
use erploc_testkit::fixture;

/// Python (numpy 2.0.2, arm64) `erploc.ift.selftest(demo_spec())`:
/// `exact_fast True, 200/200 selections, err_sel 1.17e-9, 889/889 swaps (incl. 20 neighbourhoods), err_swap 1.17e-9, N 10, K 88`.
/// The errors are the margin differences of one near-singular cell (margin about 29): relative 4e-11.
#[test]
fn b1_ift_selftest_own_stream() {
    let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
    let mut rng = Pcg64::new(1);
    let s = selftest(&mut ctx, 200, 200, &mut rng, 3, 5).unwrap();
    eprintln!(
        "b1_ift_selftest_own_stream: rust exact_fast {}, selections {}/{} exact (err {:e}), swaps+neighbourhoods {}/{} exact (err {:e}; {} neighbourhood entries), K {} N {}",
        s.exact_fast, s.n_sel - s.bad_sel, s.n_sel, s.err_sel, s.n_swap - s.bad_swap, s.n_swap, s.err_swap, s.n_nbhd, s.k, s.n
    );
    eprintln!("b1_ift_selftest_own_stream: python  exact_fast True, selections 200/200 exact (err 1.17e-9), swaps+neighbourhoods 889/889 exact (err 1.17e-9), K 88 N 10");
    assert!(s.exact_fast);
    assert_eq!((s.bad_sel, s.bad_swap), (0, 0));
    assert_eq!((s.n_sel, s.n, s.k), (200, 10, 88));
    assert!(s.n_nbhd > 0 && s.n_swap > s.n_nbhd);
    // the penalties agree to 1e-9 relative; the margin of the near-singular demo cells carries ~1e-9 absolute
    assert!(s.err_sel <= 1e-6 && s.err_swap <= 1e-6, "worst errors {:e} / {:e}", s.err_sel, s.err_swap);
}

/// Python's `erploc.ift.selftest(demo_spec())` with its draws recorded (`integers`, `choice`) and replayed: the same 200 selections,
/// 200 swaps x 3 candidates and 20 neighbourhoods, so the counts are Python's (200/200, 889/889), the criterion "all equal" is
/// Python's (`bad == 0` with its 1e-6 threshold), and the error is the one the near-singular demo cell gives on Rust's own tables.
#[test]
fn b1_ift_selftest_demo() {
    let f = fixture::load("ift.txt");
    let name = "ift_selftest_demo";
    let (draws, choices, res) = (f.get("draws", name).unwrap(), f.get("choices", name).unwrap(), f.get("res", name).unwrap());
    let rows: Vec<[f64; 4]> = (0..draws.rows()).map(|i| [draws[(i, 0)], draws[(i, 1)], draws[(i, 2)], draws[(i, 3)]]).collect();
    let mut rp = Replay::new(&rows, choices.as_slice());
    let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
    let s = selftest(&mut ctx, 200, 200, &mut rp, 3, 5).unwrap();
    rp.assert_exhausted();
    let r = res.as_slice();
    // python: n_sel, bad_sel, err_sel, n_swap, bad_swap, err_swap, N, K, exact_fast
    eprintln!(
        "b1_ift_selftest_demo: rust selections {}/{} (err {:e}), swaps+neighbourhoods {}/{} (err {:e}); python {}/{} (err {:e}), {}/{} (err {:e})",
        s.n_sel - s.bad_sel, s.n_sel, s.err_sel, s.n_swap - s.bad_swap, s.n_swap, s.err_swap,
        r[0] - r[1], r[0], r[2], r[3] - r[4], r[3], r[5]
    );
    assert_eq!((s.n_sel as f64, s.bad_sel as f64, s.n_swap as f64, s.bad_swap as f64), (r[0], r[1], r[3], r[4]), "counts equal Python's");
    assert_eq!((s.bad_sel, s.bad_swap), (0, 0), "Python's criterion: all equal");
    assert_eq!((s.n, s.k, s.exact_fast), (r[6] as usize, r[7] as usize, r[8] != 0.0));
    // for the margin of the near-singular demo cell: 1e-9 relative; Python's own error is r[2], r[5]
    assert!(s.err_sel <= 1e-8 && s.err_swap <= 1e-8, "{:e} / {:e}", s.err_sel, s.err_swap);
}

/// The four checks of `erploc.ift.run_self_checks` (reference detail values: 200/200, 889/889, IFT (15, 312.58171441648005, 19.0) vs
/// brute force (15, 312.58171441647755, 19.0), minimal count 4 vs 4 with 0 infeasible cells).
#[test]
fn b1_ift_demo() {
    let lines = run_self_checks_ift().unwrap();
    for l in &lines {
        eprintln!("{} [{}] {} ({})", l.name, if l.passed { "PASS" } else { "FAIL" }, l.value, l.criterion);
    }
    assert_eq!(lines.len(), 4);
    assert!(lines.iter().all(|l| l.passed), "{lines:?}");
    assert!(lines[0].value.starts_with("200/200 ") && lines[3].value.starts_with("4 vs 4 (infeasible 0)"));
    // the recorded Python values of checks 3 and 4
    let f = fixture::load("ift.txt");
    let (r3, kb) = (f.get("key", "ift_checks_r3").unwrap().as_slice(), f.get("key", "ift_checks_kb").unwrap().as_slice());
    assert_eq!((r3[0], kb[0]), (15.0, 15.0));
    let py = |v: &[f64]| Key3 { n_infeasible: v[0] as usize, penalty_sum: v[1], neg_margin: v[2] };
    assert!(!better(&py(kb), &py(r3)), "Python's own check");
    assert_eq!(f.get("n", "ift_checks_brute").unwrap().as_slice(), &[4.0]);
    let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
    let rust = erploc_place::search::ift_det(&mut ctx, 3, &Budget::Unlimited).unwrap();
    let kb_rust = erploc_place::search::brute_key(&ctx, 3).unwrap();
    assert_eq!(rust.best_key.n_infeasible, 15);
    assert!((rust.best_key.penalty_sum - r3[1]).abs() <= 1e-9 * r3[1] && (kb_rust.penalty_sum - kb[1]).abs() <= 1e-9 * kb[1]);
    assert!((rust.best_key.neg_margin - 19.0).abs() <= 1e-9 * 19.0 && (kb_rust.neg_margin - 19.0).abs() <= 1e-9 * 19.0);
}

/// Perturbed search with Rust's own random stream (seeds 0..4, iteration budget): the key is STRICTLY better than Python's deterministic
/// `ift_det` key at k = 4 (same count and penalty 0, larger minimum margin: the perturbation phases improve on the greedy+tabu result).
#[test]
fn statistical_demo() {
    let f = fixture::load("ift.txt");
    let k4 = f.get("key", "ift_det_demo_k4").unwrap().as_slice();
    let py = Key3 { n_infeasible: k4[0] as usize, penalty_sum: k4[1], neg_margin: k4[2] };
    let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
    for seed in 0..5u64 {
        let r =
            erploc_place::search::run_ift(&mut ctx, 4, &Budget::Iterations(150), &mut TickClock::new(), &mut Pcg64::new(seed), true, 15, 7)
                .unwrap();
        eprintln!(
            "statistical_demo seed {seed}: key {:?} (python ift_det {py:?}), phases {}, iters {}, sel {:?}",
            r.best_key, r.info.phases, r.info.iters, r.best_sel
        );
        assert!(better(&r.best_key, &py), "seed {seed}: not strictly better than python's deterministic key: {py:?} vs {:?}", r.best_key);
        assert!(r.info.fast_key_ok && r.info.phases >= 2);
    }
}

/// The site self-test (the `ift_lib.selftest` generator of the Python reference) on the synthetic site table in both visibility
/// modes with Rust's own random stream: 200 selections of k = 9 / 8, 200 swaps x 3 candidates and 20 full neighbourhoods.
/// The fast evaluator must agree with the from-scratch evaluator on all of them; counts exactly, the rest within the Python self-test threshold.
#[test]
fn selftest_site_both_modes() {
    /// The 20 neighbourhoods of the Python demo self-test (the Python site self-test has none).
    const N_NBHD: usize = 20;
    use erploc_place::search::{selftest_site, SELFTEST_TOL};
    use erploc_place::site::VisMode;
    use erploc_place::site_synth::synthetic_site;
    use erploc_place::zone::Crit;
    let s = synthetic_site().unwrap();
    for mode in [VisMode::Centre, VisMode::Samples] {
        let mut ctx = IftContext::from_site(&s, mode, Crit::N);
        let t0 = std::time::Instant::now();
        let r = selftest_site(&mut ctx, 200, 200, N_NBHD, &mut Pcg64::new(1)).unwrap();
        eprintln!(
            "selftest_site {mode:?}: exact_fast {}, selections {}/{} exact (err {:e}), swaps+neighbourhood {}/{} exact (err {:e}; {} neighbourhood entries), K {} N {} ({:.1} s)",
            r.exact_fast, r.n_sel - r.bad_sel, r.n_sel, r.err_sel, r.n_swap - r.bad_swap, r.n_swap, r.err_swap, r.n_nbhd, r.k, r.n, t0.elapsed().as_secs_f64()
        );
        assert!(r.exact_fast);
        assert_eq!((r.bad_sel, r.bad_swap), (0, 0));
        assert_eq!((r.k, r.n), (92, 54));
        // the acceptance threshold of the Python self-test; the margins of cells with a nearly singular information matrix move by
        // about 1e-8 (absolute) between the two evaluation orders, the counts and the exact-rule branches agree everywhere
        assert!(r.err_sel <= SELFTEST_TOL && r.err_swap <= SELFTEST_TOL, "{:e} / {:e}", r.err_sel, r.err_swap);
    }
}
