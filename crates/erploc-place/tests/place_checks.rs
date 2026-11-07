//! Release-only self-checks (Python `placement.run_self_checks`, with the IFT selection replaced by `optimize`).
use erploc_core::jacobian::{Axis, CamPose};
use erploc_core::markers::MarkerPose;
use erploc_est::fisher::{info_camera, sigma_pos, SIGMA_RTOL};
use erploc_linalg::DMat;
use erploc_place::placement::*;
use std::time::Instant;

/// Python 2026-10-09 (`pl.optimize(pl.demo_spec())`, 0.35 s): sel, survey, 5 history steps, survey score, n_skip 0.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn b1_optimize_demo() {
    let spec = demo_spec(10);
    let r = optimize(&spec, 0.005).unwrap();
    // Python picks [6, 7, 1, 2] / survey (6, 7, 2); this port picks the x -> 8-x mirror image [6, 5, 1, 0] / (6, 5, 0)
    // because key2 of the mirror candidates ties to ~1e-13 relative and last-ulp noise decides the strict `<`.
    if erploc_testkit::strict() {
        let mirror = r.sel == vec![6, 5, 1, 0];
        assert!(mirror || r.sel == vec![6, 7, 1, 2], "{:?}", r.sel);
        assert_eq!(r.survey_ids, if mirror { vec![6, 5, 0] } else { vec![6, 7, 2] });
    }
    // (another libm may settle a tie of the key on another layout: only the keys of the steps are compared there)
    assert!(r.stats.feasible.iter().all(|&f| f));
    let HistoryStep::Survey { ids, score, n_skip } = r.history.last().unwrap() else { panic!() };
    if erploc_testkit::strict() {
        assert_eq!(r.history.len(), 5);
        let want = [(1usize, 88usize, 13869.546674904634), (2, 47, 8919.43427380697), (3, 20, 409.9381944430356), (4, 0, 0.0)];
        for (st, (len, ninf, pen)) in r.history.iter().zip(want) {
            let HistoryStep::Greedy { sel, n_infeasible, penalty } = st else { panic!("{st:?}") };
            assert_eq!((sel.len(), *n_infeasible), (len, ninf));
            assert!((penalty - pen).abs() <= 1e-9 * pen.max(1.0), "{penalty} vs {pen}");
        }
        assert_eq!((ids.clone(), *n_skip), (r.survey_ids.clone(), 0));
    } else {
        // another libm can take a different (tied) path with another number of steps; the end must be feasible with a survey step
        assert!(r.history.len() >= 5);
        assert_eq!(ids, &r.survey_ids);
    }
    let tb = precompute(&spec).unwrap();
    if erploc_testkit::strict() {
        assert!((score - 0.005974825439844254).abs() <= 1e-9 * 0.006, "{score}");
        assert_eq!(brute_force(&tb, &spec, None).unwrap(), vec![1, 3, 6, 8]);
        assert_eq!(brute_force(&tb, &spec, Some(4)).unwrap(), vec![1, 3, 6, 8]);
    } else {
        // tied subsets may resolve differently: same size and the same key
        for k in [None, Some(4)] {
            let got = brute_force(&tb, &spec, k).unwrap();
            let (kg, kw) = (key2(&evaluate(&tb, &got, &spec).unwrap(), &spec), key2(&evaluate(&tb, &[1, 3, 6, 8], &spec).unwrap(), &spec));
            assert_eq!((got.len(), kg.n_infeasible), (4, kw.n_infeasible), "{got:?}");
            assert!((kg.penalty_sum - kw.penalty_sum).abs() <= 1e-7, "{got:?}");
        }
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn b1_runtime() {
    let t0 = Instant::now();
    let spec = demo_spec(10);
    let tb = precompute(&spec).unwrap();
    let r = optimize(&spec, 0.005).unwrap();
    let csv = report_csv(&r, &spec);
    assert_eq!(csv.lines().count(), 1 + r.sel.len());
    // survey triplet: 3 distinct ids from the selection and a finite score (Python check 2)
    let mut ids = r.survey_ids.clone();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 3, "survey ids {:?}", r.survey_ids);
    assert!(ids.iter().all(|m| r.sel.contains(m)));
    match r.history.last() {
        Some(HistoryStep::Survey { score, .. }) => assert!(score.is_finite(), "survey score {score}"),
        h => panic!("last history step is not Survey: {h:?}"),
    }
    // additivity and yaw invariance of the information table (Python check 1)
    let mks: Vec<MarkerPose> = markers_from_candidates(&spec.candidates);
    let mut worst = 0.0f64;
    for k in [3usize, 17, 40, 62, 85] {
        for h in 0..tb.hd {
            let vis: Vec<MarkerPose> = r.sel.iter().filter(|&&m| tb.vis_at(k, h, m)).map(|&m| mks[m]).collect();
            if vis.is_empty() {
                continue;
            }
            let cam = CamPose::euler(tb.cells[k], spec.ws.headings[h], 0.0, 0.0);
            let iref = info_camera(&cam, &vis, spec.a, &spec.cam_model, spec.sigma_px, spec.noise_model, &[Axis::Z]);
            let mut acc = [0.0; 16];
            for &m in r.sel.iter().filter(|&&m| tb.vis_at(k, h, m)) {
                for (a, x) in acc.iter_mut().zip(tb.info_at(k, m)) {
                    *a += x;
                }
            }
            let (a, b) = (sigma_pos(&iref, SIGMA_RTOL).unwrap(), sigma_pos(&DMat::from_rows(4, 4, &acc), SIGMA_RTOL).unwrap());
            if a.is_finite() && b.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
    }
    println!("additivity max |dsigma| = {worst:e}, survey ids {:?}", r.survey_ids);
    assert!(worst < 1e-9, "{worst:e}");
    let el = t0.elapsed().as_secs_f64();
    println!("flow: {el:.2} s (Python optimize 0.35 s + precompute; criterion < 60 s)");
    assert!(el < 60.0);
}

/// Performance record (reference, no timing assertion): Python reference on the same machine, best of 3
/// (evaluate best of 20): demo precompute 117.6 ms / evaluate 1.638 ms / triplet 82.3 ms; big scene (20 x 12 m workspace
/// at 0.25 m, z_cam (1.2, 1.5), 40 wall candidates on a 22 x 14 m room, 4 headings, 4 x 2 m box, 1.5 m high) tables (7424, 4, 40):
/// precompute 28796 ms / evaluate 66.683 ms / triplet 4333 ms.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn perf_record() {
    use erploc_place::scene::{candidates_from_walls, Obstacle, Workspace};
    fn best<T>(n: usize, mut f: impl FnMut() -> T) -> (f64, T) {
        let mut b = f64::INFINITY;
        let mut last = None;
        for _ in 0..n {
            let t0 = Instant::now();
            let r = f();
            b = b.min(t0.elapsed().as_secs_f64() * 1e3);
            last = Some(r);
        }
        (b, last.unwrap())
    }
    let run = |label: &str, spec: &PlacementSpec, sel: &[usize]| {
        let (tp, tb) = best(3, || precompute(spec).unwrap());
        let (te, _) = best(20, || evaluate(&tb, sel, spec).unwrap());
        let (tt, tr) = best(3, || choose_survey_triplet(&tb, spec, sel, 0.005).unwrap());
        println!(
            "PERF {label}: tables ({}, {}, {}) precompute {tp:.1} ms, evaluate {te:.3} ms, triplet {tt:.1} ms, ids {:?}",
            tb.k, tb.hd, tb.n, tr.0
        );
    };
    let d = demo_spec(10);
    run("demo", &d, &[6, 7, 1, 2]);
    let ws = Workspace {
        poly: vec![[1., 1.], [21., 1.], [21., 13.], [1., 13.]],
        spacing: 0.25,
        z_cam: vec![1.2, 1.5],
        headings: d.ws.headings.clone(),
    };
    let room = Obstacle { poly: vec![[0., 0.], [22., 0.], [22., 14.], [0., 14.]], height: 3.0 };
    let candidates = candidates_from_walls(&[room], 1.8, 1.5, 0.05, true).unwrap();
    assert_eq!(candidates.len(), 40);
    let obstacles = vec![Obstacle { poly: vec![[9., 6.], [13., 6.], [13., 8.], [9., 8.]], height: 1.5 }];
    let big = PlacementSpec { ws, obstacles, candidates, ..d };
    run("big", &big, &[0, 10, 20, 30]);
}
