//! Public functions that return a `Result` give an `Err` for arguments they cannot work with; they do not panic.
use erploc_est::fisher::{sigma_pos, SIGMA_RTOL};
use erploc_linalg::{DMat, LinalgError};
use erploc_place::fast_eval::IftContext;
use erploc_place::placement::*;
use erploc_place::scene::{candidates_from_walls, cells, Obstacle, Workspace};
use erploc_place::search::{brute_key, ift_det, optimize_ift, run_ift, Budget, TickClock};

fn invalid<T: std::fmt::Debug>(r: Result<T, PlaceError>, what: &str) {
    match r {
        Err(PlaceError::InvalidInput(_)) => {}
        other => panic!("{what}: expected InvalidInput, got {other:?}"),
    }
}

fn ws() -> Workspace {
    demo_spec(10).ws
}

#[test]
fn cells_reject_bad_spacing_and_polygon() {
    for spacing in [0.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        invalid(cells(&Workspace { spacing, ..ws() }, &[]), &format!("spacing {spacing}"));
    }
    invalid(cells(&Workspace { poly: vec![], ..ws() }, &[]), "empty polygon");
    invalid(cells(&Workspace { poly: vec![[0., 0.], [1., 1.]], ..ws() }, &[]), "two vertices");
    invalid(cells(&Workspace { poly: vec![[0., 0.], [f64::NAN, 1.], [1., 0.]], ..ws() }, &[]), "NaN vertex");
    // a grid that would not fit in memory is an error, not an allocation failure
    invalid(cells(&Workspace { spacing: 1e-9, ..ws() }, &[]), "grid too large");
    invalid(cells(&Workspace { poly: vec![[0., 0.], [1e300, 0.], [1e300, 1e300]], ..ws() }, &[]), "huge polygon");
    assert!(cells(&ws(), &[]).is_ok());
}

#[test]
fn candidates_from_walls_reject_bad_spacing() {
    let room = Obstacle { poly: vec![[0., 0.], [8., 0.], [8., 6.], [0., 6.]], height: 3.0 };
    for spacing in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12] {
        invalid(candidates_from_walls(std::slice::from_ref(&room), spacing, 1.5, 0.05, true), &format!("spacing {spacing}"));
    }
    invalid(candidates_from_walls(std::slice::from_ref(&room), 1.0, f64::NAN, 0.05, true), "NaN z");
    assert_eq!(candidates_from_walls(&[room], 2.0, 1.5, 0.05, true).unwrap().len(), 14);
}

#[test]
fn optimize_rejects_empty_pieces() {
    let spec = demo_spec(10);
    let mut s = spec.clone();
    s.ws.headings.clear();
    invalid(optimize(&s, 0.005), "optimize: empty headings");
    let mut s = spec.clone();
    s.ws.z_cam.clear();
    invalid(optimize(&s, 0.005), "optimize: empty z_cam");
    let mut s = spec.clone();
    s.ws.poly.clear();
    invalid(optimize(&s, 0.005), "optimize: empty polygon");
    let mut s = spec.clone();
    s.candidates.clear();
    invalid(optimize(&s, 0.005), "optimize: no candidates");
    invalid(optimize_ift(&s, 0.005, &Budget::Unlimited), "optimize_ift: no candidates");
    let mut s = spec;
    s.ws.poly.clear();
    invalid(optimize_ift(&s, 0.005, &Budget::Unlimited), "optimize_ift: empty polygon");
    invalid(precompute(&s), "precompute: empty polygon");
}

#[test]
fn index_out_of_range_is_an_error() {
    let spec = demo_spec(10);
    let tb = precompute(&spec).unwrap();
    let n = tb.n;
    invalid(evaluate(&tb, &[0, n], &spec), "evaluate");
    invalid(evaluate(&tb, &[usize::MAX], &spec), "evaluate MAX");
    invalid(local_search_1swap(&tb, &spec, &[1, n + 3]), "local_search_1swap");
    invalid(prune(&tb, &spec, &[n]), "prune");
    assert!(evaluate(&tb, &[0, n - 1], &spec).is_ok());
}

#[test]
fn subset_size_too_large_is_an_error() {
    let spec = demo_spec(10);
    let tb = precompute(&spec).unwrap();
    invalid(brute_force(&tb, &spec, Some(tb.n + 1)), "brute_force k > N");
    let mut ctx = IftContext::from_spec(&spec).unwrap();
    let n = ctx.n;
    invalid(ift_det(&mut ctx, 0, &Budget::Unlimited), "ift_det k = 0");
    invalid(ift_det(&mut ctx, n + 1, &Budget::Unlimited), "ift_det k > N");
    invalid(brute_key(&ctx, n + 1), "brute_key k > N");
    // a perturbed run without a budget
    let mut rng = erploc_linalg::rng::Pcg64::new(1);
    invalid(run_ift(&mut ctx, 3, &Budget::Unlimited, &mut TickClock::new(), &mut rng, true, 15, 7), "perturbed run without budget");
}

#[test]
fn sigma_pos_needs_a_matrix_of_at_least_3x3() {
    for (r, c) in [(0, 0), (1, 1), (2, 2), (3, 4), (4, 3)] {
        let m = DMat::zeros(r, c);
        assert_eq!(sigma_pos(&m, SIGMA_RTOL).unwrap_err(), LinalgError::Shape, "{r}x{c}");
    }
    assert!(sigma_pos(&DMat::from_rows(3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]), SIGMA_RTOL).is_ok());
}
