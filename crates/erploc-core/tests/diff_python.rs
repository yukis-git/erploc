//! Differential tests against the Python `erploc` package (fixtures/core.txt, tools/export_core_fixtures.py).
//! Tolerance 1e-12 per entry relative to max(1, |value|) unless noted.
use erploc_core::erp::{self, ErpCamera, NoiseModel};
use erploc_core::jacobian::{self, Axis, CamPose};
use erploc_core::lie::{self, Se3};
use erploc_core::markers::{self, MarkerPose};
use erploc_core::rot;
use erploc_core::{DMat, Mat, M3, V2, V3};
use erploc_testkit::fixture::{self, Fixture};

const TOL: f64 = 1e-12;

/// Worst `|a-b| / max(1,|b|)` over entries.
fn err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0, f64::max)
}

struct Worst(f64, String);
impl Worst {
    fn new() -> Self {
        Worst(0.0, String::new())
    }
    fn check(&mut self, case: &str, a: &[f64], b: &[f64], tol: f64) {
        let e = err(a, b);
        assert!(e <= tol, "{case}: error {e:e} > {tol:e}");
        if e >= self.0 {
            *self = Worst(e, case.to_string());
        }
    }
    fn report(&self, label: &str, n: usize) {
        eprintln!("{label}: {n} cases, worst error {:e} ({})", self.0, self.1);
    }
}

fn names<'a>(f: &'a Fixture, prefix: &str) -> Vec<&'a String> {
    f.names.iter().filter(|n| n.starts_with(prefix)).collect()
}
fn m<'a>(f: &'a Fixture, tag: &str, name: &str) -> &'a DMat {
    f.get(tag, name).unwrap_or_else(|| panic!("missing {tag} {name}"))
}
fn row(d: &DMat, i: usize) -> Vec<f64> {
    (0..d.cols()).map(|j| d[(i, j)]).collect()
}
fn v2s(d: &DMat) -> Vec<V2> {
    (0..d.rows()).map(|i| [d[(i, 0)], d[(i, 1)]]).collect()
}
fn v3s(d: &DMat) -> Vec<V3> {
    (0..d.rows()).map(|i| [d[(i, 0)], d[(i, 1)], d[(i, 2)]]).collect()
}
fn flat2(v: &[V2]) -> Vec<f64> {
    v.iter().flatten().copied().collect()
}
fn flat3(v: &[V3]) -> Vec<f64> {
    v.iter().flatten().copied().collect()
}
fn m3(r: &[f64]) -> M3 {
    assert_eq!(r.len(), 9);
    Mat([[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
}
fn f9<const R: usize, const C: usize>(a: &Mat<R, C>) -> Vec<f64> {
    a.0.iter().flatten().copied().collect()
}
fn v3(r: &[f64]) -> V3 {
    [r[0], r[1], r[2]]
}
fn se3(r: &[f64]) -> Se3 {
    assert_eq!(r.len(), 16);
    Se3 { r: m3(&[r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9], r[10]]), t: [r[3], r[7], r[11]] }
}

#[test]
fn erp_matches_python() {
    let f = fixture::load("core.txt");
    let mut w = Worst::new();
    let mut n = 0;
    for tag in [1, 0] {
        let nm = format!("erp_px{tag}");
        let ls = m(&f, "ls", &nm)[(0, 0)] as i32;
        assert_eq!(ls > 0, tag == 1);
        let cam = ErpCamera::new(6528, 3264, ls);
        let uv = v2s(m(&f, "uv", &nm));
        w.check(&nm, &flat2(&erp::px_to_lamphi(&uv, &cam)), m(&f, "lp", &nm).as_slice(), TOL);
        let nm = format!("erp_lp{tag}");
        assert_eq!(m(&f, "ls", &nm)[(0, 0)] as i32, ls);
        let lp = v2s(m(&f, "lp", &nm));
        w.check(&nm, &flat2(&erp::lamphi_to_px(&lp, &cam)), m(&f, "uv", &nm).as_slice(), TOL);
        n += uv.len() + lp.len();
    }
    let lp = v2s(m(&f, "lp", "erp_brg"));
    w.check("erp_brg", &flat3(&erp::lamphi_to_bearing(&lp)), m(&f, "d", "erp_brg").as_slice(), TOL);
    let d = v3s(m(&f, "d", "erp_b2l"));
    w.check("erp_b2l", &flat2(&erp::bearing_to_lamphi(&d)), m(&f, "lp", "erp_b2l").as_slice(), TOL);
    let dl = m(&f, "d", "erp_dl");
    let jl = m(&f, "J", "erp_dl");
    for (i, d) in v3s(dl).iter().enumerate() {
        w.check(&format!("erp_dl{i}"), &f9(&Mat(erp::d_lamphi_d_bearing(d))), &row(jl, i), TOL);
    }
    n += lp.len() + d.len() + dl.rows();
    let x = m(&f, "x", "erp_wrap");
    let y = m(&f, "y", "erp_wrap");
    for j in 0..x.cols() {
        // exact agreement is expected: same floor-mod arithmetic
        assert_eq!(erp::wrap_pi(x[(0, j)]), y[(0, j)], "wrap_pi({:e})", x[(0, j)]);
        n += 1;
    }
    for (mdl, nmod, tag) in [
        ("tangent", NoiseModel::Tangent, 1),
        ("pixel", NoiseModel::Pixel, 1),
        ("tangent", NoiseModel::Tangent, 0),
        ("pixel", NoiseModel::Pixel, 0),
    ] {
        let nm = format!("erp_res_{mdl}_{tag}");
        let cam = ErpCamera::new(6528, 3264, m(&f, "ls", &nm)[(0, 0)] as i32);
        let (obs, pred) = (v2s(m(&f, "obs", &nm)), v2s(m(&f, "pred", &nm)));
        let (r, dd) = erp::residual_lamphi(&obs, &pred, &cam, nmod);
        w.check(&format!("{nm} r"), &flat2(&r), m(&f, "r", &nm).as_slice(), TOL);
        w.check(&format!("{nm} D"), &flat2(&dd), m(&f, "D", &nm).as_slice(), TOL);
        n += obs.len();
    }
    assert!(n >= 100);
    w.report("erp", n);
}

#[test]
fn rot_matches_python() {
    let f = fixture::load("core.txt");
    let mut w = Worst::new();
    let mut n = 0;
    let (ws, rs) = (m(&f, "w", "rot_exp"), m(&f, "R", "rot_exp"));
    for i in 0..ws.rows() {
        w.check(&format!("rot_exp{i}"), &f9(&rot::exp_so3(&v3(&row(ws, i)))), &row(rs, i), TOL);
        n += 1;
    }
    let (rs, ws) = (m(&f, "R", "rot_log"), m(&f, "w", "rot_log"));
    for i in 0..rs.rows() {
        let r = m3(&row(rs, i));
        let got = rot::log_so3(&r);
        let want = row(ws, i);
        let th = want.iter().map(|x| x * x).sum::<f64>().sqrt();
        if th > 3.0 {
            // near pi the sign of w is ambiguous: compare the rotations and the angle
            let e1 = rot::exp_so3(&got);
            let e2 = rot::exp_so3(&v3(&want));
            w.check(&format!("rot_log{i} exp"), &f9(&e1), &f9(&e2), TOL);
            let tg = got.iter().map(|x| x * x).sum::<f64>().sqrt();
            w.check(&format!("rot_log{i} angle"), &[tg], &[th], TOL);
        } else {
            w.check(&format!("rot_log{i}"), &got, &want, TOL);
        }
        n += 1;
    }
    let (rs, ys) = (m(&f, "R", "rot_yaw"), m(&f, "y", "rot_yaw"));
    for i in 0..rs.rows() {
        w.check(&format!("rot_yaw{i}"), &[rot::yaw_of(&m3(&row(rs, i)))], &[ys[(0, i)]], TOL);
        n += 1;
    }
    let (a, rs) = (m(&f, "ypr", "rot_zyx"), m(&f, "R", "rot_zyx"));
    for i in 0..a.rows() {
        w.check(&format!("rot_zyx{i}"), &f9(&rot::rot_zyx(a[(i, 0)], a[(i, 1)], a[(i, 2)])), &row(rs, i), TOL);
        n += 1;
    }
    assert!(n >= 80);
    w.report("rot", n);
}

#[test]
fn lie_matches_python() {
    let f = fixture::load("core.txt");
    let mut w = Worst::new();
    let mut n = 0;
    let (ws, js, jis) = (m(&f, "w", "lie_jl"), m(&f, "J", "lie_jl"), m(&f, "Ji", "lie_jl"));
    for i in 0..ws.rows() {
        let wv = v3(&row(ws, i));
        w.check(&format!("lie_jl{i}"), &f9(&lie::jl_so3(&wv)), &row(js, i), TOL);
        w.check(&format!("lie_jli{i}"), &f9(&lie::jl_inv_so3(&wv)), &row(jis, i), TOL);
        n += 1;
    }
    let (xis, ts) = (m(&f, "xi", "lie_se3"), m(&f, "T", "lie_se3"));
    for i in 0..xis.rows() {
        let xi = row(xis, i);
        let t = lie::exp_se3(&[xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]]);
        let mut got = f9(&t.r);
        got.extend_from_slice(&t.t);
        let want = row(ts, i);
        let want9: Vec<f64> = [0, 1, 2, 4, 5, 6, 8, 9, 10, 3, 7, 11].iter().map(|&k| want[k]).collect();
        w.check(&format!("lie_se3_{i}"), &got, &want9, TOL);
        n += 1;
    }
    let (ts, xs) = (m(&f, "T", "lie_log"), m(&f, "xi", "lie_log"));
    let (tad, ad) = (m(&f, "T", "lie_ad"), m(&f, "Ad", "lie_ad"));
    for i in 0..ts.rows() {
        w.check(&format!("lie_log{i}"), &lie::log_se3(&se3(&row(ts, i))), &row(xs, i), TOL);
        w.check(&format!("lie_ad{i}"), &f9(&lie::adjoint_se3(&se3(&row(tad, i)))), &row(ad, i), TOL);
        n += 1;
    }
    assert!(n >= 60);
    w.report("lie", n);
}

#[test]
fn markers_match_python() {
    let f = fixture::load("core.txt");
    let mut w = Worst::new();
    let mut n = 0;
    let (p, r, a, x) = (m(&f, "p", "mk_cw"), m(&f, "R", "mk_cw"), m(&f, "A", "mk_cw"), m(&f, "X", "mk_cw"));
    for i in 0..p.rows() {
        let got = markers::corners_world(&v3(&row(p, i)), &m3(&row(r, i)), a[(0, i)]);
        w.check(&format!("mk_cw{i}"), &flat3(&got), &row(x, i), TOL);
        n += 1;
    }
    let (r, a, dp, dw) = (m(&f, "R", "mk_dc"), m(&f, "A", "mk_dc"), m(&f, "dp", "mk_dc"), m(&f, "dw", "mk_dc"));
    for i in 0..r.rows() {
        let (gp, gw) = markers::d_corners(&m3(&row(r, i)), a[(0, i)]);
        let gp: Vec<f64> = gp.iter().flat_map(f9).collect();
        let gw: Vec<f64> = gw.iter().flat_map(f9).collect();
        w.check(&format!("mk_dc{i} dp"), &gp, &row(dp, i), TOL);
        w.check(&format!("mk_dc{i} dw"), &gw, &row(dw, i), TOL);
        n += 1;
    }
    let (p, az, r) = (m(&f, "p", "mk_up"), m(&f, "az", "mk_up"), m(&f, "R", "mk_up"));
    for i in 0..p.rows() {
        let got = markers::upright_pose(i as i64, &v3(&row(p, i)), az[(0, i)]);
        assert_eq!(got.id, i as i64);
        w.check(&format!("mk_up{i}"), &f9(&got.r), &row(r, i), TOL);
        n += 1;
    }
    let (p, t, args, r) = (m(&f, "p", "mk_fp"), m(&f, "target", "mk_fp"), m(&f, "args", "mk_fp"), m(&f, "R", "mk_fp"));
    for i in 0..p.rows() {
        let got = markers::facing_pose(0, &v3(&row(p, i)), &v3(&row(t, i)), args[(i, 0)], args[(i, 1)], args[(i, 2)]);
        w.check(&format!("mk_fp{i}"), &f9(&got.r), &row(r, i), TOL);
        n += 1;
    }
    // planar_pnp: <= 1e-11 for both (measured worst 8.7e-14; ~100x headroom for Jacobi-vs-LAPACK conditioning)
    let (mut wf, mut wn) = (Worst::new(), Worst::new());
    for noisy in [0, 1] {
        for i in 0..24 {
            let nm = format!("mk_pnp{noisy}_{i}");
            let d = v3s(m(&f, "d", &nm));
            let a = m(&f, "A", &nm)[(0, 0)];
            let (t, r) = markers::planar_pnp(&[d[0], d[1], d[2], d[3]], a).unwrap();
            let (wt, wr) = (row(m(&f, "t", &nm), 0), m(&f, "R", &nm).as_slice().to_vec());
            let (wk, tol) = if noisy == 0 { (&mut wf, 1e-11) } else { (&mut wn, 1e-11) };
            wk.check(&format!("{nm} t"), &t, &wt, tol);
            wk.check(&format!("{nm} R"), &f9(&r), &wr, tol);
            n += 1;
        }
    }
    assert!(n >= 100);
    w.report("markers (non-PnP)", n);
    wf.report("planar_pnp noise-free", 24);
    wn.report("planar_pnp noisy", 24);
}

#[test]
fn jacobian_matches_python() {
    let f = fixture::load("core.txt");
    let mut w = Worst::new();
    let mut n = 0;
    for nm in names(&f, "jac_") {
        let c = v3(&row(m(&f, "c", nm), 0));
        let cam = match (f.get("eul", nm), f.get("rm", nm)) {
            (Some(e), None) => CamPose::euler(c, e[(0, 0)], e[(0, 1)], e[(0, 2)]),
            (None, Some(r)) => CamPose::from_r(c, m3(r.as_slice())),
            _ => panic!("{nm}: need exactly one of eul / rm"),
        };
        let mk = MarkerPose { id: 0, p: v3(&row(m(&f, "mp", nm), 0)), r: m3(m(&f, "mR", nm).as_slice()) };
        let a = m(&f, "A", nm)[(0, 0)];
        let dof: Vec<Axis> = row(m(&f, "dof", nm), 0).iter().map(|&k| [Axis::X, Axis::Y, Axis::Z][k as usize]).collect();
        let x = v3s(m(&f, "X", nm));
        let (lp, dv) = jacobian::project(&cam, &x);
        w.check(&format!("{nm} lp"), &flat2(&lp), m(&f, "lp", nm).as_slice(), TOL);
        w.check(&format!("{nm} d"), &flat3(&dv), m(&f, "dv", nm).as_slice(), TOL);
        w.check(&format!("{nm} pred"), &flat2(&jacobian::predict_marker(&cam, &mk, a)), m(&f, "pred", nm).as_slice(), TOL);
        let (jc, jm) = jacobian::jac_marker(&cam, &mk, a, &dof);
        assert_eq!((jc.rows(), jc.cols()), (8, 3 + dof.len()), "{nm}");
        w.check(&format!("{nm} Jc"), jc.as_slice(), m(&f, "Jc", nm).as_slice(), TOL);
        w.check(&format!("{nm} Jm"), jm.as_slice(), m(&f, "Jm", nm).as_slice(), TOL);
        n += 1;
    }
    assert!(n >= 40);
    w.report("jacobian", n);
}
