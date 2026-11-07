//! Differential tests against the Python `erploc` package (fixtures/place.txt, tools/export_place_fixtures.py).
//! Counts, masks and selections are exact; float terms are compared within the tolerance named in each test.
//! Exact-tie selections (mirror candidates with equal penalty up to ~1e-13 relative; measured genuine ties <= 8e-13) are
//! accepted when the keys agree within the tie band `TIE` = 1e-11 (the Rust-side greedy check is exact: first strict minimum;
//! other Python-key comparisons use 1e-9, the tolerance of the other comparisons); the Rust info entries differ from numpy by ulps via the
//! Rust libm path. Every accepted tie is printed.
//! Every exported record must be read: each group ends with `finish` (records read == records exported).
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::markers::MarkerPose;
use erploc_core::{Mat, V3};
use erploc_linalg::DMat;
use erploc_place::npcompat::arange;
use erploc_place::placement::*;
use erploc_place::scene::*;
use erploc_place::visibility::*;
use erploc_testkit::fixture::{self, Fixture};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// `|a-b| / max(|b|, floor)`; equal values (including inf == inf) give 0; any other non-finite value gives NaN.
/// The floor is chosen per quantity scale: 1 for the O(1) dimensionless margins, penalties and keys; 1e-4 [m] for sigma
/// (observed values 1e-3..10, so the comparison is truly relative there).
fn rel_f(a: f64, b: f64, floor: f64) -> f64 {
    if a == b {
        0.0
    } else if !a.is_finite() || !b.is_finite() {
        f64::NAN
    } else {
        (a - b).abs() / b.abs().max(floor)
    }
}
fn rel(a: f64, b: f64) -> f64 {
    rel_f(a, b, 1.0)
}
const SIGMA_FLOOR: f64 = 1e-4;
/// Tie band for accepting an equal-key alternative (measured genuine ties: 1.7e-14 and 8e-13).
const TIE: f64 = 1e-11;

/// Worst error per quantity (`what`), with the case that produced it.
struct Worst(BTreeMap<String, (f64, String)>);
impl Worst {
    fn new() -> Self {
        Worst(BTreeMap::new())
    }
    fn check(&mut self, case: &str, what: &str, e: f64, tol: f64) {
        assert!(e <= tol, "{case} {what}: error {e:e} > {tol:e}");
        let w = self.0.entry(what.to_string()).or_insert((0.0, String::new()));
        if e >= w.0 {
            *w = (e, case.to_string());
        }
    }
    fn report(&self, label: &str, n: usize) {
        for (what, (e, case)) in &self.0 {
            eprintln!("{label}: {n} cases, {what}: worst error {e:e} ({case})");
        }
    }
}

/// Fixture wrapper that remembers which records were read.
struct Fx {
    f: Fixture,
    read: RefCell<BTreeSet<(String, String)>>,
}
impl Fx {
    fn new() -> Self {
        Fx { f: fixture::load("place.txt"), read: RefCell::new(BTreeSet::new()) }
    }
    fn m(&self, tag: &str, name: &str) -> &DMat {
        let d = self.f.get(tag, name).unwrap_or_else(|| panic!("missing {tag} {name}"));
        self.read.borrow_mut().insert((tag.to_string(), name.to_string()));
        d
    }
    fn v(&self, tag: &str, name: &str) -> Vec<f64> {
        self.m(tag, name).as_slice().to_vec()
    }
    fn s(&self, tag: &str, name: &str) -> f64 {
        let d = self.m(tag, name);
        assert_eq!((d.rows(), d.cols()), (1, 1), "{tag} {name} is not a scalar");
        d.as_slice()[0]
    }
    fn names(&self, prefix: &str) -> Vec<String> {
        self.f.names.iter().filter(|n| n.starts_with(prefix)).cloned().collect()
    }
    fn poly(&self, tag: &str, name: &str) -> Vec<[f64; 2]> {
        let d = self.m(tag, name);
        assert_eq!(d.cols(), 2);
        d.as_slice().chunks(2).map(|c| [c[0], c[1]]).collect()
    }
    fn bools(&self, tag: &str, name: &str) -> Vec<bool> {
        self.m(tag, name).as_slice().iter().map(|&x| x != 0.0).collect()
    }
    fn obstacles(&self, name: &str) -> Vec<Obstacle> {
        let d = self.m("ob", name);
        assert_eq!(d.cols(), 4);
        let mut out: Vec<Obstacle> = Vec::new();
        for r in d.as_slice().chunks(4) {
            let idx = r[0] as usize;
            if idx == out.len() {
                out.push(Obstacle { poly: Vec::new(), height: r[3] });
            }
            assert_eq!(idx + 1, out.len());
            assert_eq!(out[idx].height, r[3]);
            out[idx].poly.push([r[1], r[2]]);
        }
        out
    }
    fn sectors(&self, name: &str) -> Vec<Sector> {
        self.m("sec", name).as_slice().chunks(4).map(|c| Sector { lam_min: c[0], lam_max: c[1], phi_min: c[2], phi_max: c[3] }).collect()
    }
    /// n_read == n_records for the records whose case name starts with one of `prefixes`.
    fn finish(&self, label: &str, prefixes: &[&str]) {
        let all: Vec<_> = self.f.recs.keys().filter(|(_, n)| prefixes.iter().any(|p| n.starts_with(p))).collect();
        let read = self.read.borrow();
        let n_read = all.iter().filter(|k| read.contains(**k)).count();
        assert_eq!(n_read, all.len(), "{label}: read {n_read} of {} records", all.len());
        eprintln!("{label}: n_read == n_records == {n_read}");
    }
}

fn cam_model() -> ErpCamera {
    ErpCamera::new(6528, 3264, 1)
}
fn v3(d: &[f64]) -> V3 {
    [d[0], d[1], d[2]]
}
fn marker(r: &[f64]) -> MarkerPose {
    assert_eq!(r.len(), 13);
    MarkerPose { id: r[0] as i64, p: v3(&r[1..4]), r: Mat([[r[4], r[5], r[6]], [r[7], r[8], r[9]], [r[10], r[11], r[12]]]) }
}
fn reason_code(r: Reason) -> f64 {
    match r {
        Reason::Back => 0.0,
        Reason::Incidence => 1.0,
        Reason::Size => 2.0,
        Reason::Occluded => 3.0,
        Reason::SelfOcclusion => 4.0,
        Reason::Ok => 5.0,
    }
}

// ---------------------------------------------------------------- sc_
#[test]
fn sc_pip() {
    let fx = Fx::new();
    let names = fx.names("sc_pip_");
    assert_eq!(names.len(), 4);
    let mut npts = 0;
    for n in &names {
        let poly = fx.poly("poly", n);
        let pts = fx.poly("pts", n);
        let want = fx.bools("mask", n);
        assert_eq!(point_in_polygon(&pts, &poly), want, "{n}");
        npts += pts.len();
    }
    eprintln!("sc_pip: {} polygons, {npts} points, masks exact", names.len());
    fx.finish("sc_pip", &["sc_pip_"]);
}

#[test]
fn sc_area() {
    let fx = Fx::new();
    let names = fx.names("sc_area_");
    for n in &names {
        let poly = fx.poly("poly", n);
        let want = fx.s("area", n);
        assert_eq!(polygon_signed_area(&poly).to_bits(), want.to_bits(), "{n}: {} vs {want}", polygon_signed_area(&poly));
    }
    eprintln!("sc_area: {} polygons (incl. 300-vertex star), bit-identical", names.len());
    fx.finish("sc_area", &["sc_area_"]);
}

#[test]
fn sc_seg() {
    let fx = Fx::new();
    let names = fx.names("sc_seg_");
    let mut w = Worst::new();
    let (mut nseg, mut nint) = (0, 0);
    for n in &names {
        let poly = fx.poly("poly", n);
        let segs = fx.m("segs", n);
        let cnt = fx.v("nint", n);
        let tt = fx.m("tt", n);
        assert_eq!(cnt.len(), segs.rows());
        let mut at = 0;
        for (i, s) in segs.as_slice().chunks(4).enumerate() {
            let iv = segment_inside_intervals([s[0], s[1]], [s[2], s[3]], &poly);
            assert_eq!(iv.len() as f64, cnt[i], "{n} segment {i}: interval count");
            for (t0, t1) in iv {
                let r = &tt.as_slice()[at * 2..at * 2 + 2];
                w.check(n, "interval t", (t0 - r[0]).abs().max((t1 - r[1]).abs()), 1e-15);
                at += 1;
                nint += 1;
            }
        }
        assert_eq!(at, tt.rows(), "{n}: interval total");
        nseg += segs.rows();
    }
    w.report("sc_seg", names.len());
    eprintln!("sc_seg: {nseg} segments, {nint} intervals, counts exact");
    fx.finish("sc_seg", &["sc_seg_"]);
}

#[test]
fn sc_arange() {
    let fx = Fx::new();
    let names = fx.names("sc_arange_");
    for n in &names {
        let a = fx.v("args", n);
        let want = fx.v("vals", n);
        let got = arange(a[0], a[1], a[2]);
        assert_eq!(got.len(), want.len(), "{n} {a:?}: length");
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), wv.to_bits(), "{n} {a:?}[{i}]: {g:e} vs {wv:e}");
        }
    }
    eprintln!("sc_arange: {} cases bit-identical", names.len());
    fx.finish("sc_arange", &["sc_arange_"]);
}

#[test]
fn sc_cells() {
    let fx = Fx::new();
    let names = fx.names("sc_cells_");
    let mut total = 0;
    for n in &names {
        let ws = Workspace { poly: fx.poly("wspoly", n), spacing: fx.s("par", n), z_cam: fx.v("zc", n), headings: vec![0.0] };
        let obs = fx.obstacles(n);
        let want = fx.poly("cells", n);
        let got = cells(&ws, &obs).unwrap();
        assert_eq!(got.len(), want.len(), "{n}: K");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g[0].to_bits() == w[0].to_bits() && g[1].to_bits() == w[1].to_bits(), "{n} cell {i}: {g:?} vs {w:?}");
        }
        total += got.len();
    }
    eprintln!("sc_cells: {} workspaces, {total} cells, exact and in order", names.len());
    fx.finish("sc_cells", &["sc_cells_"]);
}

#[test]
fn sc_cands() {
    let fx = Fx::new();
    let names = fx.names("sc_cands_");
    let mut w = Worst::new();
    let mut total = 0;
    // coordinates pass through sin/cos: one ulp of libm moves them by about 1e-15
    let tol = if erploc_testkit::strict() { 1e-15 } else { 1e-12 };
    for n in &names {
        let poly = fx.poly("poly", n);
        let p = fx.v("par", n);
        let want = fx.m("out", n);
        // an edge whose length over the spacing is a half integer (4.55 / 0.7 = 6.5) sits on a rounding tie: one ulp in the length (another
        // libm's hypot) moves the candidate count, so such a case is compared only on the reference platform
        let on_tie = (0..poly.len()).any(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            let x = (b[0] - a[0]).hypot(b[1] - a[1]) / p[0];
            (x - x.floor() - 0.5).abs() < 1e-9
        });
        let got = candidates_from_walls(&[Obstacle { poly, height: 3.0 }], p[0], p[1], p[2], p[3] != 0.0).unwrap();
        if on_tie && !erploc_testkit::strict() {
            continue;
        }
        assert_eq!(got.len(), want.rows(), "{n}: count");
        for (c, r) in got.iter().zip(want.as_slice().chunks(4)) {
            w.check(n, "xy", (c.xy[0] - r[0]).abs().max((c.xy[1] - r[1]).abs()), tol);
            w.check(n, "z", (c.z - r[2]).abs(), tol);
            w.check(n, "azimuth", (c.azimuth - r[3]).abs(), tol);
        }
        total += got.len();
    }
    w.report("sc_cands", names.len());
    eprintln!("sc_cands: {total} candidates, counts exact");
    fx.finish("sc_cands", &["sc_cands_"]);
}

// ---------------------------------------------------------------- vi_
#[test]
fn vi_single() {
    let fx = Fx::new();
    let names = fx.names("vi_single_");
    let cam = cam_model();
    let mut n_tri = 0;
    let mut hist = [0usize; 6];
    for n in &names {
        let vpr = fx.v("vp", n);
        let vp = VisParams { psi_max: vpr[0], px_min: vpr[1], sectors: fx.sectors(n), eps: vpr[2] };
        let a = vpr[3];
        let obs = fx.obstacles(n);
        let tri = fx.m("tri", n);
        let out = fx.m("out", n);
        let border = fx.v("border", n);
        for (i, r) in tri.as_slice().chunks(16).enumerate() {
            let c = v3(&r[0..3]);
            let mk = MarkerPose { id: 0, p: v3(&r[4..7]), r: Mat([[r[7], r[8], r[9]], [r[10], r[11], r[12]], [r[13], r[14], r[15]]]) };
            let got = [
                visible(&c, r[3], &mk, a, &cam, &obs, &vp, true, FRONT_GAP_WALL).1,
                visible(&c, r[3], &mk, a, &cam, &obs, &vp, true, FRONT_GAP_STAND).1,
                visible(&c, r[3], &mk, a, &cam, &obs, &vp, false, FRONT_GAP_WALL).1,
            ];
            for (k, g) in got.iter().enumerate() {
                assert_eq!(reason_code(*g), out[(i, k)], "{n} triple {i} mode {k}: {g:?}");
                hist[out[(i, k)] as usize] += 1;
            }
            n_tri += 1;
        }
        eprintln!(
            "{n}: smallest |psi-psi_max| {:.3e}, |px-px_min| {:.3e}, interval-end z margin to obstacle height {:.3e}",
            border[0], border[1], border[2]
        );
    }
    eprintln!("vi_single: {n_tri} triples x 3 modes exact; reason histogram (back,inc,size,occl,self,ok) {hist:?}");
    fx.finish("vi_single", &["vi_single_"]);
}

#[test]
fn vi_batch_geom() {
    let fx = Fx::new();
    let names = fx.names("vi_batch_");
    assert_eq!(names.len(), 3);
    let cam = cam_model();
    let mut w = Worst::new();
    for n in &names {
        let vpr = fx.v("vp", n);
        let vp = VisParams { psi_max: vpr[0], px_min: vpr[1], sectors: fx.sectors(n), eps: vpr[2] };
        let a = vpr[3];
        let obs = fx.obstacles(n);
        let c: Vec<V3> = fx.m("C", n).as_slice().chunks(3).map(v3).collect();
        let hd = fx.v("hd", n);
        let hp = fx.v("hp", n);
        let mks: Vec<MarkerPose> = fx.m("mk", n).as_slice().chunks(13).map(marker).collect();
        let gaps = fx.v("gaps", n);
        let want = |tag: &str| fx.bools(tag, n);
        let ge = Some(gaps.as_slice());
        assert_eq!(visible_batch(&c, &hd, &mks, a, &cam, &obs, &vp, true, ge), want("vbs"), "{n} visible_batch samples");
        assert_eq!(visible_batch(&c, &hd, &mks, a, &cam, &obs, &vp, false, ge), want("vbn"), "{n} visible_batch centre");
        assert_eq!(visible_batch(&c, &hd, &mks, a, &cam, &obs, &vp, true, None), want("vbd"), "{n} visible_batch default gap");
        assert_eq!(visible_pairs(&c, &hp, &mks, a, &cam, &obs, &vp, true, ge), want("vps"), "{n} visible_pairs samples");
        assert_eq!(visible_pairs(&c, &hp, &mks, a, &cam, &obs, &vp, false, ge), want("vpn"), "{n} visible_pairs centre");
        let g = n.replace("vi_batch_", "vi_geom_");
        for (tag, samples) in [("s", true), ("n", false)] {
            let geom = geom_mask(&c, &mks, a, &cam, &obs, &vp, samples, ge);
            assert_eq!(geom.base, fx.bools(&format!("base{tag}"), &g), "{g} base {tag}");
            let dw = fx.v(&format!("dw{tag}"), &g);
            let got: Vec<f64> = geom.d_world.iter().flatten().copied().collect();
            assert_eq!(got.len(), dw.len());
            for (x, y) in got.iter().zip(&dw) {
                w.check(&g, "d_world abs", (x - y).abs(), 1e-15);
            }
        }
        eprintln!("{n}: K={} N={} Hd={} masks exact", c.len(), mks.len(), hd.len());
    }
    w.report("vi_geom", names.len());
    fx.finish("vi_batch_geom", &["vi_batch_", "vi_geom_"]);
}

// ---------------------------------------------------------------- pl_
fn load_spec(fx: &Fx, id: &str) -> PlacementSpec {
    let n = format!("pl_spec_{id}");
    let sc = fx.v("sc", &n);
    assert_eq!(sc.len(), 15);
    let cand = fx.m("cand", &n);
    PlacementSpec {
        ws: Workspace { poly: fx.poly("wspoly", &n), spacing: fx.s("ws", &n), z_cam: fx.v("zc", &n), headings: fx.v("hd", &n) },
        obstacles: fx.obstacles(&n),
        candidates: cand.as_slice().chunks(4).map(|c| Candidate { xy: [c[0], c[1]], z: c[2], azimuth: c[3] }).collect(),
        n_max: sc[0] as usize,
        a: sc[1],
        cam_model: ErpCamera::new(sc[2] as usize, sc[3] as usize, sc[4] as i32),
        sigma_px: sc[5],
        noise_model: if sc[6] == 0.0 { NoiseModel::Tangent } else { NoiseModel::Pixel },
        vp: VisParams { psi_max: sc[7], px_min: sc[8], sectors: fx.sectors(&n), eps: sc[9] },
        tau: sc[10],
        k_min: sc[11] as usize,
        tau_loo: if sc[12].is_nan() { None } else { Some(sc[12]) },
        sigma_cap: sc[13],
        w_k: sc[14],
    }
}

const SPEC_IDS: [&str; 7] = ["demo", "demo6", "tau2", "inf", "demo7", "demo7inf", "s2"];

#[test]
fn pl_spec() {
    let fx = Fx::new();
    for id in SPEC_IDS {
        let s = load_spec(&fx, id);
        assert!(!s.candidates.is_empty() && s.tau > 0.0, "{id}");
    }
    // The demo spec equals Rust's `demo_spec` field by field.
    let (a, b) = (load_spec(&fx, "demo"), demo_spec(10));
    assert_eq!(a.candidates.len(), b.candidates.len());
    for (x, y) in a.candidates.iter().zip(&b.candidates) {
        assert!((x.xy[0] - y.xy[0]).abs() <= 1e-15 && (x.xy[1] - y.xy[1]).abs() <= 1e-15, "{x:?} {y:?}");
        assert!(x.z == y.z && (x.azimuth - y.azimuth).abs() <= 1e-15);
    }
    assert_eq!((a.n_max, a.a, a.sigma_px, a.tau, a.k_min, a.tau_loo, a.w_k), (b.n_max, b.a, b.sigma_px, b.tau, b.k_min, b.tau_loo, b.w_k));
    assert_eq!(a.ws.poly, b.ws.poly);
    assert_eq!(a.ws.headings, b.ws.headings);
    assert_eq!(a.vp.psi_max, b.vp.psi_max);
    fx.finish("pl_spec", &["pl_spec_"]);
}

fn keep_going(tb: Result<Tables, PlaceError>) -> Tables {
    tb.expect("precompute")
}

#[test]
fn pl_tables() {
    let fx = Fx::new();
    let mut w = Worst::new();
    for id in ["demo", "s2"] {
        let n = format!("pl_tables_{id}");
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        let want_cells = fx.m("cells", &n);
        assert_eq!(tb.cells.len(), want_cells.rows(), "{n} K");
        for (c, r) in tb.cells.iter().zip(want_cells.as_slice().chunks(3)) {
            assert!(c.iter().zip(r).all(|(x, y)| x.to_bits() == y.to_bits()), "{n} cell {c:?} vs {r:?}");
        }
        assert_eq!(tb.vis, fx.bools("vis", &n), "{n} vis");
        let want_i = fx.v("I", &n);
        assert_eq!(tb.info.len(), want_i.len());
        // elementwise |a-b|/max(1,|b|) (the metric used throughout) and the same error scaled by the 4x4 block's largest entry
        // (entries that are exactly 0 in exact arithmetic carry cancellation noise of ~1e-16 of the block scale)
        let (mut e_el, mut e_blk) = (0.0f64, 0.0f64);
        for (bt, bw) in tb.info.chunks(16).zip(want_i.chunks(16)) {
            let scale = bw.iter().fold(1.0f64, |m, y| m.max(y.abs()));
            for (x, y) in bt.iter().zip(bw) {
                let d = (x - y).abs();
                e_el = if d.is_nan() || e_el.is_nan() { f64::NAN } else { e_el.max(d / y.abs().max(1.0)) };
                e_blk = if d.is_nan() || e_blk.is_nan() { f64::NAN } else { e_blk.max(d / scale) };
            }
        }
        w.check(&n, "I elementwise rel (zero entries carry cancellation noise)", e_el, 1e-6);
        w.check(&n, "I rel to block max", e_blk, 1e-12);
        // samples = false table: visible_batch directly (precompute has no samples switch, as in Python)
        let mks = markers_from_candidates(&spec.candidates);
        let c3: Vec<V3> = tb.cells.clone();
        let vn = visible_batch(&c3, &spec.ws.headings, &mks, spec.a, &spec.cam_model, &spec.obstacles, &spec.vp, false, None);
        assert_eq!(vn, fx.bools("visn", &n), "{n} vis samples=false");
        eprintln!("{n}: K={} Hd={} N={} cells/vis exact", tb.k, tb.hd, tb.n);
    }
    w.report("pl_tables", 2);
    fx.finish("pl_tables", &["pl_tables_"]);
}

/// Feasible cells with n_vis < 3 seen by `check_eval` (demo has k_min = 2): compared at the strict bound.
static FEAS_LOW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Compares Rust `evaluate`/`penalty_of`/keys on every recorded selection of `case`. Returns the number of selections.
fn check_eval(fx: &Fx, case: &str, tb: &Tables, spec: &PlacementSpec, w: &mut Worst) -> usize {
    let sels = fx.m("sel", case);
    let (ns, wd) = (sels.rows(), sels.cols());
    let cols: BTreeMap<&str, Vec<f64>> = ["sigma", "nvis", "sloo", "feas", "margin", "pen"].iter().map(|t| (*t, fx.v(t, case))).collect();
    let (k2, k3, gp, ninf) = (fx.v("key2", case), fx.v("key3", case), fx.v("gap", case), fx.v("ninf", case));
    let kk = tb.k;
    for i in 0..ns {
        let sel: Vec<usize> = sels.as_slice()[i * wd..(i + 1) * wd].iter().filter(|&&x| x >= 0.0).map(|&x| x as usize).collect();
        let st = evaluate(tb, &sel, spec).expect("evaluate");
        let name = format!("{case}[{i}] sel={sel:?}");
        let slice = |t: &str| cols[t][i * kk..(i + 1) * kk].to_vec();
        let (want_feas, want_nvis) = (slice("feas"), slice("nvis"));
        for c in 0..kk {
            if (st.feasible[c] as u8 as f64) != want_feas[c] {
                let tl = spec.tau_loo_eff();
                let gap_s = (st.sigma[c] - spec.tau).abs() / spec.tau;
                let gap_l = (st.sigma_loo[c] - tl).abs() / tl;
                panic!(
                    "{name} cell {c}: feasible {} vs {} (rust gaps |sigma-tau|/tau {gap_s:e}, loo {gap_l:e}; python min gaps {:e} {:e})",
                    st.feasible[c],
                    want_feas[c],
                    gp[i * 2],
                    gp[i * 2 + 1]
                );
            }
            assert_eq!(st.n_vis[c], want_nvis[c], "{name} n_vis cell {c}");
        }
        let n_inf = st.feasible.iter().filter(|&&f| !f).count();
        assert_eq!(n_inf as f64, ninf[i], "{name} n_infeasible");
        // Cells with few visible markers have a (nearly) rank-deficient information matrix: sigma_pos is then fixed by
        // the roundoff of the null directions (the Rust libm path differs from numpy by ulps there), so 1e-9 is not attainable.
        // Strict 1e-9 for cells with n_vis >= 3 (sigma: >= 2; sigma_loo is a leave-one-out), 1e-7 below, but ONLY for
        // cells that are infeasible (asserted via the exact feasible flags); a feasible low-n_vis cell is moved to strict.
        let low3: Vec<bool> = want_nvis.iter().zip(&want_feas).map(|(&v, &f)| v < 3.0 && f == 0.0).collect();
        let low2: Vec<bool> = want_nvis.iter().zip(&want_feas).map(|(&v, &f)| v < 2.0 && f == 0.0).collect();
        let feas_low = want_nvis.iter().zip(&want_feas).filter(|(&v, &f)| v < 3.0 && f != 0.0).count();
        FEAS_LOW.fetch_add(feas_low, std::sync::atomic::Ordering::Relaxed);
        let chk = |w: &mut Worst, what: &str, a: &[f64], b: &[f64], loose: &[bool], floor: f64| {
            let (mut s_, mut r_) = (0.0f64, 0.0f64);
            for c in 0..kk {
                let e = rel_f(a[c], b[c], floor);
                let t = if loose[c] { &mut r_ } else { &mut s_ };
                *t = if e.is_nan() || t.is_nan() { f64::NAN } else { t.max(e) };
            }
            w.check(case, &format!("{what} rel (strict cells)"), s_, 1e-9);
            w.check(case, &format!("{what} rel (rank-deficient infeasible cells)"), r_, 1e-7);
        };
        let (py_sigma, py_sloo, py_margin, py_pen) = (slice("sigma"), slice("sloo"), slice("margin"), slice("pen"));
        chk(w, "sigma", &st.sigma, &py_sigma, &low2, SIGMA_FLOOR);
        chk(w, "sigma_loo", &st.sigma_loo, &py_sloo, &low3, SIGMA_FLOOR);
        chk(w, "margin", &st.margin, &py_margin, &low3, 1.0);
        let rpen = penalty_of(&st, spec);
        chk(w, "penalty_of", &rpen, &py_pen, &low3, 1.0);
        let (a2, a3) = (key2(&st, spec), key3(&st, spec));
        assert_eq!(a2.n_infeasible as f64, k2[i * 2], "{name} key2 count");
        assert_eq!(a3.n_infeasible as f64, k3[i * 3], "{name} key3 count");
        // penalty_sum: 1e-9 on the strict part, 1e-7 only on the terms that come from loose cells
        let allow: f64 =
            1e-9 * k2[i * 2 + 1].abs().max(1.0) + (0..kk).filter(|&c| low3[c]).map(|c| 1e-7 * py_pen[c].abs().max(1.0)).sum::<f64>();
        let d = (a2.penalty_sum - k2[i * 2 + 1]).abs();
        assert!(d <= allow, "{name} key2 penalty_sum: |diff| {d:e} > allowance {allow:e}");
        w.check(case, "key2 penalty_sum |diff| / allowance", d / allow, 1.0);
        let d3 = (a3.penalty_sum - k3[i * 3 + 1]).abs();
        assert!(d3 <= allow, "{name} key3 penalty_sum: |diff| {d3:e} > allowance {allow:e}");
        // neg_margin = -min(margin): loose tolerance only if a loose cell is within 1e-6 of the minimum
        let mn = py_margin.iter().cloned().fold(f64::INFINITY, f64::min);
        let loose_min = (0..kk).any(|c| low3[c] && py_margin[c] <= mn + 1e-6 * mn.abs().max(1.0));
        let tol_m = if loose_min { 1e-7 } else { 1e-9 };
        w.check(
            case,
            &format!("key3 neg_margin rel ({})", if loose_min { "min from a loose cell" } else { "strict" }),
            rel(a3.neg_margin, k3[i * 3 + 2]),
            tol_m,
        );
    }
    ns
}

#[test]
fn pl_eval() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut n = 0;
    for id in ["demo", "s2"] {
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        n += check_eval(&fx, &format!("pl_eval_{id}"), &tb, &spec, &mut w);
    }
    w.report("pl_eval", n);
    eprintln!("pl_eval: {n} selections; feasible/n_vis/infeasible counts exact");
    eprintln!(
        "pl_eval: LOOSE-FEASIBLE cells (n_vis < 3 but feasible, demo has k_min = 2) moved to the strict bound: {}",
        FEAS_LOW.load(std::sync::atomic::Ordering::Relaxed)
    );
    fx.finish("pl_eval", &["pl_eval_"]);
}

fn sel_of(row: &[f64]) -> Vec<usize> {
    row.iter().filter(|&&x| x >= 0.0).map(|&x| x as usize).collect()
}

/// Rows of a padded selection record.
fn sel_rows(d: &DMat) -> Vec<Vec<usize>> {
    if d.rows() == 0 {
        return Vec::new();
    }
    d.as_slice().chunks(d.cols()).map(sel_of).collect()
}

/// End selection of a stage: must equal the last history selection (the input if there was no step) and Python's end,
/// unless a step of this stage was already accepted as a key-equal tie (a mirror trajectory then ends elsewhere).
fn stage_end(case: &str, what: &str, input: &[usize], hist: &[HistoryStep], rust_end: &[usize], py_end: &[usize], ties: &mut Vec<String>) {
    let last = hist.iter().rev().find_map(|h| match h {
        HistoryStep::OneSwap { sel, .. } | HistoryStep::Prune { sel, .. } => Some(sel.clone()),
        _ => None,
    });
    assert_eq!(rust_end, last.as_deref().unwrap_or(input), "{case} {what}: returned selection != last history selection");
    if rust_end != py_end {
        let tied_here = ties.iter().any(|t| t.starts_with(&format!("{case} {what} step")));
        assert!(tied_here, "{case} {what}: end {rust_end:?} != python {py_end:?} without a tie-accepted step");
        ties.push(format!("{case} {what} end: rust {rust_end:?} vs python {py_end:?}"));
    }
}

/// Relative tolerance of the off-reference gates (a libm that differs in the last ulp moves keys by about 1e-13 relative, a
/// different tie resolution can move a key between mirror-equivalent selections by the same order).
const LOOSE_REL: f64 = 1e-7;

/// Off the reference platform the end of a search stage is compared by its objective key, not by selection or step count.
fn stage_end_loose(case: &str, what: &str, tb: &Tables, spec: &PlacementSpec, rust_end: &[usize], py_end: &[usize], w: &mut Worst) {
    let (kr, kp) = (key3(&evaluate(tb, rust_end, spec).unwrap(), spec), key3(&evaluate(tb, py_end, spec).unwrap(), spec));
    assert_eq!(kr.n_infeasible, kp.n_infeasible, "{case} {what}: end count");
    if what.starts_with("prune") && rust_end.len() != py_end.len() {
        // a prune that resolved a tie differently stops at another size; both ends are feasible, the sizes may differ by one
        assert!(rust_end.len().abs_diff(py_end.len()) <= 1, "{case} {what}: end size {} vs {}", rust_end.len(), py_end.len());
        return;
    }
    assert_eq!(rust_end.len(), py_end.len(), "{case} {what}: end size");
    w.check(case, "stage end penalty rel", rel(kr.penalty_sum, kp.penalty_sum), LOOSE_REL);
}

#[test]
fn pl_search() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut ties: Vec<String> = Vec::new();
    let strict = erploc_testkit::strict();
    for id in ["demo", "demo6", "tau2", "inf"] {
        let case = format!("pl_search_{id}");
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        let n = tb.n;
        // greedy
        let (gsel, ghist) = greedy(&tb, &spec).expect("greedy");
        let (want_sel, want_key, ck) = (sel_rows(fx.m("g_sel", &case)), fx.v("g_key", &case), fx.v("g_ck", &case));
        if strict {
            assert_eq!(ghist.len(), want_sel.len(), "{case} greedy steps");
        }
        let mut diverged = false;
        for (i, h) in ghist.iter().enumerate() {
            let HistoryStep::Greedy { sel, n_infeasible, penalty } = h else { panic!("not a greedy step") };
            if strict {
                assert_eq!(*n_infeasible as f64, want_key[i * 2], "{case} greedy step {i} n_infeasible");
            }
            if strict {
                w.check(&case, "greedy penalty rel", rel(*penalty, want_key[i * 2 + 1]), 1e-9);
            }
            // Rust-side check on Rust's own prefix: the chosen candidate is the first strict minimum of Rust's key2
            {
                let prefix = &sel[..sel.len() - 1];
                let j = *sel.last().unwrap();
                let key_of = |j: usize| {
                    let mut t = prefix.to_vec();
                    t.push(j);
                    key2(&evaluate(&tb, &t, &spec).unwrap(), &spec)
                };
                let kc = key_of(j);
                for j2 in (0..n).filter(|j2| !prefix.contains(j2) && *j2 != j) {
                    let k2_ = key_of(j2);
                    if j2 < j {
                        assert!(kc.lt(&k2_), "{case} step {i}: earlier candidate {j2} {k2_:?} not worse than chosen {j} {kc:?}");
                    } else {
                        assert!(!k2_.lt(&kc), "{case} step {i}: later candidate {j2} {k2_:?} better than chosen {j} {kc:?}");
                    }
                }
            }
            if strict && *sel != want_sel[i] {
                if !diverged && sel[..sel.len() - 1] == want_sel[i][..sel.len() - 1] {
                    // first divergence with an identical prefix: Python's own key of Rust's choice must tie its best
                    let j = *sel.last().unwrap();
                    let (cn, cp) = (ck[(i * n + j) * 2], ck[(i * n + j) * 2 + 1]);
                    assert_eq!(cn, want_key[i * 2], "{case} step {i}: Rust choice {j} count");
                    w.check(&case, "greedy tie key rel", rel(cp, want_key[i * 2 + 1]), TIE);
                }
                diverged = true;
                ties.push(format!("{case} greedy step {i}: rust {sel:?} vs python {:?} (keys agree)", want_sel[i]));
            }
        }
        // 1-swap and prune start from Python's previous stage
        let g_end: Vec<usize> = fx.v("g_end", &case).iter().map(|&x| x as usize).collect();
        if !strict {
            // off the reference platform: the greedy end is compared by its key
            let (kr, kp) = (key2(&evaluate(&tb, &gsel, &spec).unwrap(), &spec), key2(&evaluate(&tb, &g_end, &spec).unwrap(), &spec));
            assert_eq!(kr.n_infeasible, kp.n_infeasible, "{case} greedy end count");
            w.check(&case, "greedy end penalty rel", rel(kr.penalty_sum, kp.penalty_sum), LOOSE_REL);
        }
        let (ssel, shist) = local_search_1swap(&tb, &spec, &g_end).expect("1swap");
        let (want_sel, want_key) = (sel_rows(fx.m("s_sel", &case)), fx.v("s_key", &case));
        if strict {
            assert_eq!(shist.len(), want_sel.len(), "{case} 1swap steps");
        }
        for (i, h) in shist.iter().enumerate() {
            let HistoryStep::OneSwap { sel, n_infeasible, penalty, neg_margin } = h else { panic!("not a 1swap step") };
            if strict {
                assert_eq!(*n_infeasible as f64, want_key[i * 3], "{case} 1swap step {i} n_infeasible");
            }
            if strict {
                w.check(&case, "1swap penalty rel", rel(*penalty, want_key[i * 3 + 1]), 1e-9);
            }
            if strict {
                w.check(&case, "1swap neg_margin rel", rel(*neg_margin, want_key[i * 3 + 2]), 1e-9);
            }
            if strict && *sel != want_sel[i] {
                ties.push(format!("{case} 1swap step {i}: rust {sel:?} vs python {:?} (keys agree)", want_sel[i]));
            }
        }
        let s_end: Vec<usize> = fx.v("s_end", &case).iter().map(|&x| x as usize).collect();
        if strict {
            stage_end(&case, "1swap", &g_end, &shist, &ssel, &s_end, &mut ties);
        } else {
            stage_end_loose(&case, "1swap", &tb, &spec, &ssel, &s_end, &mut w);
        }
        let (psel, phist) = prune(&tb, &spec, &s_end).expect("prune");
        let (want_sel, want_key) = (sel_rows(fx.m("p_sel", &case)), fx.v("p_key", &case));
        if strict {
            assert_eq!(phist.len(), want_sel.len(), "{case} prune steps");
        }
        for (i, h) in phist.iter().enumerate() {
            let HistoryStep::Prune { sel, min_margin } = h else { panic!("not a prune step") };
            if strict {
                w.check(&case, "prune min_margin rel", rel(*min_margin, want_key[i]), 1e-9);
            }
            if strict && *sel != want_sel[i] {
                ties.push(format!("{case} prune step {i}: rust {sel:?} vs python {:?} (keys agree)", want_sel[i]));
            }
        }
        let p_end: Vec<usize> = fx.v("p_end", &case).iter().map(|&x| x as usize).collect();
        if strict {
            stage_end(&case, "prune", &s_end, &phist, &psel, &p_end, &mut ties);
        } else {
            stage_end_loose(&case, "prune", &tb, &spec, &psel, &p_end, &mut w);
        }
        // 1-swap from a poor start (real swaps) and prune from the full set (real removals)
        let w_start: Vec<usize> = fx.v("w_start", &case).iter().map(|&x| x as usize).collect();
        let (wsel, whist) = local_search_1swap(&tb, &spec, &w_start).expect("1swap poor start");
        let (want_sel, want_key) = (sel_rows(fx.m("w_sel", &case)), fx.v("w_key", &case));
        if strict {
            assert_eq!(whist.len(), want_sel.len(), "{case} poor-start 1swap steps");
        }
        for (i, h) in whist.iter().enumerate() {
            let HistoryStep::OneSwap { sel, n_infeasible, penalty, neg_margin } = h else { panic!("not a 1swap step") };
            if strict {
                assert_eq!(*n_infeasible as f64, want_key[i * 3], "{case} poor 1swap step {i} n_infeasible");
            }
            if strict {
                w.check(&case, "1swap(poor start) penalty rel", rel(*penalty, want_key[i * 3 + 1]), 1e-9);
            }
            if strict {
                w.check(&case, "1swap(poor start) neg_margin rel", rel(*neg_margin, want_key[i * 3 + 2]), 1e-9);
            }
            if strict && *sel != want_sel[i] {
                ties.push(format!("{case} poor-start 1swap step {i}: rust {sel:?} vs python {:?} (keys agree)", want_sel[i]));
            }
        }
        let w_end: Vec<usize> = fx.v("w_end", &case).iter().map(|&x| x as usize).collect();
        if strict {
            stage_end(&case, "poor-start 1swap", &w_start, &whist, &wsel, &w_end, &mut ties);
        } else {
            stage_end_loose(&case, "poor-start 1swap", &tb, &spec, &wsel, &w_end, &mut w);
        }
        let (qsel, qhist) = prune(&tb, &spec, &(0..n).collect::<Vec<_>>()).expect("prune full");
        let (want_sel, want_key) = (sel_rows(fx.m("q_sel", &case)), fx.v("q_key", &case));
        if strict {
            assert_eq!(qhist.len(), want_sel.len(), "{case} prune-from-full steps");
        }
        for (i, h) in qhist.iter().enumerate() {
            let HistoryStep::Prune { sel, min_margin } = h else { panic!("not a prune step") };
            if strict {
                w.check(&case, "prune(full) min_margin rel", rel(*min_margin, want_key[i]), 1e-9);
            }
            if strict && *sel != want_sel[i] {
                ties.push(format!("{case} prune-from-full step {i}: rust {sel:?} vs python {:?} (keys agree)", want_sel[i]));
            }
        }
        let q_end: Vec<usize> = fx.v("q_end", &case).iter().map(|&x| x as usize).collect();
        if strict {
            stage_end(&case, "prune-from-full", &(0..n).collect::<Vec<_>>(), &qhist, &qsel, &q_end, &mut ties);
        } else {
            stage_end_loose(&case, "prune-from-full", &tb, &spec, &qsel, &q_end, &mut w);
        }
        eprintln!("{case}: poor-start 1swap {} steps, prune-from-full {} steps", whist.len(), qhist.len());
        eprintln!(
            "{case}: greedy {gsel:?} ({} steps), 1swap {} steps, prune {} steps; python greedy {:?}",
            ghist.len(),
            shist.len(),
            phist.len(),
            fx.v("g_end", &case)
        );
    }
    // brute force on the 7-candidate sub-specs
    for id in ["demo7", "demo7inf"] {
        let case = format!("pl_search_bf_{id}");
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        for (tag, k) in [("k3", Some(3)), ("kn", None)] {
            let got = brute_force(&tb, &spec, k).expect("brute_force");
            let want: Vec<usize> = fx.v(&format!("sel_{tag}"), &case).iter().map(|&x| x as usize).collect();
            let key = fx.v(&format!("key_{tag}"), &case);
            let st = evaluate(&tb, &got, &spec).unwrap();
            let n_inf = st.feasible.iter().filter(|&&f| !f).count();
            let cmax = st.sigma.iter().fold(f64::NEG_INFINITY, |m, &x| {
                let c = cap(x, spec.sigma_cap);
                if c.is_nan() || m.is_nan() {
                    f64::NAN
                } else {
                    m.max(c)
                }
            });
            assert_eq!(got.len(), want.len(), "{case} {tag}: selection size");
            assert_eq!(n_inf as f64, key[0], "{case} {tag}: infeasible count");
            w.check(&case, "brute_force max sigma rel", rel(cmax, key[1]), 1e-9);
            if got != want {
                ties.push(format!("{case} {tag}: rust {got:?} vs python {want:?} (keys agree)"));
            }
        }
    }
    w.report("pl_search", 6);
    for t in &ties {
        eprintln!("TIE-EQUIVALENT: {t}");
    }
    eprintln!("pl_search: {} tie-equivalent selections accepted", ties.len());
    fx.finish("pl_search", &["pl_search_"]);
}

/// CSV text as numbers: header string, then per row the 6 fields.
fn csv_rows(s: &str) -> (String, Vec<Vec<f64>>) {
    let mut it = s.lines();
    let head = it.next().unwrap().to_string();
    (head, it.map(|l| l.split(',').map(|x| x.parse::<f64>().unwrap()).collect()).collect())
}

fn check_tri(fx: &Fx, case: &str, tb: &Tables, spec: &PlacementSpec, w: &mut Worst) {
    let sel: Vec<usize> = fx.v("sel", case).iter().map(|&x| x as usize).collect();
    let ts = fx.s("ts", case);
    let (ids, score, nskip) = choose_survey_triplet(tb, spec, &sel, ts).expect("triplet");
    let want: Vec<usize> = fx.v("ids", case).iter().map(|&x| x as usize).collect();
    let want_nskip = fx.s("nskip", case);
    if erploc_testkit::strict() {
        assert_eq!(ids, want, "{case}: survey ids");
        assert_eq!(nskip as f64, want_nskip, "{case}: n_skip");
    } else {
        // tied triplets may resolve differently: the score is the objective
        assert_eq!(ids.len(), want.len(), "{case}: survey size");
    }
    w.check(case, "triplet score rel", rel(score, fx.s("score", case)), if erploc_testkit::strict() { 1e-9 } else { LOOSE_REL });
}

#[test]
fn pl_survey() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let mut notes: Vec<String> = Vec::new();
    for (id, ts) in [("demo", 5), ("demo7", 5), ("demo", 20)] {
        let case = format!("pl_survey_opt_{id}_ts{ts}");
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        let sigma_ts = fx.s("ts", &case);
        let res = optimize(&spec, sigma_ts).expect("optimize");
        let py_sel: Vec<usize> = fx.v("sel", &case).iter().map(|&x| x as usize).collect();
        let py_ids: Vec<usize> = fx.v("ids", &case).iter().map(|&x| x as usize).collect();
        let k3 = fx.v("key3", &case);
        let a = key3(&evaluate(&tb, &res.sel, &spec).unwrap(), &spec);
        assert_eq!(a.n_infeasible as f64, k3[0], "{case}: final key3 count");
        w.check(&case, "optimize key3 penalty rel", rel(a.penalty_sum, k3[1]), 1e-9);
        if erploc_testkit::strict() {
            // (the margin is the tie-breaking third part of the key: another libm may settle a tie of the first two on another layout)
            w.check(&case, "optimize key3 neg_margin rel", rel(a.neg_margin, k3[2]), 1e-9);
        }
        let HistoryStep::Survey { ids, score, n_skip } = res.history.last().unwrap().clone() else { panic!("no survey step") };
        let (py_nskip, py_score) = (fx.s("nskip", &case), fx.s("score", &case));
        if res.sel == py_sel {
            assert_eq!(res.survey_ids, py_ids, "{case}: survey ids");
            assert_eq!(ids, py_ids);
            assert_eq!(n_skip as f64, py_nskip, "{case}: n_skip");
            w.check(&case, "optimize score rel", rel(score, py_score), 1e-9);
        } else {
            // mirror selection: Rust-side self-consistency of ids/score/n_skip with choose_survey_triplet on its own sel
            let (i2, s2, n2) = choose_survey_triplet(&tb, &spec, &res.sel, sigma_ts).unwrap();
            assert_eq!((&res.survey_ids, &ids, n_skip), (&i2, &i2, n2), "{case}: optimize survey != triplet on own sel");
            assert_eq!(score.to_bits(), s2.to_bits(), "{case}: optimize score != triplet score");
            notes.push(format!("{case}: optimize sel rust {:?} vs python {py_sel:?} (keys agree); ids not compared", res.sel));
        }
        // CSV: Python's sel/ids through Rust's report_csv, compared numerically
        let st = evaluate(&tb, &py_sel, &spec).unwrap();
        let pr = PlacementResult { sel: py_sel.clone(), stats: st, survey_ids: py_ids.clone(), history: Vec::new() };
        let got = report_csv(&pr, &spec);
        let want: String = fx.v("csv", &case).iter().map(|&x| x as u8 as char).collect();
        let (gh, gr) = csv_rows(&got);
        let (wh, wr) = csv_rows(&want);
        assert_eq!(gh, wh, "{case}: csv header");
        assert_eq!(gr.len(), wr.len(), "{case}: csv rows");
        let quantum = [0.0, 1e-4, 1e-4, 1e-4, 1e-2, 0.0];
        for (g, p) in gr.iter().zip(&wr) {
            for c in 0..6 {
                assert!((g[c] - p[c]).abs() <= quantum[c] * 1.0001, "{case}: csv field {c}: {} vs {}", g[c], p[c]);
            }
        }
        if erploc_testkit::strict() {
            assert_eq!(got, want, "{case}: report_csv string");
        }
    }
    for (id, ts_t, ts_o) in [("demo", 5, 5), ("demo", 20, 5), ("demo7", 5, 5), ("demo7", 20, 5), ("demo", 5, 20), ("demo", 20, 20)] {
        let case = format!("pl_survey_tri_{id}_ts{ts_t}_on{ts_o}");
        let spec = load_spec(&fx, id);
        let tb = keep_going(precompute(&spec));
        check_tri(&fx, &case, &tb, &spec, &mut w);
    }
    w.report("pl_survey", 9);
    for t in &notes {
        eprintln!("TIE-EQUIVALENT: {t}");
    }
    fx.finish("pl_survey", &["pl_survey_"]);
}

#[test]
fn pl_python_sel() {
    let fx = Fx::new();
    let mut w = Worst::new();
    let spec = load_spec(&fx, "demo");
    let tb = keep_going(precompute(&spec));
    let meta: Vec<usize> = fx.v("sel", "pl_python_sel_meta").iter().map(|&x| x as usize).collect();
    let n = check_eval(&fx, "pl_python_sel_eval", &tb, &spec, &mut w);
    assert_eq!(sel_rows(fx.m("sel", "pl_python_sel_eval")), vec![meta.clone()]);
    for ts in [5, 20] {
        check_tri(&fx, &format!("pl_python_sel_tri_ts{ts}"), &tb, &spec, &mut w);
    }
    w.report("pl_python_sel", n);
    eprintln!("pl_python_sel: Python IFT selection {meta:?} evaluated and surveyed in Rust");
    fx.finish("pl_python_sel", &["pl_python_sel_"]);
}
