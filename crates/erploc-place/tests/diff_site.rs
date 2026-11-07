//! Differential tests of the site model against Python (fixtures/ift.txt: outputs recorded from the Python reference, thinned; the
//! recording scripts depend on private modules and are not included; see fixtures/README.md).
//! Group `oc_`: occlusion geometry, blocked triples, blocked matrix, typed precompute (the hit paths have no recorded reference in
//! public; `oc_hit_self_check` compares the fast and the scalar implementation on the synthetic site).
//! Groups `sg_` and `fe_`: `sigma10` (3-way rule) and the `FastEval` core (`set_key`, `add_keys`) with the AMB counts.
//! Counts, masks and triples are exact. A mismatch is first classified by the margins the exporter recorded for the
//! comparison that decides it: below 1e-12 a genuine rounding tie (reported with the case name), otherwise a bug.
//! Group `ne_`: `FastEval::neighbourhood` (demo room, sub-scene, full site) and the recorded `ift_lib.selftest` draws.
//! Group `ift_`: `ift_det`, the perturbed runs by recorded-draw replay under the tick clock, `ift_min_count`, `optimize_ift`.
//! Every exported record must be read: each test ends with `finish`.
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::V3;
use erploc_linalg::DMat;
use erploc_place::fast_eval::{FastEval, IftContext, BIG};
use erploc_place::occlusion::*;
use erploc_place::placement::*;
use erploc_place::scene::*;
use erploc_place::search::{ift_det, ift_min_count, optimize_ift, run_ift, Budget, IftRun, Replay, TickClock};
use erploc_place::sigma10::{sigma10_one, Sigma10Counts};
use erploc_place::site::CandKind;
use erploc_place::visibility::{Sector, VisParams};
use erploc_testkit::fixture::{self, Fixture};
use std::cell::RefCell;
use std::collections::BTreeSet;

struct Fx {
    f: Fixture,
    read: RefCell<BTreeSet<(String, String)>>,
}
impl Fx {
    fn new() -> Self {
        Fx::open("ift.txt")
    }
    fn open(file: &str) -> Self {
        Fx { f: fixture::load(file), read: RefCell::new(BTreeSet::new()) }
    }
    fn m(&self, tag: &str, name: &str) -> &DMat {
        let d = self.f.get(tag, name).unwrap_or_else(|| panic!("missing {tag} {name}"));
        self.read.borrow_mut().insert((tag.to_string(), name.to_string()));
        d
    }
    fn v(&self, tag: &str, name: &str) -> Vec<f64> {
        self.m(tag, name).as_slice().to_vec()
    }
    fn usizes(&self, tag: &str, name: &str) -> Vec<usize> {
        self.v(tag, name).iter().map(|&x| x as usize).collect()
    }
    fn bools(&self, tag: &str, name: &str) -> Vec<bool> {
        self.m(tag, name).as_slice().iter().map(|&x| x != 0.0).collect()
    }
    fn s(&self, tag: &str, name: &str) -> f64 {
        let d = self.m(tag, name);
        assert_eq!((d.rows(), d.cols()), (1, 1), "{tag} {name} is not a scalar");
        d.as_slice()[0]
    }
    fn poly(&self, tag: &str, name: &str) -> Vec<[f64; 2]> {
        self.m(tag, name).as_slice().chunks(2).map(|c| [c[0], c[1]]).collect()
    }
    fn obstacles(&self, name: &str) -> Vec<Obstacle> {
        let d = self.m("ob", name);
        let mut out: Vec<Obstacle> = Vec::new();
        for r in d.as_slice().chunks(4) {
            let idx = r[0] as usize;
            if idx == out.len() {
                out.push(Obstacle { poly: Vec::new(), height: r[3] });
            }
            out[idx].poly.push([r[1], r[2]]);
        }
        out
    }
    fn sectors(&self, name: &str) -> Vec<Sector> {
        self.m("sec", name).as_slice().chunks(4).map(|c| Sector { lam_min: c[0], lam_max: c[1], phi_min: c[2], phi_max: c[3] }).collect()
    }
    fn finish(&self, label: &str, prefixes: &[&str]) {
        let all: Vec<_> = self.f.recs.keys().filter(|(_, n)| prefixes.iter().any(|p| n.starts_with(p))).collect();
        let read = self.read.borrow();
        let n_read = all.iter().filter(|k| read.contains(**k)).count();
        assert_eq!(n_read, all.len(), "{label}: read {n_read} of {} records", all.len());
        eprintln!("{label}: n_read == n_records == {n_read}");
    }
}

fn v3(d: &[f64]) -> V3 {
    [d[0], d[1], d[2]]
}

#[test]
fn oc_samples() {
    let fx = Fx::new();
    let mut worst: f64 = 0.0;
    for nm in ["oc_samples_40", "oc_samples_16"] {
        let a = fx.s("A", nm);
        let cand = fx.m("cand", nm);
        let ps = fx.v("ps", nm);
        assert_eq!((cand.rows(), ps.len()), (20, 20 * 13 * 3));
        for (i, c) in cand.as_slice().chunks(4).enumerate() {
            let s = site_face_samples(&Candidate { xy: [c[0], c[1]], z: c[2], azimuth: c[3] }, a);
            for (q, p) in s.iter().enumerate() {
                for d in 0..3 {
                    let e = (p[d] - ps[(i * 13 + q) * 3 + d]).abs();
                    assert!(e <= 1e-15, "{nm} cand {i} sample {q}: {e:e}");
                    worst = worst.max(e);
                }
            }
        }
    }
    eprintln!("oc_samples: worst abs error {worst:e}");
    fx.finish("oc_samples", &["oc_samples_"]);
}

/// No recorded Python reference exists in public for the occlusion hit paths. Self-check on the synthetic site instead: the
/// blocked triples (with the plan-view prefilter in `Samples` mode) equal an unfiltered scalar enumeration over every stand and
/// every visible pair, in both modes.
#[test]
fn oc_hit_self_check() {
    let s = erploc_place::site_synth::synthetic_site().unwrap();
    let n = s.cands.len();
    for (mode, om) in [(erploc_place::site::VisMode::Centre, OccMode::Centre), (erploc_place::site::VisMode::Samples, OccMode::Samples)] {
        let vis = &s.vis[mode.index()];
        let occ = blocked_triples(&s.cells, &s.cands, &s.kinds, s.spec0.a, vis, om);
        let mut fast = BTreeSet::new();
        for st in 0..n {
            for i in occ.bs_ptr[st]..occ.bs_ptr[st + 1] {
                fast.insert((st, occ.bs_t[i] as usize, occ.bs_c[i] as usize));
            }
        }
        assert_eq!(fast.len(), occ.n_triples);
        let mut scalar = BTreeSet::new();
        for k in 0..s.cells.len() {
            for j in 0..n {
                if !vis[k * n + j] {
                    continue;
                }
                let cj = &s.cands[j];
                let centre = [cj.xy[0], cj.xy[1], cj.z];
                let ps = site_face_samples(cj, s.spec0.a);
                for st in (0..n).filter(|&st| st != j && s.kinds[st] == CandKind::Stand) {
                    let hit = match om {
                        OccMode::Centre => hit_board(&s.cells[k], &centre, &s.cands[st]),
                        OccMode::Samples => ps
                            .iter()
                            .any(|p| hit_board(&s.cells[k], p, &s.cands[st]) || hit_base(&s.cells[k], p, &s.cands[st], &STAND_BASE)),
                    };
                    if hit {
                        scalar.insert((st, j, k));
                    }
                }
            }
        }
        assert_eq!(fast, scalar, "{mode:?}: fast triples differ from the scalar enumeration");
        assert!(!fast.is_empty(), "{mode:?}: the synthetic site must contain blocked triples");
    }
}

fn load_spec(fx: &Fx, id: &str) -> PlacementSpec {
    let n = format!("oc_typed_{id}");
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

#[test]
fn oc_typed() {
    let fx = Fx::open("typed.txt");
    // the information blocks of `demo` and `s2` do not depend on the kinds; they are the blocks of `placement.precompute`
    let place = Fx::open("place.txt");
    for id in ["demo", "s2", "pole"] {
        let n = format!("oc_typed_{id}");
        let spec = load_spec(&fx, id);
        let kinds: Vec<CandKind> = fx.v("kinds", &n).iter().map(|&x| if x != 0.0 { CandKind::Stand } else { CandKind::Wall }).collect();
        let tb = precompute_typed(&spec, &kinds).unwrap();
        let want_cells = fx.m("cells", &n);
        assert_eq!(tb.cells.len(), want_cells.rows(), "{n} K");
        for (c, r) in tb.cells.iter().zip(want_cells.as_slice().chunks(3)) {
            assert!(c.iter().zip(r).all(|(x, y)| x.to_bits() == y.to_bits()), "{n} cell {c:?} vs {r:?}");
        }
        assert_eq!(tb.vis, fx.bools("vis", &n), "{n} vis");
        let want_i = if id == "pole" { fx.v("I", &n) } else { place.v("I", &format!("pl_tables_{id}")) };
        assert_eq!(tb.info.len(), want_i.len());
        let (mut e_el, mut e_blk) = (0.0f64, 0.0f64);
        for (bt, bw) in tb.info.chunks(16).zip(want_i.chunks(16)) {
            let scale = bw.iter().fold(1.0f64, |m, y| m.max(y.abs()));
            for (x, y) in bt.iter().zip(bw) {
                let d = (x - y).abs();
                assert!(d.is_finite(), "{n}: non-finite info difference");
                e_el = e_el.max(d / y.abs().max(1.0));
                e_blk = e_blk.max(d / scale);
            }
        }
        assert!(e_el <= 1e-6 && e_blk <= 1e-12, "{n}: I errors {e_el:e} (elementwise) {e_blk:e} (block max)");
        eprintln!("{n}: K={} Hd={} N={} cells/vis exact; I worst rel {e_el:e} elementwise, {e_blk:e} to block max", tb.k, tb.hd, tb.n);
    }
    fx.finish("oc_typed", &["oc_typed_"]);
}

/// Compares Rust and Python `sigma10` on `a` (n x 10): the branch flag exactly, an error exactly, the value to `tol_sure`
/// relative on the sure branches (the arithmetic is identical there) and to `tol_amb` on the exact-rule entries
/// (eigenvalue test + LU inverse of two different LAPACK/LU implementations: error of order cond * eps).
/// Returns `(amb entries, worst rel diff sure, worst rel diff amb, capped)`.
fn compare_sigma10(name: &str, a: &DMat, v: &[f64], amb: &[f64], err: &[f64], tol_sure: f64, tol_amb: f64) -> (usize, f64, f64, usize) {
    let cap = 10.0;
    let (mut n_amb, mut w_sure, mut w_amb, mut n_cap) = (0, 0.0f64, 0.0f64, 0);
    let mut cnt = Sigma10Counts::default();
    for (i, r) in a.as_slice().chunks(10).enumerate() {
        let row: [f64; 10] = r.try_into().unwrap();
        let before = cnt.amb;
        let got = sigma10_one(&row, cap, &mut cnt);
        let is_amb = cnt.amb > before;
        if err[i] != 0.0 {
            assert!(got.is_err(), "{name} row {i}: python raised, rust returned {got:?}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name} row {i}: rust error {e:?}, python value {}", v[i]));
        assert_eq!(is_amb, amb[i] != 0.0, "{name} row {i}: branch (amb) rust {is_amb} python {} for {row:?}", amb[i]);
        n_amb += is_amb as usize;
        n_cap += (v[i] == cap) as usize;
        if got.to_bits() == v[i].to_bits() {
            continue;
        }
        assert_eq!(got == cap, v[i] == cap, "{name} row {i}: cap rust {got} python {}", v[i]);
        assert!(got.is_finite() && v[i].is_finite(), "{name} row {i}: non-finite mismatch {got} vs {}", v[i]);
        let rel = (got - v[i]).abs() / v[i].abs().max(1.0);
        if is_amb {
            assert!(rel <= tol_amb, "{name} row {i} (exact rule): rel {rel:e} > {tol_amb:e}: {got} vs {} for {row:?}", v[i]);
            w_amb = w_amb.max(rel);
        } else {
            assert!(rel <= tol_sure, "{name} row {i} (sure branch): rel {rel:e} > {tol_sure:e}: {got} vs {} for {row:?}", v[i]);
            w_sure = w_sure.max(rel);
        }
    }
    assert_eq!((cnt.amb, cnt.total), (amb.iter().filter(|&&x| x != 0.0).count(), a.rows()), "{name}: counts");
    (n_amb, w_sure, w_amb, n_cap)
}

#[test]
fn sg_random() {
    let fx = Fx::new();
    let a = fx.m("a", "sg_random");
    let (v, amb, err, cls) = (fx.v("v", "sg_random"), fx.v("amb", "sg_random"), fx.v("err", "sg_random"), fx.v("cls", "sg_random"));
    let (n_amb, ws, wa, nc) = compare_sigma10("sg_random", a, &v, &amb, &err, 1e-12, 1e-8);
    let per_class: Vec<usize> = (0..8).map(|c| (0..a.rows()).filter(|&i| cls[i] as usize == c && amb[i] != 0.0).count()).collect();
    eprintln!("sg_random: {} matrices exact in branch (AMB {n_amb}: per class {per_class:?}), {} python errors; worst rel diff sure {ws:e}, exact rule {wa:e}; capped {nc}",
        a.rows(), err.iter().filter(|&&e| e != 0.0).count());
    fx.finish("sg_random", &["sg_random"]);
}

/// The demo room on Python's tables (`pl_tables_demo` of place.txt): Rust `precompute` differs from them at 1e-16 of the block max, which the
/// near-singular cells of the demo room amplify to margin differences of 1e-8 (at margin 29); on the same tables the evaluators agree
/// to the bit (sub-scenes, which load the tables from `site_small.bin`, are bit-identical either way).
fn demo_python_tables(ctx: &mut IftContext) {
    let place = Fx::open("place.txt");
    let (vis, info) = (place.bools("vis", "pl_tables_demo"), place.v("I", "pl_tables_demo"));
    let (k, hd, n) = (ctx.tb.k, ctx.tb.hd, ctx.tb.n);
    ctx.tb = Tables::from_raw(ctx.tb.cells.clone(), hd, n, vis, info);
    assert_eq!(ctx.tb.k, k);
}

fn fe_context(name: &str) -> IftContext {
    match name {
        "demo" => IftContext::from_spec(&demo_spec(10)).unwrap(),
        _ => panic!("{name}"),
    }
}

fn unpad(m: &DMat) -> Vec<Vec<usize>> {
    m.as_slice().chunks(m.cols()).map(|r| r.iter().filter(|&&x| x >= 0.0).map(|&x| x as usize).collect()).collect()
}

#[test]
fn fe_set() {
    fe_set_run(false);
}

/// Demo room on Rust's own `precompute` tables: same records, margin tolerance relative (1e-9 * max(1, |margin|)).
#[test]
fn fe_set_demo_own_tables() {
    fe_set_run(true);
}

fn fe_set_run(own: bool) {
    let fx = Fx::new();
    for case in ["demo"] {
        let name = format!("fe_set_{case}");
        let mut ctx = fe_context(case);
        if case == "demo" && !own {
            demo_python_tables(&mut ctx);
        }
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let sels = unpad(fx.m("sel", &name));
        let key = fx.m("key", &name);
        let gap = fx.v("gap", &name);
        let (mut w_pen, mut w_m) = (0.0f64, 0.0f64);
        for (i, s) in sels.iter().enumerate() {
            let k = fe.set_key(s).unwrap();
            let (pn, pp, pm) = (key[(i, 0)] as usize, key[(i, 1)], key[(i, 2)]);
            assert_eq!(
                k.n_infeasible,
                pn,
                "{name} sel {i} {s:?}: count (smallest threshold gap of the case {:e}, of this selection {:e})",
                gap.iter().cloned().fold(f64::INFINITY, f64::min),
                gap[i]
            );
            let rp = (k.penalty_sum - pp).abs() / pp.abs().max(1.0);
            let dm = (k.neg_margin - pm).abs();
            assert!(rp <= 1e-9, "{name} sel {i}: penalty rust {} python {pp} (rel {rp:e}, gap {:e})", k.penalty_sum, gap[i]);
            let tm = if own { 1e-9 * pm.abs().max(1.0) } else { 1e-9 };
            assert!(dm <= tm, "{name} sel {i}: margin rust {} python {pm} (abs {dm:e}, gap {:e})", k.neg_margin, gap[i]);
            w_pen = w_pen.max(rp);
            w_m = w_m.max(dm);
        }
        let amb = fx.v("amb", &name);
        assert_eq!((fe.amb.amb, fe.amb.total), (amb[0] as usize, amb[1] as usize), "{name}: AMB counts (rust vs python)");
        eprintln!("{name}: {} selections equal; worst penalty rel {w_pen:e}, margin abs {w_m:e}; AMB {} of {} (python equal); smallest threshold gap {:e}", sels.len(), fe.amb.amb, fe.amb.total, gap.iter().cloned().fold(f64::INFINITY, f64::min));
    }
    if !own {
        fx.finish("fe_set", &["fe_set_"]);
    }
}

#[test]
fn fe_add() {
    fe_add_run(false);
}

/// Demo room on Rust's own `precompute` tables: same records, margin tolerance relative (1e-9 * max(1, |margin|)).
#[test]
fn fe_add_demo_own_tables() {
    fe_add_run(true);
}

fn fe_add_run(own: bool) {
    let fx = Fx::new();
    for case in ["demo"] {
        let name = format!("fe_add_{case}");
        let mut ctx = fe_context(case);
        if case == "demo" && !own {
            demo_python_tables(&mut ctx);
        }
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let sels = unpad(fx.m("sel", &name));
        let (inf, pen, nm) = (fx.m("inf", &name), fx.m("pen", &name), fx.m("nm", &name));
        // smallest threshold gap |sigma/tau - 1| (and the loo analogue) over 8 sampled S' + {j} per S': a count mismatch below
        // 1e-12 would be a rounding tie of a cell sitting on a feasibility threshold (as `fe_set`)
        let gap = fx.v("gap", &name);
        assert_eq!(gap.len(), sels.len());
        let (mut w_pen, mut w_m, mut n_big) = (0.0f64, 0.0f64, 0);
        for (i, sp) in sels.iter().enumerate() {
            let ak = fe.add_keys(sp).unwrap();
            assert_eq!(ak.inf.len(), inf.cols());
            for j in 0..ak.inf.len() {
                let (pi, pp, pm) = (inf[(i, j)] as i64, pen[(i, j)], nm[(i, j)]);
                if sp.contains(&j) {
                    assert_eq!((ak.inf[j], ak.pen[j], ak.nm[j]), (BIG, f64::INFINITY, f64::INFINITY));
                    assert_eq!((pi, pp, pm), (BIG, f64::INFINITY, f64::INFINITY), "{name}: python selected entry");
                    n_big += 1;
                    continue;
                }
                assert_eq!(
                    ak.inf[j],
                    pi,
                    "{name} S' {sp:?} j {j}: infeasible count (smallest sampled threshold gap of this S' {:e}, of the case {:e})",
                    gap[i],
                    gap.iter().cloned().fold(f64::INFINITY, f64::min)
                );
                let rp = (ak.pen[j] - pp).abs() / pp.abs().max(1.0);
                let dm = (ak.nm[j] - pm).abs();
                assert!(rp <= 1e-9, "{name} S' {sp:?} j {j}: penalty rust {} python {pp} (rel {rp:e})", ak.pen[j]);
                let tm = if own { 1e-9 * pm.abs().max(1.0) } else { 1e-9 };
                assert!(dm <= tm || (ak.nm[j] == pm), "{name} S' {sp:?} j {j}: margin rust {} python {pm} (abs {dm:e})", ak.nm[j]);
                w_pen = w_pen.max(rp);
                if dm.is_finite() {
                    w_m = w_m.max(dm);
                }
            }
        }
        let amb = fx.v("amb", &name);
        assert_eq!((fe.amb.amb, fe.amb.total), (amb[0] as usize, amb[1] as usize), "{name}: AMB counts (rust vs python)");
        eprintln!("{name}: {} S' x {} entries equal ({n_big} selected = BIG); worst penalty rel {w_pen:e}, margin abs {w_m:e}; AMB {} of {} (python equal); smallest sampled threshold gap {:e}", sels.len(), inf.cols(), fe.amb.amb, fe.amb.total, gap.iter().cloned().fold(f64::INFINITY, f64::min));
    }
    if !own {
        fx.finish("fe_add", &["fe_add_"]);
    }
}

/// Compares `Nbhd` with the recorded Python `(k, N)` matrices of selection `i` of `name`: `inf` exact incl. `BIG`, penalty
/// `<= 1e-9` relative (`inf` entries exact), margin `<= tol_m` (absolute, or relative above 1 when `own` tables are used).
fn compare_nbhd(fx: &Fx, name: &str, ctx: &mut IftContext, k_eval: usize, own: bool) -> (usize, f64, f64) {
    let sels = unpad(fx.m("sel", name));
    let (inf, pen, nm) = (fx.m("inf", name), fx.m("pen", name), fx.m("nm", name));
    let mut fe = FastEval::new(ctx, k_eval).unwrap();
    let (mut w_pen, mut w_m, mut n) = (0.0f64, 0.0f64, 0usize);
    for (r, s) in sels.iter().enumerate() {
        let nb = fe.neighbourhood(s).unwrap();
        assert_eq!((nb.k, nb.n), (s.len(), ctx.n));
        for i in 0..s.len() {
            for j in 0..ctx.n {
                let (at, c) = (i * ctx.n + j, i * ctx.n + j);
                let (pi, pp, pm) = (inf[(r, c)] as i64, pen[(r, c)], nm[(r, c)]);
                assert_eq!(nb.inf[at], pi, "{name} sel {r} {s:?} (i {i}, j {j}): infeasible count");
                if pp.is_infinite() || pm.is_infinite() {
                    assert_eq!((nb.pen[at], nb.nm[at]), (pp, pm), "{name} sel {r} (i {i}, j {j}): inf entries");
                    continue;
                }
                let rp = (nb.pen[at] - pp).abs() / pp.abs().max(1.0);
                let dm = (nb.nm[at] - pm).abs();
                let tm = if own { 1e-9 * pm.abs().max(1.0) } else { 1e-9 };
                assert!(rp <= 1e-9, "{name} sel {r} (i {i}, j {j}): penalty rust {} python {pp} (rel {rp:e})", nb.pen[at]);
                assert!(dm <= tm, "{name} sel {r} (i {i}, j {j}): margin rust {} python {pm} (abs {dm:e})", nb.nm[at]);
                w_pen = w_pen.max(rp);
                w_m = w_m.max(dm);
                n += 1;
            }
        }
    }
    eprintln!(
        "{name}: {} neighbourhoods, {n} finite entries equal (worst penalty rel {w_pen:e}, margin abs {w_m:e}); AMB {} of {}",
        sels.len(),
        fe.amb.amb,
        fe.amb.total
    );
    (n, w_pen, w_m)
}

#[test]
fn ne_demo() {
    let fx = Fx::new();
    let mut ctx = fe_context("demo");
    demo_python_tables(&mut ctx);
    compare_nbhd(&fx, "ne_demo", &mut ctx, 10, false);
    fx.finish("ne_demo", &["ne_demo"]);
}

/// Demo room on Rust's own `precompute` tables (margin tolerance relative above 1, as `fe_*_demo_own_tables`).
#[test]
fn ne_demo_own_tables() {
    let fx = Fx::new();
    let mut ctx = fe_context("demo");
    compare_nbhd(&fx, "ne_demo", &mut ctx, 10, true);
}

/// Count exact, penalty `<= 1e-9` relative (`max(1, |pen|)`), margin `<= 1e-9 * max(1, |margin|)`.
fn key_close3(a: &Key3, count: usize, pen: f64, nm: f64) -> bool {
    a.n_infeasible == count
        && (a.penalty_sum - pen).abs() <= 1e-9 * pen.abs().max(1.0)
        && (a.neg_margin - nm).abs() <= 1e-9 * nm.abs().max(1.0)
}

fn rec_key(fx: &Fx, name: &str) -> (usize, f64, f64) {
    let k = fx.v("key", name);
    (k[0] as usize, k[1], k[2])
}

/// Python tables for the demo room (bit-comparable); other names as `fe_context`.
fn ift_context(case: &str) -> IftContext {
    let mut ctx = match case {
        "demo" => fe_context("demo"),
        _ => panic!("{case}"),
    };
    if case == "demo" {
        demo_python_tables(&mut ctx);
    }
    ctx
}

/// Compares a run with the record `name`: returns "EXACT" or "TIE-EQUIVALENT" (the path differs, but the key of every `pick` is
/// equal within the tie tolerance and so is the final key). With `exact_only` a differing selection fails.
fn compare_run(fx: &Fx, name: &str, r: &IftRun, exact_only: bool) -> &'static str {
    let sel = fx.usizes("sel", name);
    let (c, p, m) = rec_key(fx, name);
    let meta = fx.v("meta", name);
    let picks = fx.m("picks", name);
    let fk = fx.v("fkey", name);
    let trace = fx.m("trace", name);
    assert!(key_close3(&r.best_key, c, p, m), "{name}: best_key rust {:?} python ({c}, {p}, {m})", r.best_key);
    assert!(key_close3(&r.info.fast_key, fk[0] as usize, fk[1], fk[2]), "{name}: fast_key");
    assert_eq!(r.info.picks.len(), picks.rows(), "{name}: number of picks");
    let mut worst = (0.0f64, 0.0f64);
    for (i, pk) in r.info.picks.iter().enumerate() {
        let (pc, pp, pm) = (picks[(i, 0)] as usize, picks[(i, 1)], picks[(i, 2)]);
        assert!(key_close3(pk, pc, pp, pm), "{name}: pick {i}: rust {pk:?} python ({pc}, {pp}, {pm})");
        worst.0 = worst.0.max((pk.penalty_sum - pp).abs() / pp.abs().max(1.0));
        worst.1 = worst.1.max((pk.neg_margin - pm).abs());
    }
    let (multi, distinct) =
        (0..picks.rows()).fold((0, 0), |(a, b), i| (a + (picks[(i, 3)] > 1.0) as usize, b + (picks[(i, 4)] > 1.0) as usize));
    let same = r.best_sel == sel;
    if same {
        assert_eq!((r.evals, r.info.iters, r.info.phases), (meta[0] as u64, meta[1] as u64, meta[2] as u64), "{name}: evals/iters/phases");
        assert_eq!((r.time, r.info.t_construction), (meta[4], meta[5]), "{name}: virtual times");
        assert_eq!(r.info.n_triples.map_or(-1.0, |x| x as f64), meta[6], "{name}: n_triples");
        assert_eq!((r.info.fast_key_ok, r.info.exact_fast), (meta[7] != 0.0, meta[8] != 0.0));
        assert_eq!(r.trace.len(), trace.rows(), "{name}: trace length");
        for (i, &(e, f)) in r.trace.iter().enumerate() {
            assert_eq!((e as f64, f), (trace[(i, 0)], trace[(i, 1)]), "{name}: trace {i}");
        }
        if fx.f.get("trace_time", name).is_some() {
            let tt = fx.m("trace_time", name);
            assert_eq!(r.trace_time.len(), tt.rows());
            for (i, &(t, e, f)) in r.trace_time.iter().enumerate() {
                assert_eq!((t, e as f64, f), (tt[(i, 0)], tt[(i, 1)], tt[(i, 2)]), "{name}: trace_time {i}");
            }
        }
    } else {
        assert!(!exact_only, "{name}: selection rust {:?} python {sel:?}", r.best_sel);
        // a tie-equivalent path is the mirror image step for step: the counters and the virtual clock follow Python's
        assert_eq!(
            (r.evals, r.info.iters, r.info.phases),
            (meta[0] as u64, meta[1] as u64, meta[2] as u64),
            "{name}: evals/iters/phases (tie branch)"
        );
        assert_eq!((r.time, r.info.t_construction), (meta[4], meta[5]), "{name}: virtual times (tie branch)");
        if fx.f.get("trace_time", name).is_some() {
            fx.m("trace_time", name);
        }
    }
    let verdict = if same { "EXACT" } else { "TIE-EQUIVALENT" };
    eprintln!(
        "{name}: {verdict} sel {:?} key {:?}; {} picks equal (worst pen rel {:e}, margin abs {:e}); picks with >=2 tied {multi}, with distinct tied penalties {distinct}",
        r.best_sel, r.best_key, picks.rows(), worst.0, worst.1
    );
    verdict
}

#[test]
fn det_parity_demo() {
    let fx = Fx::new();
    let mut ctx = ift_context("demo");
    let mut n_tie = 0;
    for k in 1..=6 {
        let r = ift_det(&mut ctx, k, &Budget::Unlimited).unwrap();
        n_tie += (compare_run(&fx, &format!("ift_det_demo_k{k}"), &r, false) == "TIE-EQUIVALENT") as usize;
    }
    eprintln!("det_parity_demo: {n_tie} TIE-EQUIVALENT of 6 runs");
    fx.finish("det_parity_demo", &["ift_det_demo_"]);
}

/// Demo room on Rust's own tables (margins of the near-singular cells differ by ~1e-9 relative; ties may flip).
#[test]
fn det_parity_demo_own_tables() {
    let fx = Fx::new();
    for k in 1..=6 {
        let mut ctx = fe_context("demo");
        let r = ift_det(&mut ctx, k, &Budget::Unlimited).unwrap();
        compare_run(&fx, &format!("ift_det_demo_k{k}"), &r, false);
    }
}

fn replay_run(fx: &Fx, name: &str, ctx: &mut IftContext) -> IftRun {
    let b = fx.v("budget", name);
    let (draws, k) = (fx.m("draws", name), b[1] as usize);
    let rows: Vec<[f64; 4]> = (0..draws.rows()).map(|i| [draws[(i, 0)], draws[(i, 1)], draws[(i, 2)], draws[(i, 3)]]).collect();
    let mut rp = Replay::new(&rows, &[]);
    let r = run_ift(ctx, k, &Budget::Ticks(b[0]), &mut TickClock::new(), &mut rp, true, 15, 7).unwrap();
    rp.assert_exhausted();
    eprintln!(
        "{name}: k {k}, budget {} ticks, {} draws consumed, phases {}, iters {}, evals {}",
        b[0],
        rp.consumed(),
        r.info.phases,
        r.info.iters,
        r.evals
    );
    r
}

#[test]
fn replay_parity_demo() {
    let fx = Fx::new();
    let mut ctx = ift_context("demo");
    let r = replay_run(&fx, "ift_replay_demo", &mut ctx);
    assert!(r.info.phases >= 3, "the record has at least 3 perturbation phases");
    compare_run(&fx, "ift_replay_demo", &r, true);
    fx.finish("replay_parity_demo", &["ift_replay_demo"]);
}

#[test]
fn min_count_parity_demo() {
    let fx = Fx::new();
    let mut ctx = ift_context("demo");
    let mc = ift_min_count(&mut ctx, None, &Budget::Unlimited).unwrap();
    let (meta, perk) = (fx.v("meta", "ift_min_count_demo"), fx.m("perk", "ift_min_count_demo"));
    let (c, p, m) = rec_key(&fx, "ift_min_count_demo");
    assert_eq!((mc.k, mc.found), (meta[0] as usize, meta[1] != 0.0));
    assert!(key_close3(&mc.key, c, p, m));
    assert_eq!(mc.per_k.len(), perk.rows());
    for (i, pk) in mc.per_k.iter().enumerate() {
        assert_eq!(pk.k as f64, perk[(i, 0)]);
        assert!(key_close3(&pk.key, perk[(i, 1)] as usize, perk[(i, 2)], perk[(i, 3)]), "per_k {i}: {:?}", pk.key);
        assert_eq!(pk.evals as f64, perk[(i, 4)], "per_k {i}: evals");
    }
    let sel = fx.usizes("sel", "ift_min_count_demo");
    eprintln!("min_count_parity_demo: {} sel rust {:?} python {sel:?}", if mc.sel == sel { "EXACT" } else { "TIE-EQUIVALENT" }, mc.sel);
    fx.finish("min_count_parity_demo", &["ift_min_count_demo"]);
}

/// `optimize_ift` on the demo room (Rust's own tables; the selection may be the mirror image of Python's).
#[test]
fn optimize_parity_demo() {
    let fx = Fx::new();
    let name = "ift_optimize_demo";
    let spec = demo_spec(10);
    let r = optimize_ift(&spec, 0.005, &Budget::Unlimited).unwrap();
    let hist = fx.m("hist", name);
    let steps: Vec<_> = r
        .history
        .iter()
        .filter_map(|h| if let HistoryStep::Ift { k, n_infeasible, penalty } = h { Some((*k, *n_infeasible, *penalty)) } else { None })
        .collect();
    let strict = erploc_testkit::strict();
    if strict {
        assert_eq!(steps.len(), hist.rows());
        for (i, &(k, ni, pen)) in steps.iter().enumerate() {
            assert_eq!((k as f64, ni as f64), (hist[(i, 0)], hist[(i, 1)]), "history {i}");
            assert!((pen - hist[(i, 2)]).abs() <= 1e-9 * hist[(i, 2)].abs().max(1.0), "history {i}: penalty {pen} vs {}", hist[(i, 2)]);
        }
    } else {
        // off the reference platform only the final key of the history is compared (the path to it may differ in tied steps)
        let (&(k, ni, pen), last) = (steps.last().unwrap(), hist.rows() - 1);
        assert_eq!((k as f64, ni as f64), (hist[(last, 0)], hist[(last, 1)]), "final history step");
        assert!((pen - hist[(last, 2)]).abs() <= 1e-7 * hist[(last, 2)].abs().max(1.0), "final penalty {pen} vs {}", hist[(last, 2)]);
    }
    let HistoryStep::Survey { ids, score, n_skip } = r.history.last().unwrap().clone() else { panic!("no survey step") };
    assert_eq!(r.history.len(), steps.len() + 1);
    let sel = fx.usizes("sel", name);
    let (pids, psv) = (fx.usizes("ids", name), fx.v("survey", name));
    assert_eq!(r.sel.len(), sel.len());
    assert!(r.stats.feasible.iter().all(|&f| f), "every cell feasible");
    let exact = strict && r.sel == sel;
    if exact {
        assert_eq!(ids, pids);
        assert!((score - psv[0]).abs() <= 1e-9 * psv[0].abs().max(1.0) && n_skip as f64 == psv[1], "survey {score} {n_skip} vs {psv:?}");
        let want: String = fx.v("csv", name).iter().map(|&x| x as u8 as char).collect();
        assert_eq!(report_csv(&r, &spec), want, "report_csv string");
    } else {
        fx.m("csv", name);
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|i| r.sel.contains(i)));
        assert!((score - psv[0]).abs() <= 1e-6 * psv[0].abs().max(1.0), "survey score {score} vs {}", psv[0]);
    }
    eprintln!(
        "optimize_parity_demo: {} sel rust {:?} python {sel:?}, survey rust {ids:?} python {pids:?}",
        if exact { "EXACT" } else { "TIE-EQUIVALENT" },
        r.sel
    );
    fx.finish("optimize_parity_demo", &[name]);
}

// ----------------------------------------------------------------------------------------------------------------------------
// zone statistics (`zn_`), partial rendering (`rp_`)
// ----------------------------------------------------------------------------------------------------------------------------
use erploc_core::jacobian::CamPose;
use erploc_core::markers::upright_pose;
use erploc_est::synth::SynthConfig;
use erploc_place::partial::render_partial;
use erploc_place::zone::{Grid, ZoneStats};

fn zone_matches(got: &ZoneStats, res: &[f64], mask: &[f64], sizes: &[f64], label: &str) {
    assert_eq!(
        [got.zone, got.total, got.outside, got.holes, got.holes_nonfloor, got.n_components].map(|v| v as f64),
        [res[0], res[1], res[2], res[3], res[4], res[5]],
        "{label}: zone statistics"
    );
    assert_eq!(got.mask.iter().map(|&b| b as u8 as f64).collect::<Vec<_>>(), mask, "{label}: zone mask");
    assert_eq!(sizes[0] as usize, got.component_sizes.len(), "{label}: component count");
    assert_eq!(got.component_sizes.iter().map(|&s| s as f64).collect::<Vec<_>>(), sizes[1..], "{label}: component sizes");
}

#[test]
fn zn_zone() {
    let fx = Fx::new();
    let (mut holes, mut nonfloor, mut multi) = (0, 0, 0);
    for i in 0..16 {
        let name = format!("zn_rand_{i}");
        let cells: Vec<V3> = fx.v("cells", &name).chunks(3).map(v3).collect();
        let g = Grid::new(&cells);
        let z = g.zone(&fx.bools("f", &name));
        zone_matches(&z, &fx.v("res", &name), &fx.v("mask", &name), &fx.v("sizes", &name), &name);
        holes += (z.holes > 0) as usize;
        nonfloor += (z.holes_nonfloor > 0) as usize;
        multi += (z.n_components > 1) as usize;
    }
    eprintln!("zn_zone: 16 random grids identical (holes>0 in {holes}, holes_nonfloor>0 in {nonfloor}, several components in {multi})");
    assert!(holes > 2 && nonfloor > 2 && multi > 2);
    fx.finish("zn_zone", &["zn_"]);
}

#[test]
fn rp_render() {
    let fx = Fx::new();
    let cams: Vec<CamPose> = fx.v("cams", "rp_inputs").chunks(4).map(|r| CamPose::euler([r[0], r[1], r[2]], r[3], 0.0, 0.0)).collect();
    let mks: Vec<_> = fx.v("markers", "rp_inputs").chunks(5).map(|r| upright_pose(r[0] as i64, &[r[1], r[2], r[3]], r[4])).collect();
    let c = fx.v("cfg", "rp_inputs");
    let cfg = SynthConfig {
        cam: ErpCamera::new(c[1] as usize, c[2] as usize, 1),
        a: c[0],
        sigma_px: 0.3,
        noise_model: NoiseModel::Tangent,
        seed: 0,
    };
    let vps = [None, Some(VisParams::new(70f64.to_radians(), 12.0))];
    let (mut worst, mut n_obs, mut n_cases) = (0.0f64, 0usize, 0usize);
    let mut by_clear = [0usize; 5];
    for kind in ["box", "pole", "two", "combo"] {
        let obs = fx.obstacles(&format!("rp_scene_{kind}"));
        for (vi, vp) in vps.iter().enumerate() {
            for mv in [0usize, 2, 4] {
                for dt in [0, 1] {
                    let name = format!("rp_{kind}_v{vi}_mv{mv}_d{dt}");
                    let det = |_: &CamPose, _: &erploc_core::markers::MarkerPose, clear: [bool; 4]| {
                        (clear.iter().filter(|&&c| c).count() >= 3, clear.map(|c| if c { 2.0 } else { 1.0 }))
                    };
                    let frames = render_partial(&cams, &mks, &cfg, &obs, vp.as_ref(), mv, if dt == 1 { Some(&det) } else { None });
                    let want = fx.m("obs", &name);
                    let got: Vec<(usize, &erploc_est::synth::Observation)> =
                        frames.iter().enumerate().flat_map(|(f, fr)| fr.iter().map(move |o| (f, o))).collect();
                    assert_eq!(got.len(), want.rows(), "{name}: observation count");
                    for (&(f, o), w) in got.iter().zip(want.as_slice().chunks(18)) {
                        assert_eq!((f as f64, o.id as f64), (w[0], w[1]), "{name}: frame / id");
                        for k in 0..4 {
                            assert_eq!(o.corner_mask[k] as u8 as f64, w[2 + k], "{name}: mask {k}");
                            assert_eq!(o.corner_sigma_scale[k].to_bits(), w[6 + k].to_bits(), "{name}: scale {k}");
                            for d in 0..2 {
                                let e = (o.uv[k][d] - w[10 + 2 * k + d]).abs();
                                assert!(e.is_finite());
                                worst = worst.max(e);
                            }
                        }
                        by_clear[o.corner_mask.iter().filter(|&&c| c).count()] += 1;
                        n_obs += 1;
                    }
                    n_cases += 1;
                }
            }
        }
    }
    eprintln!("rp_render: {n_cases} cases, {n_obs} observations, masks and scales exact, worst uv error {worst:.3e}, clear-corner histogram {by_clear:?}");
    assert!(worst <= if erploc_testkit::strict() { 1e-12 } else { 1e-9 }, "uv error {worst:e}");
    assert!(by_clear[..4].iter().all(|&c| c > 0));
    fx.finish("rp_render", &["rp_"]);
}
