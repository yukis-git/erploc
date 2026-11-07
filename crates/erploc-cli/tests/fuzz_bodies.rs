//! Fuzz-target bodies as plain functions, driven by the in-tree `Pcg64`.
//!
//! Random smoke test, not coverage-guided: there is no `fuzz/` crate and no `cargo fuzz`. It gives no-panic / no-overflow /
//! termination evidence for the five target properties below on a fixed budget of random (and adversarially mixed) inputs, nothing
//! more: it cannot find inputs a coverage-guided fuzzer would reach.
//!
//! Inputs per target: 5000 in release (`FULL` tests, ignored in debug; override with the env var `ERPLOC_FUZZ_N`, e.g.
//! `ERPLOC_FUZZ_N=100000`; the slow `run_ba` / `run_place` targets take about 50 / 12 minutes at 100000), 200 in debug (`small` tests,
//! same code path; 4 for ba, 10 for place). Counts actually run are printed per target; cite those, not the default.
//!
//! Known non-panic behaviour of the library-direct `resect` (io-rejected inputs only): it returns `Ok` with a non-finite result in 202 of
//! 100000 inputs (158 + 44 in the direct table, at 100000 inputs); the validated path never does. `log_so3` returns non-finite output for a finite
//! non-rotation matrix (documented in `rot.rs`).
//! Every input runs under `catch_unwind`; a panic or a violated property is recorded (message, first minimal-looking input) and the
//! test fails at the end with the full table. A panic is a finding: the owning crate gets a regression test and the library or the
//! validation is fixed, never this harness.
//!
//! Targets:
//! 1. `resect_init` / `resect`, validated through the `erploc_cli::io` conversions (the exact path of `run_resect`), and the same
//!    inputs sent straight to the library; the direct variant records which panics are precondition panics (the input is rejected by
//!    `io`) and flags a VALIDATION GAP when `io` accepts an input that panics.
//! 2. `log_so3` (any 3x3 f64) with the round-trip property `exp_so3(log_so3(R)) = R` within 1e-9 for R within 1e-9 of SO(3).
//! 3. `segment_inside_intervals` / `point_in_polygon` (<= 8 vertices): intervals inside [0, 1], sorted, non-overlapping.
//! 4. `SiteTables::from_bytes` / `load` on mutated bytes of the synthetic site table: `Ok` or `Err(PlaceError::Site(_))`. The allocation
//!    bound (about 640x the file size) is NOT measured (the workspace forbids `unsafe`, so no counting allocator); it follows from
//!    the load's own size guards: every section length is checked against the bytes left before allocation, `unpack_bits` runs only
//!    after `take(ceil(KN/8))` succeeded, so `KN <= 8 * len` and the largest buffer `i10 = vec![0.0; KN * 10]` is `<= 640 * len`
//!    bytes. This harness bounds the input size (<= base + 8 KiB) and fires header attacks (huge K, N) that must fail cleanly.
//! 5. CLI JSON: garbled valid fixtures (text and value level) for `run_place`, `run_resect`, `run_ba`, `run_site`: `Ok` or
//!    `CliError::{Input, Numeric}`, never `Internal` / any other kind, never a panic. One restriction of the harness: for `site` the
//!    mutated `budget_ms` and `k`/`k_max` are clamped (<= 20 ms, <= 8) because a legitimately huge wall-clock budget or `k` is a
//!    long run, not a hang.
use erploc_cli::io::{self, CamIn, CameraJson, DetectionJson, Flag, LossJson, MarkerJson};
use erploc_cli::{run_ba, run_place, run_resect, run_site, CliError, Method, PlaceOpts};
use erploc_core::erp::{lamphi_to_px, ErpCamera, NoiseModel};
use erploc_core::jacobian::{predict_marker, CamPose};
use erploc_core::markers::{corners_world, MarkerPose};
use erploc_core::rot::{exp_so3, log_so3};
use erploc_core::{Mat, M3};
use erploc_est::lm::Loss;
use erploc_est::resection::{resect, resect_init, CamDof, ResectError, ResectOptions};
use erploc_est::synth::Observation;
use erploc_linalg::rng::Pcg64;
use erploc_place::placement::PlaceError;
use erploc_place::scene::{point_in_polygon, segment_inside_intervals};
use erploc_place::site::SiteTables;
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;

const FULL: u64 = 5_000;

/// Per-target count of the `FULL` tests: `ERPLOC_FUZZ_N` if set and parseable, else 5000.
fn full_n() -> u64 {
    std::env::var("ERPLOC_FUZZ_N").ok().and_then(|v| v.parse().ok()).unwrap_or(FULL)
}
const SMALL: u64 = 200;

// ------------------------------------------------------------------------------------------------ panic capture and tally
thread_local! {
    static IN_GUARD: Cell<bool> = const { Cell::new(false) };
    static LAST: RefCell<String> = const { RefCell::new(String::new()) };
}
static HOOK: Once = Once::new();

/// Silent hook while a guarded body runs (the message and location are kept); every other panic keeps the default output.
fn install_hook() {
    HOOK.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if IN_GUARD.with(|g| g.get()) {
                LAST.with(|l| *l.borrow_mut() = info.to_string());
            } else {
                prev(info);
            }
        }));
    });
}

/// `Err(message with location)` when the body panicked.
fn guarded<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    install_hook();
    IN_GUARD.with(|g| g.set(true));
    let r = catch_unwind(AssertUnwindSafe(f));
    IN_GUARD.with(|g| g.set(false));
    r.map_err(|p| {
        let m = LAST.with(|l| l.borrow().clone());
        if m.is_empty() {
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into())
        } else {
            m
        }
    })
}

fn clip(s: String) -> String {
    if s.len() > 1500 {
        let mut t: String = s.chars().take(1500).collect();
        t.push_str("...");
        t
    } else {
        s
    }
}

#[derive(Default)]
struct Tally {
    name: String,
    n: u64,
    classes: BTreeMap<String, u64>,
    /// message -> (count, first input)
    panics: BTreeMap<String, (u64, String)>,
    /// property violations (not panics): message -> (count, first input)
    violations: BTreeMap<String, (u64, String)>,
    /// panics that are precondition panics of the library-direct variant: message -> (count, first input)
    preconds: BTreeMap<String, (u64, String)>,
    secs: f64,
}

impl Tally {
    fn new(name: &str) -> Self {
        Tally { name: name.to_string(), ..Tally::default() }
    }
    fn class(&mut self, c: impl Into<String>) {
        *self.classes.entry(c.into()).or_insert(0) += 1;
    }
    fn add(map: &mut BTreeMap<String, (u64, String)>, msg: String, input: impl FnOnce() -> String) {
        map.entry(msg).and_modify(|e| e.0 += 1).or_insert_with(|| (1, clip(input())));
    }
    fn panic(&mut self, msg: String, input: impl FnOnce() -> String) {
        Tally::add(&mut self.panics, msg, input);
    }
    fn violation(&mut self, msg: String, input: impl FnOnce() -> String) {
        Tally::add(&mut self.violations, msg, input);
    }
    fn precond(&mut self, msg: String, input: impl FnOnce() -> String) {
        // numbers in the message are normalised so that one precondition is one row (first input kept as an example)
        let key: String = msg
            .split('\n')
            .enumerate()
            .map(|(i, l)| if i == 0 { l.to_string() } else { l.split(['[', '(']).next().unwrap_or("").to_string() })
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .map(|c| if c.is_ascii_digit() { '#' } else { c })
            .collect();
        Tally::add(&mut self.preconds, key, || input().chars().take(300).collect());
    }
    fn n_panics(&self) -> u64 {
        self.panics.values().map(|e| e.0).sum()
    }
    fn n_violations(&self) -> u64 {
        self.violations.values().map(|e| e.0).sum()
    }
    fn report(&self) {
        println!("[random smoke, not coverage-guided] target {}: {} inputs in {:.1} s", self.name, self.n, self.secs);
        for (c, k) in &self.classes {
            println!("    {c}: {k}");
        }
        let pre: u64 = self.preconds.values().map(|e| e.0).sum();
        println!("    precondition panics recorded (library-direct, io rejects the input): {pre}");
        for (m, (k, inp)) in &self.preconds {
            println!("      x{k}: {m}\n        first input: {inp}");
        }
        println!("    panics: {}   property violations: {}", self.n_panics(), self.n_violations());
        for (m, (k, inp)) in &self.panics {
            println!("      PANIC x{k}: {m}\n        first input: {inp}");
        }
        for (m, (k, inp)) in &self.violations {
            println!("      VIOLATION x{k}: {m}\n        first input: {inp}");
        }
    }
    /// Reports, then fails when anything was found.
    fn finish(self, t0: Instant) {
        let mut s = self;
        s.secs = t0.elapsed().as_secs_f64();
        s.report();
        assert_eq!(s.n_panics(), 0, "{}: panics found (see the table above)", s.name);
        assert_eq!(s.n_violations(), 0, "{}: property violations found (see the table above)", s.name);
    }
}

// ------------------------------------------------------------------------------------------------ random values
fn pick(rng: &mut Pcg64, n: u64) -> u64 {
    rng.next_u64() % n
}

fn chance(rng: &mut Pcg64, p: f64) -> bool {
    rng.uniform01() < p
}

/// An adversarial f64: NaN, +-inf, +-0, subnormals, huge, tiny, and ordinary values.
fn adv(rng: &mut Pcg64) -> f64 {
    match pick(rng, 18) {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => 0.0,
        4 => -0.0,
        5 => f64::MIN_POSITIVE / 2.0,
        6 => 5e-324,
        7 => 1e300,
        8 => -1e300,
        9 => f64::MAX,
        10 => f64::MIN,
        11 => 1e10,
        12 => 1e-300,
        13 => 1.0,
        14 => -1.0,
        15 => 1e6 + 1.0,
        16 => f64::from_bits(rng.next_u64()),
        _ => rng.normal() * 10.0,
    }
}

fn pick_vec3(rng: &mut Pcg64, lo: f64, hi: f64) -> [f64; 3] {
    [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi)]
}

/// A random rotation: Rodrigues of a random axis-angle vector with angle in [0, pi].
fn rand_rot(rng: &mut Pcg64) -> M3 {
    let w = pick_vec3(rng, -1.0, 1.0);
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1e-9);
    let ang = rng.uniform(0.0, std::f64::consts::PI);
    exp_so3(&[w[0] / n * ang, w[1] / n * ang, w[2] / n * ang])
}

fn max_abs_diff(a: &M3, b: &M3) -> f64 {
    let mut m = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a.0[i][j] - b.0[i][j]).abs());
        }
    }
    m
}

// ================================================================================================ target 1: resect
/// A raw resection problem in the CLI's own JSON field types (so the validated path is `io` itself).
#[derive(Clone, Debug)]
struct RawResect {
    camera: CameraJson,
    a: f64,
    sigma_px: f64,
    noise: NoiseModel,
    huber: Option<f64>,
    dof: CamDof,
    roll: f64,
    pitch: f64,
    x0: Option<CamIn>,
    markers: Vec<MarkerJson>,
    dets: Vec<DetectionJson>,
}

fn opt(x: f64) -> Option<f64> {
    if x.is_nan() {
        None
    } else {
        Some(x)
    }
}

fn gen_resect(rng: &mut Pcg64) -> RawResect {
    let h = [16u64, 64, 256, 512, 1024][pick(rng, 5) as usize];
    let mut camera = CameraJson { w: 2 * h, h, model: if chance(rng, 0.5) { 1 } else { -1 } };
    let a = [0.16, 0.4, 0.05][pick(rng, 3) as usize] * rng.uniform(0.5, 1.5);
    let sigma_px = rng.uniform(0.2, 3.0);
    let noise = if chance(rng, 0.5) { NoiseModel::Tangent } else { NoiseModel::Pixel };
    let huber = if chance(rng, 0.3) { Some(rng.uniform(0.5, 3.0)) } else { None };
    let dof = if chance(rng, 0.5) { CamDof::Yaw } else { CamDof::Full };
    let cam = CamPose::euler(
        [rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(0.5, 2.0)],
        rng.uniform(-3.0, 3.0),
        rng.normal() * 0.05,
        rng.normal() * 0.05,
    );
    let nm = pick(rng, 6) as usize; // 0 .. 5 markers (0: empty map)
    let mut markers: Vec<MarkerJson> = Vec::new();
    for i in 0..nm {
        let p = [rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0), rng.uniform(0.0, 3.0)];
        markers.push(MarkerJson { id: i as i64, p, r: rand_rot(rng).0 });
    }
    let consistent = chance(rng, 0.7);
    let ecam = ErpCamera::new(camera.w as usize, camera.h as usize, camera.model as i32);
    let mut dets = Vec::new();
    for m in &markers {
        let mp = MarkerPose { id: m.id, p: m.p, r: Mat(m.r) };
        let lp = predict_marker(&cam, &mp, a);
        let px = lamphi_to_px(&lp, &ecam);
        let mut corners = [[None; 2]; 4];
        for k in 0..4 {
            let (mut u, mut v) = (px[k][0], px[k][1]);
            if !consistent {
                u = rng.uniform(0.0, camera.w as f64);
                v = rng.uniform(0.0, camera.h as f64);
            } else {
                u += rng.normal();
                v += rng.normal();
            }
            corners[k] = [Some(u), Some(v)];
        }
        let masked = chance(rng, 0.2);
        let valid: Vec<Flag> = (0..4).map(|_| Flag::B(!masked || chance(rng, 0.6))).collect();
        let scale: Vec<Option<f64>> = (0..4).map(|_| Some(if chance(rng, 0.7) { 1.0 } else { rng.uniform(0.5, 4.0) })).collect();
        for k in 0..4 {
            if matches!(valid[k], Flag::B(false)) && chance(rng, 0.5) {
                corners[k] = [None, None];
            }
        }
        dets.push(DetectionJson { id: m.id, corners, corner_valid: Some(valid), corner_sigma_scale: Some(scale), ok: None });
    }
    if chance(rng, 0.15) {
        // an observation of an unknown marker
        dets.push(DetectionJson {
            id: 100 + pick(rng, 5) as i64,
            corners: [[Some(10.0), Some(10.0)]; 4],
            corner_valid: None,
            corner_sigma_scale: None,
            ok: None,
        });
    }
    let x0 = if chance(rng, 0.3) {
        let c = [cam.c[0] + rng.normal() * 0.2, cam.c[1] + rng.normal() * 0.2, cam.c[2] + rng.normal() * 0.1];
        Some(CamIn { c, yaw: cam.yaw + rng.normal() * 0.1, roll: cam.roll, pitch: cam.pitch, r: None })
    } else {
        None
    };
    let (mut roll, mut pitch) = (cam.roll, cam.pitch);
    let mut raw_a = a;
    let mut raw_sigma = sigma_px;
    let mut raw_huber = huber;
    // adversarial edits (35 % of the inputs get 1 .. 3 of them)
    let mut x0 = x0;
    if chance(rng, 0.35) {
        for _ in 0..1 + pick(rng, 3) {
            match pick(rng, 22) {
                0 => raw_a = adv(rng),
                1 => raw_sigma = adv(rng),
                2 => roll = adv(rng),
                3 => pitch = adv(rng),
                4 => raw_huber = Some(adv(rng)),
                5 => {
                    camera.w = pick(rng, 9);
                    camera.h = pick(rng, 5);
                }
                6 => camera.model = pick(rng, 5) as i64 - 2,
                7 => {
                    if !markers.is_empty() {
                        let i = pick(rng, markers.len() as u64) as usize;
                        markers[i].p[pick(rng, 3) as usize] = adv(rng);
                    }
                }
                8 => {
                    if !markers.is_empty() {
                        let i = pick(rng, markers.len() as u64) as usize;
                        markers[i].r[pick(rng, 3) as usize][pick(rng, 3) as usize] = adv(rng);
                    }
                }
                9 => {
                    // non-orthonormal rotation: small drift, scale, zero matrix, reflection
                    if !markers.is_empty() {
                        let i = pick(rng, markers.len() as u64) as usize;
                        let r = &mut markers[i].r;
                        match pick(rng, 4) {
                            0 => r[0][0] += rng.normal() * 10f64.powi(-(pick(rng, 8) as i32)),
                            1 => *r = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                            2 => *r = [[0.0; 3]; 3],
                            _ => *r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                        }
                    }
                }
                10 => {
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        dets[i].corners[pick(rng, 4) as usize][pick(rng, 2) as usize] = opt(adv(rng));
                    }
                }
                11 => {
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        dets[i].corner_valid = Some(vec![Flag::B(false); 4]);
                    }
                }
                12 => {
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        let mut s = dets[i].corner_sigma_scale.clone().unwrap_or_else(|| vec![Some(1.0); 4]);
                        s[pick(rng, 4) as usize] = opt(adv(rng));
                        dets[i].corner_sigma_scale = Some(s);
                    }
                }
                13 => {
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        dets[i].id = [-1i64, i64::MAX, i64::MIN, 0, 7][pick(rng, 5) as usize];
                    }
                }
                14 => {
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        let d = dets[i].clone();
                        dets.push(d); // duplicate id
                    }
                }
                15 => dets.clear(),
                16 => markers.clear(),
                17 => {
                    if let Some(x) = x0.as_mut() {
                        x.c[pick(rng, 3) as usize] = adv(rng);
                    } else {
                        x0 = Some(CamIn { c: [adv(rng), adv(rng), adv(rng)], yaw: adv(rng), roll: 0.0, pitch: 0.0, r: None });
                    }
                }
                18 => {
                    // start exactly on a marker corner
                    if let Some(m) = markers.first() {
                        let c = corners_world(&m.p, &Mat(m.r), a)[pick(rng, 4) as usize];
                        x0 = Some(CamIn { c, yaw: 0.0, roll: 0.0, pitch: 0.0, r: None });
                    }
                }
                19 => {
                    // a camera exactly on a marker corner (zero bearing) via the true camera centre of the observations is not
                    // reachable here; instead a start with a non-rotation matrix
                    x0 = Some(CamIn {
                        c: [0.0; 3],
                        yaw: 0.0,
                        roll: 0.0,
                        pitch: 0.0,
                        r: Some([[adv(rng), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                    });
                }
                20 => {
                    if !markers.is_empty() {
                        let i = pick(rng, markers.len() as u64) as usize;
                        markers[i].p = [1e6, -1e6, 1e6]; // extreme positions
                    }
                }
                _ => {
                    // huge-scale pixels
                    if !dets.is_empty() {
                        let i = pick(rng, dets.len() as u64) as usize;
                        dets[i].corners = [[Some(1e6), Some(-1e6)]; 4];
                    }
                }
            }
        }
    }
    RawResect { camera, a: raw_a, sigma_px: raw_sigma, noise, huber: raw_huber, dof, roll, pitch, x0, markers, dets }
}

/// The validated problem exactly as `run_resect` builds it from the same fields.
struct Valid {
    cam: ErpCamera,
    a: f64,
    sigma_px: f64,
    roll: f64,
    pitch: f64,
    loss: Loss,
    markers: std::collections::BTreeMap<i64, MarkerPose>,
    obs: Vec<Observation>,
    x0: Option<CamPose>,
}

/// Mirrors the validation of `cmd_resect::run_resect` step by step (same `io` functions, same x0-on-corner rejection).
fn validate_resect(raw: &RawResect) -> Result<Valid, CliError> {
    let cam = io::camera(&raw.camera)?;
    let a = io::within(raw.a, io::A_RANGE, "a")?;
    let sigma_px = io::within(raw.sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
    let (roll, pitch) = (io::finite(raw.roll, "roll")?, io::finite(raw.pitch, "pitch")?);
    let loss = io::loss(raw.huber.map(|_| LossJson::Huber), raw.huber)?;
    if raw.markers.is_empty() {
        return Err(CliError::Input("markers is empty".into()));
    }
    let markers = io::marker_map(&raw.markers)?;
    let (obs, _dropped) = io::frame(&raw.dets, &cam, "detections")?;
    let x0 = match &raw.x0 {
        Some(c) => Some(io::cam_pose(c, "x0")?),
        None => None,
    };
    if let Some(x) = &x0 {
        for m in markers.values() {
            for c in corners_world(&m.p, &m.r, a).iter() {
                if (0..3).map(|i| (c[i] - x.c[i]).powi(2)).sum::<f64>().sqrt() < 1e-6 {
                    return Err(CliError::Input("x0 coincides with a marker corner".into()));
                }
            }
        }
    }
    Ok(Valid { cam, a, sigma_px, roll, pitch, loss, markers, obs, x0 })
}

fn err_name(e: &ResectError) -> &'static str {
    match e {
        ResectError::NoKnownMarkers => "Err(NoKnownMarkers)",
        ResectError::NoForwardCandidate => "Err(NoForwardCandidate)",
        ResectError::Linalg(_) => "Err(Linalg)",
        ResectError::InvalidInput(_) => "Err(InvalidInput)",
    }
}

fn pose_finite(c: &CamPose) -> bool {
    c.c.iter().all(|x| x.is_finite())
        && c.yaw.is_finite()
        && c.roll.is_finite()
        && c.pitch.is_finite()
        && c.r().0.iter().flatten().all(|x| x.is_finite())
}

fn init_class(r: &Result<CamPose, ResectError>) -> String {
    match r {
        Ok(p) if pose_finite(p) => "resect_init Ok".into(),
        Ok(_) => "resect_init Ok(non-finite pose)".into(),
        Err(e) => format!("resect_init {}", err_name(e)),
    }
}

fn resect_class(r: &Result<erploc_est::resection::ResectionResult, ResectError>) -> String {
    match r {
        Ok(p) if pose_finite(&p.cam) && p.rms_px.is_finite() => "resect Ok".into(),
        Ok(_) => "resect Ok(non-finite result)".into(),
        Err(e) => format!("resect {}", err_name(e)),
    }
}

fn run_validated(raw: &RawResect, t: &mut Tally) {
    let v = match validate_resect(raw) {
        Ok(v) => v,
        Err(e) => {
            t.class(format!("rejected by io ({})", if e.exit_code() == 3 { "Input" } else { "other kind" }));
            return;
        }
    };
    t.class("accepted by io");
    let dof = raw.dof;
    match guarded(|| resect_init(&v.obs, &v.markers, v.a, &v.cam, v.roll, v.pitch, dof)) {
        Ok(r) => t.class(init_class(&r)),
        Err(m) => t.panic(format!("resect_init (validated): {m}"), || format!("{raw:?}")),
    }
    let opts = ResectOptions { x0: v.x0, sigma_px: v.sigma_px, noise_model: raw.noise, loss: v.loss, dof };
    match guarded(|| resect(&v.obs, &v.markers, v.a, &v.cam, v.roll, v.pitch, &opts)) {
        Ok(r) => t.class(resect_class(&r)),
        Err(m) => t.panic(format!("resect (validated): {m}"), || format!("{raw:?}")),
    }
}

/// The library called with the raw values, no validation at all (camera construction included). A panic on an input that `io`
/// rejects is a PRECONDITION panic (recorded); a panic on an input that `io` accepts is a validation gap (a violation).
fn run_direct(raw: &RawResect, t: &mut Tally) {
    let accepted = validate_resect(raw).is_ok();
    let res = guarded(|| {
        let cam = ErpCamera::new(raw.camera.w as usize, raw.camera.h as usize, raw.camera.model as i32);
        let mut markers = std::collections::BTreeMap::new();
        for m in &raw.markers {
            markers.insert(m.id, MarkerPose { id: m.id, p: m.p, r: Mat(m.r) });
        }
        let obs: Vec<Observation> = raw
            .dets
            .iter()
            .map(|d| {
                let mut uv = [[f64::NAN; 2]; 4];
                for (o, c) in uv.iter_mut().zip(&d.corners) {
                    *o = [c[0].unwrap_or(f64::NAN), c[1].unwrap_or(f64::NAN)];
                }
                let mut mask = [true; 4];
                if let Some(v) = &d.corner_valid {
                    for (m, f) in mask.iter_mut().zip(v) {
                        *m = matches!(f, Flag::B(true) | Flag::I(1));
                    }
                }
                let mut scale = [1.0; 4];
                if let Some(s) = &d.corner_sigma_scale {
                    for (o, x) in scale.iter_mut().zip(s) {
                        *o = x.unwrap_or(f64::NAN);
                    }
                }
                Observation { id: d.id, uv, corner_mask: mask, corner_sigma_scale: scale }
            })
            .collect();
        let init = resect_init(&obs, &markers, raw.a, &cam, raw.roll, raw.pitch, raw.dof);
        let x0 = raw.x0.as_ref().map(|c| CamPose { c: c.c, yaw: c.yaw, roll: c.roll, pitch: c.pitch, r_mat: c.r.map(Mat) });
        let loss = match raw.huber {
            Some(d) => Loss::Huber { delta: d },
            None => Loss::Linear,
        };
        let opts = ResectOptions { x0, sigma_px: raw.sigma_px, noise_model: raw.noise, loss, dof: raw.dof };
        let full = resect(&obs, &markers, raw.a, &cam, raw.roll, raw.pitch, &opts);
        (init_class(&init), resect_class(&full))
    });
    match res {
        Ok((a, b)) => {
            t.class(format!("direct: {a} / {b}"));
        }
        Err(m) => {
            if accepted {
                t.violation(format!("VALIDATION GAP (io accepts, library panics): {m}"), || format!("{raw:?}"));
            } else {
                t.class("direct: precondition panic (io rejects the input)");
                t.precond(m, || format!("{raw:?}"));
            }
        }
    }
}

fn resect_validated(n: u64) {
    let t0 = Instant::now();
    let mut rng = Pcg64::new(0xF022_0001);
    let mut t = Tally::new("resect_init/resect (validated through io)");
    for _ in 0..n {
        t.n += 1;
        let raw = gen_resect(&mut rng);
        run_validated(&raw, &mut t);
    }
    t.finish(t0);
}

fn resect_direct(n: u64) {
    let t0 = Instant::now();
    let mut rng = Pcg64::new(0xF022_0002);
    let mut t = Tally::new("resect_init/resect (library-direct, precondition panics recorded)");
    for _ in 0..n {
        t.n += 1;
        let raw = gen_resect(&mut rng);
        run_direct(&raw, &mut t);
    }
    t.finish(t0);
}

// ================================================================================================ target 2: log_so3
fn gen_log_so3(rng: &mut Pcg64) -> ([[f64; 3]; 3], bool) {
    let kind = pick(rng, 20);
    if kind < 7 {
        let mut m = [[0.0; 3]; 3];
        for row in m.iter_mut() {
            for x in row.iter_mut() {
                *x = adv(rng);
            }
        }
        return (m, false);
    }
    if kind < 10 {
        let mut m = [[0.0; 3]; 3];
        for row in m.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.normal();
            }
        }
        return (m, false);
    }
    // near SO(3): exp of a rotation vector, then a perturbation of at most 3e-10 per entry (Frobenius <= 9e-10 < 1e-9)
    let w = pick_vec3(rng, -1.0, 1.0);
    let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1e-12);
    let ang = match pick(rng, 7) {
        0 => rng.uniform(0.0, std::f64::consts::PI),
        1 => std::f64::consts::PI,
        2 => std::f64::consts::PI - 10f64.powi(-(1 + pick(rng, 14) as i32)),
        3 => 10f64.powi(-(1 + pick(rng, 14) as i32)),
        4 => 0.0,
        5 => rng.uniform(1.5, 1.7),
        _ => rng.uniform(0.0, 0.1),
    };
    let mut axis = [w[0] / n, w[1] / n, w[2] / n];
    if chance(rng, 0.2) {
        axis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][pick(rng, 3) as usize];
        // axis-aligned: diagonal-dominance branches
    }
    let mut r = exp_so3(&[axis[0] * ang, axis[1] * ang, axis[2] * ang]).0;
    if chance(rng, 0.7) {
        for row in r.iter_mut() {
            for x in row.iter_mut() {
                *x += rng.uniform(-3e-10, 3e-10);
            }
        }
    }
    (r, true)
}

fn log_so3_body(n: u64) {
    let t0 = Instant::now();
    let mut rng = Pcg64::new(0xF022_0003);
    let mut t = Tally::new("log_so3");
    for _ in 0..n {
        t.n += 1;
        let (m, near) = gen_log_so3(&mut rng);
        let rm = Mat(m);
        let finite_in = m.iter().flatten().all(|x| x.is_finite());
        let out = guarded(|| {
            let w = log_so3(&rm);
            let back = exp_so3(&w);
            (w, back)
        });
        match out {
            Err(msg) => t.panic(format!("log_so3/exp_so3: {msg}"), || format!("{m:?}")),
            Ok((w, back)) => {
                let w_fin = w.iter().all(|x| x.is_finite());
                if w_fin {
                    let nrm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                    if nrm > std::f64::consts::PI * (1.0 + 1e-12) {
                        t.violation(format!("|log_so3(R)| = {nrm} > pi"), || format!("{m:?}"));
                    }
                }
                match (finite_in, w_fin) {
                    (true, true) => t.class("finite in, finite out"),
                    (true, false) => t.class("finite in, non-finite out (not a rotation; not documented)"),
                    (false, true) => t.class("non-finite in, finite out"),
                    (false, false) => t.class("non-finite in, non-finite out"),
                }
                if near {
                    t.class("near-SO(3) round-trip checked");
                    let d = max_abs_diff(&back, &rm);
                    if !(w_fin && d <= 1e-9) {
                        t.violation(
                            format!("round trip exp(log R) differs from R: max |diff| = {d:e}, log finite = {w_fin} (limit 1e-9)"),
                            || format!("{m:?}"),
                        );
                    }
                }
            }
        }
    }
    t.finish(t0);
}

// ================================================================================================ target 3: segment / polygon
fn coord(rng: &mut Pcg64, mode: u64) -> f64 {
    match mode {
        0 => rng.uniform(-5.0, 5.0),
        1 => pick(rng, 5) as f64 - 2.0, // integer grid: collinear, touching and shared vertices
        _ => {
            if chance(rng, 0.5) {
                adv(rng)
            } else {
                rng.uniform(-5.0, 5.0)
            }
        }
    }
}

fn rand_pt(rng: &mut Pcg64, mode: u64) -> [f64; 2] {
    [coord(rng, mode), coord(rng, mode)]
}

fn interval_violation(iv: &[(f64, f64)]) -> Option<String> {
    let mut prev = 0.0f64;
    for (i, &(lo, hi)) in iv.iter().enumerate() {
        if !(lo.is_finite() && hi.is_finite()) {
            return Some(format!("interval {i} not finite: ({lo}, {hi})"));
        }
        if !(0.0 <= lo && lo <= hi && hi <= 1.0) {
            return Some(format!("interval {i} not inside [0, 1] or reversed: ({lo}, {hi})"));
        }
        if i > 0 && lo < prev {
            return Some(format!("interval {i} overlaps or is unsorted: ({lo}, {hi}) after end {prev}"));
        }
        prev = hi;
    }
    None
}

fn segment_body(n: u64) {
    let t0 = Instant::now();
    let mut rng = Pcg64::new(0xF022_0004);
    let mut t = Tally::new("segment_inside_intervals / point_in_polygon");
    for _ in 0..n {
        t.n += 1;
        let mode = pick(&mut rng, 3);
        let nv = pick(&mut rng, 9) as usize; // 0 .. 8 vertices
        let mut poly: Vec<[f64; 2]> = (0..nv).map(|_| rand_pt(&mut rng, mode)).collect();
        if nv > 2 && chance(&mut rng, 0.2) {
            poly[1] = poly[0]; // repeated vertex
        }
        let a = rand_pt(&mut rng, mode);
        let b = if chance(&mut rng, 0.1) {
            a
        } else if nv > 0 && chance(&mut rng, 0.15) {
            poly[pick(&mut rng, nv as u64) as usize] // endpoint on a vertex
        } else {
            rand_pt(&mut rng, mode)
        };
        let finite = a.iter().chain(b.iter()).chain(poly.iter().flatten()).all(|x| x.is_finite());
        let pts: Vec<[f64; 2]> = (0..pick(&mut rng, 5)).map(|_| rand_pt(&mut rng, mode)).collect();
        let out = guarded(|| (segment_inside_intervals(a, b, &poly), segment_inside_intervals(b, a, &poly), point_in_polygon(&pts, &poly)));
        match out {
            Err(m) => t.panic(m, || format!("a={a:?} b={b:?} poly={poly:?} pts={pts:?}")),
            Ok((iv, iv_rev, pip)) => {
                for (name, v) in [("forward", &iv), ("reversed", &iv_rev)] {
                    if let Some(msg) = interval_violation(v) {
                        t.violation(format!("{name}: {msg}"), || format!("a={a:?} b={b:?} poly={poly:?}"));
                    }
                }
                if pip.len() != pts.len() {
                    t.violation("point_in_polygon length differs from the number of points".into(), || format!("{pts:?} {poly:?}"));
                }
                t.class(format!("{} input, {} interval(s)", if finite { "finite" } else { "non-finite" }, iv.len().min(3)));
            }
        }
    }
    t.finish(t0);
}

// ================================================================================================ target 4: site load
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A string that differs between the test threads of one process (temporary file names).
fn thread_tag() -> String {
    format!("{:?}", std::thread::current().id()).chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

fn site_small() -> Vec<u8> {
    erploc_place::site_synth::synthetic_site_bytes().expect("synthetic site table")
}

fn u32_adv(rng: &mut Pcg64, len: usize) -> u32 {
    match pick(rng, 9) {
        0 => 0,
        1 => 1,
        2 => u32::MAX,
        3 => 0x8000_0000,
        4 => 0x7FFF_FFFF,
        5 => len as u32,
        6 => (len / 8) as u32,
        7 => 1 << 24,
        _ => rng.next_u64() as u32,
    }
}

/// Offset of the `K` field in the header of a valid file.
fn kn_offset(base: &[u8]) -> usize {
    let nlen = u32::from_le_bytes([base[12], base[13], base[14], base[15]]) as usize;
    // magic 8, version 4, nlen 4, numpy string, A f64, W u32, H u32, sigma f64, noise u32, tau f64, tau_loo f64, k_min u32,
    // 6 x f64 (sigma_cap, w_k, psi_max, px_min, local_r, spacing), z_cam f64
    8 + 4 + 4 + nlen + 8 + 4 + 4 + 8 + 4 + 8 + 8 + 4 + 8 * 6 + 8
}

fn mutate_bytes(base: &[u8], kn_at: usize, rng: &mut Pcg64) -> Vec<u8> {
    let mut b = base.to_vec();
    match pick(rng, 11) {
        0 => {
            for _ in 0..1 + pick(rng, 8) {
                let i = pick(rng, b.len() as u64) as usize;
                b[i] = rng.next_u64() as u8;
            }
        }
        1 => {
            let i = pick(rng, b.len() as u64) as usize;
            b[i] ^= 1 << pick(rng, 8);
        }
        2 => b.truncate(pick(rng, b.len() as u64) as usize),
        3 => {
            // adversarial u32 at a random offset inside the header region
            let i = pick(rng, 200) as usize;
            let v = u32_adv(rng, b.len()).to_le_bytes();
            b[i..i + 4].copy_from_slice(&v);
        }
        4 => {
            // adversarial f64 bit pattern at a random offset inside the header region
            let i = pick(rng, 200) as usize;
            let v = adv(rng).to_le_bytes();
            b[i..i + 8].copy_from_slice(&v);
        }
        5 => {
            // random bytes, half of them behind a valid magic + version
            let len = pick(rng, 4096) as usize;
            b = (0..len).map(|_| rng.next_u64() as u8).collect();
            if chance(rng, 0.5) && len >= 12 {
                b[..8].copy_from_slice(&base[..8]);
                b[8..12].copy_from_slice(&base[8..12]);
            }
        }
        6 => {
            let extra = 1 + pick(rng, 64) as usize;
            b.extend((0..extra).map(|_| rng.next_u64() as u8));
        }
        7 => {
            let i = pick(rng, b.len() as u64) as usize;
            let j = (i + 1 + pick(rng, 4096) as usize).min(b.len());
            b.drain(i..j);
        }
        8 => b.clear(),
        9 => {
            // the K, N counts attacked, file cut right after them half of the time
            let (k, n) = (u32_adv(rng, b.len()), u32_adv(rng, b.len()));
            b[kn_at..kn_at + 4].copy_from_slice(&k.to_le_bytes());
            b[kn_at + 4..kn_at + 8].copy_from_slice(&n.to_le_bytes());
            if chance(rng, 0.5) {
                b.truncate(kn_at + 8 + pick(rng, 64) as usize);
            }
        }
        _ => {
            // one K or N changed by +-1 / doubled (consistent-looking but wrong sizes)
            let off = kn_at + 4 * pick(rng, 2) as usize;
            let cur = u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]);
            let v = [cur.wrapping_add(1), cur.wrapping_sub(1), cur.wrapping_mul(2)][pick(rng, 3) as usize];
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    b
}

fn site_class(r: &Result<SiteTables, PlaceError>) -> Result<&'static str, String> {
    match r {
        Ok(_) => Ok("Ok"),
        Err(PlaceError::Site(_)) => Ok("Err(Site)"),
        Err(e) => Err(format!("returned {e:?}, which is not PlaceError::Site")),
    }
}

fn site_body(n: u64) {
    let t0 = Instant::now();
    let base = site_small();
    let kn_at = kn_offset(&base);
    let mut rng = Pcg64::new(0xF022_0005);
    let mut t = Tally::new("SiteTables::from_bytes / load on mutated synthetic site table");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fuzz_site_{}_{}.bin", std::process::id(), thread_tag()));
    let max_len = base.len() + 8192;
    for i in 0..n {
        t.n += 1;
        let bytes = mutate_bytes(&base, kn_at, &mut rng);
        if bytes.len() > max_len {
            t.violation(format!("harness bug: input longer than the bound {max_len}"), String::new);
            continue;
        }
        let repr = || format!("{} bytes, head {:02x?}", bytes.len(), &bytes[..bytes.len().min(48)]);
        match guarded(|| SiteTables::from_bytes(&bytes)) {
            Err(m) => t.panic(format!("from_bytes: {m}"), repr),
            Ok(r) => match site_class(&r) {
                Ok(c) => t.class(format!("from_bytes {c}")),
                Err(m) => t.violation(format!("from_bytes {m}"), repr),
            },
        }
        // the file path variant for one in 100 inputs, cross-checked against from_bytes
        if i % 100 == 0 {
            let wrote = std::fs::write(&tmp, &bytes).is_ok();
            if wrote {
                match guarded(|| (SiteTables::load(&tmp), SiteTables::from_bytes(&bytes))) {
                    Err(m) => t.panic(format!("load: {m}"), repr),
                    Ok((a, b)) => match site_class(&a) {
                        Ok(c) => {
                            t.class(format!("load (temp file) {c}"));
                            if c != site_class(&b).unwrap_or("?") {
                                t.violation("load and from_bytes disagree".into(), repr);
                            }
                        }
                        Err(m) => t.violation(format!("load {m}"), repr),
                    },
                }
            }
        }
    }
    // a missing file is an Err(Site), not a panic
    match guarded(|| SiteTables::load(&tmp.with_extension("absent"))) {
        Ok(Err(PlaceError::Site(_))) => t.class("load of a missing file: Err(Site)"),
        other => t.violation(format!("missing file: {:?}", other.map(|r| r.map(|_| ()))), String::new),
    }
    let _ = std::fs::remove_file(&tmp);
    t.finish(t0);
}

// ================================================================================================ target 5: CLI JSON
const TOKENS: [&str; 18] = [
    "null",
    "NaN",
    "Infinity",
    "-Infinity",
    "1e999",
    "-1e999",
    "[]",
    "{}",
    "\"\"",
    "0",
    "-0",
    "1e-320",
    "18446744073709551616",
    "-1",
    "[[",
    "]",
    ",",
    "\"a\":",
];

fn mutate_text(s: &str, rng: &mut Pcg64) -> String {
    let mut b = s.as_bytes().to_vec();
    for _ in 0..1 + pick(rng, 4) {
        if b.is_empty() {
            break;
        }
        let i = pick(rng, b.len() as u64) as usize;
        match pick(rng, 7) {
            0 => b[i] = rng.next_u64() as u8,
            1 => b[i] = b"{}[],:\"0123456789.-eE tfn"[pick(rng, 23) as usize],
            2 => {
                let j = (i + 1 + pick(rng, 40) as usize).min(b.len());
                b.drain(i..j);
            }
            3 => {
                let tok = TOKENS[pick(rng, TOKENS.len() as u64) as usize];
                let tail = b.split_off(i);
                b.extend_from_slice(tok.as_bytes());
                b.extend_from_slice(&tail);
            }
            4 => b.truncate(i),
            5 => {
                let j = (i + 1 + pick(rng, 60) as usize).min(b.len());
                let chunk = b[i..j].to_vec();
                let tail = b.split_off(j);
                b.extend_from_slice(&chunk);
                b.extend_from_slice(&tail);
            }
            _ => {
                let j = pick(rng, b.len() as u64) as usize;
                b.swap(i, j);
            }
        }
    }
    String::from_utf8_lossy(&b).into_owned()
}

fn adv_json(rng: &mut Pcg64) -> Value {
    const LITS: [&str; 24] = [
        "null",
        "true",
        "false",
        "\"x\"",
        "0",
        "-0.0",
        "5e-324",
        "1e300",
        "1.7976931348623157e308",
        "-1e300",
        "1e10",
        "1e6",
        "1e-9",
        "9007199254740993",
        "18446744073709551615",
        "-9223372036854775808",
        "-1",
        "3",
        "0.5",
        "[]",
        "{}",
        "[null,null]",
        "\"1e999\"",
        "1e15",
    ];
    serde_json::from_str(LITS[pick(rng, LITS.len() as u64) as usize]).unwrap_or(Value::Null)
}

fn mutate_value(v: &mut Value, rng: &mut Pcg64) {
    let stop = chance(rng, 0.25);
    match v {
        Value::Object(m) if !m.is_empty() && !stop => {
            let i = pick(rng, m.len() as u64) as usize;
            let Some(key) = m.keys().nth(i).cloned() else { return };
            if chance(rng, 0.1) {
                m.remove(&key);
            } else if let Some(c) = m.get_mut(&key) {
                mutate_value(c, rng);
            }
        }
        Value::Array(a) if !a.is_empty() && !stop => {
            let i = pick(rng, a.len() as u64) as usize;
            if chance(rng, 0.08) {
                a.remove(i);
            } else if chance(rng, 0.08) {
                let d = a[i].clone();
                a.insert(i, d);
            } else {
                mutate_value(&mut a[i], rng);
            }
        }
        Value::Array(a) => match pick(rng, 3) {
            0 => a.clear(),
            1 => a.push(Value::Null),
            _ => *v = adv_json(rng),
        },
        Value::Object(m) => {
            m.clear();
        }
        Value::Number(_) if chance(rng, 0.7) => *v = adv_json(rng),
        _ => *v = adv_json(rng),
    }
}

fn garble(base: &str, rng: &mut Pcg64) -> String {
    match pick(rng, 20) {
        0 => base.to_string(), // pristine control
        1..=5 => mutate_text(base, rng),
        _ => {
            let mut v: Value = serde_json::from_str(base).unwrap_or(Value::Null);
            for _ in 0..1 + pick(rng, 3) {
                mutate_value(&mut v, rng);
            }
            v.to_string()
        }
    }
}

fn cli_class(r: Result<String, CliError>) -> Result<String, String> {
    match r {
        Ok(s) => match serde_json::from_str::<Value>(&s) {
            Ok(_) => Ok("Ok".into()),
            Err(e) => Err(format!("Ok output is not valid JSON: {e}")),
        },
        Err(CliError::Input(_)) => Ok("Err(Input)".into()),
        Err(CliError::Numeric(_)) => Ok("Err(Numeric)".into()),
        Err(e) => Err(format!("forbidden error kind: {e:?}")),
    }
}

fn cli_loop(
    name: &str,
    seed: u64,
    n: u64,
    bases: &[String],
    mut run: impl FnMut(&str, &mut Pcg64, &mut String) -> Result<String, CliError>,
) {
    let t0 = Instant::now();
    let mut rng = Pcg64::new(seed);
    let mut t = Tally::new(name);
    for _ in 0..n {
        t.n += 1;
        let base = &bases[pick(&mut rng, bases.len() as u64) as usize];
        let text = garble(base, &mut rng);
        let opts = RefCell::new(String::new());
        let repr = || {
            let body = if text.len() > 1500 {
                format!("{}... ({} bytes)", text.chars().take(1500).collect::<String>(), text.len())
            } else {
                text.clone()
            };
            format!("harness options: [{}]\n{body}", opts.borrow())
        };
        match guarded(|| run(&text, &mut rng.clone(), &mut opts.borrow_mut())) {
            Err(m) => t.panic(m, repr),
            Ok(r) => match cli_class(r) {
                Ok(c) => t.class(c),
                Err(m) => t.violation(m, repr),
            },
        }
    }
    t.finish(t0);
}

fn read_cli(prefix: &str, names: &[&str]) -> Vec<String> {
    names.iter().map(|n| std::fs::read_to_string(fixtures_dir().join("cli").join(format!("{prefix}{n}.json"))).expect("fixture")).collect()
}

fn cli_resect_body(n: u64) {
    let bases = read_cli("resect_", &["yaw_0", "full_0", "huber_0", "masked2_0", "pixel_0", "zerovalid_0"]);
    cli_loop("cli JSON: run_resect", 0xF022_0006, n, &bases, |s, _, _| run_resect(s));
}

fn cli_ba_body(n: u64) {
    let bases = read_cli("ba_", &["yaw_0", "full_0", "pixel_0"]);
    cli_loop("cli JSON: run_ba", 0xF022_0007, n, &bases, |s, _, _| run_ba(s));
}

fn cli_place_body(n: u64) {
    let bases = read_cli("place_", &["demo_0"]);
    cli_loop("cli JSON: run_place", 0xF022_0008, n, &bases, |s, rng, note| {
        let opts = match pick(rng, 5) {
            0 => PlaceOpts { method: Method::Ift, budget_ms: Some(5), ..PlaceOpts::default() },
            1 => PlaceOpts { sigma_ts: adv(rng), ..PlaceOpts::default() },
            _ => PlaceOpts::default(),
        };
        *note = format!("{opts:?}");
        run_place(s, &opts).map(|o| o.json)
    });
}

/// Valid site queries (the site command has no fixture file); the mutations act on these.
fn site_queries() -> Vec<String> {
    let q = |m: &str, c: &str, a: &str| format!(r#"{{"mode":"{m}","crit":"{c}","action":{a}}}"#);
    vec![
        q("centre", "N", r#"{"min_count":{"k_max":4}}"#),
        q("centre", "K3", r#"{"det":{"k":3}}"#),
        q("samples", "Loo", r#"{"ift":{"k":3,"budget_ms":10,"seed":7}}"#),
        q("centre", "N", r#"{"zone":{"sel":[0,1,2,3,4,5]}}"#),
        q("samples", "Both", r#"{"zone":{"sel":[3,9,12,40]}}"#),
    ]
}

/// Compute bound of the harness (see the module doc): clamp `budget_ms` and `k` / `k_max` of a parsed query.
fn clamp_site(text: &str) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(text) else { return text.to_string() };
    if let Some(Value::Object(act)) = v.get_mut("action") {
        for (_, body) in act.iter_mut() {
            if let Value::Object(b) = body {
                for (key, cap) in [("budget_ms", 20u64), ("k", 8), ("k_max", 8)] {
                    if let Some(x) = b.get_mut(key) {
                        if x.as_u64().is_some_and(|u| u > cap) || x.as_f64().is_some_and(|f| f > cap as f64) {
                            *x = Value::from(cap);
                        }
                    }
                }
            }
        }
    }
    v.to_string()
}

fn cli_site_body(n: u64) {
    let bases = site_queries();
    let tables = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fuzz_cli_site_{}_{}.bin", std::process::id(), thread_tag()));
    std::fs::write(&tables, site_small()).expect("write the synthetic site table");
    cli_loop("cli JSON: run_site", 0xF022_0009, n, &bases, |s, _rng, note| {
        // always clamped (text that does not parse as JSON is returned unchanged and is rejected by run_site)
        let text = clamp_site(s);
        *note = format!("clamped query: {text}");
        run_site(&text, &tables)
    });
    let _ = std::fs::remove_file(&tables);
}

// ================================================================================================ tests
macro_rules! target {
    ($full:ident, $small:ident, $body:ident) => {
        target!($full, $small, $body, SMALL);
    };
    ($full:ident, $small:ident, $body:ident, $debug_n:expr) => {
        #[test]
        #[cfg_attr(debug_assertions, ignore = "100000 inputs: run with --release")]
        fn $full() {
            $body(full_n());
        }
        #[test]
        fn $small() {
            $body($debug_n);
        }
    };
}

target!(resect_validated_full, resect_validated_small, resect_validated);
target!(resect_direct_full, resect_direct_small, resect_direct);
target!(log_so3_full, log_so3_small, log_so3_body);
target!(segment_full, segment_small, segment_body);
target!(site_load_full, site_load_small, site_body);
target!(cli_resect_full, cli_resect_small, cli_resect_body);
target!(cli_ba_full, cli_ba_small, cli_ba_body, 4);
target!(cli_place_full, cli_place_small, cli_place_body, 10);
target!(cli_site_full, cli_site_small, cli_site_body);
