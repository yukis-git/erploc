//! Malformed input: every case must end in a controlled error (exit code 3 or 4, never 70 and never a panic).
mod common;
use common::*;
use erploc_cli::{run_ba, run_place, run_resect, run_site, CliError, PlaceOpts};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

type Mutation = Box<dyn Fn(&mut Value)>;

struct Case {
    name: &'static str,
    base: &'static str,
    mutate: Mutation,
    code: i32,
}

fn case(name: &'static str, base: &'static str, code: i32, mutate: impl Fn(&mut Value) + 'static) -> Case {
    Case { name, base, mutate: Box::new(mutate), code }
}

fn run_any(base: &str, text: &str) -> Result<(), CliError> {
    if base.starts_with("resect") {
        run_resect(text).map(|_| ())
    } else if base.starts_with("ba_") {
        run_ba(text).map(|_| ())
    } else {
        run_place(text, &PlaceOpts::default()).map(|_| ())
    }
}

fn check(c: &Case, text: &str) {
    match run_any(c.base, text) {
        Ok(()) => panic!("{}: expected exit code {}, got success", c.name, c.code),
        Err(e) => {
            assert_ne!(e.exit_code(), 70, "{}: internal error {e}", c.name);
            assert!([3, 4].contains(&e.exit_code()), "{}: unexpected exit code {} ({e})", c.name, e.exit_code());
            assert_eq!(e.exit_code(), c.code, "{}: {e}", c.name);
        }
    }
}

fn cases() -> Vec<Case> {
    let r = "resect_yaw_0.json";
    let rf = "resect_full_0.json";
    let b = "ba_yaw_0.json";
    let p = "place_demo_0.json";
    let bad_rot = json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.001]]);
    let reflection = json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    let (bad_rot2, bad_rot3) = (bad_rot.clone(), bad_rot.clone());
    vec![
        // ---- JSON level
        // ---- size limit of the dense Hessian: 1100 yaw frames are 4400 unknowns, over the 16 million entry ceiling
        case("too many frames for the dense Hessian", b, 3, |v| {
            let (f0, r0) = (v["frames"][0].clone(), v["rp"][0].clone());
            v["frames"] = Value::Array(vec![f0; 1100]);
            v["rp"] = Value::Array(vec![r0; 1100]);
        }),
        case("wrong type a", r, 3, |v| v["a"] = json!("0.16")),
        case("missing camera", r, 3, |v| drop(v.as_object_mut().unwrap().remove("camera"))),
        case("missing markers", r, 3, |v| drop(v.as_object_mut().unwrap().remove("markers"))),
        case("missing detections", r, 3, |v| drop(v.as_object_mut().unwrap().remove("detections"))),
        case("unknown field", r, 3, |v| v["bogus"] = json!(1)),
        case("detections not an array", r, 3, |v| v["detections"] = json!({"id": 1})),
        case("a is null", r, 3, |v| v["a"] = Value::Null),
        // ---- detections
        case("corners length 3", r, 3, |v| drop(v["detections"][0]["corners"].as_array_mut().unwrap().pop())),
        case("corner with one coordinate", r, 3, |v| v["detections"][0]["corners"][1] = json!([5.0])),
        case("corner_valid length 5", r, 3, |v| v["detections"][0]["corner_valid"] = json!([1, 1, 1, 1, 1])),
        case("corner_valid length 3", r, 3, |v| v["detections"][0]["corner_valid"] = json!([1, 1, 1])),
        case("corner_valid value 2", r, 3, |v| v["detections"][0]["corner_valid"] = json!([1, 1, 2, 1])),
        case("corner_sigma_scale 0", r, 3, |v| v["detections"][0]["corner_sigma_scale"] = json!([1, 0, 1, 1])),
        case("corner_sigma_scale negative", r, 3, |v| v["detections"][0]["corner_sigma_scale"] = json!([1, 1, -2, 1])),
        case("corner_sigma_scale NaN (null)", r, 3, |v| v["detections"][0]["corner_sigma_scale"] = json!([1, 1, null, 1])),
        case("corner_sigma_scale length 3", r, 3, |v| v["detections"][0]["corner_sigma_scale"] = json!([1, 1, 1])),
        case("corner_sigma_scale bad on an INVALID corner", r, 3, |v| {
            v["detections"][0]["corner_valid"] = json!([1, 1, 0, 1]);
            v["detections"][0]["corner_sigma_scale"] = json!([1, 1, 0, 1]);
        }),
        case("valid corner with null uv", r, 3, |v| v["detections"][0]["corners"][0] = json!([null, null])),
        case("valid corner with half-null uv", r, 3, |v| v["detections"][0]["corners"][0] = json!([1.0, null])),
        case("valid corner pixel out of range", r, 3, |v| v["detections"][0]["corners"][0] = json!([1e300, 5.0])),
        case("ok is a string", r, 3, |v| v["detections"][0]["ok"] = json!("yes")),
        case("duplicate detection ids", r, 3, |v| {
            let d = v["detections"][0].clone();
            v["detections"].as_array_mut().unwrap().push(d);
        }),
        // ---- camera and scalars
        case("w != 2h", r, 3, |v| v["camera"]["w"] = json!(6500)),
        case("h = 0, w = 0", r, 3, |v| v["camera"] = json!({"w": 0, "h": 0})),
        case("camera model 0", r, 3, |v| v["camera"]["model"] = json!(0)),
        case("camera size beyond 2^20", r, 3, |v| v["camera"] = json!({"w": 4294967296u64, "h": 2147483648u64})),
        case("a = 0", r, 3, |v| v["a"] = json!(0.0)),
        case("a negative", r, 3, |v| v["a"] = json!(-0.16)),
        case("sigma_px = 0", r, 3, |v| v["sigma_px"] = json!(0)),
        case("sigma_px negative", r, 3, |v| v["sigma_px"] = json!(-1.0)),
        case("unknown noise_model", r, 3, |v| v["noise_model"] = json!("gaussian")),
        case("unknown loss", r, 3, |v| v["loss"] = json!("cauchy")),
        case("unknown dof", r, 3, |v| v["dof"] = json!("pitch")),
        case("huber_delta 0", r, 3, |v| {
            v["loss"] = json!("huber");
            v["huber_delta"] = json!(0.0);
        }),
        case("roll is infinity (written as null)", r, 3, |v| v["roll"] = json!(f64::INFINITY)),
        // ---- markers
        case("duplicate marker ids", r, 3, |v| {
            let m = v["markers"][0].clone();
            v["markers"].as_array_mut().unwrap().push(m);
        }),
        case("markers empty", r, 3, |v| v["markers"] = json!([])),
        case("marker p non-finite (null)", r, 3, |v| v["markers"][0]["p"][1] = Value::Null),
        case("marker p wrong length", r, 3, |v| v["markers"][0]["p"] = json!([1.0, 2.0])),
        case("marker rotation far from orthonormal", r, 3, move |v| v["markers"][0]["r"] = bad_rot.clone()),
        case("marker rotation is a reflection", r, 3, move |v| v["markers"][1]["r"] = reflection.clone()),
        case("marker id negative", r, 3, |v| v["markers"][0]["id"] = json!(-4)),
        case("marker r wrong shape", r, 3, |v| v["markers"][0]["r"] = json!([[1, 0, 0], [0, 1, 0]])),
        case("x0 c non-finite", r, 3, |v| v["x0"]["c"][0] = Value::Null),
        case("x0 rotation not orthonormal", rf, 3, move |v| v["x0"] = json!({"c": [0, 0, 1], "r": bad_rot2.clone()})),
        case("x0 yaw non-finite", r, 3, |v| v["x0"]["yaw"] = json!(f64::NAN)),
        // ---- library-level failures (exit 4, not 3)
        case("no marker known to the map", r, 4, |v| {
            for m in v["markers"].as_array_mut().unwrap() {
                let id = m["id"].as_i64().unwrap();
                m["id"] = json!(id + 1000);
            }
        }),
        case("every detection dropped", r, 4, |v| {
            for d in v["detections"].as_array_mut().unwrap() {
                d["ok"] = json!(false);
            }
        }),
        case("every observation has 0 valid corners", r, 4, |v| {
            for d in v["detections"].as_array_mut().unwrap() {
                d["corner_valid"] = json!([0, 0, 0, 0]);
            }
        }),
        // ---- ba
        case("ba: rp shorter than frames (M-8)", b, 3, |v| drop(v["rp"].as_array_mut().unwrap().pop())),
        case("ba: rp longer than frames", b, 3, |v| v["rp"].as_array_mut().unwrap().push(json!([0, 0]))),
        case("ba: rp NaN", b, 3, |v| v["rp"][2][0] = Value::Null),
        case("ba: empty frames", b, 3, |v| {
            v["frames"] = json!([]);
            v["rp"] = json!([]);
        }),
        case("ba: surveyed id never observed", b, 3, |v| v["surveyed"][0]["id"] = json!(9999)),
        case("ba: duplicate surveyed id", b, 3, |v| {
            let s = v["surveyed"][0].clone();
            v["surveyed"].as_array_mut().unwrap().push(s);
        }),
        case("ba: sigma_ts 0", b, 3, |v| v["surveyed"][0]["sigma_ts"] = json!(0.0)),
        case("ba: surveyed p null", b, 3, |v| v["surveyed"][0]["p"][2] = Value::Null),
        case("ba: max_iter 0", b, 3, |v| v["max_iter"] = json!(0)),
        case("ba: x0 with the wrong number of cameras (library Shape)", b, 3, |v| v["x0"] = json!({"cams": [null], "markers": []})),
        case("ba: duplicate id inside a frame", b, 3, |v| {
            let d = v["frames"][1][0].clone();
            v["frames"][1].as_array_mut().unwrap().push(d);
        }),
        case("ba: corner_sigma_scale 0", b, 3, |v| v["frames"][0][0]["corner_sigma_scale"] = json!([1, 0, 1, 1])),
        case("ba: one surveyed marker only (library error)", b, 4, |v| v["surveyed"].as_array_mut().unwrap().truncate(1)),
        case("ba: no surveyed marker (library error)", b, 4, |v| v["surveyed"] = json!([])),
        case(
            "ba: x0 marker rotation bad",
            b,
            3,
            move |v| v["x0"] = json!({"cams": v["rp"].as_array().unwrap().iter().map(|_| Value::Null).collect::<Vec<_>>(), "markers": [{"id": 0, "p": [0, 0, 0], "r": bad_rot3.clone()}]}),
        ),
        // ---- place
        case("place: n_max > N", p, 3, |v| v["n_max"] = json!(11)),
        case("place: n_max = 0", p, 3, |v| v["n_max"] = json!(0)),
        case("place: empty workspace polygon", p, 3, |v| v["workspace"]["poly"] = json!([])),
        case("place: 2-vertex workspace polygon", p, 3, |v| v["workspace"]["poly"] = json!([[0, 0], [5, 5]])),
        case("place: workspace vertex NaN", p, 3, |v| v["workspace"]["poly"][1][0] = Value::Null),
        case("place: spacing 0", p, 3, |v| v["workspace"]["spacing"] = json!(0)),
        case("place: spacing negative", p, 3, |v| v["workspace"]["spacing"] = json!(-0.5)),
        case("place: spacing so small the grid explodes", p, 3, |v| v["workspace"]["spacing"] = json!(1e-6)),
        case("place: no cell inside the polygon", p, 3, |v| v["workspace"]["poly"] = json!([[0, 0], [0.1, 0], [0.1, 0.1]])),
        case("place: z_cam empty", p, 3, |v| v["workspace"]["z_cam"] = json!([])),
        case("place: headings empty", p, 3, |v| v["workspace"]["headings"] = json!([])),
        case("place: no candidates", p, 3, |v| v["candidates"] = json!([])),
        case("place: candidate xy null", p, 3, |v| v["candidates"][0]["xy"][0] = Value::Null),
        case("place: obstacle with 2 vertices", p, 3, |v| v["obstacles"][0]["poly"] = json!([[3, 3], [4, 4]])),
        case("place: tau 0", p, 3, |v| v["tau"] = json!(0)),
        case("place: tau_loo 0", p, 3, |v| v["tau_loo"] = json!(0)),
        case("place: sigma_cap negative", p, 3, |v| v["sigma_cap"] = json!(-1)),
        case("place: psi_max 0", p, 3, |v| v["vis"]["psi_max"] = json!(0)),
        case("place: camera w != 2h", p, 3, |v| v["camera"]["w"] = json!(6000)),
        case("place: unknown noise_model", p, 3, |v| v["noise_model"] = json!("x")),
        case("place: sector NaN", p, 3, |v| v["vis"]["sectors"] = json!([{"lam_min": null, "lam_max": 1, "phi_min": 0, "phi_max": 1}])),
    ]
}

#[test]
fn malformed_table() {
    let all = cases();
    assert!(all.len() >= 40, "{} cases", all.len());
    for c in &all {
        let mut v = read_json(c.base);
        (c.mutate)(&mut v);
        check(c, &v.to_string());
    }
    eprintln!("malformed_table: {} cases", all.len());
}

#[test]
fn not_json_and_trailing_garbage() {
    for (name, text) in [
        ("empty", ""),
        ("brace", "{"),
        ("text", "hello"),
        ("array", "[1,2,3]"),
        ("number", "1"),
        ("null", "null"),
        ("trailing", "{} x"),
        ("nan literal", "{\"a\": NaN}"),
    ] {
        for base in ["resect", "ba_", "place"] {
            match run_any(base, text) {
                Err(e) => assert_eq!(e.exit_code(), 3, "{name}/{base}: {e}"),
                Ok(()) => panic!("{name}/{base}: accepted"),
            }
        }
        assert_eq!(run_site(text, Path::new("x")).unwrap_err().exit_code(), 3, "{name}/site");
    }
    // a number outside the f64 range is a parse error, not an infinity
    let text = read_text("resect_yaw_0.json");
    let huge = text.replacen("\"a\":0.16", "\"a\":1e999", 1);
    assert_ne!(huge, text);
    assert_eq!(run_resect(&huge).unwrap_err().exit_code(), 3);
    // sigma_ts options of place
    let spec = read_text("place_demo_0.json");
    for ts in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let e = run_place(&spec, &PlaceOpts { sigma_ts: ts, ..PlaceOpts::default() }).err().unwrap();
        assert_eq!(e.exit_code(), 3, "sigma_ts {ts}");
    }
    let e = run_place(&spec, &PlaceOpts { method: erploc_cli::Method::Ift, budget_ms: Some(0), ..PlaceOpts::default() }).err().unwrap();
    assert_eq!(e.exit_code(), 3);
}

// ----------------------------------------------------------------------------------------------- site
/// The synthetic site table written to a temporary file.
fn synthetic_table(name: &str) -> (PathBuf, Vec<u8>) {
    let bytes = erploc_place::site_synth::synthetic_site_bytes().unwrap();
    (tmp(name, &bytes), bytes)
}

fn tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("erploc_cli_test_{}_{name}", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn site_malformed() {
    let (small, bytes) = synthetic_table("malformed_site.bin");
    let n = erploc_place::site::SiteTables::from_bytes(&bytes).unwrap().cands.len();
    let q = |a: Value| json!({"mode": "centre", "crit": "N", "action": a}).to_string();
    let run = |text: &str| run_site(text, &small);
    let code = |text: String| run(&text).unwrap_err().exit_code();
    // k >= N, k = 0
    assert_eq!(code(q(json!({"det": {"k": n}}))), 3);
    assert_eq!(code(q(json!({"det": {"k": 0}}))), 3);
    assert_eq!(code(q(json!({"det": {"k": n + 5}}))), 3);
    assert_eq!(code(q(json!({"ift": {"k": n, "budget_ms": 100, "seed": 1}}))), 3);
    assert_eq!(code(q(json!({"ift": {"k": 3, "budget_ms": 0, "seed": 1}}))), 3);
    assert_eq!(code(q(json!({"min_count": {"k_max": 0}}))), 3);
    assert_eq!(code(q(json!({"min_count": {"k_max": n}}))), 3);
    // zone selections
    assert_eq!(code(q(json!({"zone": {"sel": []}}))), 3);
    assert_eq!(code(q(json!({"zone": {"sel": [n]}}))), 3);
    assert_eq!(code(q(json!({"zone": {"sel": [0, 0]}}))), 3);
    // unknown names, wrong types
    assert_eq!(code(json!({"mode": "centre", "crit": "R9", "action": {"det": {"k": 3}}}).to_string()), 3);
    assert_eq!(code(json!({"mode": "rays", "crit": "N", "action": {"det": {"k": 3}}}).to_string()), 3);
    assert_eq!(code(q(json!({"fly": {}}))), 3);
    assert_eq!(code(q(json!({"det": {"k": -1}}))), 3);
    assert_eq!(code(q(json!({"det": {"k": 3, "extra": 1}}))), 3);
    // bad magic / truncated tables: PlaceError::Site -> exit 4
    let ok_query = q(json!({"det": {"k": 3}}));
    let garbage = tmp("garbage.bin", b"this is not a site table, definitely not ERPSITE1");
    assert_eq!(run_site(&ok_query, &garbage).unwrap_err().exit_code(), 4);
    let trunc = tmp("trunc.bin", &bytes[..bytes.len() / 2]);
    assert_eq!(run_site(&ok_query, &trunc).unwrap_err().exit_code(), 4);
    let empty = tmp("empty.bin", b"");
    assert_eq!(run_site(&ok_query, &empty).unwrap_err().exit_code(), 4);
    let mut flipped = bytes.clone();
    flipped[0] ^= 0xff;
    let bad_magic = tmp("magic.bin", &flipped);
    assert_eq!(run_site(&ok_query, &bad_magic).unwrap_err().exit_code(), 4);
    // unreadable files: exit 5
    assert_eq!(run_site(&ok_query, Path::new("/nonexistent/erploc/site.bin")).unwrap_err().exit_code(), 5);
    for p in [small, garbage, trunc, empty, bad_magic] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn site_small_queries() {
    let (small, _) = synthetic_table("queries_site.bin");
    let t = |a: Value| {
        let text = json!({"mode": "centre", "crit": "N", "action": a}).to_string();
        let out: Value = serde_json::from_str(&run_site(&text, &small).unwrap()).unwrap();
        out
    };
    let det = t(json!({"det": {"k": 3}}));
    assert_eq!(det["action"], "det");
    assert_eq!(det["sel"].as_array().unwrap().len(), 3);
    assert!(det["feasible_fraction"].as_f64().unwrap() <= 1.0);
    let mc = t(json!({"min_count": {"k_max": 4}}));
    assert_eq!(mc["action"], "min_count");
    assert!(mc["per_k"].as_array().unwrap().len() <= 4);
    let sel = det["sel"].clone();
    let z = t(json!({"zone": {"sel": sel}}));
    assert!(z["zone"]["zone"].as_u64().unwrap() <= z["zone"]["total"].as_u64().unwrap());
    let ift = t(json!({"ift": {"k": 3, "budget_ms": 200, "seed": 3}}));
    assert_eq!(ift["sel"].as_array().unwrap().len(), 3);
    let _ = std::fs::remove_file(small);
}

// ----------------------------------------------------------------------------------------------- random mutation sweep
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Paths of every value in `v` (arrays and objects are descended), as JSON pointers.
fn paths(v: &Value, here: String, out: &mut Vec<String>) {
    out.push(here.clone());
    match v {
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                paths(x, format!("{here}/{i}"), out);
            }
        }
        Value::Object(m) => {
            for (k, x) in m {
                paths(x, format!("{here}/{k}"), out);
            }
        }
        _ => {}
    }
}

fn sweep(base: &str, iters: usize, seed: u64, tail: usize) {
    // ERPLOC_SWEEP_MULT=n multiplies the iteration counts and offsets the seed (manual hunting runs; the default is the committed set)
    let mult: usize = std::env::var("ERPLOC_SWEEP_MULT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let (iters, seed) = (iters * mult, seed + (mult as u64 - 1) * 1000);
    let orig = read_json(base);
    let mut all = Vec::new();
    paths(&orig, String::new(), &mut all);
    let all: Vec<String> = all.into_iter().filter(|p| !p.is_empty()).collect();
    let repl = [
        json!(null),
        json!(0),
        json!(-1),
        json!(1e300),
        json!(1e-300),
        json!("x"),
        json!(4294967296u64),
        json!([]),
        json!({}),
        json!(true),
        json!(0.5),
        json!(-0.0),
        json!(2),
        json!([1, 2, 3, 4, 5]),
    ];
    let mut rng = Xs(seed);
    let mut codes = std::collections::BTreeMap::new();
    for it in 0..iters {
        let mut v = orig.clone();
        // mutate up to 3 places; later pointers may no longer exist (then the mutation is skipped)
        for _ in 0..1 + (rng.next() % 3) as usize {
            let p = &all[(rng.next() as usize) % all.len()];
            if let Some(slot) = v.pointer_mut(p) {
                *slot = repl[(rng.next() as usize) % repl.len()].clone();
            }
        }
        let _ = tail;
        let text = v.to_string();
        let res = match std::panic::catch_unwind(|| run_any(base, &text)) {
            Ok(r) => r,
            Err(p) => {
                let _ = std::fs::write(std::env::temp_dir().join("erploc_cli_panic_input.json"), &text);
                eprintln!("PANIC input of {base} iteration {it}: {text}");
                std::panic::resume_unwind(p)
            }
        };
        match res {
            Ok(()) => *codes.entry(0).or_insert(0usize) += 1,
            Err(e) => {
                assert_ne!(e.exit_code(), 70, "{base} iteration {it}: internal error {e}\ninput: {text}");
                assert!([3, 4].contains(&e.exit_code()), "{base} iteration {it}: exit code {} ({e})", e.exit_code());
                *codes.entry(e.exit_code()).or_insert(0usize) += 1;
            }
        }
    }
    eprintln!("sweep {base}: {iters} mutations, exit codes {codes:?}");
}

#[test]
fn mutation_sweep_resect() {
    for (i, f) in ["resect_yaw_0.json", "resect_full_0.json", "resect_masked2_0.json", "resect_zerovalid_0.json"].into_iter().enumerate() {
        sweep(f, 400, 0x1234 + i as u64, 0);
    }
}

#[test]
fn mutation_sweep_ba_and_place() {
    // heavier runs: fewer iterations in debug
    sweep("ba_yaw_0.json", 25, 77, 0);
    sweep("place_demo_0.json", 60, 99, 0);
}
