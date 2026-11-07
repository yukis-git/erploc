//! Round trip of the CLI against (a) the library called directly with the converted inputs (bit-identical) and (b) the Python
//! outputs stored next to the inputs (`fixtures/cli/*.expected.json`, written by `tools/export_cli_cases.py`).
mod common;
use common::*;
use erploc_cli::{run_ba, run_place, run_resect, PlaceOpts};
use erploc_est::ba::bundle_adjust;
use erploc_est::ba_types::BaProblem;
use erploc_est::resection::{resect, CamDof, ResectOptions};
use erploc_place::placement::{demo_spec, optimize, report_csv};
use serde_json::{json, Value};

fn wrap_pi(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    x - t * (x / t).round()
}

fn resect_direct(inp: &Value) -> erploc_est::resection::ResectionResult {
    let cam = camera(&inp["camera"]);
    let opts = ResectOptions {
        x0: inp.get("x0").filter(|x| !x.is_null()).map(cam_pose),
        sigma_px: f(&inp["sigma_px"]),
        noise_model: noise(&inp["noise_model"]),
        loss: loss(inp),
        dof: dof(&inp["dof"]),
    };
    resect(
        &observations(&inp["detections"]),
        &marker_map(&inp["markers"]),
        f(&inp["a"]),
        &cam,
        inp["roll"].as_f64().unwrap_or(0.0),
        inp["pitch"].as_f64().unwrap_or(0.0),
        &opts,
    )
    .unwrap()
}

fn check_resect_bits(case: &str, out: &Value, direct: &erploc_est::resection::ResectionResult) {
    bits_eq(case, "cam.c", &fs(&out["cam"]["c"]), &direct.cam.c);
    bits_eq(case, "cam.yaw", &[f(&out["cam"]["yaw"])], &[direct.cam.yaw]);
    let r = direct.cam.r();
    bits_eq(case, "cam.r", &flat(&out["cam"]["r"]), &r.0.iter().flatten().copied().collect::<Vec<_>>());
    bits_eq(case, "cov", &flat(&out["cov"]), direct.cov.as_slice());
    bits_eq(case, "info", &flat(&out["info"]), direct.info.as_slice());
    bits_eq(case, "rms_px", &[f(&out["rms_px"])], &[direct.rms_px]);
    assert_eq!(out["n_markers"].as_u64().unwrap() as usize, direct.n_markers, "{case}");
    assert_eq!(out["converged"].as_bool().unwrap(), direct.converged, "{case}");
}

const RESECT_CASES: [&str; 6] =
    ["resect_yaw_0", "resect_full_0", "resect_huber_0", "resect_masked2_0", "resect_zerovalid_0", "resect_pixel_0"];

#[test]
fn resect_roundtrip() {
    for case in RESECT_CASES {
        let text = read_text(&format!("{case}.json"));
        let inp: Value = serde_json::from_str(&text).unwrap();
        let out: Value = serde_json::from_str(&run_resect(&text).unwrap()).unwrap();
        let direct = resect_direct(&inp);
        check_resect_bits(case, &out, &direct);
        // Python reference, tolerances of the existing differential tests
        let exp = read_json(&format!("{case}.expected.json"));
        assert!(err(&fs(&out["cam"]["c"]), &fs(&exp["c"])) <= 1e-9 * lf(), "{case}: c");
        if inp["dof"] == "yaw" {
            assert!(err(&[f(&out["cam"]["yaw"])], &[f(&exp["yaw"])]) <= 1e-9 * lf(), "{case}: yaw");
        } else {
            assert!(err(&flat(&out["cam"]["r"]), &flat(&exp["r"])) <= 1e-9 * lf(), "{case}: R");
        }
        assert!(mat_rel(&flat(&out["cov"]), &flat(&exp["cov"])) <= 1e-8 * lf(), "{case}: cov");
        assert!(mat_rel(&flat(&out["info"]), &flat(&exp["info"])) <= 1e-8 * lf(), "{case}: info");
        assert!(err(&[f(&out["rms_px"])], &[f(&exp["rms_px"])]) <= 1e-9 * lf(), "{case}: rms_px");
        assert_eq!(out["n_markers"], exp["n_markers"], "{case}");
        assert_eq!(out["converged"], exp["converged"], "{case}");
        assert_eq!(out["dropped"], 0, "{case}");
    }
}

fn ba_problem(inp: &Value) -> BaProblem {
    BaProblem {
        frames: inp["frames"].as_array().unwrap().iter().map(observations).collect(),
        rp: rows(&inp["rp"]).into_iter().map(|r| [r[0], r[1]]).collect(),
        surveyed: inp["surveyed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["id"].as_i64().unwrap(), (fs(&s["p"]).try_into().unwrap(), f(&s["sigma_ts"]))))
            .collect(),
        a: f(&inp["a"]),
        cam_model: camera(&inp["camera"]),
        sigma_px: f(&inp["sigma_px"]),
        noise_model: noise(&inp["noise_model"]),
        cam_dof: dof(&inp["dof"]),
    }
}

/// Tolerance factor for the comparisons with the Python record: 1 on the reference platform, 100 elsewhere (see `lf` in
/// `erploc-est/tests/diff_python.rs`).
fn lf() -> f64 {
    if erploc_testkit::strict() {
        1.0
    } else {
        100.0
    }
}

const BA_CASES: [&str; 3] = ["ba_yaw_0", "ba_full_0", "ba_pixel_0"];

#[test]
fn ba_roundtrip() {
    for case in BA_CASES {
        let text = read_text(&format!("{case}.json"));
        let inp: Value = serde_json::from_str(&text).unwrap();
        let out: Value = serde_json::from_str(&run_ba(&text).unwrap()).unwrap();
        let problem = ba_problem(&inp);
        let direct = bundle_adjust(&problem, None, loss(&inp), inp["max_iter"].as_u64().unwrap() as usize).unwrap();
        // bit-identical to the direct call
        let cams = out["cams"].as_array().unwrap();
        assert_eq!(cams.len(), direct.state.cams.len(), "{case}");
        for (f_, (o, d)) in cams.iter().zip(&direct.state.cams).enumerate() {
            assert_eq!(o.is_null(), d.is_none(), "{case}: frame {f_}");
            if let Some(d) = d {
                bits_eq(case, "cam c", &fs(&o["c"]), &d.c);
                bits_eq(case, "cam r", &flat(&o["r"]), &d.r().0.iter().flatten().copied().collect::<Vec<_>>());
                bits_eq(case, "cam yaw", &[f(&o["yaw"])], &[d.yaw]);
            }
        }
        let mks = out["markers"].as_array().unwrap();
        assert_eq!(mks.len(), direct.state.markers.len(), "{case}");
        for (o, (id, d)) in mks.iter().zip(&direct.state.markers) {
            assert_eq!(o["id"].as_i64().unwrap(), *id);
            bits_eq(case, "marker p", &fs(&o["p"]), &d.p);
            bits_eq(case, "marker r", &flat(&o["r"]), &d.r.0.iter().flatten().copied().collect::<Vec<_>>());
        }
        bits_eq(case, "info", &flat(&out["info"]), direct.info.as_slice());
        bits_eq(case, "rms_px", &[f(&out["rms_px"])], &[direct.rms_px]);
        let cov = erploc_linalg::lu::inv(&direct.info).unwrap();
        bits_eq(case, "cov", &flat(&out["cov"]), cov.as_slice());
        for (o, (id, d)) in out["cov_markers"].as_array().unwrap().iter().zip(&direct.cov_markers) {
            assert_eq!(o["id"].as_i64().unwrap(), *id);
            bits_eq(case, "cov_markers", &flat(&o["cov"]), d.as_slice());
        }
        for (o, d) in out["cov_cams"].as_array().unwrap().iter().zip(&direct.cov_cams) {
            assert_eq!(o.is_null(), d.is_none());
            if let Some(d) = d {
                bits_eq(case, "cov_cams", &flat(o), d.as_slice());
            }
        }
        assert_eq!(out["gauge"]["null_dim_noprior"].as_u64().unwrap() as usize, direct.gauge.null_dim_noprior);
        bits_eq(case, "eig_min_ratio", &[f(&out["gauge"]["eig_min_ratio"])], &[direct.gauge.eig_min_ratio]);
        assert_eq!(out["converged"].as_bool().unwrap(), direct.converged);
        // Python reference, tolerances of `bundle_adjust_matches_python`
        let exp = read_json(&format!("{case}.expected.json"));
        let full = inp["dof"] == "full";
        let pc = rows(&exp["cams"]);
        for (f_, r) in pc.iter().enumerate() {
            assert_eq!(!cams[f_].is_null(), r[0] == 1.0, "{case}: frame {f_} presence");
            if r[0] == 1.0 {
                assert!(err(&fs(&cams[f_]["c"]), &r[1..4]) <= 1e-9 * lf(), "{case}: cam c");
                assert!(err(&flat(&cams[f_]["r"]), &r[5..14]) <= 1e-9 * lf(), "{case}: cam R");
                if !full {
                    assert!(wrap_pi(f(&cams[f_]["yaw"]) - r[4]).abs() <= 1e-9 * lf(), "{case}: cam yaw");
                }
            }
        }
        let pm = rows(&exp["mk"]);
        assert_eq!(pm.len(), mks.len(), "{case}");
        for r in &pm {
            let o = mks.iter().find(|m| m["id"].as_i64() == Some(r[0] as i64)).unwrap();
            assert!(err(&fs(&o["p"]), &r[1..4]) <= 1e-9 * lf(), "{case}: marker p");
            assert!(err(&flat(&o["r"]), &r[4..13]) <= 1e-9 * lf(), "{case}: marker R");
        }
        for r in rows(&exp["covm"]) {
            let o = out["cov_markers"].as_array().unwrap().iter().find(|m| m["id"].as_i64() == Some(r[0] as i64)).unwrap();
            assert!(mat_rel(&flat(&o["cov"]), &r[1..]) <= 1e-8 * lf(), "{case}: cov_markers");
        }
        let width = if full { 6 } else { 4 };
        let got_cc: Vec<f64> =
            out["cov_cams"].as_array().unwrap().iter().flat_map(|c| if c.is_null() { vec![0.0; width * width] } else { flat(c) }).collect();
        let want_cc: Vec<f64> = rows(&exp["covc"]).into_iter().flatten().collect();
        assert!(mat_rel(&got_cc, &want_cc) <= 1e-8 * lf(), "{case}: cov_cams");
        assert!(err(&[f(&out["rms_px"])], &[f(&exp["rms_px"])]) <= 1e-9 * lf(), "{case}: rms_px");
        let g = fs(&exp["gauge"]);
        assert_eq!(out["gauge"]["null_dim_noprior"].as_f64().unwrap(), g[0], "{case}");
        assert!((f(&out["gauge"]["eig_min_ratio"]) - g[1]).abs() <= 1e-12 * lf(), "{case}: eig_min_ratio");
        assert_eq!(out["converged"], exp["converged"], "{case}");
    }
}

#[test]
fn place_roundtrip() {
    let text = read_text("place_demo_0.json");
    let exp = read_json("place_demo_0.expected.json");
    let opts = PlaceOpts { sigma_ts: f(&exp["sigma_ts"]), ..PlaceOpts::default() };
    let out = run_place(&text, &opts).unwrap();
    let j: Value = serde_json::from_str(&out.json).unwrap();
    // bit-identical to the direct call on the library's own demo spec (equal to the fixture spec field by field: `pl_spec`)
    let spec = demo_spec(10);
    let direct = optimize(&spec, opts.sigma_ts).unwrap();
    let to_us = |v: &Value| -> Vec<usize> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect() };
    assert_eq!(to_us(&j["sel"]), direct.sel);
    assert_eq!(to_us(&j["survey_ids"]), direct.survey_ids);
    assert_eq!(out.csv, report_csv(&direct, &spec));
    bits_eq("place", "sigma", &fs(&j["stats"]["sigma"]), &direct.stats.sigma);
    bits_eq("place", "margin", &fs(&j["stats"]["margin"]), &direct.stats.margin);
    assert_eq!(j["history"].as_array().unwrap().len(), direct.history.len());
    // Python: counts exact, penalties 1e-9 relative, or the TIE-EQUIVALENT mirror rule (the keys agree, the mirror selection differs)
    let k = &j["key3"];
    let pk = &exp["key3"];
    assert_eq!(k["n_infeasible"], pk["n_infeasible"]);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
    assert!(rel(f(&k["penalty_sum"]), f(&pk["penalty_sum"])) <= 1e-9 * lf());
    // the margin is the third, tie-breaking part of the key: another libm may settle a tie of the first two parts on another layout
    if erploc_testkit::strict() {
        assert!(rel(f(&k["neg_margin"]), f(&pk["neg_margin"])) <= 1e-9);
    }
    if erploc_testkit::strict() && to_us(&j["sel"]) == to_us(&exp["sel"]) {
        assert_eq!(to_us(&j["survey_ids"]), to_us(&exp["survey_ids"]));
        assert_eq!(out.csv, exp["csv"].as_str().unwrap());
        let last = j["history"].as_array().unwrap().last().unwrap();
        assert!(rel(f(&last["score"]), f(&exp["score"])) <= 1e-9 * lf());
        assert_eq!(last["n_skip"], exp["n_skip"]);
    } else {
        eprintln!("TIE-EQUIVALENT: place_demo sel rust {:?} vs python {:?} (keys agree)", to_us(&j["sel"]), to_us(&exp["sel"]));
    }
    // the IFT method runs on the same input and reaches a feasible layout (deterministic: no budget)
    let ift = run_place(&text, &PlaceOpts { method: erploc_cli::Method::Ift, ..opts.clone() }).unwrap();
    let ij: Value = serde_json::from_str(&ift.json).unwrap();
    assert_eq!(ij["method"], "ift");
    assert_eq!(ij["key3"]["n_infeasible"], 0);
}

// ----------------------------------------------------------------------------------------------- JSON floats
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn json_floats_roundtrip() {
    let mut vals: Vec<f64> = vec![
        0.0,
        -0.0,
        1e-308,
        -1e-308,
        5e-324,
        2.2250738585072014e-308,
        2.225073858507201e-308,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        0.1,
        1.0 / 3.0,
        1.7976931348623157e308,
        9007199254740993.0,
        123456789.12345678,
    ];
    let mut rng = Xs(0x9E3779B97F4A7C15);
    while vals.len() < 30_000 {
        let v = f64::from_bits(rng.next());
        if v.is_finite() {
            vals.push(v);
        }
    }
    for _ in 0..10_000 {
        // decimal-looking values near the 17-digit boundary, where an inexact parser is off by one ulp
        vals.push((rng.next() % 1_000_000_000_000_000) as f64 / 1e15 * 1.7);
    }
    let text = serde_json::to_string(&vals).unwrap();
    let back: Vec<f64> = serde_json::from_str(&text).unwrap();
    assert_eq!(back.len(), vals.len());
    for (a, b) in vals.iter().zip(&back) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a:e} came back as {b:e}");
    }
    // also from a 17-significant-digit text of every value (not only from the shortest representation)
    let wide: Vec<String> = vals.iter().take(5000).map(|v| format!("{v:.16e}")).collect();
    for (v, s) in vals.iter().zip(&wide) {
        let w: f64 = serde_json::from_str(s).unwrap();
        assert_eq!(w.to_bits(), v.to_bits(), "{s}");
    }
    assert_eq!(serde_json::to_string(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]).unwrap(), "[null,null,null]");
    // the CLI writes non-finite numbers as null too
    assert_eq!(serde_json::to_string(&json!({"x": f64::NAN})).unwrap(), "{\"x\":null}");
}

// ----------------------------------------------------------------------------------------------- detector fields
#[test]
fn detector_fields() {
    let base = read_json("resect_yaw_0.json");
    let direct_with = |mutate: &dyn Fn(&mut erploc_est::synth::Observation)| {
        let mut obs = observations(&base["detections"]);
        mutate(&mut obs[0]);
        let opts = ResectOptions {
            x0: base.get("x0").map(cam_pose),
            sigma_px: f(&base["sigma_px"]),
            noise_model: noise(&base["noise_model"]),
            loss: loss(&base),
            dof: CamDof::Yaw,
        };
        resect(
            &obs,
            &marker_map(&base["markers"]),
            f(&base["a"]),
            &camera(&base["camera"]),
            base["roll"].as_f64().unwrap(),
            base["pitch"].as_f64().unwrap(),
            &opts,
        )
        .unwrap()
    };
    // corner_valid / corner_sigma_scale / null uv of the invalid corner
    let mut inp = base.clone();
    inp["detections"][0]["corner_valid"] = json!([1, 1, 0, 1]);
    inp["detections"][0]["corner_sigma_scale"] = json!([1, 1, 2.5, 1]);
    inp["detections"][0]["corners"][2] = json!([null, null]);
    let out: Value = serde_json::from_str(&run_resect(&inp.to_string()).unwrap()).unwrap();
    let want = direct_with(&|o| {
        o.corner_mask = [true, true, false, true];
        o.corner_sigma_scale = [1.0, 1.0, 2.5, 1.0];
        o.uv[2] = [f64::NAN, f64::NAN];
    });
    check_resect_bits("detector_fields mask", &out, &want);
    // the pixel of an invalid corner is never read: a finite value gives the same bits
    let mut inp2 = inp.clone();
    inp2["detections"][0]["corners"][2] = json!([123.0, -77.0]);
    let out2: Value = serde_json::from_str(&run_resect(&inp2.to_string()).unwrap()).unwrap();
    assert_eq!(out2["cov"], out["cov"]);
    assert_eq!(out2["cam"], out["cam"]);
    // ok == false and id < 0 entries are dropped and counted
    let mut inp3 = inp.clone();
    let extra_bad = json!({"id": 3, "corners": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], "ok": false});
    let extra_neg = json!({"id": -1, "corners": [[null, null], [null, null], [null, null], [null, null]]});
    inp3["detections"].as_array_mut().unwrap().push(extra_bad);
    inp3["detections"].as_array_mut().unwrap().push(extra_neg);
    let out3: Value = serde_json::from_str(&run_resect(&inp3.to_string()).unwrap()).unwrap();
    assert_eq!(out3["dropped"], 2);
    assert_eq!(out3["cam"], out["cam"]);
    assert_eq!(out3["info"], out["info"]);
    // a marker with 0 valid corners is accepted and ignored by the library (per-corner mask rule)
    let mut inp4 = base.clone();
    inp4["detections"][0]["corner_valid"] = json!([false, false, false, false]);
    let out4: Value = serde_json::from_str(&run_resect(&inp4.to_string()).unwrap()).unwrap();
    let want4 = direct_with(&|o| o.corner_mask = [false; 4]);
    check_resect_bits("detector_fields zero-valid", &out4, &want4);
    assert_eq!(out4["n_markers"].as_u64().unwrap(), f(&read_json("resect_yaw_0.expected.json")["n_markers"]) as u64 - 1);
}

#[test]
fn dropped_entries_in_ba() {
    let mut inp = read_json("ba_yaw_0.json");
    let base: Value = serde_json::from_str(&run_ba(&inp.to_string()).unwrap()).unwrap();
    inp["frames"][1].as_array_mut().unwrap().push(json!({"id": -5, "corners": [[0, 0], [0, 0], [0, 0], [0, 0]]}));
    inp["frames"][2].as_array_mut().unwrap().push(json!({"id": 1, "corners": [[0, 0], [0, 0], [0, 0], [0, 0]], "ok": false}));
    let out: Value = serde_json::from_str(&run_ba(&inp.to_string()).unwrap()).unwrap();
    assert_eq!(out["dropped"], 2);
    assert_eq!(out["info"], base["info"]);
}
