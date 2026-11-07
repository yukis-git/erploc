//! Shared helpers of the CLI integration tests (test code: unwrap / indexing on fixture data is fine here).
#![allow(dead_code)]
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::{Mat, M3};
use erploc_est::lm::Loss;
use erploc_est::resection::CamDof;
use erploc_est::synth::Observation;
use serde_json::Value;
use std::collections::BTreeMap;

pub fn cli_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/cli").join(name)
}

pub fn read_text(name: &str) -> String {
    std::fs::read_to_string(cli_path(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

pub fn read_json(name: &str) -> Value {
    serde_json::from_str(&read_text(name)).unwrap()
}

pub fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

pub fn fs(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

pub fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(fs).collect()
}

pub fn flat(v: &Value) -> Vec<f64> {
    rows(v).into_iter().flatten().collect()
}

/// `null` -> NaN.
pub fn fnan(v: &Value) -> f64 {
    if v.is_null() {
        f64::NAN
    } else {
        f(v)
    }
}

pub fn m3(v: &Value) -> M3 {
    let r = rows(v);
    Mat([[r[0][0], r[0][1], r[0][2]], [r[1][0], r[1][1], r[1][2]], [r[2][0], r[2][1], r[2][2]]])
}

pub fn camera(v: &Value) -> ErpCamera {
    ErpCamera::new(v["w"].as_u64().unwrap() as usize, v["h"].as_u64().unwrap() as usize, v["model"].as_i64().unwrap_or(1) as i32)
}

pub fn noise(v: &Value) -> NoiseModel {
    if v.as_str() == Some("pixel") {
        NoiseModel::Pixel
    } else {
        NoiseModel::Tangent
    }
}

pub fn loss(inp: &Value) -> Loss {
    if inp["loss"].as_str() == Some("huber") {
        Loss::Huber { delta: inp["huber_delta"].as_f64().unwrap_or(1.5) }
    } else {
        Loss::Linear
    }
}

pub fn dof(v: &Value) -> CamDof {
    if v.as_str() == Some("full") {
        CamDof::Full
    } else {
        CamDof::Yaw
    }
}

pub fn markers(v: &Value) -> Vec<MarkerPose> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| MarkerPose { id: m["id"].as_i64().unwrap(), p: fs(&m["p"]).try_into().unwrap(), r: m3(&m["r"]) })
        .collect()
}

pub fn marker_map(v: &Value) -> BTreeMap<i64, MarkerPose> {
    markers(v).into_iter().map(|m| (m.id, m)).collect()
}

/// Detector entries -> observations (drops `ok == false` / `id < 0`).
pub fn observations(list: &Value) -> Vec<Observation> {
    let mut out = Vec::new();
    for d in list.as_array().unwrap() {
        if d["id"].as_i64().unwrap() < 0 || d["ok"].as_bool() == Some(false) {
            continue;
        }
        let c = d["corners"].as_array().unwrap();
        let mut uv = [[0.0; 2]; 4];
        for k in 0..4 {
            uv[k] = [fnan(&c[k][0]), fnan(&c[k][1])];
        }
        let mut ob = Observation::new(d["id"].as_i64().unwrap(), uv);
        if let Some(v) = d.get("corner_valid") {
            for (k, x) in v.as_array().unwrap().iter().enumerate() {
                ob.corner_mask[k] = x.as_i64() == Some(1) || x.as_bool() == Some(true);
            }
        }
        if let Some(s) = d.get("corner_sigma_scale") {
            for (k, x) in s.as_array().unwrap().iter().enumerate() {
                ob.corner_sigma_scale[k] = f(x);
            }
        }
        out.push(ob);
    }
    out
}

pub fn cam_pose(v: &Value) -> CamPose {
    let r_mat = v.get("r").filter(|r| !r.is_null()).map(m3);
    CamPose {
        c: fs(&v["c"]).try_into().unwrap(),
        yaw: v["yaw"].as_f64().unwrap_or(0.0),
        roll: v["roll"].as_f64().unwrap_or(0.0),
        pitch: v["pitch"].as_f64().unwrap_or(0.0),
        r_mat,
    }
}

pub fn bits_eq(case: &str, what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{case} {what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{case} {what}[{i}]: {x:e} vs {y:e}");
    }
}

fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Worst `|a-b| / max(1,|b|)`; NaN propagates.
pub fn err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).fold(0.0, |m, (x, y)| nmax(m, (x - y).abs() / y.abs().max(1.0)))
}

/// Worst `|a-b|` relative to the largest entry of `b`; NaN propagates.
pub fn mat_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let scale = b.iter().fold(0.0f64, |m, y| m.max(y.abs())).max(f64::MIN_POSITIVE);
    a.iter().zip(b).fold(0.0, |m, (x, y)| nmax(m, (x - y).abs() / scale))
}
