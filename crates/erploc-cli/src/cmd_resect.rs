//! `erploc resect`: single-frame resection (Yaw 4-dof or Full 6-dof) of one detector frame against known marker poses.
use crate::error::CliError;
use crate::io::{self, CamIn, CamOut, CameraJson, DetectionJson, DofJson, LossJson, MarkerJson, NoiseJson};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::corners_world;
use erploc_est::resection::{resect, ResectOptions};
use serde::{Deserialize, Serialize};

fn zero() -> f64 {
    0.0
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ResectIn {
    pub camera: CameraJson,
    pub a: f64,
    pub sigma_px: f64,
    pub noise_model: NoiseJson,
    pub loss: Option<LossJson>,
    pub huber_delta: Option<f64>,
    pub dof: DofJson,
    /// Known roll / pitch (Yaw) or start hints (Full).
    #[serde(default = "zero")]
    pub roll: f64,
    #[serde(default = "zero")]
    pub pitch: f64,
    pub x0: Option<CamIn>,
    pub markers: Vec<MarkerJson>,
    pub detections: Vec<DetectionJson>,
}

/// Field order matters for readability only: the summary comes first.
#[derive(Serialize, Debug)]
pub struct ResectOut {
    pub converged: bool,
    pub n_markers: usize,
    pub rms_px: f64,
    pub dropped: usize,
    pub cam: CamOut,
    pub cov: Vec<Vec<f64>>,
    pub info: Vec<Vec<f64>>,
}

pub fn run_resect(input_json: &str) -> Result<String, CliError> {
    let inp: ResectIn = io::parse(input_json)?;
    let cam = io::camera(&inp.camera)?;
    let a = io::within(inp.a, io::A_RANGE, "a")?;
    let sigma_px = io::within(inp.sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
    let (roll, pitch) = (io::finite(inp.roll, "roll")?, io::finite(inp.pitch, "pitch")?);
    let loss = io::loss(inp.loss, inp.huber_delta)?;
    if inp.markers.is_empty() {
        return Err(CliError::Input("markers is empty".into()));
    }
    let markers = io::marker_map(&inp.markers)?;
    let (obs, dropped) = io::frame(&inp.detections, &cam, "detections")?;
    let x0: Option<CamPose> = match &inp.x0 {
        Some(c) => Some(io::cam_pose(c, "x0")?),
        None => None,
    };
    if let Some(x) = &x0 {
        // the dense LM returns Err(NoConvergence) on a non-finite J^T J, which a start exactly on a marker corner (zero bearing) would violate
        for m in markers.values() {
            for (k, c) in corners_world(&m.p, &m.r, a).iter().enumerate() {
                if (0..3).map(|i| (c[i] - x.c[i]).powi(2)).sum::<f64>().sqrt() < 1e-6 {
                    return Err(CliError::Input(format!("x0 coincides with corner {k} of marker {}", m.id)));
                }
            }
        }
    }
    let opts = ResectOptions { x0, sigma_px, noise_model: inp.noise_model.get(), loss, dof: inp.dof.get() };
    let res = resect(&obs, &markers, a, &cam, roll, pitch, &opts)?;
    io::to_json(&ResectOut {
        converged: res.converged,
        n_markers: res.n_markers,
        rms_px: res.rms_px,
        dropped,
        cam: io::cam_out(&res.cam),
        cov: io::mat_rows(&res.cov),
        info: io::mat_rows(&res.info),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAW: &str = include_str!("../../../fixtures/cli/resect_yaw_0.json");
    const MASKED: &str = include_str!("../../../fixtures/cli/resect_masked2_0.json");

    fn value(text: &str) -> serde_json::Value {
        serde_json::from_str(text).expect("fixture/output is valid JSON")
    }

    #[test]
    fn resect_embedded_cases_converge() {
        for text in [YAW, MASKED] {
            let out = value(&run_resect(text).expect("embedded case runs"));
            assert_eq!(out["converged"], true);
            assert_eq!(out["dropped"], 0);
            assert_eq!(out["cov"].as_array().map(|a| a.len()), Some(4));
        }
    }

    #[test]
    fn resect_rejects_bad_input_with_exit_3() -> Result<(), String> {
        let mut v = value(YAW);
        v["a"] = serde_json::json!(0.0);
        assert_eq!(run_resect(&v.to_string()).err().map(|e| e.exit_code()), Some(3));
        let mut v = value(YAW);
        v["detections"][0]["corner_valid"] = serde_json::json!([1, 1]);
        assert_eq!(run_resect(&v.to_string()).err().map(|e| e.exit_code()), Some(3));
        // a start exactly on a marker corner is rejected before the library (zero bearing -> non-finite J^T J)
        let mut v = value(YAW);
        let m: MarkerJson = serde_json::from_value(v["markers"][0].clone()).map_err(|e| e.to_string())?;
        let pose = io::marker_pose(&m).map_err(|e| e.to_string())?;
        v["x0"]["c"] = serde_json::json!(corners_world(&pose.p, &pose.r, 0.16)[2]);
        assert_eq!(run_resect(&v.to_string()).err().map(|e| e.exit_code()), Some(3));
        assert_eq!(run_resect("not json").err().map(|e| e.exit_code()), Some(3));
        Ok(())
    }
}
