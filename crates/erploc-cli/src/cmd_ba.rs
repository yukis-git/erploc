//! `erploc ba`: bundle adjustment over detector frames with surveyed markers (Yaw or Full camera DoF).
use crate::error::CliError;
use crate::io::{self, CamIn, CamOut, CameraJson, DetectionJson, DofJson, LossJson, MarkerJson, NoiseJson};
use erploc_est::ba::bundle_adjust;
use erploc_est::ba_types::{BaProblem, BaState};
use erploc_est::resection::CamDof;
use erploc_linalg::lu;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct SurveyedJson {
    pub id: i64,
    pub p: [f64; 3],
    pub sigma_ts: f64,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct StateIn {
    pub cams: Vec<Option<CamIn>>,
    pub markers: Vec<MarkerJson>,
}

fn max_iter_default() -> usize {
    50
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct BaIn {
    pub camera: CameraJson,
    pub a: f64,
    pub sigma_px: f64,
    pub noise_model: NoiseJson,
    pub dof: DofJson,
    pub loss: Option<LossJson>,
    pub huber_delta: Option<f64>,
    #[serde(default = "max_iter_default")]
    pub max_iter: usize,
    /// Known `[roll, pitch]` per frame.
    pub rp: Vec<[f64; 2]>,
    pub surveyed: Vec<SurveyedJson>,
    pub frames: Vec<Vec<DetectionJson>>,
    pub x0: Option<StateIn>,
}

#[derive(Serialize, Debug)]
pub struct GaugeOut {
    pub null_dim_noprior: usize,
    pub eig_min_ratio: f64,
}

#[derive(Serialize, Debug)]
pub struct CovMarkerOut {
    pub id: i64,
    pub cov: Vec<Vec<f64>>,
}

#[derive(Serialize, Debug)]
pub struct BaOut {
    pub converged: bool,
    pub rms_px: f64,
    pub dropped: usize,
    pub gauge: GaugeOut,
    /// One entry per frame; `null` where the frame has no camera.
    pub cams: Vec<Option<CamOut>>,
    /// Sorted by id.
    pub markers: Vec<MarkerJson>,
    pub cov_markers: Vec<CovMarkerOut>,
    pub cov_cams: Vec<Option<Vec<Vec<f64>>>>,
    pub info: Vec<Vec<f64>>,
    /// Dense LU inverse of `info`; `null` if `info` is singular.
    pub cov: Option<Vec<Vec<f64>>>,
}

pub fn run_ba(input_json: &str) -> Result<String, CliError> {
    let inp: BaIn = io::parse(input_json)?;
    let cam = io::camera(&inp.camera)?;
    let a = io::within(inp.a, io::A_RANGE, "a")?;
    let sigma_px = io::within(inp.sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
    let loss = io::loss(inp.loss, inp.huber_delta)?;
    if inp.max_iter == 0 || inp.max_iter > 10_000 {
        return Err(CliError::Input(format!("max_iter must be in 1..=10000, got {}", inp.max_iter)));
    }
    if inp.frames.is_empty() || inp.frames.len() > io::MAX_FRAMES {
        return Err(CliError::Input(format!("frames needs 1..={} entries, got {}", io::MAX_FRAMES, inp.frames.len())));
    }
    if inp.rp.len() != inp.frames.len() {
        return Err(CliError::Input(format!("rp has {} entries but frames has {}", inp.rp.len(), inp.frames.len())));
    }
    let mut rp = Vec::with_capacity(inp.rp.len());
    for (f, r) in inp.rp.iter().enumerate() {
        rp.push([io::finite(r[0], &format!("rp[{f}] roll"))?, io::finite(r[1], &format!("rp[{f}] pitch"))?]);
    }
    let mut frames = Vec::with_capacity(inp.frames.len());
    let mut dropped = 0;
    let mut observed = BTreeSet::new();
    for (f, fr) in inp.frames.iter().enumerate() {
        let (obs, d) = io::frame(fr, &cam, &format!("frames[{f}]"))?;
        dropped += d;
        observed.extend(obs.iter().map(|o| o.id));
        frames.push(obs);
    }
    io::check_ba_size(frames.len(), observed.len(), inp.dof.get() == CamDof::Full)?;
    let mut surveyed = BTreeMap::new();
    for s in &inp.surveyed {
        if s.id < 0 {
            return Err(CliError::Input(format!("surveyed id must be >= 0, got {}", s.id)));
        }
        let p = io::v3(&s.p, "surveyed p")?;
        let sig = io::within(s.sigma_ts, io::SIGMA_TS_RANGE, "surveyed sigma_ts")?;
        if surveyed.insert(s.id, (p, sig)).is_some() {
            return Err(CliError::Input(format!("duplicate surveyed id {}", s.id)));
        }
        if !observed.contains(&s.id) {
            return Err(CliError::Input(format!("surveyed marker id {} is not observed in any frame", s.id)));
        }
    }
    let x0 = match &inp.x0 {
        Some(st) => {
            let mut cams = Vec::with_capacity(st.cams.len());
            for (f, c) in st.cams.iter().enumerate() {
                cams.push(match c {
                    Some(c) => Some(io::cam_pose(c, &format!("x0.cams[{f}]"))?),
                    None => None,
                });
            }
            Some(BaState { cams, markers: io::marker_map(&st.markers)? })
        }
        None => None,
    };
    let problem =
        BaProblem { frames, rp, surveyed, a, cam_model: cam, sigma_px, noise_model: inp.noise_model.get(), cam_dof: inp.dof.get() };
    let res = bundle_adjust(&problem, x0, loss, inp.max_iter)?;
    io::to_json(&BaOut {
        converged: res.converged,
        rms_px: res.rms_px,
        dropped,
        gauge: GaugeOut { null_dim_noprior: res.gauge.null_dim_noprior, eig_min_ratio: res.gauge.eig_min_ratio },
        cams: res.state.cams.iter().map(|c| c.as_ref().map(io::cam_out)).collect(),
        markers: res.state.markers.values().map(io::marker_out).collect(),
        cov_markers: res.cov_markers.iter().map(|(id, m)| CovMarkerOut { id: *id, cov: io::mat_rows(m) }).collect(),
        cov_cams: res.cov_cams.iter().map(|c| c.as_ref().map(io::mat_rows)).collect(),
        info: io::mat_rows(&res.info),
        cov: lu::inv(&res.info).ok().map(|m| io::mat_rows(&m)),
    })
}
