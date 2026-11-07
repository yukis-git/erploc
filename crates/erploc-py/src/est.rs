//! Estimation: resection, bundle adjustment, its initialisation and the camera Fisher information.

use crate::conv::{self, cli_err, guarded, value_err, Camera, Pose};
use erploc_cli::error::CliError;
use erploc_cli::io;
use erploc_core::jacobian::CamPose;
use erploc_core::markers::{corners_world, MarkerPose};
use erploc_core::DMat;
use erploc_est::ba::bundle_adjust as lib_bundle_adjust;
use erploc_est::ba_init;
use erploc_est::ba_types::{BaProblem, BaState};
use erploc_est::fisher::{info_camera, sigma_pos as lib_sigma_pos, SIGMA_RTOL};
use erploc_est::resection::{resect, CamDof, ResectOptions};
use erploc_linalg::lu;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

// ------------------------------------------------------------------------------------------------ resection
/// Result of `resection`: the pose, its covariance, information matrix and fit statistics.
#[pyclass(frozen, module = "erploc")]
pub struct Resection {
    pose: CamPose,
    cov: DMat,
    info: DMat,
    rms_px: f64,
    n_markers: usize,
    converged: bool,
    dof: CamDof,
}

#[pymethods]
impl Resection {
    /// The estimated camera pose.
    #[getter]
    fn pose(&self) -> Pose {
        Pose { pose: self.pose }
    }
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        conv::arr1(py, self.pose.c.to_vec())
    }
    #[getter]
    fn yaw(&self) -> f64 {
        self.pose().euler().0
    }
    /// Covariance: 4x4 (x, y, z, yaw) for `dof="yaw"`, 6x6 (position, rotation vector) for `dof="full"`.
    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        conv::dmat_py(py, &self.cov)
    }
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        conv::dmat_py(py, &self.info)
    }
    /// `sqrt(trace(I^-1[:3, :3]))` [m]; `inf` if the information is singular.
    #[getter]
    fn sigma_pos(&self) -> f64 {
        lib_sigma_pos(&self.info, SIGMA_RTOL).unwrap_or(f64::NAN)
    }
    #[getter]
    fn rms_px(&self) -> f64 {
        self.rms_px
    }
    #[getter]
    fn n_markers(&self) -> usize {
        self.n_markers
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    #[getter]
    fn dof(&self) -> &'static str {
        conv::dof_name(self.dof)
    }
    fn __repr__(&self) -> String {
        format!(
            "Resection(position={:?}, n_markers={}, rms_px={:.4}, converged={})",
            self.pose.c,
            self.n_markers,
            self.rms_px,
            py_bool(self.converged)
        )
    }
}

/// Single-frame resection: camera position and orientation from observed marker corners and known marker poses.
#[pyfunction]
#[pyo3(signature = (observations, markers, camera, a, *, sigma_px=0.3, noise_model="tangent", loss="linear", huber_delta=1.5, dof="yaw", roll=0.0, pitch=0.0, x0=None))]
#[allow(clippy::too_many_arguments)]
pub fn resection(
    py: Python<'_>,
    observations: &Bound<'_, PyAny>,
    markers: &Bound<'_, PyDict>,
    camera: &Camera,
    a: f64,
    sigma_px: f64,
    noise_model: &str,
    loss: &str,
    huber_delta: f64,
    dof: &str,
    roll: f64,
    pitch: f64,
    x0: Option<PyRef<'_, Pose>>,
) -> PyResult<Resection> {
    let dets = conv::detections(observations, "observations")?;
    let map = conv::marker_map(markers)?;
    let (nm, dof, loss) = (conv::noise_model(noise_model)?, conv::dof(dof)?, conv::loss(loss, huber_delta)?);
    let x0 = x0.map(|p| conv::pose_in(&p));
    let cam_json = camera.json();
    let r = guarded(py, move || {
        let cam = io::camera(&cam_json)?;
        let a = io::within(a, io::A_RANGE, "a")?;
        let sigma_px = io::within(sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
        let (roll, pitch) = (io::finite(roll, "roll")?, io::finite(pitch, "pitch")?);
        if map.is_empty() {
            return Err(CliError::Input("markers is empty".into()));
        }
        let (obs, _dropped) = io::frame(&dets, &cam, "observations")?;
        if let Some(x) = &x0 {
            // a start exactly on a marker corner has a zero bearing and makes J^T J non-finite
            for m in map.values() {
                for (k, c) in corners_world(&m.p, &m.r, a).iter().enumerate() {
                    if (0..3).map(|i| (c[i] - x.c[i]).powi(2)).sum::<f64>().sqrt() < 1e-6 {
                        return Err(CliError::Input(format!("x0 coincides with corner {k} of marker {}", m.id)));
                    }
                }
            }
        }
        let opts = ResectOptions { x0, sigma_px, noise_model: nm, loss, dof };
        Ok(resect(&obs, &map, a, &cam, roll, pitch, &opts)?)
    })?;
    Ok(Resection { pose: r.cam, cov: r.cov, info: r.info, rms_px: r.rms_px, n_markers: r.n_markers, converged: r.converged, dof })
}

// ------------------------------------------------------------------------------------------------ bundle state
/// Camera poses (one per frame, `None` where a frame has no camera) and marker poses: a start or a result of a bundle adjustment.
#[pyclass(frozen, module = "erploc")]
#[derive(Clone)]
pub struct BundleState {
    cams: Vec<Option<CamPose>>,
    markers: BTreeMap<i64, MarkerPose>,
}

#[pymethods]
impl BundleState {
    #[new]
    fn new(cameras: &Bound<'_, PyAny>, markers: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cams = Vec::new();
        for c in cameras.try_iter()? {
            let c = c?;
            cams.push(if c.is_none() { None } else { Some(c.extract::<PyRef<'_, Pose>>()?.pose) });
        }
        Ok(BundleState { cams, markers: conv::marker_map(markers)? })
    }
    #[getter]
    fn cameras<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        conv::pose_list(py, &self.cams)
    }
    /// `{id: (position, rotation)}`.
    #[getter]
    fn markers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        conv::marker_dict(py, &self.markers)
    }
    fn __repr__(&self) -> String {
        format!("BundleState(cameras={}, markers={})", self.cams.len(), self.markers.len())
    }
}

/// Result of `bundle_adjust`.
#[pyclass(frozen, module = "erploc")]
pub struct Bundle {
    state: BundleState,
    cov_markers: BTreeMap<i64, DMat>,
    cov_cams: Vec<Option<DMat>>,
    info: DMat,
    cov: Option<DMat>,
    rms_px: f64,
    converged: bool,
    null_dim: usize,
    eig_min_ratio: f64,
    dropped: usize,
}

#[pymethods]
impl Bundle {
    /// The adjusted cameras and map; can be passed as `x0` of another `bundle_adjust`.
    #[getter]
    fn state(&self) -> BundleState {
        self.state.clone()
    }
    #[getter]
    fn cameras<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.state.cameras(py)
    }
    #[getter]
    fn markers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.state.markers(py)
    }
    /// `{id: (6, 6)}` covariance of each marker pose (position, rotation vector).
    #[getter]
    fn cov_markers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (id, m) in &self.cov_markers {
            d.set_item(*id, conv::dmat_py(py, m)?)?;
        }
        Ok(d)
    }
    /// One covariance per frame (4x4 for yaw, 6x6 for full), `None` for a frame without a camera.
    #[getter]
    fn cov_cams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: PyResult<Vec<Option<Bound<'py, PyArray2<f64>>>>> =
            self.cov_cams.iter().map(|m| m.as_ref().map(|m| conv::dmat_py(py, m)).transpose()).collect();
        PyList::new(py, items?)
    }
    /// Information matrix `J^T J` of all parameters (survey priors included).
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        conv::dmat_py(py, &self.info)
    }
    /// Dense inverse of `info`, `None` if it is singular.
    #[getter]
    fn cov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.cov.as_ref().map(|m| conv::dmat_py(py, m)).transpose()
    }
    #[getter]
    fn rms_px(&self) -> f64 {
        self.rms_px
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    /// Null-space dimension of the information without the survey priors (the gauge freedom: 6 for a free network).
    #[getter]
    fn null_dim(&self) -> usize {
        self.null_dim
    }
    /// Smallest over largest eigenvalue of the information with priors.
    #[getter]
    fn eig_min_ratio(&self) -> f64 {
        self.eig_min_ratio
    }
    /// Detections skipped because they were flagged not ok or had a negative id.
    #[getter]
    fn dropped(&self) -> usize {
        self.dropped
    }
    fn __repr__(&self) -> String {
        format!(
            "Bundle(cameras={}, markers={}, rms_px={:.4}, converged={})",
            self.state.cams.len(),
            self.state.markers.len(),
            self.rms_px,
            py_bool(self.converged)
        )
    }
}

struct Problem {
    problem: BaProblem,
    dropped: usize,
}

/// Validates and builds a bundle-adjustment problem. Same checks, in the same order, as `erploc ba`.
#[allow(clippy::too_many_arguments)]
fn build_problem(
    frames: &Bound<'_, PyAny>,
    camera: &Camera,
    a: f64,
    survey: &Bound<'_, PyDict>,
    rp: Option<&Bound<'_, PyAny>>,
    sigma_px: f64,
    noise_model: &str,
    dof: &str,
) -> PyResult<Box<dyn FnOnce() -> Result<Problem, CliError> + Send>> {
    let mut frame_dets = Vec::new();
    for f in frames.try_iter().map_err(|_| value_err("frames must be an iterable of lists of observations"))? {
        frame_dets.push(conv::detections(&f?, "each frame")?);
    }
    let rp: Vec<[f64; 2]> = match rp {
        Some(r) if !r.is_none() => {
            let (n, v) = conv::f64_rows(r, "rp", 2)?;
            (0..n).map(|i| [v[2 * i], v[2 * i + 1]]).collect()
        }
        _ => vec![[0.0, 0.0]; frame_dets.len()],
    };
    let mut surveyed_in = Vec::new();
    for (k, v) in survey.iter() {
        let id: i64 = k.extract().map_err(|_| value_err("survey_markers keys must be integer marker ids"))?;
        let bad = || value_err(format!("survey_markers[{id}] must be a pair (position (3,), sigma_ts)"));
        if v.len().map_err(|_| bad())? != 2 {
            return Err(bad());
        }
        let p = conv::arr3(&conv::f64s(&v.get_item(0)?, &format!("survey_markers[{id}] position"), &[3])?);
        let sigma: f64 = v.get_item(1)?.extract().map_err(|_| bad())?;
        surveyed_in.push((id, p, sigma));
    }
    let (nm, dof) = (conv::noise_model(noise_model)?, conv::dof(dof)?);
    let cam_json = camera.json();
    Ok(Box::new(move || {
        let cam = io::camera(&cam_json)?;
        let a = io::within(a, io::A_RANGE, "a")?;
        let sigma_px = io::within(sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
        if frame_dets.is_empty() || frame_dets.len() > io::MAX_FRAMES {
            return Err(CliError::Input(format!("frames needs 1..={} entries, got {}", io::MAX_FRAMES, frame_dets.len())));
        }
        if rp.len() != frame_dets.len() {
            return Err(CliError::Input(format!("rp has {} entries but frames has {}", rp.len(), frame_dets.len())));
        }
        for (f, r) in rp.iter().enumerate() {
            io::finite(r[0], &format!("rp[{f}] roll"))?;
            io::finite(r[1], &format!("rp[{f}] pitch"))?;
        }
        let mut frames = Vec::with_capacity(frame_dets.len());
        let mut dropped = 0;
        let mut observed = BTreeSet::new();
        for (f, fr) in frame_dets.iter().enumerate() {
            let (obs, d) = io::frame(fr, &cam, &format!("frames[{f}]"))?;
            dropped += d;
            observed.extend(obs.iter().map(|o| o.id));
            frames.push(obs);
        }
        io::check_ba_size(frames.len(), observed.len(), dof == CamDof::Full)?;
        let mut surveyed = BTreeMap::new();
        for (id, p, sig) in surveyed_in {
            if id < 0 {
                return Err(CliError::Input(format!("survey marker id must be >= 0, got {id}")));
            }
            let p = io::v3(&p, "survey position")?;
            let sig = io::within(sig, io::SIGMA_TS_RANGE, "survey sigma_ts")?;
            surveyed.insert(id, (p, sig));
            if !observed.contains(&id) {
                return Err(CliError::Input(format!("survey marker id {id} is not observed in any frame")));
            }
        }
        let problem = BaProblem { frames, rp, surveyed, a, cam_model: cam, sigma_px, noise_model: nm, cam_dof: dof };
        Ok(Problem { problem, dropped })
    }))
}

fn x0_state(x0: &BundleState) -> Result<BaState, CliError> {
    for (f, c) in x0.cams.iter().enumerate() {
        if let Some(c) = c {
            io::cam_pose(
                &io::CamIn { c: c.c, yaw: c.yaw, roll: c.roll, pitch: c.pitch, r: c.r_mat.map(|r| r.0) },
                &format!("x0.cameras[{f}]"),
            )?;
        }
    }
    Ok(BaState { cams: x0.cams.clone(), markers: x0.markers.clone() })
}

/// Bundle adjustment over a sequence of frames with surveyed markers fixing the gauge.
///
/// `frames`: one list of observations per frame. `survey_markers`: `{id: (position, sigma_ts)}` (at least two). `rp`:
/// known `(roll, pitch)` per frame, shape (n_frames, 2), default zeros (a hint only for `dof="full"`).
#[pyfunction]
#[pyo3(signature = (frames, camera, a, survey_markers, *, rp=None, sigma_px=0.3, noise_model="tangent", dof="yaw", loss="linear", huber_delta=1.5, max_iter=50, x0=None))]
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust(
    py: Python<'_>,
    frames: &Bound<'_, PyAny>,
    camera: &Camera,
    a: f64,
    survey_markers: &Bound<'_, PyDict>,
    rp: Option<&Bound<'_, PyAny>>,
    sigma_px: f64,
    noise_model: &str,
    dof: &str,
    loss: &str,
    huber_delta: f64,
    max_iter: usize,
    x0: Option<PyRef<'_, BundleState>>,
) -> PyResult<Bundle> {
    let build = build_problem(frames, camera, a, survey_markers, rp, sigma_px, noise_model, dof)?;
    let loss = conv::loss(loss, huber_delta)?;
    let x0 = x0.map(|s| s.clone());
    let res = guarded(py, move || {
        if max_iter == 0 || max_iter > 10_000 {
            return Err(CliError::Input(format!("max_iter must be in 1..=10000, got {max_iter}")));
        }
        let Problem { problem, dropped } = build()?;
        let x0 = x0.as_ref().map(x0_state).transpose()?;
        let r = lib_bundle_adjust(&problem, x0, loss, max_iter)?;
        let cov = lu::inv(&r.info).ok();
        Ok((r, dropped, cov))
    })?;
    let (r, dropped, cov) = res;
    Ok(Bundle {
        state: BundleState { cams: r.state.cams, markers: r.state.markers },
        cov_markers: r.cov_markers,
        cov_cams: r.cov_cams,
        info: r.info,
        cov,
        rms_px: r.rms_px,
        converged: r.converged,
        null_dim: r.gauge.null_dim_noprior,
        eig_min_ratio: r.gauge.eig_min_ratio,
        dropped,
    })
}

/// Initial cameras and map for a bundle adjustment (propagation from the surveyed markers by PnP and resection).
#[pyfunction]
#[pyo3(signature = (frames, camera, a, survey_markers, *, rp=None, sigma_px=0.3, noise_model="tangent", dof="yaw", refine_rounds=1))]
#[allow(clippy::too_many_arguments)]
pub fn init_bundle(
    py: Python<'_>,
    frames: &Bound<'_, PyAny>,
    camera: &Camera,
    a: f64,
    survey_markers: &Bound<'_, PyDict>,
    rp: Option<&Bound<'_, PyAny>>,
    sigma_px: f64,
    noise_model: &str,
    dof: &str,
    refine_rounds: usize,
) -> PyResult<BundleState> {
    let build = build_problem(frames, camera, a, survey_markers, rp, sigma_px, noise_model, dof)?;
    let st = guarded(py, move || {
        if refine_rounds > 100 {
            return Err(CliError::Input(format!("refine_rounds must be <= 100, got {refine_rounds}")));
        }
        let Problem { problem, .. } = build()?;
        Ok(ba_init::initialize(&problem, refine_rounds)?)
    })?;
    Ok(BundleState { cams: st.cams, markers: st.markers })
}

// ------------------------------------------------------------------------------------------------ Fisher information
/// Fisher information of a camera pose given the markers it sees: shape (4, 4) for `dof="yaw"` (x, y, z, yaw) or (6, 6) for
/// `dof="full"`. Every marker in `markers` is assumed fully visible (visibility is the caller's business; the placement
/// functions model it).
#[pyfunction]
#[pyo3(signature = (pose, markers, camera, a, *, sigma_px=0.3, noise_model="tangent", dof="yaw"))]
#[allow(clippy::too_many_arguments)]
pub fn fisher_camera<'py>(
    py: Python<'py>,
    pose: PyRef<'_, Pose>,
    markers: &Bound<'py, PyDict>,
    camera: &Camera,
    a: f64,
    sigma_px: f64,
    noise_model: &str,
    dof: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let map = conv::marker_map(markers)?;
    let (nm, dof) = (conv::noise_model(noise_model)?, conv::dof(dof)?);
    let cam = camera.erp().map_err(cli_err)?;
    let pose = conv::pose_in(&pose);
    let m = guarded(py, move || {
        let a = io::within(a, io::A_RANGE, "a")?;
        let sigma_px = io::within(sigma_px, io::SIGMA_PX_RANGE, "sigma_px")?;
        let list: Vec<MarkerPose> = map.values().copied().collect();
        Ok(info_camera(&pose, &list, a, &cam, sigma_px, nm, dof.axes()))
    })?;
    conv::dmat_py(py, &m)
}

/// Corner pixels `{id: (4, 2)}` (TL, TR, BR, BL) that a camera at `pose` would see for each marker of `markers`, noise-free. There is no
/// visibility test: markers behind the camera plane, seen from the back or hidden by an obstacle are returned like any other.
#[pyfunction]
pub fn predict_corners<'py>(
    py: Python<'py>,
    pose: PyRef<'_, Pose>,
    markers: &Bound<'py, PyDict>,
    camera: &Camera,
    a: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let map = conv::marker_map(markers)?;
    let cam = camera.erp().map_err(cli_err)?;
    let pose = conv::pose_in(&pose);
    let out = guarded(py, move || {
        let a = io::within(a, io::A_RANGE, "a")?;
        let mut out = Vec::with_capacity(map.len());
        for (id, mk) in &map {
            let lp = erploc_core::jacobian::predict_marker(&pose, mk, a);
            let lp: Vec<[f64; 2]> = lp.iter().map(|p| [erploc_core::erp::wrap_pi(p[0]), p[1]]).collect();
            let px = erploc_core::erp::lamphi_to_px(&lp, &cam);
            out.push((*id, px.iter().flatten().copied().collect::<Vec<f64>>()));
        }
        Ok(out)
    })?;
    let d = PyDict::new(py);
    for (id, v) in out {
        d.set_item(id, conv::arr2(py, 4, 2, v)?)?;
    }
    Ok(d)
}

/// `sqrt(trace(I^-1[:3, :3]))` of an information matrix whose first three rows and columns are the position; `inf` if singular.
#[pyfunction]
pub fn sigma_pos(py: Python<'_>, info: &Bound<'_, PyAny>) -> PyResult<f64> {
    let arr: numpy::PyArrayLikeDyn<'_, f64, numpy::AllowTypeChange> =
        info.extract().map_err(|_| pyo3::exceptions::PyTypeError::new_err("info must be a numeric square matrix"))?;
    let a = arr.as_array();
    let n = match a.shape() {
        [r, c] if r == c && *r >= 3 && *r <= 64 => *r,
        s => return Err(value_err(format!("info must be a square matrix of size 3..=64, got shape {s:?}"))),
    };
    let v: Vec<f64> = a.iter().copied().collect();
    guarded(py, move || {
        if v.iter().any(|x| !x.is_finite()) {
            return Err(CliError::Input("info must be finite".into()));
        }
        Ok(lib_sigma_pos(&DMat::from_rows(n, n, &v), SIGMA_RTOL)?)
    })
}
