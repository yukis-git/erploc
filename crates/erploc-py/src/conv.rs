//! Conversions between Python objects and the validated library types, and the error mapping.
//!
//! All input validation reuses the checks of `erploc_cli::io` (the same obligations as the command line), so a value that
//! passed through here cannot reach a library `assert!`. The compute closures are additionally wrapped in `catch_unwind`
//! (`guarded`), which turns a panic that should not happen into a `RuntimeError` instead of a `PanicException`.

use erploc_cli::error::CliError;
use erploc_cli::io::{self, CamIn, CameraJson, DetectionJson, Flag, MarkerJson, NoiseJson};
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::MarkerPose;
use erploc_core::{DMat, M3};
use erploc_est::lm::Loss;
use erploc_est::resection::CamDof;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

create_exception!(
    erploc,
    NumericalError,
    PyRuntimeError,
    "A computation failed numerically (singular system, no convergence, no usable seed frame, ...)."
);

pub fn cli_err(e: CliError) -> PyErr {
    match e {
        CliError::Input(m) | CliError::Usage(m) => PyValueError::new_err(m),
        CliError::Numeric(m) => NumericalError::new_err(m),
        CliError::Io(m) => PyOSError::new_err(m),
        CliError::Internal(m) | CliError::CheckFailed(m) => PyRuntimeError::new_err(m),
    }
}

pub fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Runs `f` without the GIL; a panic becomes a `RuntimeError`.
pub fn guarded<T: Send>(py: Python<'_>, f: impl FnOnce() -> Result<T, CliError> + Send) -> PyResult<T> {
    let r = py.allow_threads(|| catch_unwind(AssertUnwindSafe(f)));
    match r {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(cli_err(e)),
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".into()
            };
            Err(PyRuntimeError::new_err(format!("internal error (this is a bug in erploc): {msg}")))
        }
    }
}

// ------------------------------------------------------------------------------------------------ arrays
/// A float array of exactly `shape` (lists, tuples and arrays of any numeric dtype are accepted), flattened row-major.
pub fn f64s(obj: &Bound<'_, PyAny>, what: &str, shape: &[usize]) -> PyResult<Vec<f64>> {
    let arr: PyArrayLikeDyn<'_, f64, AllowTypeChange> =
        obj.extract().map_err(|_| PyTypeError::new_err(format!("{what} must be a numeric array-like of shape {shape:?}")))?;
    let a = arr.as_array();
    if a.shape() != shape {
        return Err(value_err(format!("{what} must have shape {shape:?}, got {:?}", a.shape())));
    }
    Ok(a.iter().copied().collect())
}

/// A float array with `ncols` columns and any number of rows (rows, ncols).
pub fn f64_rows(obj: &Bound<'_, PyAny>, what: &str, ncols: usize) -> PyResult<(usize, Vec<f64>)> {
    let arr: PyArrayLikeDyn<'_, f64, AllowTypeChange> =
        obj.extract().map_err(|_| PyTypeError::new_err(format!("{what} must be a numeric array-like of shape (n, {ncols})")))?;
    let a = arr.as_array();
    match a.shape() {
        [n, c] if *c == ncols => Ok((*n, a.iter().copied().collect())),
        s => Err(value_err(format!("{what} must have shape (n, {ncols}), got {s:?}"))),
    }
}

pub fn arr1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

pub fn arr2<'py>(py: Python<'py>, rows: usize, cols: usize, v: Vec<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_vec(py, v).reshape([rows, cols])
}

pub fn dmat_py<'py>(py: Python<'py>, m: &DMat) -> PyResult<Bound<'py, PyArray2<f64>>> {
    arr2(py, m.rows(), m.cols(), m.as_slice().to_vec())
}

pub fn m3_py<'py>(py: Python<'py>, r: &M3) -> PyResult<Bound<'py, PyArray2<f64>>> {
    arr2(py, 3, 3, r.0.iter().flatten().copied().collect())
}

pub fn rows3(v: &[f64]) -> [[f64; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

pub fn arr3(v: &[f64]) -> [f64; 3] {
    [v[0], v[1], v[2]]
}

// ------------------------------------------------------------------------------------------------ options
pub fn noise(s: &str) -> PyResult<NoiseJson> {
    match s {
        "tangent" => Ok(NoiseJson::Tangent),
        "pixel" => Ok(NoiseJson::Pixel),
        _ => Err(value_err(format!("noise_model must be \"tangent\" or \"pixel\", got {s:?}"))),
    }
}

pub fn noise_model(s: &str) -> PyResult<NoiseModel> {
    Ok(noise(s)?.get())
}

pub fn dof(s: &str) -> PyResult<CamDof> {
    match s {
        "yaw" => Ok(CamDof::Yaw),
        "full" => Ok(CamDof::Full),
        _ => Err(value_err(format!("dof must be \"yaw\" or \"full\", got {s:?}"))),
    }
}

pub fn dof_name(d: CamDof) -> &'static str {
    match d {
        CamDof::Yaw => "yaw",
        CamDof::Full => "full",
    }
}

pub fn loss(s: &str, huber_delta: f64) -> PyResult<Loss> {
    match s {
        "linear" => Ok(Loss::Linear),
        "huber" => io::loss(Some(io::LossJson::Huber), Some(huber_delta)).map_err(cli_err),
        _ => Err(value_err(format!("loss must be \"linear\" or \"huber\", got {s:?}"))),
    }
}

// ------------------------------------------------------------------------------------------------ camera
#[pyclass(frozen, module = "erploc", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub w: u64,
    pub h: u64,
    pub lam_sign: i64,
}

#[pymethods]
impl Camera {
    /// An equirectangular camera of `width` x `height` pixels (`width == 2 * height`).
    #[new]
    #[pyo3(signature = (width, height=None, lam_sign=1))]
    fn new(width: u64, height: Option<u64>, lam_sign: i64) -> PyResult<Self> {
        let height = height.unwrap_or(width / 2);
        let c = Camera { w: width, h: height, lam_sign };
        c.erp().map_err(cli_err)?;
        Ok(c)
    }
    #[getter]
    fn width(&self) -> u64 {
        self.w
    }
    #[getter]
    fn height(&self) -> u64 {
        self.h
    }
    #[getter]
    fn lam_sign(&self) -> i64 {
        self.lam_sign
    }
    fn __repr__(&self) -> String {
        format!("Camera(width={}, height={}, lam_sign={})", self.w, self.h, self.lam_sign)
    }
}

impl Camera {
    pub fn erp(&self) -> Result<ErpCamera, CliError> {
        io::camera(&CameraJson { w: self.w, h: self.h, model: self.lam_sign })
    }
    pub fn json(&self) -> CameraJson {
        CameraJson { w: self.w, h: self.h, model: self.lam_sign }
    }
}

// ------------------------------------------------------------------------------------------------ pose
/// Euler angles (yaw, roll, pitch) of `R = Rz(yaw) Ry(pitch) Rx(roll)`.
pub fn euler_of(r: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    let pitch = (-r[2][0]).clamp(-1.0, 1.0).asin();
    if r[2][0].abs() > 1.0 - 1e-12 {
        ((-r[0][1]).atan2(r[1][1]), 0.0, pitch)
    } else {
        (r[1][0].atan2(r[0][0]), r[2][1].atan2(r[2][2]), pitch)
    }
}

/// A camera pose: position (3,), `rotation` (3, 3) = camera-to-world, and its Euler angles.
#[pyclass(frozen, module = "erploc")]
#[derive(Clone, Copy)]
pub struct Pose {
    pub pose: CamPose,
}

#[pymethods]
impl Pose {
    /// `Pose(position, yaw=0, roll=0, pitch=0, rotation=None)`: Euler `R = Rz(yaw) Ry(pitch) Rx(roll)`; a given `rotation` wins.
    #[new]
    #[pyo3(signature = (position, yaw=0.0, roll=0.0, pitch=0.0, rotation=None))]
    fn new(position: &Bound<'_, PyAny>, yaw: f64, roll: f64, pitch: f64, rotation: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let c = arr3(&f64s(position, "position", &[3])?);
        let r = match rotation {
            Some(r) if !r.is_none() => Some(rows3(&f64s(r, "rotation", &[3, 3])?)),
            _ => None,
        };
        let pose = io::cam_pose(&CamIn { c, yaw, roll, pitch, r }, "pose").map_err(cli_err)?;
        Ok(Pose { pose })
    }
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        arr1(py, self.pose.c.to_vec())
    }
    #[getter]
    fn yaw(&self) -> f64 {
        self.euler().0
    }
    #[getter]
    fn roll(&self) -> f64 {
        self.euler().1
    }
    #[getter]
    fn pitch(&self) -> f64 {
        self.euler().2
    }
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        m3_py(py, &self.pose.r())
    }
    fn __repr__(&self) -> String {
        let (yaw, roll, pitch) = self.euler();
        format!("Pose(position={:?}, yaw={yaw:.6}, roll={roll:.6}, pitch={pitch:.6})", self.pose.c)
    }
}

impl Pose {
    pub fn euler(&self) -> (f64, f64, f64) {
        match self.pose.r_mat {
            Some(r) => euler_of(&r.0),
            None => (self.pose.yaw, self.pose.roll, self.pose.pitch),
        }
    }
}

pub fn pose_in(p: &Pose) -> CamPose {
    p.pose
}

// ------------------------------------------------------------------------------------------------ markers
pub fn parse_marker(id: i64, v: &Bound<'_, PyAny>) -> PyResult<MarkerJson> {
    let bad = || value_err(format!("marker {id}: expected a pair (position (3,), rotation (3, 3))"));
    if v.len().map_err(|_| bad())? != 2 {
        return Err(bad());
    }
    let p = arr3(&f64s(&v.get_item(0)?, &format!("marker {id} position"), &[3])?);
    let r = rows3(&f64s(&v.get_item(1)?, &format!("marker {id} rotation"), &[3, 3])?);
    Ok(MarkerJson { id, p, r })
}

/// `{id: (position (3,), rotation (3, 3))}` -> validated marker map. Rotation columns are the marker x (right), y (up) and
/// normal (viewing side) axes in world coordinates.
pub fn marker_map(d: &Bound<'_, PyDict>) -> PyResult<BTreeMap<i64, MarkerPose>> {
    let mut list = Vec::new();
    for (k, v) in d.iter() {
        let id: i64 = k.extract().map_err(|_| value_err("marker ids must be integers"))?;
        list.push(parse_marker(id, &v)?);
    }
    io::marker_map(&list).map_err(cli_err)
}

pub fn marker_dict<'py>(py: Python<'py>, m: &BTreeMap<i64, MarkerPose>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (id, mk) in m {
        d.set_item(*id, (arr1(py, mk.p.to_vec()), m3_py(py, &mk.r)?))?;
    }
    Ok(d)
}

pub fn pose_list<'py>(py: Python<'py>, cams: &[Option<CamPose>]) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<Option<Pose>> = cams.iter().map(|c| c.map(|pose| Pose { pose })).collect();
    PyList::new(py, items)
}

// ------------------------------------------------------------------------------------------------ observations
/// One detected marker: `corners` (4, 2) ERP pixels (TL, TR, BR, BL), per-corner validity and sigma scale.
#[pyclass(frozen, module = "erploc")]
#[derive(Clone)]
pub struct Observation {
    pub id: i64,
    pub corners: [[f64; 2]; 4],
    pub valid: [bool; 4],
    pub scale: [f64; 4],
}

#[pymethods]
impl Observation {
    #[new]
    #[pyo3(signature = (id, corners, corner_valid=None, corner_sigma_scale=None))]
    fn new(
        id: i64,
        corners: &Bound<'_, PyAny>,
        corner_valid: Option<&Bound<'_, PyAny>>,
        corner_sigma_scale: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let c = f64s(corners, "corners", &[4, 2])?;
        let corners = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]];
        let mut valid = [true; 4];
        if let Some(v) = corner_valid.filter(|v| !v.is_none()) {
            valid = flags(&f64s(v, "corner_valid", &[4])?)?;
        }
        let mut scale = [1.0; 4];
        if let Some(s) = corner_sigma_scale.filter(|s| !s.is_none()) {
            scale.copy_from_slice(&f64s(s, "corner_sigma_scale", &[4])?);
        }
        Ok(Observation { id, corners, valid, scale })
    }
    #[getter]
    fn id(&self) -> i64 {
        self.id
    }
    #[getter]
    fn corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        arr2(py, 4, 2, self.corners.iter().flatten().copied().collect())
    }
    #[getter]
    fn corner_valid(&self) -> Vec<bool> {
        self.valid.to_vec()
    }
    #[getter]
    fn corner_sigma_scale<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        arr1(py, self.scale.to_vec())
    }
    fn __repr__(&self) -> String {
        format!("Observation(id={}, corner_valid={:?})", self.id, self.valid)
    }
}

fn flags(v: &[f64]) -> PyResult<[bool; 4]> {
    let mut out = [true; 4];
    for (o, x) in out.iter_mut().zip(v) {
        *o = if *x == 0.0 {
            false
        } else if *x == 1.0 {
            true
        } else {
            return Err(value_err(format!("corner_valid must hold booleans or 0/1, got {x}")));
        };
    }
    Ok(out)
}

fn field<'py>(o: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(d) = o.downcast::<PyDict>() {
        return Ok(d.get_item(name)?.filter(|v| !v.is_none()));
    }
    if o.hasattr(name)? {
        return Ok(Some(o.getattr(name)?).filter(|v| !v.is_none()));
    }
    if o.hasattr("keys")? && o.hasattr("__getitem__")? {
        return match o.get_item(name) {
            Ok(v) => Ok(Some(v).filter(|v| !v.is_none())),
            Err(e) if e.is_instance_of::<PyKeyError>(o.py()) => Ok(None),
            Err(e) => Err(e),
        };
    }
    Ok(None)
}

/// An observation-like object -> detection entry. Accepts `Observation`, an eqar `Marker`, or a dict with the keys `id`,
/// `corners`, and optionally `corner_valid`, `corner_sigma_scale`, `ok` (the layout of `eqar.Marker.to_dict()`).
pub fn detection(o: &Bound<'_, PyAny>) -> PyResult<DetectionJson> {
    if let Ok(ob) = o.extract::<PyRef<'_, Observation>>() {
        return Ok(DetectionJson {
            id: ob.id,
            corners: ob.corners.map(|c| c.map(|x| Some(x).filter(|x| x.is_finite()))),
            corner_valid: Some(ob.valid.iter().map(|b| Flag::B(*b)).collect()),
            corner_sigma_scale: Some(ob.scale.iter().map(|s| Some(*s)).collect()),
            ok: None,
        });
    }
    let missing =
        |n: &str| PyTypeError::new_err(format!("observation has no `{n}` (expected an erploc.Observation, an eqar.Marker or a dict)"));
    let id: i64 = field(o, "id")?.ok_or_else(|| missing("id"))?.extract().map_err(|_| value_err("observation id must be an integer"))?;
    let c = f64s(&field(o, "corners")?.ok_or_else(|| missing("corners"))?, "corners", &[4, 2])?;
    let corners = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]].map(|p: [f64; 2]| p.map(|x| Some(x).filter(|x| x.is_finite())));
    let corner_valid = match field(o, "corner_valid")? {
        Some(v) => Some(f64s(&v, "corner_valid", &[4])?.into_iter().map(Flag::F).collect()),
        None => None,
    };
    let corner_sigma_scale = match field(o, "corner_sigma_scale")? {
        Some(s) => Some(f64s(&s, "corner_sigma_scale", &[4])?.into_iter().map(|x| Some(x).filter(|x| x.is_finite())).collect()),
        None => None,
    };
    let ok = match field(o, "ok")? {
        Some(v) => Some(Flag::B(v.is_truthy()?)),
        None => None,
    };
    Ok(DetectionJson { id, corners, corner_valid, corner_sigma_scale, ok })
}

pub fn detections(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<DetectionJson>> {
    let it = obj.try_iter().map_err(|_| PyTypeError::new_err(format!("{what} must be an iterable of observations")))?;
    it.map(|o| detection(&o?)).collect()
}

/// Converts eqar detections into a list of `Observation`s (duck typing: no eqar import).
pub fn observations_from_eqar<'py>(py: Python<'py>, markers: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut out: Vec<Observation> = Vec::new();
    for d in detections(markers, "markers")? {
        // same rule as the command line: entries with ok == false or id < 0 are not detections
        if let Some(Flag::B(false)) = d.ok {
            continue;
        }
        if d.id < 0 {
            continue;
        }
        let mut corners = [[f64::NAN; 2]; 4];
        for (c, s) in corners.iter_mut().zip(&d.corners) {
            *c = [s[0].unwrap_or(f64::NAN), s[1].unwrap_or(f64::NAN)];
        }
        let mut valid = [true; 4];
        if let Some(v) = &d.corner_valid {
            for (o, f) in valid.iter_mut().zip(v) {
                *o = match f {
                    Flag::F(x) => flags(&[*x, 1.0, 1.0, 1.0])?[0],
                    Flag::B(b) => *b,
                    Flag::I(i) => flags(&[*i as f64, 1.0, 1.0, 1.0])?[0],
                };
            }
        }
        let mut scale = [1.0; 4];
        if let Some(s) = &d.corner_sigma_scale {
            for (o, x) in scale.iter_mut().zip(s) {
                *o = x.unwrap_or(f64::NAN);
            }
        }
        out.push(Observation { id: d.id, corners, valid, scale });
    }
    PyList::new(py, out)
}
