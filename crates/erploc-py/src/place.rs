//! Marker placement: `Scene`, `place_markers`, `evaluate`, `wall_candidates`.

use crate::conv::{self, cli_err, guarded, value_err, Camera};
use erploc_cli::error::CliError;
use erploc_cli::io::{self, CandidateJson, ObstacleJson, PlaceSpecJson, SectorJson, VisJson, WorkspaceJson};
use erploc_place::placement::{
    choose_survey_triplet, evaluate as lib_evaluate, greedy, local_search_1swap, precompute, prune, CellStats, PlacementSpec, Tables,
};
use erploc_place::scene::{candidates_from_walls, default_headings, Obstacle};
use erploc_place::search::{optimize_ift, Budget};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// A placement problem: workspace polygon, candidate marker positions, obstacles, camera and visibility model.
///
/// The score tables (visibility and Fisher information of every cell/candidate pair) are computed on first use and cached.
#[pyclass(frozen, module = "erploc")]
pub struct Scene {
    json: PlaceSpecJson,
    tables: Mutex<Option<Arc<Tables>>>,
}

#[pymethods]
impl Scene {
    /// `workspace`: polygon (n, 2) of the area where the camera moves. `candidates`: (m, 4) rows `[x, y, z, azimuth]` of
    /// possible upright markers (azimuth = direction of the marker normal, rad). `obstacles`: list of `(polygon (k, 2), height)`.
    /// Cells are a `spacing` grid inside the workspace and outside the obstacles, at each height of `z_cam`, with `headings`
    /// (default: 8 evenly spaced).
    #[new]
    #[pyo3(signature = (workspace, candidates, obstacles=None, *, camera, a, sigma_px=0.3, noise_model="tangent", psi_max_deg=70.0, px_min=12.0, eps=1e-3, sectors=None, spacing=0.5, z_cam=None, headings=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        workspace: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        obstacles: Option<&Bound<'_, PyAny>>,
        camera: &Camera,
        a: f64,
        sigma_px: f64,
        noise_model: &str,
        psi_max_deg: f64,
        px_min: f64,
        eps: f64,
        sectors: Option<&Bound<'_, PyAny>>,
        spacing: f64,
        z_cam: Option<&Bound<'_, PyAny>>,
        headings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let poly = poly2(workspace, "workspace")?;
        let (m, c) = conv::f64_rows(candidates, "candidates", 4)?;
        let cands: Vec<CandidateJson> =
            (0..m).map(|i| CandidateJson { xy: [c[4 * i], c[4 * i + 1]], z: c[4 * i + 2], azimuth: c[4 * i + 3] }).collect();
        let mut obs = Vec::new();
        if let Some(o) = obstacles.filter(|o| !o.is_none()) {
            for (i, item) in o.try_iter()?.enumerate() {
                let item = item?;
                let bad = || value_err(format!("obstacles[{i}] must be a pair (polygon (k, 2), height)"));
                if item.len().map_err(|_| bad())? != 2 {
                    return Err(bad());
                }
                let height: f64 = item.get_item(1)?.extract().map_err(|_| bad())?;
                obs.push(ObstacleJson { poly: poly2(&item.get_item(0)?, &format!("obstacles[{i}] polygon"))?, height });
            }
        }
        let z_cam = match z_cam.filter(|z| !z.is_none()) {
            Some(z) => vec1(z, "z_cam")?,
            None => vec![1.2],
        };
        let headings = match headings.filter(|h| !h.is_none()) {
            Some(h) => vec1(h, "headings")?,
            None => default_headings(),
        };
        let mut secs = Vec::new();
        if let Some(s) = sectors.filter(|s| !s.is_none()) {
            let (n, v) = conv::f64_rows(s, "sectors", 4)?;
            secs = (0..n)
                .map(|i| SectorJson { lam_min: v[4 * i], lam_max: v[4 * i + 1], phi_min: v[4 * i + 2], phi_max: v[4 * i + 3] })
                .collect();
        }
        let json = PlaceSpecJson {
            workspace: WorkspaceJson { poly, spacing, z_cam, headings },
            obstacles: obs,
            n_max: cands.len(),
            candidates: cands,
            a,
            camera: camera.json(),
            sigma_px,
            noise_model: conv::noise(noise_model)?,
            vis: VisJson { psi_max: psi_max_deg.to_radians(), px_min, eps, sectors: secs },
            tau: 1.0,
            k_min: 2,
            tau_loo: None,
            sigma_cap: 10.0,
            w_k: 1.0,
        };
        io::place_spec(&json).map_err(cli_err)?;
        Ok(Scene { json, tables: Mutex::new(None) })
    }

    /// Number of candidate markers.
    #[getter]
    fn n_candidates(&self) -> usize {
        self.json.candidates.len()
    }

    /// Number of evaluation cells (grid points x heights).
    #[getter]
    fn n_cells(&self) -> PyResult<usize> {
        let spec = io::place_spec(&self.json).map_err(cli_err)?;
        Ok(erploc_place::scene::cells(&spec.ws, &spec.obstacles).map_err(|e| cli_err(e.into()))?.len() * spec.ws.z_cam.len())
    }

    /// The candidates as an (m, 4) array `[x, y, z, azimuth]`.
    #[getter]
    fn candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let v: Vec<f64> = self.json.candidates.iter().flat_map(|c| [c.xy[0], c.xy[1], c.z, c.azimuth]).collect();
        conv::arr2(py, self.json.candidates.len(), 4, v)
    }

    fn __repr__(&self) -> String {
        format!("Scene(candidates={}, obstacles={})", self.json.candidates.len(), self.json.obstacles.len())
    }
}

impl Scene {
    fn spec(&self, p: &Params) -> Result<PlacementSpec, CliError> {
        let mut j = self.json.clone();
        j.tau = p.tau;
        j.k_min = p.k_min;
        j.tau_loo = p.tau_loo;
        j.sigma_cap = p.sigma_cap;
        j.w_k = p.w_k;
        j.n_max = p.n_max.unwrap_or(j.candidates.len());
        io::place_spec(&j)
    }

    /// The tables depend on the geometry and the camera model only, not on the thresholds.
    fn tables(&self, spec: &PlacementSpec) -> Result<Arc<Tables>, CliError> {
        let mut g = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = g.as_ref() {
            return Ok(t.clone());
        }
        let t = Arc::new(precompute(spec)?);
        *g = Some(t.clone());
        Ok(t)
    }
}

struct Params {
    tau: f64,
    k_min: usize,
    tau_loo: Option<f64>,
    sigma_cap: f64,
    w_k: f64,
    n_max: Option<usize>,
}

fn poly2(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[f64; 2]>> {
    let (n, v) = conv::f64_rows(obj, what, 2)?;
    Ok((0..n).map(|i| [v[2 * i], v[2 * i + 1]]).collect())
}

fn vec1(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let arr: numpy::PyArrayLikeDyn<'_, f64, numpy::AllowTypeChange> =
        obj.extract().map_err(|_| pyo3::exceptions::PyTypeError::new_err(format!("{what} must be a numeric array-like")))?;
    let a = arr.as_array();
    if a.ndim() != 1 {
        return Err(value_err(format!("{what} must be one-dimensional, got shape {:?}", a.shape())));
    }
    Ok(a.iter().copied().collect())
}

/// Chosen markers and the per-cell accuracy statistics of a layout (result of `place_markers` and `evaluate`).
#[pyclass(frozen, module = "erploc")]
pub struct Placement {
    method: &'static str,
    sel: Vec<usize>,
    survey: Vec<usize>,
    chosen: Vec<[f64; 4]>,
    cells: Vec<[f64; 3]>,
    stats: CellStats,
}

#[pymethods]
impl Placement {
    /// Indices into the candidate array of the chosen markers (in selection order for `greedy`).
    #[getter]
    fn chosen<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.sel.iter().map(|&i| i as i64).collect())
    }
    /// The chosen candidates as rows `[x, y, z, azimuth]`.
    #[getter]
    fn chosen_candidates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        conv::arr2(py, self.chosen.len(), 4, self.chosen.iter().flatten().copied().collect())
    }
    /// Up to three chosen markers to survey (empty for `evaluate`).
    #[getter]
    fn survey_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.survey.iter().map(|&i| i as i64).collect())
    }
    /// Number of chosen markers.
    #[getter]
    fn count(&self) -> usize {
        self.sel.len()
    }
    /// Fraction of the cells that meet the three conditions (σ_pos, visible count, leave-one-out σ_pos).
    #[getter]
    fn feasible_fraction(&self) -> f64 {
        self.stats.feasible.iter().filter(|f| **f).count() as f64 / self.stats.feasible.len().max(1) as f64
    }
    /// Largest per-cell σ_pos [m] over the headings (capped at `sigma_cap`).
    #[getter]
    fn worst_sigma_pos(&self) -> f64 {
        self.stats.sigma.iter().fold(0.0, |m, &s| if s > m { s } else { m })
    }
    /// Evaluation cells (n_cells, 3): x, y, camera height.
    #[getter]
    fn cells<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        conv::arr2(py, self.cells.len(), 3, self.cells.iter().flatten().copied().collect())
    }
    /// Worst-case σ_pos of every cell [m].
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        conv::arr1(py, self.stats.sigma.clone())
    }
    /// Smallest number of visible markers of every cell over the headings.
    #[getter]
    fn n_vis<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        conv::arr1(py, self.stats.n_vis.clone())
    }
    /// Worst σ_pos of every cell with any one visible marker removed.
    #[getter]
    fn sigma_loo<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        conv::arr1(py, self.stats.sigma_loo.clone())
    }
    #[getter]
    fn feasible<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.stats.feasible.clone())
    }
    /// Relative margin `min(1 - sigma/tau, 1 - sigma_loo/tau_loo)` of every cell (negative = infeasible).
    #[getter]
    fn margin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        conv::arr1(py, self.stats.margin.clone())
    }
    /// `"greedy"`, `"ift"` or `"evaluate"`.
    #[getter]
    fn method(&self) -> &'static str {
        self.method
    }
    fn __repr__(&self) -> String {
        format!(
            "Placement(method={:?}, count={}, feasible_fraction={:.4}, worst_sigma_pos={:.5})",
            self.method,
            self.sel.len(),
            self.feasible_fraction(),
            self.worst_sigma_pos()
        )
    }
}

fn placement(spec: &PlacementSpec, tb: &Tables, method: &'static str, sel: Vec<usize>, survey: Vec<usize>, stats: CellStats) -> Placement {
    let chosen = sel
        .iter()
        .map(|&j| [spec.candidates[j].xy[0], spec.candidates[j].xy[1], spec.candidates[j].z, spec.candidates[j].azimuth])
        .collect();
    Placement { method, chosen, sel, survey, cells: tb.cells.clone(), stats }
}

/// Chooses markers for `scene`: a cell is feasible when `sigma_pos <= tau`, at least `k_min` markers are visible, and
/// `sigma_pos <= tau_loo` (default `tau`) with any one of them removed. `method="greedy"` (greedy forward selection, 1-swap,
/// pruning) or `"ift"` (the minimal-count search; `budget_ms` limits its wall-clock time). At most `n_max` markers
/// (default: all candidates). Also picks three markers to survey (`sigma_ts` = their survey sigma, m).
#[pyfunction]
#[pyo3(signature = (scene, tau, *, k_min=2, n_max=None, tau_loo=None, sigma_cap=10.0, w_k=1.0, method="greedy", budget_ms=None, sigma_ts=0.005))]
#[allow(clippy::too_many_arguments)]
pub fn place_markers(
    py: Python<'_>,
    scene: &Bound<'_, Scene>,
    tau: f64,
    k_min: usize,
    n_max: Option<usize>,
    tau_loo: Option<f64>,
    sigma_cap: f64,
    w_k: f64,
    method: &str,
    budget_ms: Option<u64>,
    sigma_ts: f64,
) -> PyResult<Placement> {
    let ift = match method {
        "greedy" => false,
        "ift" => true,
        _ => return Err(value_err(format!("method must be \"greedy\" or \"ift\", got {method:?}"))),
    };
    if budget_ms == Some(0) {
        return Err(value_err("budget_ms must be >= 1"));
    }
    let sc = scene.get();
    let p = Params { tau, k_min, tau_loo, sigma_cap, w_k, n_max };
    let (spec, tb, sel, survey, stats) = guarded(py, || {
        let spec = sc.spec(&p)?;
        let sigma_ts = io::within(sigma_ts, io::SIGMA_TS_RANGE, "sigma_ts")?;
        if ift {
            let budget = match budget_ms {
                Some(ms) => Budget::Wall(ms as f64 / 1000.0),
                None => Budget::Unlimited,
            };
            let r = optimize_ift(&spec, sigma_ts, &budget)?;
            let tb = sc.tables(&spec)?;
            Ok((spec, tb, r.sel, r.survey_ids, r.stats))
        } else {
            let tb = sc.tables(&spec)?;
            let (sel, _) = greedy(&tb, &spec)?;
            let (sel, _) = local_search_1swap(&tb, &spec, &sel)?;
            let (sel, _) = prune(&tb, &spec, &sel)?;
            let stats = lib_evaluate(&tb, &sel, &spec)?;
            let (ids, _, _) = choose_survey_triplet(&tb, &spec, &sel, sigma_ts)?;
            Ok((spec, tb, sel, ids, stats))
        }
    })?;
    Ok(placement(&spec, &tb, if ift { "ift" } else { "greedy" }, sel, survey, stats))
}

/// Evaluates a given layout under the same model as `place_markers`. `chosen`: indices into the candidates.
#[pyfunction]
#[pyo3(signature = (scene, chosen, tau, *, k_min=2, tau_loo=None, sigma_cap=10.0, w_k=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    py: Python<'_>,
    scene: &Bound<'_, Scene>,
    chosen: &Bound<'_, PyAny>,
    tau: f64,
    k_min: usize,
    tau_loo: Option<f64>,
    sigma_cap: f64,
    w_k: f64,
) -> PyResult<Placement> {
    let sc = scene.get();
    let idx = vec1(chosen, "chosen")?;
    let n = sc.json.candidates.len();
    let mut sel = Vec::with_capacity(idx.len());
    for &x in &idx {
        if !(x.is_finite() && x.fract() == 0.0 && x >= 0.0 && (x as usize) < n) {
            return Err(value_err(format!("chosen must hold integers in 0..{n}, got {x}")));
        }
        if sel.contains(&(x as usize)) {
            return Err(value_err(format!("chosen contains candidate {x} twice")));
        }
        sel.push(x as usize);
    }
    let p = Params { tau, k_min, tau_loo, sigma_cap, w_k, n_max: None };
    let (spec, tb, stats) = guarded(py, || {
        let spec = sc.spec(&p)?;
        let tb = sc.tables(&spec)?;
        let stats = lib_evaluate(&tb, &sel, &spec)?;
        Ok((spec, tb, stats))
    })?;
    Ok(placement(&spec, &tb, "evaluate", sel, Vec::new(), stats))
}

/// Candidate markers spaced along the edges of a polygon (a room outline), facing inward by default. Rows `[x, y, z, azimuth]`.
#[pyfunction]
#[pyo3(signature = (polygon, spacing, *, z=1.5, offset=0.05, inward=true))]
pub fn wall_candidates<'py>(
    py: Python<'py>,
    polygon: &Bound<'py, PyAny>,
    spacing: f64,
    z: f64,
    offset: f64,
    inward: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let poly = poly2(polygon, "polygon")?;
    if poly.len() < 3 || poly.len() > 10_000 {
        return Err(value_err(format!("polygon needs 3..=10000 vertices, got {}", poly.len())));
    }
    if poly.iter().flatten().any(|x| !x.is_finite() || x.abs() > 1e6) {
        return Err(value_err("polygon coordinates must be finite and within 1e6 m"));
    }
    if !(spacing.is_finite() && spacing > 0.0) || !z.is_finite() || !offset.is_finite() {
        return Err(value_err("spacing must be finite and > 0; z and offset must be finite"));
    }
    let perimeter: f64 =
        (0..poly.len()).map(|i| (poly[(i + 1) % poly.len()][0] - poly[i][0]).hypot(poly[(i + 1) % poly.len()][1] - poly[i][1])).sum();
    if perimeter / spacing + poly.len() as f64 > io::MAX_CANDS as f64 {
        return Err(value_err(format!("spacing {spacing} would give more than {} candidates", io::MAX_CANDS)));
    }
    let c = guarded(py, move || Ok(candidates_from_walls(&[Obstacle { poly, height: 0.0 }], spacing, z, offset, inward)?))?;
    conv::arr2(py, c.len(), 4, c.iter().flat_map(|c| [c.xy[0], c.xy[1], c.z, c.azimuth]).collect())
}
