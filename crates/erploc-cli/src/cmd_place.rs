//! `erploc place`: marker placement for a workspace (greedy + 1-swap + prune, or the IFT minimal-count search), survey triplet,
//! report CSV (`placement::report_csv` verbatim).
use crate::error::CliError;
use crate::io::{self, PlaceSpecJson};
use erploc_place::placement::{key3, optimize, report_csv, HistoryStep, Key3, PlacementResult, PlacementSpec};
use erploc_place::scene::cells;
use erploc_place::search::{optimize_ift, Budget};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Greedy,
    Ift,
}

#[derive(Clone, Debug)]
pub struct PlaceOpts {
    pub method: Method,
    /// Wall-clock budget of the IFT k-loop in milliseconds; `None` runs the deterministic search to its own stop.
    pub budget_ms: Option<u64>,
    pub sigma_ts: f64,
}

impl Default for PlaceOpts {
    fn default() -> Self {
        PlaceOpts { method: Method::Greedy, budget_ms: None, sigma_ts: 0.005 }
    }
}

pub struct PlaceOut {
    pub json: String,
    pub csv: String,
}

#[derive(Serialize, Debug)]
pub struct KeyOut {
    pub n_infeasible: usize,
    pub penalty_sum: f64,
    pub neg_margin: f64,
}

#[derive(Serialize, Debug)]
pub struct StatsOut {
    pub sigma: Vec<f64>,
    pub n_vis: Vec<f64>,
    pub sigma_loo: Vec<f64>,
    pub feasible: Vec<bool>,
    pub margin: Vec<f64>,
}

#[derive(Serialize, Debug)]
pub struct HistoryOut {
    pub step: &'static str,
    pub sel: Option<Vec<usize>>,
    pub ids: Option<Vec<usize>>,
    pub k: Option<usize>,
    pub n_infeasible: Option<usize>,
    pub penalty: Option<f64>,
    pub neg_margin: Option<f64>,
    pub min_margin: Option<f64>,
    pub score: Option<f64>,
    pub n_skip: Option<usize>,
}

impl HistoryOut {
    fn empty(step: &'static str) -> Self {
        HistoryOut {
            step,
            sel: None,
            ids: None,
            k: None,
            n_infeasible: None,
            penalty: None,
            neg_margin: None,
            min_margin: None,
            score: None,
            n_skip: None,
        }
    }
}

fn history_out(h: &HistoryStep) -> HistoryOut {
    match h {
        HistoryStep::Greedy { sel, n_infeasible, penalty } => {
            HistoryOut { sel: Some(sel.clone()), n_infeasible: Some(*n_infeasible), penalty: Some(*penalty), ..HistoryOut::empty("greedy") }
        }
        HistoryStep::OneSwap { sel, n_infeasible, penalty, neg_margin } => HistoryOut {
            sel: Some(sel.clone()),
            n_infeasible: Some(*n_infeasible),
            penalty: Some(*penalty),
            neg_margin: Some(*neg_margin),
            ..HistoryOut::empty("one_swap")
        },
        HistoryStep::Prune { sel, min_margin } => {
            HistoryOut { sel: Some(sel.clone()), min_margin: Some(*min_margin), ..HistoryOut::empty("prune") }
        }
        HistoryStep::Survey { ids, score, n_skip } => {
            HistoryOut { ids: Some(ids.clone()), score: Some(*score), n_skip: Some(*n_skip), ..HistoryOut::empty("survey") }
        }
        HistoryStep::Ift { k, n_infeasible, penalty } => {
            HistoryOut { k: Some(*k), n_infeasible: Some(*n_infeasible), penalty: Some(*penalty), ..HistoryOut::empty("ift") }
        }
    }
}

#[derive(Serialize, Debug)]
pub struct PlaceJsonOut {
    pub method: &'static str,
    pub sel: Vec<usize>,
    pub survey_ids: Vec<usize>,
    pub key3: KeyOut,
    pub cells: Vec<[f64; 3]>,
    pub stats: StatsOut,
    pub history: Vec<HistoryOut>,
}

/// Evaluation cells in table order (z-major stacking of the workspace grid, as in `placement::precompute`).
fn eval_cells(spec: &PlacementSpec) -> Result<Vec<[f64; 3]>, CliError> {
    let xy = cells(&spec.ws, &spec.obstacles)?;
    Ok(spec.ws.z_cam.iter().flat_map(|&z| xy.iter().map(move |p| [p[0], p[1], z])).collect())
}

fn key_out(k: &Key3) -> KeyOut {
    KeyOut { n_infeasible: k.n_infeasible, penalty_sum: k.penalty_sum, neg_margin: k.neg_margin }
}

pub fn run_place(input_json: &str, opts: &PlaceOpts) -> Result<PlaceOut, CliError> {
    let j: PlaceSpecJson = io::parse(input_json)?;
    let spec = io::place_spec(&j)?;
    let sigma_ts = io::within(opts.sigma_ts, io::SIGMA_TS_RANGE, "sigma_ts")?;
    let (res, method): (PlacementResult, &'static str) = match opts.method {
        Method::Greedy => (optimize(&spec, sigma_ts)?, "greedy"),
        Method::Ift => {
            let budget = match opts.budget_ms {
                Some(0) => return Err(CliError::Input("budget_ms must be >= 1".into())),
                Some(ms) => Budget::Wall(ms as f64 / 1000.0),
                None => Budget::Unlimited,
            };
            (optimize_ift(&spec, sigma_ts, &budget)?, "ift")
        }
    };
    let key = key3(&res.stats, &spec);
    let out = PlaceJsonOut {
        method,
        sel: res.sel.clone(),
        survey_ids: res.survey_ids.clone(),
        key3: key_out(&key),
        cells: eval_cells(&spec)?,
        stats: StatsOut {
            sigma: res.stats.sigma.clone(),
            n_vis: res.stats.n_vis.clone(),
            sigma_loo: res.stats.sigma_loo.clone(),
            feasible: res.stats.feasible.clone(),
            margin: res.stats.margin.clone(),
        },
        history: res.history.iter().map(history_out).collect(),
    };
    Ok(PlaceOut { json: io::to_json(&out)?, csv: report_csv(&res, &spec) })
}
