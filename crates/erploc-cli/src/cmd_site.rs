//! `erploc site`: queries against a site table (`site.bin`, format `ERPSITE1`): IFT searches, minimal count, zone
//! statistics of a layout. A survey triplet needs full-heading tables and is out of scope.
use crate::error::CliError;
use crate::io;
use erploc_core::V3;
use erploc_linalg::rng::Pcg64;
use erploc_place::fast_eval::IftContext;
use erploc_place::occlusion::{occ_tables, OccMode};
use erploc_place::placement::{evaluate, Key3};
use erploc_place::search::{ift_det, ift_min_count, run_ift, Budget, WallClock};
use erploc_place::site::{SiteTables, VisMode};
use erploc_place::zone::{Crit, Grid, ZoneStats};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModeJson {
    Centre,
    Samples,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CritJson {
    N,
    K3,
    Loo,
    Both,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ActionJson {
    MinCount { k_max: Option<usize> },
    Det { k: usize },
    Ift { k: usize, budget_ms: u64, seed: u64 },
    Zone { sel: Vec<usize> },
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct SiteIn {
    pub mode: ModeJson,
    pub crit: CritJson,
    pub action: ActionJson,
}

#[derive(Serialize, Debug)]
pub struct KeyOut {
    pub n_infeasible: usize,
    pub penalty_sum: f64,
    pub neg_margin: f64,
}

fn key_out(k: &Key3) -> KeyOut {
    KeyOut { n_infeasible: k.n_infeasible, penalty_sum: k.penalty_sum, neg_margin: k.neg_margin }
}

#[derive(Serialize, Debug)]
pub struct PerKOut {
    pub k: usize,
    pub key: KeyOut,
    pub evaluations: u64,
}

#[derive(Serialize, Debug)]
pub struct ZoneOut {
    pub zone: usize,
    pub total: usize,
    pub outside: usize,
    pub holes: usize,
    pub holes_nonfloor: usize,
    pub n_components: usize,
    pub component_sizes: Vec<usize>,
}

fn zone_out(z: &ZoneStats) -> ZoneOut {
    ZoneOut {
        zone: z.zone,
        total: z.total,
        outside: z.outside,
        holes: z.holes,
        holes_nonfloor: z.holes_nonfloor,
        n_components: z.n_components,
        component_sizes: z.component_sizes.clone(),
    }
}

/// One output shape for every action; fields that do not apply are `null`.
#[derive(Serialize, Debug, Default)]
pub struct SiteOut {
    pub action: &'static str,
    pub mode: &'static str,
    pub crit: &'static str,
    pub n_cells: usize,
    pub n_candidates: usize,
    pub sel: Option<Vec<usize>>,
    pub key: Option<KeyOut>,
    pub feasible_fraction: Option<f64>,
    pub evaluations: Option<u64>,
    pub found: Option<bool>,
    pub k: Option<usize>,
    pub per_k: Option<Vec<PerKOut>>,
    pub zone: Option<ZoneOut>,
    pub n_infeasible: Option<usize>,
}

fn read(path: &Path) -> Result<Vec<u8>, CliError> {
    std::fs::read(path).map_err(|e| CliError::Io(format!("{}: {e}", path.display())))
}

/// `sel`: non-empty, strictly valid candidate indices, no repeats.
fn check_sel(sel: &[usize], n: usize) -> Result<(), CliError> {
    if sel.is_empty() || sel.len() > n {
        return Err(CliError::Input(format!("sel needs 1..={n} candidate indices, got {}", sel.len())));
    }
    let mut seen = vec![false; n];
    for &s in sel {
        if s >= n {
            return Err(CliError::Input(format!("sel index {s} is out of range (N = {n})")));
        }
        if std::mem::replace(&mut seen[s], true) {
            return Err(CliError::Input(format!("sel index {s} appears twice")));
        }
    }
    Ok(())
}

fn check_k(k: usize, n: usize, what: &str) -> Result<(), CliError> {
    if k == 0 || k >= n {
        return Err(CliError::Input(format!("{what} must be in 1..N-1 (N = {n}), got {k}")));
    }
    Ok(())
}

pub fn run_site(input_json: &str, tables: &Path) -> Result<String, CliError> {
    let q: SiteIn = io::parse(input_json)?;
    let site = SiteTables::from_bytes(&read(tables)?)?;
    let n = site.cands.len();
    let (mode, mode_name) = match q.mode {
        ModeJson::Centre => (VisMode::Centre, "centre"),
        ModeJson::Samples => (VisMode::Samples, "samples"),
    };
    let (crit, crit_name) = match q.crit {
        CritJson::N => (Crit::N, "N"),
        CritJson::K3 => (Crit::K3, "K3"),
        CritJson::Loo => (Crit::Loo, "Loo"),
        CritJson::Both => (Crit::Both, "Both"),
    };
    let mut ctx = IftContext::from_site(&site, mode, crit);
    let kc = ctx.kc;
    let frac = |k: &Key3| 1.0 - k.n_infeasible as f64 / kc as f64;
    let mut out = SiteOut { mode: mode_name, crit: crit_name, n_cells: kc, n_candidates: n, ..SiteOut::default() };
    match q.action {
        ActionJson::MinCount { k_max } => {
            if let Some(km) = k_max {
                check_k(km, n, "k_max")?;
            }
            let mc = ift_min_count(&mut ctx, k_max, &Budget::Unlimited)?;
            out.action = "min_count";
            out.feasible_fraction = Some(frac(&mc.key));
            out.evaluations = Some(mc.per_k.iter().map(|p| p.evals).sum());
            out.per_k = Some(mc.per_k.iter().map(|p| PerKOut { k: p.k, key: key_out(&p.key), evaluations: p.evals }).collect());
            out.found = Some(mc.found);
            out.k = Some(mc.k);
            out.key = Some(key_out(&mc.key));
            out.sel = Some(mc.sel);
        }
        ActionJson::Det { k } => {
            check_k(k, n, "k")?;
            let r = ift_det(&mut ctx, k, &Budget::Unlimited)?;
            out.action = "det";
            out.k = Some(k);
            out.feasible_fraction = Some(frac(&r.best_key));
            out.evaluations = Some(r.evals);
            out.key = Some(key_out(&r.best_key));
            out.sel = Some(r.best_sel);
        }
        ActionJson::Ift { k, budget_ms, seed } => {
            check_k(k, n, "k")?;
            if budget_ms == 0 {
                return Err(CliError::Input("budget_ms must be >= 1".into()));
            }
            let budget = Budget::Wall(budget_ms as f64 / 1000.0);
            let r = run_ift(&mut ctx, k, &budget, &mut WallClock, &mut Pcg64::new(seed), true, 15, 7)?;
            out.action = "ift";
            out.k = Some(k);
            out.feasible_fraction = Some(frac(&r.best_key));
            out.evaluations = Some(r.evals);
            out.key = Some(key_out(&r.best_key));
            out.sel = Some(r.best_sel);
        }
        ActionJson::Zone { sel } => {
            check_sel(&sel, n)?;
            let cells: Vec<V3> = site.cells.clone();
            if !io::cells_on_1m_grid(&cells) {
                return Err(CliError::Input("the site cells are not on a 1 m grid; zone statistics are undefined".into()));
            }
            let om = if mode == VisMode::Samples { OccMode::Samples } else { OccMode::Centre };
            let sk = ctx.spec_k(sel.len());
            let (tb2, _, _) = occ_tables(&ctx.tb, &ctx.cands, &ctx.kinds, &sel, ctx.a, om);
            let feasible = evaluate(&tb2, &sel, &sk)?.feasible;
            let z = Grid::new(&cells).zone(&feasible);
            out.action = "zone";
            let n_inf = feasible.iter().filter(|&&f| !f).count();
            out.n_infeasible = Some(n_inf);
            out.feasible_fraction = Some(1.0 - n_inf as f64 / kc as f64);
            out.zone = Some(zone_out(&z));
            out.sel = Some(sel);
        }
    }
    io::to_json(&out)
}
