//! IFT evaluators: `IftContext` (`IFTContext` + site `Ctx`) and `FastEval` (`_base`, `_pairs`, `set_key`,
//! `add_keys`, `min_margin`) of `erploc/ift.py`. `FastEval` evaluates `S' + {j}` for every candidate `j` at once by touching only
//! the (cell, candidate) pairs that can change; it is exact, not an approximation.
//!
//! Summation orders follow numpy: `sum_weighted10` is the measured `einsum` order, `np_sum` the pairwise sum, `bincount` adds
//! sequentially, `np.maximum/minimum` propagate NaN (`nmax/nmin`). `neighbourhood` (shared base), `SlowEval`,
//! the `Evaluator` enum and `get_eval`.
use crate::npcompat::{bincount_into, nmax, nmin, np_sum, searchsorted_left, sum_weighted10};
use crate::occlusion::{blocked_triples, occ_tables, Occ, OccMode};
use crate::placement::{evaluate, key3, precompute, Key3, PlaceError, PlacementSpec, Tables};
use crate::scene::Candidate;
use crate::sigma10::{sigma10_one, Sigma10Counts};
use crate::site::{near_lists, CandKind, SiteTables, VisMode, IU};
use crate::zone::{criterion_spec, Crit};
use std::collections::HashMap;

/// Key entry of a selected candidate in `add_keys` (the count term).
pub const BIG: i64 = 1_000_000_000;

/// Key of the pair (candidate `j`, cell `c`) in the sorted pair list: `j*K + c` (Python `bs*K + bc`). Float-free (Kani).
pub(crate) fn pair_key(j: usize, c: usize, kc: usize) -> usize {
    j * kc + c
}

/// Inverse of [`pair_key`] for `c < K`: `(u / K, u % K)`. Float-free (Kani).
pub(crate) fn unpair(u: usize, kc: usize) -> (usize, usize) {
    (u / kc, u % kc)
}

/// Flat index `c*N + t` of the blocked-triple flag of (cell `c`, candidate `t`) in `Bl` (`K x N`, cell-major). Float-free (Kani).
pub(crate) fn bl_index(c: usize, t: usize, nn: usize) -> usize {
    c * nn + t
}

/// `IFTContext` plus the fields of the site `Ctx`.
pub struct IftContext {
    /// Criterion specification (read through [`IftContext::spec0`]; private so that the evaluator cache cannot go stale: the only
    /// way to change the criterion is [`IftContext::with_crit`], which starts with fresh caches).
    spec0: PlacementSpec,
    /// Specification before the criterion overrides, the base of `with_crit`.
    base_spec: PlacementSpec,
    pub n: usize,
    pub kc: usize,
    pub kinds: Vec<CandKind>,
    pub near: Vec<Vec<usize>>,
    /// Raw reference tables (full 4x4 blocks), heading axis as imported.
    pub tb: Tables,
    /// `None`: no stands (synthetic room), `Some`: occlusion by triples / the raw blocked matrix.
    pub occ_mode: Option<OccMode>,
    pub a: f64,
    pub cands: Vec<Candidate>,
    occ: Option<Occ>,
    /// `get_eval` cache, one evaluator per `k` (Python `ctx._fe`).
    evals: HashMap<usize, Evaluator>,
    /// Number of evaluators `get_eval` has built (cache misses); lets tests check the cache without comparing addresses.
    n_built: usize,
}

impl IftContext {
    /// `IFTContext(spec)`: `precompute(spec)`, walls only (no selectable stands).
    pub fn from_spec(spec: &PlacementSpec) -> Result<IftContext, PlaceError> {
        let tb = precompute(spec)?;
        let n = spec.candidates.len();
        Ok(IftContext {
            spec0: spec.clone(),
            base_spec: spec.clone(),
            n,
            kc: tb.cells.len(),
            kinds: vec![CandKind::Wall; n],
            near: near_lists(&spec.candidates),
            tb,
            occ_mode: None,
            a: spec.a,
            cands: spec.candidates.clone(),
            occ: None,
            evals: HashMap::new(),
            n_built: 0,
        })
    }

    /// The site context of one visibility mode and one criterion (`SiteIftContext`; `rl.with_spec(ctx, **CRITS[c][0])`): the
    /// specification is `criterion_spec(site.spec0, crit)`, the tables are shared by all criteria.
    pub fn from_site(site: &SiteTables, mode: VisMode, crit: Crit) -> IftContext {
        let tb = site.tables(mode);
        IftContext {
            spec0: criterion_spec(&site.spec0, crit),
            base_spec: site.spec0.clone(),
            n: site.cands.len(),
            kc: site.cells.len(),
            kinds: site.kinds.clone(),
            near: site.near.clone(),
            tb,
            occ_mode: Some(match mode {
                VisMode::Centre => OccMode::Centre,
                VisMode::Samples => OccMode::Samples,
            }),
            a: site.spec0.a,
            cands: site.cands.clone(),
            occ: None,
            evals: HashMap::new(),
            n_built: 0,
        }
    }

    /// The specification of this context (with the criterion's overrides).
    pub fn spec0(&self) -> &PlacementSpec {
        &self.spec0
    }

    /// `robust_lib.with_spec(ctx, **CRITS[c][0])`: the same tables and blocked triples under another criterion. The `FastEval`
    /// cache is dropped (it holds `tau` / `k_min` / `tau_loo` of the old specification) and the occlusion triples are shared
    /// (they depend on visibility only). The overrides apply to the base specification, so `with_crit(Crit::N)` undoes `Both`.
    pub fn with_crit(&self, crit: Crit) -> IftContext {
        IftContext {
            spec0: criterion_spec(&self.base_spec, crit),
            base_spec: self.base_spec.clone(),
            n: self.n,
            kc: self.kc,
            kinds: self.kinds.clone(),
            near: self.near.clone(),
            tb: self.tb.clone(),
            occ_mode: self.occ_mode,
            a: self.a,
            cands: self.cands.clone(),
            occ: self.occ.clone(),
            evals: HashMap::new(),
            n_built: 0,
        }
    }

    /// `dataclasses.replace(spec0, n_max=k)`.
    pub fn spec_k(&self, k: usize) -> PlacementSpec {
        self.spec0.with_n_max(k)
    }

    /// `f(S)` from scratch with the reference evaluator: `key3(evaluate(occ_tables(tb, S), spec_k))`, or without occlusion
    /// when `occ_mode` is `None`.
    pub fn raw_key(&self, s: &[usize], k: usize) -> Result<Key3, PlaceError> {
        let sk = self.spec_k(k);
        match self.occ_mode {
            Some(m) => {
                let (tb2, _, _) = occ_tables(&self.tb, &self.cands, &self.kinds, s, self.a, m);
                Ok(key3(&evaluate(&tb2, s, &sk)?, &sk))
            }
            None => Ok(key3(&evaluate(&self.tb, s, &sk)?, &sk)),
        }
    }

    /// Number of evaluators built by `get_eval` so far (cache misses).
    pub fn n_built(&self) -> usize {
        self.n_built
    }

    /// `n_triples` of the cached blocked-triple table, `None` while it has not been built (Python
    /// `ctx._occ['n_triples'] if ctx._occ else None`; a built empty table of a room without stands is `Some(0)`).
    pub fn n_triples(&self) -> Option<usize> {
        self.occ.as_ref().map(|o| o.n_triples)
    }

    /// `build_occ` (cached): blocked triples of the mode; an empty table (visible pairs only) when there are no stands.
    /// Python's `IFTContext.occ_triples` raises `NotImplementedError` when stands exist and no mode is set; here that is a panic.
    pub fn occ(&mut self) -> &Occ {
        if self.occ.is_none() {
            assert!(
                self.occ_mode.is_some() || !self.kinds.contains(&CandKind::Stand),
                "IftContext::occ: selectable stands need an occlusion mode"
            );
            let (k, n) = (self.tb.k, self.tb.n);
            let vis0: Vec<bool> = (0..k).flat_map(|kk| (0..n).map(move |j| (kk, j))).map(|(kk, j)| self.tb.vis_at(kk, 0, j)).collect();
            let mode = self.occ_mode.unwrap_or(OccMode::Centre);
            self.occ = Some(blocked_triples(&self.tb.cells, &self.cands, &self.kinds, self.a, &vis0, mode));
        }
        self.occ.as_ref().expect("occ built above")
    }
}

/// `add_keys` result, per candidate `j`.
#[derive(Clone, Debug, PartialEq)]
pub struct AddKeys {
    pub inf: Vec<i64>,
    pub pen: Vec<f64>,
    pub nm: Vec<f64>,
}

/// `_base`: everything about the selection `S'` itself.
#[derive(Default)]
pub struct Base {
    /// Blocked-by-`S'`-boards mask `(K, N)`.
    pub bl: Vec<bool>,
    /// Effective-marker weights `(K, n)`.
    pub w: Vec<bool>,
    /// `I10[:, S', :]`, `(K, n, 10)`.
    pub ip: Vec<f64>,
    /// `(K, 10)`.
    pub isum: Vec<f64>,
    pub feas: Vec<bool>,
    pub pen: Vec<f64>,
    pub marg: Vec<f64>,
}

/// `_pairs` result: per affected pair (cell `pc`, candidate `pj`), sorted by `pj * K + pc`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Pairs {
    pub pj: Vec<usize>,
    pub pc: Vec<usize>,
    pub d_inf: Vec<i64>,
    pub d_pen: Vec<f64>,
    pub marg: Vec<f64>,
    /// Number of pair rows re-summed exactly (`rb`: rows where a candidate's board blocks a marker of `S'`), a diagnostic.
    pub n_resum: usize,
}

/// Reusable work buffers of `pairs` (cleared, never freed).
#[derive(Default)]
pub(crate) struct PairsScratch {
    in_s: Vec<bool>,
    keys: Vec<i64>,
    bk: Vec<i64>,
    uk: Vec<i64>,
    blk: Vec<(usize, usize, usize)>,
    bidx: Vec<usize>,
    wj: Vec<bool>,
    isum: Vec<f64>,
    wn: Vec<bool>,
    rb: Vec<usize>,
}

/// Reusable work buffers of `min_margin`.
#[derive(Default)]
pub(crate) struct MmScratch {
    mm: Vec<f64>,
    cells: Vec<u32>,
    ptr: Vec<u32>,
    order: Vec<usize>,
}

/// Work buffers of one position `i` of `neighbourhood` (one set per thread with the `parallel` feature).
#[derive(Default)]
pub(crate) struct RowScratch {
    pr: PairsScratch,
    mm: MmScratch,
    bp: Base,
    p2: Pairs,
    sp: Vec<usize>,
    a: Vec<bool>,
    bi1: Vec<f64>,
    bp1: Vec<f64>,
    bi2: Vec<f64>,
    bp2: Vec<f64>,
    m1: Vec<f64>,
    m2: Vec<f64>,
}

/// Everything `FastEval` reuses between calls.
#[derive(Default)]
pub(crate) struct Scratch {
    bs: Base,
    ps: Pairs,
    dis: Vec<f64>,
    di_s: Vec<f64>,
    dp_s: Vec<f64>,
    m1: Vec<f64>,
    row: RowScratch,
}

pub struct FastEval {
    pub k: usize,
    pub kc: usize,
    pub n: usize,
    pub(crate) v: Vec<bool>,
    /// `I10` in candidate-major layout `[(j*K + c)*10 ..]` (the pair-key order: consecutive pairs touch consecutive rows).
    pub(crate) i10t: Vec<f64>,
    /// `I10` in the cell-major layout `[(c*N + j)*10 ..]`, kept for the frozen reference only.
    #[cfg(test)]
    pub(crate) i10: Vec<f64>,
    pub(crate) cap: f64,
    pub(crate) tau: f64,
    pub(crate) kmin: f64,
    pub(crate) wk: f64,
    pub(crate) tl: f64,
    pub(crate) is_stand: Vec<bool>,
    pub(crate) occ: Occ,
    /// Visible cells of each candidate `j`, ascending, deduplicated: `vj_c[vj_ptr[j]..vj_ptr[j + 1]]` (the visibility list in pair-key order).
    vj_ptr: Vec<u32>,
    vj_c: Vec<u32>,
    sc: Scratch,
    /// `AMB_COUNT`: entries sent to the exact rule / evaluated, summed over every call so far.
    pub amb: Sigma10Counts,
}

/// Candidate-major CSR of the visible pairs: `vk` holds the strictly ascending pair keys `j*K + c` (`j < N`, `c < K`); the result is
/// `ptr` (`N + 1` entries, monotone, `ptr[N] = vk.len()`) and `c` (the cell of each key), so `c[ptr[j]..ptr[j + 1]]` are the cells of
/// candidate `j`, ascending. Float-free (Kani).
pub(crate) fn vis_csr(vk: &[u64], kc: usize, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut ptr = vec![0u32; n + 1];
    let mut c = vec![0u32; vk.len()];
    vis_csr_into(vk, kc, n, &mut ptr, &mut c);
    (ptr, c)
}

/// [`vis_csr`] into caller-provided slices (`ptr.len() == n + 1`, `c.len() == vk.len()`, `ptr` zeroed); allocation-free, the part
/// the Kani harness checks.
pub(crate) fn vis_csr_into(vk: &[u64], kc: usize, n: usize, ptr: &mut [u32], c: &mut [u32]) {
    for (i, &u) in vk.iter().enumerate() {
        ptr[(u / kc as u64) as usize + 1] += 1;
        c[i] = (u % kc as u64) as u32;
    }
    for j in 0..n {
        ptr[j + 1] += ptr[j];
    }
}

/// Python builtin `min(a, b)` on floats: `b` replaces `a` only if `b < a` (a NaN `b` never does, a NaN `a` sticks).
fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `FastEval._min_margin`: for every candidate `j` the minimum over the cells in `mask` of the cell margin after adding `j`:
/// the affected pairs `(pj, pc)` carry the new margin `mn`, every other cell keeps its base margin `m0`; `+inf` for cells
/// outside `mask`. Transcription of the Python (stable ascending order of the masked base margins, NaN last; first finite
/// unaffected cell); the unit test compares it with a set-based literal transcription.
///
/// Precondition (panics otherwise, O(P) check): the pairs `(pj[t], pc[t])` are strictly increasing in lexicographic order, as
/// `FastEval` produces them. The hot `min_margin_into` does not re-check this.
pub fn min_margin(m0: &[f64], pj: &[usize], pc: &[usize], mn: &[f64], mask: Option<&[bool]>, n: usize) -> Vec<f64> {
    assert!(
        pj.len() == pc.len() && (1..pj.len()).all(|t| (pj[t - 1], pc[t - 1]) < (pj[t], pc[t])),
        "min_margin: pairs (pj, pc) must be strictly sorted"
    );
    let mut out = Vec::new();
    min_margin_into(m0, pj, pc, mn, mask, false, n, &mut out, &mut MmScratch::default());
    out
}

/// `min_margin` into reusable buffers; `invert` uses the complement of `mask` (so the caller needs no negated copy).
/// The pairs are sorted by `(pj, pc)`, so the cells of one candidate are a contiguous, ascending run: a CSR over `pj` replaces
/// the per-candidate `Vec` and its sort (the lists the Python sorts are already sorted).
#[allow(clippy::too_many_arguments)]
fn min_margin_into(
    m0: &[f64],
    pj: &[usize],
    pc: &[usize],
    mn: &[f64],
    mask: Option<&[bool]>,
    invert: bool,
    n: usize,
    out: &mut Vec<f64>,
    sc: &mut MmScratch,
) {
    assert!(!m0.is_empty(), "min_margin: no cells");
    assert!(pj.len() == pc.len() && pj.len() == mn.len(), "min_margin: pair array lengths");
    out.clear();
    out.resize(n, f64::INFINITY);
    let MmScratch { mm, cells, ptr, order } = sc;
    let keep = |c: usize| mask.map_or(true, |mk| mk[c] != invert);
    mm.clear();
    mm.extend(m0.iter().enumerate().map(|(c, &m)| if keep(c) { m } else { f64::INFINITY }));
    cells.clear();
    ptr.clear();
    ptr.resize(n + 1, 0);
    let mut last = 0usize; // candidates 0..=last have their ptr set
    for i in 0..pj.len() {
        if keep(pc[i]) {
            let j = pj[i];
            debug_assert!(i == 0 || (pj[i - 1], pc[i - 1]) < (j, pc[i]), "pairs must be sorted by (pj, pc)");
            out[j] = nmin(out[j], mn[i]); // np.minimum.at
            while last < j {
                last += 1;
                ptr[last] = cells.len() as u32;
            }
            cells.push(pc[i] as u32);
        }
    }
    while last < n {
        last += 1;
        ptr[last] = cells.len() as u32;
    }
    // np.argsort(kind='stable'): ascending, NaN last, ties by index (the index tie-break makes the unstable sort equal to it)
    order.clear();
    order.extend(0..mm.len());
    order.sort_unstable_by(|&a, &b| match (mm[a].is_nan(), mm[b].is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => mm[a].partial_cmp(&mm[b]).expect("non-NaN").then(a.cmp(&b)),
    });
    let c0 = order[0];
    let first = mm[c0];
    for j in 0..n {
        let aff = &cells[ptr[j] as usize..ptr[j + 1] as usize];
        if aff.is_empty() || aff.binary_search(&(c0 as u32)).is_err() {
            out[j] = pymin(out[j], first);
            continue;
        }
        for &c in order.iter() {
            if !mm[c].is_finite() {
                break;
            }
            if aff.binary_search(&(c as u32)).is_err() {
                out[j] = pymin(out[j], mm[c]);
                break;
            }
        }
    }
}

fn sig10(row: &[f64], cap: f64, amb: &mut Sigma10Counts) -> Result<f64, PlaceError> {
    sigma10_one(row.try_into().expect("10 entries"), cap, amb)
}

fn sub10(a: &[f64], b: &[f64]) -> [f64; 10] {
    let mut r = [0.0; 10];
    for d in 0..10 {
        r[d] = a[d] - b[d];
    }
    r
}

impl FastEval {
    /// `FastEval(ctx, k)`: `Err(NotHeadingIndependent)` when the visibility depends on the heading (the caller then uses the
    /// slow evaluator).
    pub fn new(ctx: &mut IftContext, k: usize) -> Result<FastEval, PlaceError> {
        let spec = ctx.spec_k(k);
        if !ctx.tb.heading_independent() {
            return Err(PlaceError::NotHeadingIndependent);
        }
        let (kc, n) = (ctx.tb.k, ctx.tb.n);
        let mut v = Vec::with_capacity(kc * n);
        let mut i10 = Vec::with_capacity(kc * n * 10);
        for kk in 0..kc {
            for j in 0..n {
                v.push(ctx.tb.vis_at(kk, 0, j));
                let blk = ctx.tb.info_at(kk, j);
                i10.extend(IU.iter().map(|&(a, b)| blk[a * 4 + b]));
            }
        }
        let mut i10t = vec![0.0; kc * n * 10];
        for kk in 0..kc {
            for j in 0..n {
                let (src, dst) = ((kk * n + j) * 10, (j * kc + kk) * 10);
                i10t[dst..dst + 10].copy_from_slice(&i10[src..src + 10]);
            }
        }
        let occ = ctx.occ().clone();
        let mut vk: Vec<u64> = (0..occ.v_j.len()).map(|i| occ.v_j[i] as u64 * kc as u64 + occ.v_c[i] as u64).collect();
        vk.sort_unstable();
        vk.dedup();
        let (vj_ptr, vj_c) = vis_csr(&vk, kc, n);
        Ok(FastEval {
            k,
            kc,
            n,
            v,
            i10t,
            #[cfg(test)]
            i10,
            cap: spec.sigma_cap,
            tau: spec.tau,
            kmin: spec.k_min as f64,
            wk: spec.w_k,
            tl: spec.tau_loo.unwrap_or(spec.tau),
            is_stand: ctx.kinds.iter().map(|&t| t == CandKind::Stand).collect(),
            occ,
            vj_ptr,
            vj_c,
            sc: Scratch::default(),
            amb: Sigma10Counts::default(),
        })
    }

    /// `_feas_pen` for one row.
    pub(crate) fn feas_pen(&self, sig: f64, nv: f64, loo: f64) -> (bool, f64, f64) {
        let feas = sig <= self.tau && nv >= self.kmin && loo <= self.tl;
        let pen = nmax(0.0, sig / self.tau - 1.0) + nmax(0.0, loo / self.tl - 1.0) + self.wk * nmax(0.0, self.kmin - nv);
        let marg = nmin(1.0 - sig / self.tau, 1.0 - loo / self.tl);
        (feas, pen, marg)
    }

    /// `_base(S')`.
    pub fn base(&mut self, sp: &[usize]) -> Result<Base, PlaceError> {
        let mut b = Base::default();
        let mut amb = std::mem::take(&mut self.amb);
        let r = self.base_into(sp, &mut b, &mut amb);
        self.amb = amb;
        r.map(|()| b)
    }

    /// `_base(S')` into reusable buffers (cleared first).
    fn base_into(&self, sp: &[usize], b: &mut Base, amb: &mut Sigma10Counts) -> Result<(), PlaceError> {
        let (kc, nn, n) = (self.kc, self.n, sp.len());
        let cap = self.cap;
        let Base { bl, w, ip, isum, feas, pen, marg } = b;
        bl.clear();
        bl.resize(kc * nn, false);
        for &s in sp {
            if self.is_stand[s] {
                for t in self.occ.bs_ptr[s]..self.occ.bs_ptr[s + 1] {
                    bl[bl_index(self.occ.bs_c[t] as usize, self.occ.bs_t[t] as usize, nn)] = true;
                }
            }
        }
        w.clear();
        ip.clear();
        for kk in 0..kc {
            for &s in sp {
                w.push(self.v[kk * nn + s] && !bl[kk * nn + s]);
                ip.extend_from_slice(&self.i10t[(s * kc + kk) * 10..(s * kc + kk) * 10 + 10]);
            }
        }
        isum.clear();
        for kk in 0..kc {
            isum.extend_from_slice(&sum_weighted10(&w[kk * n..(kk + 1) * n], &ip[kk * n * 10..(kk + 1) * n * 10]));
        }
        feas.clear();
        pen.clear();
        marg.clear();
        for kk in 0..kc {
            let nv = w[kk * n..(kk + 1) * n].iter().filter(|&&b| b).count() as f64;
            let row = &isum[kk * 10..kk * 10 + 10];
            let sg = sig10(row, cap, amb)?;
            // _loo: max over the visible markers of the capped sigma of Isum - I_m; cap if none visible
            let mut loo = self.cap;
            let mut first = true;
            for m in 0..n {
                if w[kk * n + m] {
                    let s = sig10(&sub10(row, &ip[(kk * n + m) * 10..(kk * n + m) * 10 + 10]), cap, amb)?;
                    loo = if first { s } else { nmax(loo, s) };
                    first = false;
                }
            }
            let (f, p, mg) = self.feas_pen(sg, nv, loo);
            feas.push(f);
            pen.push(p);
            marg.push(mg);
        }
        Ok(())
    }

    /// `set_key(S)`: `f(S)` for a complete selection.
    pub fn set_key(&mut self, s: &[usize]) -> Result<Key3, PlaceError> {
        let b = self.base(s)?;
        let neg_margin = -b.marg[1..].iter().fold(b.marg[0], |m, &x| nmin(m, x));
        Ok(Key3 { n_infeasible: b.feas.iter().filter(|&&f| !f).count(), penalty_sum: np_sum(&b.pen), neg_margin })
    }

    /// `_pairs(S', base, cell_mask)`: per affected pair (cell `c`, candidate `j` not in `S'`) the change of cell `c` when `j` is
    /// added to `S'`. Affected = `j` visible at `c`, or `j`'s board blocks a currently effective marker of `S'` at `c`; every other
    /// pair has change 0. `cell_mask` restricts the pairs to those cells. The information sum is updated sparsely
    /// (`I(c) = I_S'(c) + w_j I_j(c)`); rows where `j`'s board blocks a marker are re-summed exactly (a subtraction would leave
    /// a rounding residue where the true sum is the zero matrix, and the singularity test needs that exact zero).
    pub fn pairs(&mut self, sp: &[usize], base: &Base, cell_mask: Option<&[bool]>) -> Result<Pairs, PlaceError> {
        let mut sc = std::mem::take(&mut self.sc.row.pr);
        let mut out = Pairs::default();
        let mut amb = std::mem::take(&mut self.amb);
        let r = self.pairs_into(sp, base, cell_mask, &mut sc, &mut out, &mut amb);
        self.amb = amb;
        self.sc.row.pr = sc;
        r.map(|()| out)
    }

    fn pairs_into(
        &self,
        sp: &[usize],
        base: &Base,
        cell_mask: Option<&[bool]>,
        sc: &mut PairsScratch,
        out: &mut Pairs,
        amb: &mut Sigma10Counts,
    ) -> Result<(), PlaceError> {
        let (kc, nn, n) = (self.kc, self.n, sp.len());
        let cap = self.cap;
        let PairsScratch { in_s, keys, bk, uk, blk, bidx, wj, isum, wn, rb } = sc;
        let Pairs { pj, pc, d_inf, d_pen, marg, n_resum } = out;
        pj.clear();
        pc.clear();
        d_inf.clear();
        d_pen.clear();
        marg.clear();
        *n_resum = 0;
        in_s.clear();
        in_s.resize(nn, false);
        for &s in sp {
            in_s[s] = true;
        }
        let o = &self.occ;
        // visible pairs: already in pair-key order (candidate-major CSR), so no sort is needed
        keys.clear();
        for (j, &sel) in in_s.iter().enumerate() {
            if sel {
                continue;
            }
            for &c in &self.vj_c[self.vj_ptr[j] as usize..self.vj_ptr[j + 1] as usize] {
                if cell_mask.map_or(true, |m| m[c as usize]) {
                    keys.push((j * kc + c as usize) as i64);
                }
            }
        }
        // (blocker candidate j, cell c, position of the blocked marker in S')
        blk.clear();
        bk.clear();
        for (pos, &m) in sp.iter().enumerate() {
            for t in o.bt_ptr[m]..o.bt_ptr[m + 1] {
                let (bs, bc) = (o.bt_s[t] as usize, o.bt_c[t] as usize);
                if !in_s[bs] && base.w[bc * n + pos] && cell_mask.map_or(true, |mk| mk[bc]) {
                    blk.push((bs, bc, pos));
                    bk.push(pair_key(bs, bc, kc) as i64);
                }
            }
        }
        bk.sort_unstable();
        bk.dedup();
        // sorted union of the two sorted unique key lists = np.unique of their concatenation
        uk.clear();
        let (mut x, mut y) = (0, 0);
        while x < keys.len() && y < bk.len() {
            if keys[x] < bk[y] {
                uk.push(keys[x]);
                x += 1;
            } else if keys[x] > bk[y] {
                uk.push(bk[y]);
                y += 1;
            } else {
                uk.push(keys[x]);
                x += 1;
                y += 1;
            }
        }
        uk.extend_from_slice(&keys[x..]);
        uk.extend_from_slice(&bk[y..]);
        let uk: &[i64] = uk;
        let p = uk.len();
        if p == 0 {
            return Ok(());
        }
        pj.extend(uk.iter().map(|&u| unpair(u as usize, kc).0));
        pc.extend(uk.iter().map(|&u| unpair(u as usize, kc).1));
        bidx.clear();
        bidx.extend(blk.iter().map(|&(j, c, _)| searchsorted_left(uk, pair_key(j, c, kc) as i64)));
        wj.clear();
        wj.extend((0..p).map(|i| self.v[pc[i] * nn + pj[i]] && !base.bl[pc[i] * nn + pj[i]]));
        let i10t = &self.i10t;
        let ij = |i: usize| -> &[f64] {
            let at = uk[i] as usize * 10;
            &i10t[at..at + 10]
        };
        isum.clear();
        isum.resize(p * 10, 0.0);
        for i in 0..p {
            let wf = if wj[i] { 1.0 } else { 0.0 };
            for d in 0..10 {
                isum[i * 10 + d] = base.isum[pc[i] * 10 + d] + wf * ij(i)[d];
            }
        }
        wn.clear();
        wn.resize(p * n, false);
        for i in 0..p {
            wn[i * n..(i + 1) * n].copy_from_slice(&base.w[pc[i] * n..(pc[i] + 1) * n]);
        }
        for (b, &(_, _, pos)) in blk.iter().enumerate() {
            wn[bidx[b] * n + pos] = false;
        }
        let ip_at = |c: usize, m: usize| -> &[f64] { &base.ip[(c * n + m) * 10..(c * n + m) * 10 + 10] };
        rb.clear();
        rb.extend(bidx.iter());
        rb.sort_unstable();
        rb.dedup();
        for &r in rb.iter() {
            let e = sum_weighted10(&wn[r * n..(r + 1) * n], &base.ip[pc[r] * n * 10..(pc[r] + 1) * n * 10]);
            let wf = if wj[r] { 1.0 } else { 0.0 };
            for d in 0..10 {
                isum[r * 10 + d] = e[d] + wf * ij(r)[d];
            }
        }
        d_inf.reserve(p);
        d_pen.reserve(p);
        marg.reserve(p);
        for i in 0..p {
            let nv = (wn[i * n..(i + 1) * n].iter().filter(|&&b| b).count() + wj[i] as usize) as f64;
            let row = &isum[i * 10..i * 10 + 10];
            let sg = sig10(row, cap, amb)?;
            // leave-one-out over the effective markers: those of S' (wn) and j itself (wj)
            let mut loo = f64::NEG_INFINITY;
            for m in 0..n {
                if wn[i * n + m] {
                    let s = sig10(&sub10(row, ip_at(pc[i], m)), cap, amb)?;
                    loo = nmax(loo, s);
                }
            }
            if wj[i] {
                let s = sig10(&sub10(row, ij(i)), cap, amb)?;
                loo = nmax(loo, s);
            }
            if loo == f64::NEG_INFINITY {
                loo = self.cap;
            }
            let (f, pn, mg) = self.feas_pen(sg, nv, loo);
            d_inf.push(!f as i64 - !base.feas[pc[i]] as i64);
            d_pen.push(pn - base.pen[pc[i]]);
            marg.push(mg);
        }
        *n_resum = rb.len();
        Ok(())
    }

    /// `add_keys(S')`: key of `S' + {j}` for every `j`; entries `j` in `S'` are `(BIG, inf, inf)`.
    pub fn add_keys(&mut self, sp: &[usize]) -> Result<AddKeys, PlaceError> {
        let mut sc = std::mem::take(&mut self.sc);
        let r = self.add_keys_with(sp, &mut sc);
        self.sc = sc;
        r
    }

    fn add_keys_with(&mut self, sp: &[usize], sc: &mut Scratch) -> Result<AddKeys, PlaceError> {
        let Scratch { bs, ps, dis, di_s, dp_s, m1, row, .. } = sc;
        let RowScratch { pr, mm, .. } = row;
        let mut amb = std::mem::take(&mut self.amb);
        let r = self.base_into(sp, bs, &mut amb).and_then(|()| self.pairs_into(sp, bs, None, pr, ps, &mut amb));
        self.amb = amb;
        r?;
        let nn = self.n;
        let n_inf0 = bs.feas.iter().filter(|&&f| !f).count() as i64;
        dis.clear();
        dis.extend(ps.d_inf.iter().map(|&d| d as f64));
        bincount_into(&ps.pj, dis, nn, di_s);
        bincount_into(&ps.pj, &ps.d_pen, nn, dp_s);
        let pen0 = np_sum(&bs.pen);
        let mut inf: Vec<i64> = di_s.iter().map(|&x| n_inf0 + x.round_ties_even() as i64).collect();
        let mut pen: Vec<f64> = dp_s.iter().map(|&x| pen0 + x).collect();
        min_margin_into(&bs.marg, &ps.pj, &ps.pc, &ps.marg, None, false, nn, m1, mm);
        let mut nm: Vec<f64> = m1.iter().map(|&x| -x).collect();
        for &s in sp {
            inf[s] = BIG;
            pen[s] = f64::INFINITY;
            nm[s] = f64::INFINITY;
        }
        Ok(AddKeys { inf, pen, nm })
    }
}

/// `neighbourhood` result: the key of `S - S[i] + {j}` for every position `i` and candidate `j`, row-major `[i * n + j]`.
/// Identity moves and `j` in `S` are `(BIG, inf, inf)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Nbhd {
    pub inf: Vec<i64>,
    pub pen: Vec<f64>,
    pub nm: Vec<f64>,
    pub k: usize,
    pub n: usize,
}

impl FastEval {
    /// `neighbourhood(S)`: key of `S - S[i] + {j}` for all `i`, `j`. Shared base: at a cell `c`
    /// where `S[i]` is not effective and its board blocks nothing, `f(c, S - i + j) = f(c, S + j)` and `f(c, S - i) = f(c, S)`.
    /// So the pairs of `S + j` are computed once, and for each `i` only the cells `A_i` where `S[i]` matters are recomputed with
    /// base `S - i`:
    /// `key(S - i + j) = F(S - i) + D_S(j) - sum_{c in A_i} d_S(c, j) + sum_{c in A_i} d_{S-i}(c, j)`
    /// (evaluated left to right, each `bincount` adding sequentially in array order, as in Python). The minimum margin is not
    /// additive: cells outside `A_i` follow `S + j`, cells in `A_i` follow `S - i + j`.
    pub fn neighbourhood(&mut self, s: &[usize]) -> Result<Nbhd, PlaceError> {
        let mut sc = std::mem::take(&mut self.sc);
        let r = self.neighbourhood_with(s, &mut sc);
        self.sc = sc;
        r
    }

    fn neighbourhood_with(&mut self, s: &[usize], sc: &mut Scratch) -> Result<Nbhd, PlaceError> {
        let (k, nn) = (s.len(), self.n);
        let Scratch { bs, ps, di_s, dp_s, row, .. } = sc;
        let mut amb = std::mem::take(&mut self.amb);
        let r = self.base_into(s, bs, &mut amb).and_then(|()| self.pairs_into(s, bs, None, &mut row.pr, ps, &mut amb));
        if let Err(e) = r {
            self.amb = amb;
            return Err(e);
        }
        // one sweep per sum; every bin still receives its additions in array order (np.bincount)
        di_s.clear();
        di_s.resize(nn, 0.0);
        dp_s.clear();
        dp_s.resize(nn, 0.0);
        for q in 0..ps.pj.len() {
            di_s[ps.pj[q]] += ps.d_inf[q] as f64;
            dp_s[ps.pj[q]] += ps.d_pen[q];
        }
        let mut out = Nbhd { inf: Vec::with_capacity(k * nn), pen: Vec::with_capacity(k * nn), nm: Vec::with_capacity(k * nn), k, n: nn };
        #[cfg(feature = "parallel")]
        let threaded = self.parallel_rows(s, bs, ps, di_s, dp_s, &mut out, &mut amb);
        #[cfg(not(feature = "parallel"))]
        let threaded: Option<Result<(), PlaceError>> = None;
        let res = match threaded {
            Some(r) => r,
            None => (0..k).try_for_each(|i| self.nbhd_row(s, i, bs, ps, di_s, dp_s, row, &mut amb, &mut out)),
        };
        self.amb = amb;
        res.map(|()| out)
    }

    /// Row `i` of the neighbourhood (positions are independent given the shared `bs`, `ps`): appends `n` entries to `out`.
    #[allow(clippy::too_many_arguments)]
    fn nbhd_row(
        &self,
        s: &[usize],
        i: usize,
        bs: &Base,
        ps: &Pairs,
        di_s: &[f64],
        dp_s: &[f64],
        rs: &mut RowScratch,
        amb: &mut Sigma10Counts,
        out: &mut Nbhd,
    ) -> Result<(), PlaceError> {
        let (k, nn, kc) = (s.len(), self.n, self.kc);
        let RowScratch { pr, mm, bp, p2, sp, a, bi1, bp1, bi2, bp2, m1, m2 } = rs;
        let si = s[i];
        sp.clear();
        sp.extend(s[..i].iter().chain(&s[i + 1..]).copied());
        // cells where S[i] is effective, plus (stand) the cells where its board blocks any sight line
        a.clear();
        a.extend((0..kc).map(|c| bs.w[c * k + i]));
        if self.is_stand[si] {
            for t in self.occ.bs_ptr[si]..self.occ.bs_ptr[si + 1] {
                a[self.occ.bs_c[t] as usize] = true;
            }
        }
        self.base_into(sp, bp, amb)?;
        self.pairs_into(sp, bp, Some(a), pr, p2, amb)?;
        bi1.clear();
        bi1.resize(nn, 0.0);
        bp1.clear();
        bp1.resize(nn, 0.0);
        for q in 0..ps.pj.len() {
            if a[ps.pc[q]] {
                bi1[ps.pj[q]] += ps.d_inf[q] as f64;
                bp1[ps.pj[q]] += ps.d_pen[q];
            }
        }
        bi2.clear();
        bi2.resize(nn, 0.0);
        bp2.clear();
        bp2.resize(nn, 0.0);
        for q in 0..p2.pj.len() {
            bi2[p2.pj[q]] += p2.d_inf[q] as f64;
            bp2[p2.pj[q]] += p2.d_pen[q];
        }
        let n_inf_p = bp.feas.iter().filter(|&&f| !f).count() as f64;
        let pen_p = np_sum(&bp.pen);
        min_margin_into(&bs.marg, &ps.pj, &ps.pc, &ps.marg, Some(a), true, nn, m1, mm);
        min_margin_into(&bp.marg, &p2.pj, &p2.pc, &p2.marg, Some(a), false, nn, m2, mm);
        let base_at = out.inf.len();
        for j in 0..nn {
            out.inf.push((n_inf_p + di_s[j] - bi1[j] + bi2[j]).round_ties_even() as i64);
            out.pen.push(pen_p + dp_s[j] - bp1[j] + bp2[j]);
            out.nm.push(-nmin(m1[j], m2[j]));
        }
        for &x in s {
            out.inf[base_at + x] = BIG;
            out.pen[base_at + x] = f64::INFINITY;
            out.nm[base_at + x] = f64::INFINITY;
        }
        Ok(())
    }

    /// `parallel` feature: the `k` independent positions on std threads (own scratch and `AMB` counters each), rows concatenated
    /// in `i` order, so the result is bit-identical to the single-threaded one. `None` when one thread would do.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn parallel_rows(
        &self,
        s: &[usize],
        bs: &Base,
        ps: &Pairs,
        di_s: &[f64],
        dp_s: &[f64],
        out: &mut Nbhd,
        amb: &mut Sigma10Counts,
    ) -> Option<Result<(), PlaceError>> {
        let k = s.len();
        let nt = std::thread::available_parallelism().map_or(1, |x| x.get()).min(k);
        if nt < 2 {
            return None;
        }
        let results: Vec<(usize, Result<Nbhd, PlaceError>, Sigma10Counts)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|t| {
                    sc.spawn(move || {
                        let mut rs = RowScratch::default();
                        let mut cnt = Sigma10Counts::default();
                        let mut part = Nbhd { inf: Vec::new(), pen: Vec::new(), nm: Vec::new(), k, n: self.n };
                        // contiguous chunk of positions [lo, hi)
                        let (lo, hi) = (t * k / nt, (t + 1) * k / nt);
                        let r = (lo..hi).try_for_each(|i| self.nbhd_row(s, i, bs, ps, di_s, dp_s, &mut rs, &mut cnt, &mut part));
                        (t, r.map(|()| part), cnt)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("worker panicked")).collect()
        });
        // On an error path the merged counts also include the work threads did after the failing row `i` (diagnostic only).
        let mut first_err = None;
        for (_, r, cnt) in results {
            *amb += cnt;
            match r {
                Ok(part) if first_err.is_none() => {
                    out.inf.extend(part.inf);
                    out.pen.extend(part.pen);
                    out.nm.extend(part.nm);
                }
                Ok(_) => {}
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        Some(first_err.map_or(Ok(()), Err))
    }
}

/// Fallback evaluator (heading-dependent visibility): exact but slow, same interface (`SlowEval`). It holds no tables: the
/// reference evaluation is `IftContext::raw_key`, so every method takes the context.
#[derive(Clone, Debug)]
pub struct SlowEval {
    pub k: usize,
    pub n: usize,
}

impl SlowEval {
    pub fn new(ctx: &IftContext, k: usize) -> SlowEval {
        SlowEval { k, n: ctx.n }
    }

    pub fn set_key(&self, ctx: &IftContext, s: &[usize]) -> Result<Key3, PlaceError> {
        ctx.raw_key(s, self.k)
    }

    /// `(BIG, inf, inf)` for `j` in `S'`, `raw_key(S' + [j], |S'| + 1)` for every other `j`.
    pub fn add_keys(&self, ctx: &IftContext, sp: &[usize]) -> Result<AddKeys, PlaceError> {
        let n = self.n;
        let mut r = AddKeys { inf: vec![BIG; n], pen: vec![f64::INFINITY; n], nm: vec![f64::INFINITY; n] };
        for j in 0..n {
            if sp.contains(&j) {
                continue;
            }
            let mut s2 = sp.to_vec();
            s2.push(j);
            let key = ctx.raw_key(&s2, sp.len() + 1)?;
            r.inf[j] = key.n_infeasible as i64;
            r.pen[j] = key.penalty_sum;
            r.nm[j] = key.neg_margin;
        }
        Ok(r)
    }

    /// Row `i` is `add_keys(S - S[i])`; the identity move and `j` in `S` are `(BIG, inf, inf)` (the identity entry is not computed).
    pub fn neighbourhood(&self, ctx: &IftContext, s: &[usize]) -> Result<Nbhd, PlaceError> {
        let (k, n) = (s.len(), self.n);
        let mut out = Nbhd { inf: Vec::with_capacity(k * n), pen: Vec::with_capacity(k * n), nm: Vec::with_capacity(k * n), k, n };
        for i in 0..k {
            let sp: Vec<usize> = s[..i].iter().chain(&s[i + 1..]).copied().collect();
            let ak = self.add_keys(ctx, &sp)?;
            let at = out.inf.len();
            out.inf.extend(ak.inf);
            out.pen.extend(ak.pen);
            out.nm.extend(ak.nm);
            for &x in s {
                out.inf[at + x] = BIG;
                out.pen[at + x] = f64::INFINITY;
                out.nm[at + x] = f64::INFINITY;
            }
        }
        Ok(out)
    }
}

/// `FastEval`, or `SlowEval` when the visibility depends on the heading.
pub enum Evaluator {
    Fast(Box<FastEval>),
    Slow(SlowEval),
}

impl Evaluator {
    /// Python class attribute `exact_fast`.
    pub fn exact_fast(&self) -> bool {
        matches!(self, Evaluator::Fast(_))
    }

    pub fn set_key(&mut self, ctx: &IftContext, s: &[usize]) -> Result<Key3, PlaceError> {
        match self {
            Evaluator::Fast(f) => f.set_key(s),
            Evaluator::Slow(e) => e.set_key(ctx, s),
        }
    }

    pub fn add_keys(&mut self, ctx: &IftContext, sp: &[usize]) -> Result<AddKeys, PlaceError> {
        match self {
            Evaluator::Fast(f) => f.add_keys(sp),
            Evaluator::Slow(e) => e.add_keys(ctx, sp),
        }
    }

    pub fn neighbourhood(&mut self, ctx: &IftContext, s: &[usize]) -> Result<Nbhd, PlaceError> {
        match self {
            Evaluator::Fast(f) => f.neighbourhood(s),
            Evaluator::Slow(e) => e.neighbourhood(ctx, s),
        }
    }
}

/// `get_eval(ctx, k)`: the cached evaluator of size `k`; `FastEval`, else `SlowEval` exactly when `FastEval::new` refuses
/// (`NotHeadingIndependent`, Python's `AssertionError`). Any other error is returned.
///
/// The returned borrow holds `ctx` mutably, so the evaluator cannot be called with `&ctx` while it lives; use
/// [`IftContext::with_eval`] for that.
pub fn get_eval(ctx: &mut IftContext, k: usize) -> Result<&mut Evaluator, PlaceError> {
    if !ctx.evals.contains_key(&k) {
        let ev = match FastEval::new(ctx, k) {
            Ok(f) => Evaluator::Fast(Box::new(f)),
            Err(PlaceError::NotHeadingIndependent) => Evaluator::Slow(SlowEval::new(ctx, k)),
            Err(e) => return Err(e),
        };
        ctx.evals.insert(k, ev);
        ctx.n_built += 1;
    }
    Ok(ctx.evals.get_mut(&k).expect("inserted above"))
}

impl IftContext {
    /// Runs `f` with the cached evaluator of size `k` (built by [`get_eval`]) and a shared borrow of the context, which
    /// `get_eval`'s `&mut Evaluator` cannot offer (both would borrow the same `IftContext`). The evaluator is moved out of the
    /// cache for the call and put back afterwards. If `f` panics the evaluator is dropped with the unwinding closure frame and is
    /// not in the cache any more: the next `get_eval` / `with_eval` of that `k` rebuilds it (state such as AMB counters restarts).
    pub fn with_eval<R>(&mut self, k: usize, f: impl FnOnce(&mut Evaluator, &IftContext) -> R) -> Result<R, PlaceError> {
        get_eval(self, k)?;
        let mut ev = self.evals.remove(&k).expect("cached by get_eval");
        let r = f(&mut ev, self);
        self.evals.insert(k, ev);
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::demo_spec;
    use crate::visibility::Sector;
    use erploc_linalg::rng::Pcg64;
    use std::collections::{HashMap, HashSet};

    fn site_small() -> SiteTables {
        crate::site_synth::synthetic_site().unwrap()
    }

    /// Distinct random selection of `size` indices below `n` (partial Fisher-Yates), in random order.
    fn rsel(rng: &mut Pcg64, n: usize, size: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..size {
            let j = i + (rng.next_u64() % (n - i) as u64) as usize;
            pool.swap(i, j);
        }
        pool.truncate(size);
        pool
    }

    fn key_close(a: &Key3, b: &Key3) {
        assert_eq!(a.n_infeasible, b.n_infeasible, "count {a:?} vs {b:?}");
        assert!((a.penalty_sum - b.penalty_sum).abs() <= 1e-9 * b.penalty_sum.abs().max(1.0), "penalty {a:?} vs {b:?}");
        // abs 1e-9 for margins of order 1; the margin of a nearly infeasible cell is large and its Python error (fast vs raw)
        // is of the same size (1.2e-9 at margin 28.9 on this room), so the tolerance is relative above 1
        assert!((a.neg_margin - b.neg_margin).abs() <= 1e-9 * b.neg_margin.abs().max(1.0), "margin {a:?} vs {b:?}");
    }

    fn check_set_and_add(ctx: &mut IftContext, n_sets: usize, n_adds: usize, max_size: usize, seed: u64) {
        let nn = ctx.n;
        let mut rng = Pcg64::new(seed);
        let mut fe = FastEval::new(ctx, max_size).unwrap();
        for i in 0..n_sets {
            let size = 1 + i % max_size.min(nn);
            let s = rsel(&mut rng, nn, size);
            let got = fe.set_key(&s).unwrap();
            let want = ctx.raw_key(&s, size).unwrap();
            key_close(&got, &want);
        }
        for i in 0..n_adds {
            let size = i % max_size.min(nn);
            let sp = rsel(&mut rng, nn, size);
            let ak = fe.add_keys(&sp).unwrap();
            for j in 0..nn {
                if sp.contains(&j) {
                    assert_eq!((ak.inf[j], ak.pen[j], ak.nm[j]), (BIG, f64::INFINITY, f64::INFINITY));
                    continue;
                }
                let mut s = sp.clone();
                s.push(j);
                // the spec of the evaluated size, as the Python selftest swap check
                let want = ctx.raw_key(&s, s.len()).unwrap();
                key_close(&Key3 { n_infeasible: ak.inf[j] as usize, penalty_sum: ak.pen[j], neg_margin: ak.nm[j] }, &want);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn set_key_equals_raw() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let mut rng = Pcg64::new(11);
        for i in 0..50 {
            let size = 1 + i % 10;
            let s = rsel(&mut rng, 10, size);
            let got = fe.set_key(&s).unwrap();
            let want = ctx.raw_key(&s, 10).unwrap();
            key_close(&got, &want);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn add_keys_equals_raw() {
        // add_keys uses spec_k(k) fixed at construction; raw_key with k = |S'| + 1 equals it because n_max is not read by evaluate
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        check_set_and_add(&mut ctx, 0, 50, 10, 12);
    }

    #[test]
    fn add_keys_selected_are_big() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut fe = FastEval::new(&mut ctx, 5).unwrap();
        let ak = fe.add_keys(&[2, 7, 3]).unwrap();
        for &s in &[2, 7, 3] {
            assert_eq!((ak.inf[s], ak.pen[s], ak.nm[s]), (BIG, f64::INFINITY, f64::INFINITY));
        }
        assert!(ak.inf.iter().enumerate().all(|(j, &v)| [2, 7, 3].contains(&j) || v < BIG));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn empty_selection_add_keys() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut fe = FastEval::new(&mut ctx, 1).unwrap();
        let ak = fe.add_keys(&[]).unwrap();
        for j in 0..10 {
            let want = ctx.raw_key(&[j], 1).unwrap();
            key_close(&Key3 { n_infeasible: ak.inf[j] as usize, penalty_sum: ak.pen[j], neg_margin: ak.nm[j] }, &want);
        }
        // the base of the empty selection: no marker visible anywhere, every cell infeasible with sigma = cap
        let b = fe.base(&[]).unwrap();
        assert_eq!((b.w.len(), b.ip.len()), (0, 0));
        assert!(b.feas.iter().all(|&f| !f));
    }

    #[test]
    fn refuses_heading_dependent() {
        let mut spec = demo_spec(10);
        spec.vp.sectors = vec![Sector { lam_min: -0.5, lam_max: 0.5, phi_min: -0.4, phi_max: 0.4 }];
        let mut ctx = IftContext::from_spec(&spec).unwrap();
        assert!(!ctx.tb.heading_independent());
        assert!(matches!(FastEval::new(&mut ctx, 4), Err(PlaceError::NotHeadingIndependent)));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn fast_eval_stands_small() {
        let s = site_small();
        for mode in [VisMode::Centre, VisMode::Samples] {
            let mut ctx = IftContext::from_site(&s, mode, Crit::N);
            assert!(ctx.occ().n_triples > 0);
            let nn = ctx.n;
            let stands: Vec<usize> = (0..nn).filter(|&j| ctx.kinds[j] == CandKind::Stand).collect();
            let mut rng = Pcg64::new(21);
            let mut fe = FastEval::new(&mut ctx, 10).unwrap();
            let (mut n_blocked_sets, mut n_resum) = (0, 0);
            for i in 0..40 {
                // stand-heavy selections: half from the stands, so boards block markers
                let size = 2 + i % 8;
                let mut sel = rsel(&mut rng, stands.len(), size.min(stands.len())).iter().map(|&x| stands[x]).collect::<Vec<_>>();
                if i % 3 == 0 {
                    sel[0] = (rng.next_u64() % nn as u64) as usize;
                    sel.sort();
                    sel.dedup();
                }
                let got = fe.set_key(&sel).unwrap();
                let want = ctx.raw_key(&sel, sel.len()).unwrap();
                key_close(&got, &want);
                let b = fe.base(&sel).unwrap();
                n_blocked_sets += b.bl.iter().any(|&x| x) as usize;
                if i % 4 == 0 {
                    let sp = &sel[..sel.len() - 1];
                    let ak = fe.add_keys(sp).unwrap();
                    let pr = {
                        let b0 = fe.base(sp).unwrap();
                        fe.pairs(sp, &b0, None).unwrap()
                    };
                    n_resum += pr.n_resum;
                    for j in (0..nn).step_by(3) {
                        if sp.contains(&j) {
                            continue;
                        }
                        let mut s2 = sp.to_vec();
                        s2.push(j);
                        let want = ctx.raw_key(&s2, s2.len()).unwrap();
                        key_close(&Key3 { n_infeasible: ak.inf[j] as usize, penalty_sum: ak.pen[j], neg_margin: ak.nm[j] }, &want);
                    }
                }
            }
            assert!(n_blocked_sets > 0, "no selection had a blocking board: the Bl branch is not exercised");
            assert!(n_resum > 0, "no pair row was re-summed: the rb branch of _pairs is not exercised");
            eprintln!("fast_eval_stands_small {mode:?}: {n_blocked_sets} selections with blocked sight lines, {n_resum} re-summed pair rows, {} AMB of {}", fe.amb.amb, fe.amb.total);
        }
    }

    /// Stand-heavy selection of `size` (half or more from the stands, 0..=3 walls), as the site self-test generator.
    fn stand_sel(rng: &mut Pcg64, ctx: &IftContext, size: usize) -> Vec<usize> {
        let stands: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Stand).collect();
        let walls: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Wall).collect();
        let nw = ((rng.next_u64() % 4) as usize).min(size).min(walls.len());
        let mut sel: Vec<usize> = rsel(rng, walls.len(), nw).iter().map(|&x| walls[x]).collect();
        sel.extend(rsel(rng, stands.len(), size - nw).iter().map(|&x| stands[x]));
        sel
    }

    /// Every `(i, j)` entry of `neighbourhood(sel)` against `raw_key(sel - sel[i] + j)`; returns the number of entries compared.
    fn check_nbhd(ctx: &IftContext, fe: &mut FastEval, sel: &[usize]) -> usize {
        let nb = fe.neighbourhood(sel).unwrap();
        assert_eq!(
            (nb.k, nb.n, nb.inf.len(), nb.pen.len(), nb.nm.len()),
            (sel.len(), ctx.n, sel.len() * ctx.n, sel.len() * ctx.n, sel.len() * ctx.n)
        );
        let mut cnt = 0;
        for i in 0..sel.len() {
            for j in 0..ctx.n {
                let at = i * ctx.n + j;
                if sel.contains(&j) {
                    assert_eq!((nb.inf[at], nb.pen[at], nb.nm[at]), (BIG, f64::INFINITY, f64::INFINITY));
                    continue;
                }
                let mut s2 = sel.to_vec();
                s2[i] = j;
                let want = ctx.raw_key(&s2, s2.len()).unwrap();
                key_close(&Key3 { n_infeasible: nb.inf[at] as usize, penalty_sum: nb.pen[at], neg_margin: nb.nm[at] }, &want);
                cnt += 1;
            }
        }
        cnt
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn neighbourhood_equals_raw_demo() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let mut rng = Pcg64::new(31);
        let mut n = 0;
        for t in 0..10 {
            let sel = rsel(&mut rng, 10, 3 + t % 3);
            n += check_nbhd(&ctx, &mut fe, &sel);
        }
        eprintln!("neighbourhood_equals_raw_demo: {n} entries equal to raw_key");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn neighbourhood_equals_raw_site_small() {
        let s = site_small();
        for mode in [VisMode::Centre, VisMode::Samples] {
            let mut ctx = IftContext::from_site(&s, mode, Crit::N);
            let mut fe = FastEval::new(&mut ctx, 10).unwrap();
            let mut rng = Pcg64::new(32);
            let mut n = 0;
            for t in 0..8 {
                let sel = stand_sel(&mut rng, &ctx, 4 + t % 7);
                n += check_nbhd(&ctx, &mut fe, &sel);
            }
            eprintln!(
                "neighbourhood_equals_raw_site_small {mode:?}: {n} entries equal to raw_key ({} AMB of {})",
                fe.amb.amb, fe.amb.total
            );
        }
    }

    /// `with_crit` starts with fresh caches and the new criterion; keys equal a context built directly with that criterion.
    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn with_crit_drops_evaluators_and_applies_the_criterion() {
        let s = site_small();
        let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let sel = stand_sel(&mut Pcg64::new(34), &ctx, 5);
        get_eval(&mut ctx, 6).unwrap();
        assert_eq!(ctx.n_built(), 1);
        let mut c3 = ctx.with_crit(Crit::Both);
        assert_eq!(c3.n_built(), 0, "evaluator cache is not carried over");
        assert_eq!(c3.spec0().k_min, 3);
        assert_eq!(c3.spec0().tau_loo, Some(0.05));
        assert_eq!(ctx.spec0().k_min, s.spec0.k_min, "the source context is unchanged");
        let mut direct = IftContext::from_site(&s, VisMode::Samples, Crit::Both);
        let (a, b) = (c3.raw_key(&sel, 6).unwrap(), direct.raw_key(&sel, 6).unwrap());
        assert_eq!((a.n_infeasible, a.penalty_sum.to_bits()), (b.n_infeasible, b.penalty_sum.to_bits()));
        let ka = FastEval::new(&mut c3, 6).unwrap().set_key(&sel).unwrap();
        let kb = FastEval::new(&mut direct, 6).unwrap().set_key(&sel).unwrap();
        assert_eq!((ka.n_infeasible, ka.penalty_sum.to_bits()), (kb.n_infeasible, kb.penalty_sum.to_bits()));
        // overrides apply to the base spec: Both -> N restores the original criterion
        let back = c3.with_crit(Crit::N);
        assert_eq!((back.spec0().k_min, back.spec0().tau_loo), (s.spec0.k_min, s.spec0.tau_loo));
        // blocked triples are shared, not rebuilt
        let mut c = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let nt = c.occ().n_triples;
        assert_eq!(c.with_crit(Crit::K3).n_triples(), Some(nt));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn neighbourhood_identity_and_members_are_big() {
        let s = site_small();
        let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let sel = stand_sel(&mut Pcg64::new(33), &ctx, 6);
        let nb = fe.neighbourhood(&sel).unwrap();
        for i in 0..sel.len() {
            for j in 0..ctx.n {
                let at = i * ctx.n + j;
                if sel.contains(&j) {
                    assert_eq!((nb.inf[at], nb.pen[at], nb.nm[at]), (BIG, f64::INFINITY, f64::INFINITY), "i {i} j {j}");
                } else {
                    assert!(
                        nb.inf[at] < BIG && nb.pen[at].is_finite() && nb.nm[at].is_finite(),
                        "i {i} j {j}: {} {} {}",
                        nb.inf[at],
                        nb.pen[at],
                        nb.nm[at]
                    );
                }
            }
        }
    }

    /// Row `i` of the neighbourhood equals `add_keys(S - S[i])` with `BIG` at `S`.
    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn neighbourhood_equals_loop_of_add_keys() {
        let s = site_small();
        for mode in [VisMode::Centre, VisMode::Samples] {
            let mut ctx = IftContext::from_site(&s, mode, Crit::N);
            let mut fe = FastEval::new(&mut ctx, 10).unwrap();
            let sel = stand_sel(&mut Pcg64::new(34), &ctx, 7);
            let nb = fe.neighbourhood(&sel).unwrap();
            for i in 0..sel.len() {
                let sp: Vec<usize> = sel[..i].iter().chain(&sel[i + 1..]).copied().collect();
                let ak = fe.add_keys(&sp).unwrap();
                for j in 0..ctx.n {
                    let at = i * ctx.n + j;
                    if sel.contains(&j) {
                        assert_eq!(nb.inf[at], BIG);
                        continue;
                    }
                    assert_eq!(nb.inf[at], ak.inf[j], "{mode:?} i {i} j {j}: count");
                    assert!(
                        (nb.pen[at] - ak.pen[j]).abs() <= 1e-9 * ak.pen[j].abs().max(1.0),
                        "{mode:?} i {i} j {j}: penalty {} vs {}",
                        nb.pen[at],
                        ak.pen[j]
                    );
                    assert!(
                        (nb.nm[at] - ak.nm[j]).abs() <= 1e-9 * ak.nm[j].abs().max(1.0),
                        "{mode:?} i {i} j {j}: margin {} vs {}",
                        nb.nm[at],
                        ak.nm[j]
                    );
                }
            }
        }
    }

    fn sector_ctx() -> IftContext {
        let mut spec = demo_spec(10);
        spec.vp.sectors = vec![Sector { lam_min: -0.5, lam_max: 0.5, phi_min: -0.4, phi_max: 0.4 }];
        IftContext::from_spec(&spec).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn slow_eval_equals_raw() {
        let ctx = sector_ctx();
        assert!(!ctx.tb.heading_independent());
        let slow = SlowEval::new(&ctx, 4);
        let mut rng = Pcg64::new(35);
        for t in 0..4 {
            let sel = rsel(&mut rng, 10, 2 + t);
            assert_eq!(slow.set_key(&ctx, &sel).unwrap(), ctx.raw_key(&sel, 4).unwrap());
            let sp = &sel[..sel.len() - 1];
            let ak = slow.add_keys(&ctx, sp).unwrap();
            for j in 0..10 {
                if sp.contains(&j) {
                    assert_eq!((ak.inf[j], ak.pen[j], ak.nm[j]), (BIG, f64::INFINITY, f64::INFINITY));
                    continue;
                }
                let mut s2 = sp.to_vec();
                s2.push(j);
                let want = ctx.raw_key(&s2, s2.len()).unwrap();
                assert_eq!((ak.inf[j] as usize, ak.pen[j], ak.nm[j]), (want.n_infeasible, want.penalty_sum, want.neg_margin));
            }
            let nb = slow.neighbourhood(&ctx, &sel).unwrap();
            for i in 0..sel.len() {
                for j in 0..10 {
                    let at = i * 10 + j;
                    if sel.contains(&j) {
                        assert_eq!((nb.inf[at], nb.pen[at], nb.nm[at]), (BIG, f64::INFINITY, f64::INFINITY));
                        continue;
                    }
                    let mut s2 = sel.clone();
                    s2[i] = j;
                    let want = ctx.raw_key(&s2, s2.len()).unwrap();
                    assert_eq!((nb.inf[at] as usize, nb.pen[at], nb.nm[at]), (want.n_infeasible, want.penalty_sum, want.neg_margin));
                }
            }
        }
        // the evaluator enum forwards to the same code
        let mut ev = Evaluator::Slow(slow);
        assert!(!ev.exact_fast());
        let sel = [1usize, 5, 8];
        assert_eq!(ev.set_key(&ctx, &sel).unwrap(), ctx.raw_key(&sel, 4).unwrap());
    }

    /// A sector-free scene through both evaluators: Fast and Slow agree on set / add / neighbourhood.
    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn slow_fast_agree_on_homogeneous_vis() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut fe = Evaluator::Fast(Box::new(FastEval::new(&mut ctx, 4).unwrap()));
        let mut sl = Evaluator::Slow(SlowEval::new(&ctx, 4));
        assert!(fe.exact_fast() && !sl.exact_fast());
        let mut rng = Pcg64::new(36);
        for t in 0..5 {
            let sel = rsel(&mut rng, 10, 2 + t % 3);
            key_close(&fe.set_key(&ctx, &sel).unwrap(), &sl.set_key(&ctx, &sel).unwrap());
            let (a, b) = (fe.add_keys(&ctx, &sel).unwrap(), sl.add_keys(&ctx, &sel).unwrap());
            let (na, nb) = (fe.neighbourhood(&ctx, &sel).unwrap(), sl.neighbourhood(&ctx, &sel).unwrap());
            for j in 0..10 {
                assert_eq!(a.inf[j], b.inf[j]);
                if a.pen[j].is_finite() {
                    key_close(
                        &Key3 { n_infeasible: a.inf[j] as usize, penalty_sum: a.pen[j], neg_margin: a.nm[j] },
                        &Key3 { n_infeasible: b.inf[j] as usize, penalty_sum: b.pen[j], neg_margin: b.nm[j] },
                    );
                }
            }
            assert_eq!(na.inf, nb.inf);
            for q in 0..na.pen.len() {
                if na.pen[q].is_finite() {
                    key_close(
                        &Key3 { n_infeasible: na.inf[q] as usize, penalty_sum: na.pen[q], neg_margin: na.nm[q] },
                        &Key3 { n_infeasible: nb.inf[q] as usize, penalty_sum: nb.pen[q], neg_margin: nb.nm[q] },
                    );
                } else {
                    assert_eq!((na.pen[q], na.nm[q]), (nb.pen[q], nb.nm[q]));
                }
            }
        }
    }

    #[test]
    fn get_eval_falls_back() {
        let mut ctx = sector_ctx();
        let ev = get_eval(&mut ctx, 4).unwrap();
        assert!(matches!(ev, Evaluator::Slow(_)) && !ev.exact_fast());
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let ev = get_eval(&mut ctx, 4).unwrap();
        assert!(matches!(ev, Evaluator::Fast(_)) && ev.exact_fast());
    }

    #[test]
    fn get_eval_is_cached() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        assert_eq!(ctx.n_built(), 0);
        get_eval(&mut ctx, 4).unwrap();
        get_eval(&mut ctx, 4).unwrap();
        assert_eq!(ctx.n_built(), 1, "same k: built once");
        get_eval(&mut ctx, 5).unwrap();
        assert_eq!(ctx.n_built(), 2, "other k: other evaluator");
        ctx.with_eval(4, |_, _| ()).unwrap();
        assert_eq!(ctx.n_built(), 2, "with_eval uses the cache");
        // with_eval hands out the cached evaluator together with a shared context, and puts it back
        let sel = [0usize, 3, 6];
        let amb0 = ctx
            .with_eval(4, |e, c| {
                e.set_key(c, &sel).unwrap();
                match e {
                    Evaluator::Fast(f) => f.amb.total,
                    _ => 0,
                }
            })
            .unwrap();
        let amb1 = ctx
            .with_eval(4, |e, _| match e {
                Evaluator::Fast(f) => f.amb.total,
                _ => 0,
            })
            .unwrap();
        assert!(amb0 > 0 && amb0 == amb1, "the evaluator state survives with_eval ({amb0} vs {amb1})");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn cell_mask_filters_pairs() {
        let s = site_small();
        let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let sp = vec![3usize, 20, 41, 47, 53];
        let b = fe.base(&sp).unwrap();
        let all = fe.pairs(&sp, &b, None).unwrap();
        let mask: Vec<bool> = (0..fe.kc).map(|c| c % 3 != 1).collect();
        let part = fe.pairs(&sp, &b, Some(&mask)).unwrap();
        let keep: Vec<usize> = (0..all.pj.len()).filter(|&i| mask[all.pc[i]]).collect();
        assert!(!keep.is_empty() && keep.len() < all.pj.len());
        assert_eq!(part.pj, keep.iter().map(|&i| all.pj[i]).collect::<Vec<_>>());
        assert_eq!(part.pc, keep.iter().map(|&i| all.pc[i]).collect::<Vec<_>>());
        assert_eq!(part.d_inf, keep.iter().map(|&i| all.d_inf[i]).collect::<Vec<_>>());
        for (a, &i) in part.d_pen.iter().zip(&keep) {
            assert_eq!(a.to_bits(), all.d_pen[i].to_bits());
        }
        for (a, &i) in part.marg.iter().zip(&keep) {
            assert_eq!(a.to_bits(), all.marg[i].to_bits());
        }
    }

    /// Literal transcription of `FastEval._min_margin` (sets, builtin `min`, stable argsort with NaN last).
    fn min_margin_literal(m0: &[f64], pj: &[usize], pc: &[usize], mn: &[f64], mask: Option<&[bool]>, n: usize) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; n];
        let mm: Vec<f64> = match mask {
            None => m0.to_vec(),
            Some(mk) => m0.iter().zip(mk).map(|(&m, &b)| if b { m } else { f64::INFINITY }).collect(),
        };
        let (mut pj2, mut pc2, mut mn2) = (vec![], vec![], vec![]);
        for i in 0..pj.len() {
            if mask.map_or(true, |mk| mk[pc[i]]) {
                pj2.push(pj[i]);
                pc2.push(pc[i]);
                mn2.push(mn[i]);
            }
        }
        for i in 0..pj2.len() {
            let j = pj2[i];
            // np.minimum.at
            out[j] = if out[j].is_nan() || mn2[i].is_nan() {
                f64::NAN
            } else if mn2[i] < out[j] {
                mn2[i]
            } else {
                out[j]
            };
        }
        let mut order: Vec<usize> = (0..mm.len()).collect();
        // stable insertion sort with numpy's NaN-last less-than
        let lt = |a: f64, b: f64| a < b || (b.is_nan() && !a.is_nan());
        for i in 1..order.len() {
            let mut k = i;
            while k > 0 && lt(mm[order[k]], mm[order[k - 1]]) {
                order.swap(k, k - 1);
                k -= 1;
            }
        }
        let c0 = order[0];
        let first = mm[c0];
        let mut aff: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in 0..pj2.len() {
            aff.entry(pj2[i]).or_default().insert(pc2[i]);
        }
        let pymin = |a: f64, b: f64| if b < a { b } else { a };
        for (j, oj) in out.iter_mut().enumerate() {
            match aff.get(&j) {
                Some(a) if a.contains(&c0) => {
                    for &c in &order {
                        if !mm[c].is_finite() {
                            break;
                        }
                        if !a.contains(&c) {
                            *oj = pymin(*oj, mm[c]);
                            break;
                        }
                    }
                }
                _ => *oj = pymin(*oj, first),
            }
        }
        out
    }

    fn same_bits(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()), "{what}: entry {i}: {x} vs {y}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn min_margin_equals_python_loop() {
        let mut rng = Pcg64::new(5);
        let mut u = |m: usize| (rng.next_u64() % m as u64) as usize;
        let special = [f64::INFINITY, f64::NAN, 0.0, -0.0, 0.5, 0.5, -1.0];
        let mut cases = 0;
        for t in 0..3000 {
            let (kc, n) = (1 + u(12), 1 + u(7));
            let mut m0: Vec<f64> = (0..kc).map(|_| (u(9) as f64 - 4.0) / 4.0).collect();
            if t % 3 == 0 {
                for x in m0.iter_mut() {
                    if u(4) == 0 {
                        *x = special[u(special.len())];
                    }
                }
            }
            let np = u(30);
            let mut keys: Vec<(usize, usize)> = (0..np).map(|_| (u(n), u(kc))).collect();
            keys.sort();
            keys.dedup();
            let pj: Vec<usize> = keys.iter().map(|k| k.0).collect();
            let pc: Vec<usize> = keys.iter().map(|k| k.1).collect();
            let mn: Vec<f64> =
                keys.iter().map(|_| if t % 5 == 0 && u(6) == 0 { special[u(special.len())] } else { (u(9) as f64 - 4.0) / 4.0 }).collect();
            let mask: Vec<bool> = (0..kc).map(|_| u(3) != 0).collect();
            for mk in [None, Some(&mask[..])] {
                same_bits(&min_margin(&m0, &pj, &pc, &mn, mk, n), &min_margin_literal(&m0, &pj, &pc, &mn, mk, n), "min_margin");
                cases += 1;
            }
        }
        assert_eq!(cases, 6000);
    }

    #[test]
    fn min_margin_semantics() {
        // cell 0 (smallest) affected for j = 0 only
        let m0 = [0.1, 0.5, 0.9];
        let r = min_margin(&m0, &[0], &[0], &[0.7], None, 2);
        assert_eq!(r, vec![0.5f64.min(0.7), 0.1]);
        // all cells masked out: +inf
        let r = min_margin(&m0, &[], &[], &[], Some(&[false, false, false]), 2);
        assert_eq!(r, vec![f64::INFINITY; 2]);
    }
}
