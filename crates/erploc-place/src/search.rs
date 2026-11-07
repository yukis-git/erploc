//! IFT search module: the random-number trait `IftRng` (with `Pcg64` and the recorded-draw `Replay`), the clocks, `swap_move`,
//! `pick` / `better`, `run_ift` (greedy construction, best-improvement tabu phases, perturbation), `ift_det`, `ift_min_count`,
//! `optimize_ift`, `brute_key`, the IFT self-checks, and the self-tests (`selftest`, `selftest_site`) of `erploc/ift.py`
//! and of the Python site IFT tooling.
//!
//! Parity with Python: numpy's `Generator` is not ported. A perturbed run is reproduced exactly by
//! replaying the recorded draws (`Replay`) under a virtual clock (`TickClock`, +1.0 per call, so the stop point is deterministic);
//! the deterministic `ift_det` needs no random numbers at all.
use crate::fast_eval::{Evaluator, IftContext, BIG};
use crate::placement::{
    brute_force, choose_survey_triplet, combinations, evaluate, HistoryStep, Key3, PlaceError, PlacementResult, PlacementSpec,
};
use crate::site::CandKind;
use erploc_linalg::rng::Pcg64;

/// The draws the search and the self-tests use (`rng.integers`, `rng.random`, `rng.choice(.., replace=False)`).
pub trait IftRng {
    /// Uniform integer in `[lo, hi)`.
    fn integers(&mut self, lo: usize, hi: usize) -> usize;
    /// Uniform double in `[0, 1)`.
    fn random(&mut self) -> f64;
    /// `rng.choice(pool_len, size, replace=False)`: `size` distinct indices below `pool_len`, in draw order. The default is a
    /// partial Fisher-Yates shuffle over `integers` (not numpy's stream); `Replay` returns the recorded indices.
    fn choice(&mut self, pool_len: usize, size: usize) -> Vec<usize> {
        assert!(size <= pool_len, "choice: size {size} > pool {pool_len}");
        let mut pool: Vec<usize> = (0..pool_len).collect();
        for i in 0..size {
            let j = self.integers(i, pool_len);
            pool.swap(i, j);
        }
        pool.truncate(size);
        pool
    }
}

impl IftRng for Pcg64 {
    fn integers(&mut self, lo: usize, hi: usize) -> usize {
        assert!(lo < hi, "integers: empty range [{lo}, {hi})");
        lo + (self.next_u64() % (hi - lo) as u64) as usize
    }
    fn random(&mut self) -> f64 {
        self.uniform01()
    }
}

/// `rng.choice(pool_len, size, replace=False)` as indices.
pub fn choice(rng: &mut dyn IftRng, pool_len: usize, size: usize) -> Vec<usize> {
    rng.choice(pool_len, size)
}

/// Recorded draws of a numpy `Generator`, replayed in call order (test-only parity). Rows of the draw list are
/// `(kind, a, b, value)`: kind 0 = `integers(a, b)` with result `value`, 1 = `random()` with result `value`, 2 = `choice(a, b)`
/// whose result is the next `b` entries of the flat `choices` list. A call whose kind or arguments differ from the record panics.
pub struct Replay {
    draws: Vec<[f64; 4]>,
    choices: Vec<f64>,
    pos: usize,
    cpos: usize,
}

impl Replay {
    pub fn new(draws: &[[f64; 4]], choices: &[f64]) -> Replay {
        Replay { draws: draws.to_vec(), choices: choices.to_vec(), pos: 0, cpos: 0 }
    }

    /// Number of recorded draws consumed so far.
    pub fn consumed(&self) -> usize {
        self.pos
    }

    /// Panics unless every recorded draw was consumed.
    pub fn assert_exhausted(&self) {
        assert_eq!(self.pos, self.draws.len(), "Replay: {} of {} draws consumed", self.pos, self.draws.len());
        assert_eq!(self.cpos, self.choices.len(), "Replay: choice values left over");
    }

    fn next(&mut self, kind: f64, a: f64, b: f64) -> [f64; 4] {
        let d = *self.draws.get(self.pos).unwrap_or_else(|| panic!("Replay: draws exhausted at call {}", self.pos));
        assert!(d[0] == kind && d[1] == a && d[2] == b, "Replay: draw {} is {:?}, requested ({kind}, {a}, {b})", self.pos, &d[..3]);
        self.pos += 1;
        d
    }
}

impl IftRng for Replay {
    fn integers(&mut self, lo: usize, hi: usize) -> usize {
        let d = self.next(0.0, lo as f64, hi as f64);
        assert!(d[3] >= lo as f64 && d[3] < hi as f64, "Replay: recorded integer {} outside [{lo}, {hi})", d[3]);
        d[3] as usize
    }
    fn random(&mut self) -> f64 {
        self.next(1.0, 0.0, 0.0)[3]
    }
    fn choice(&mut self, pool_len: usize, size: usize) -> Vec<usize> {
        self.next(2.0, pool_len as f64, size as f64);
        let v: Vec<usize> = self.choices[self.cpos..self.cpos + size].iter().map(|&x| x as usize).collect();
        self.cpos += size;
        assert!(v.iter().all(|&x| x < pool_len), "Replay: recorded choice outside the pool");
        v
    }
}

/// Source of `time.time()` for the search (the call sites are part of the replay contract).
pub trait Clock {
    fn now(&mut self) -> f64;
}

/// Seconds since the Unix epoch (`time.time()`).
pub struct WallClock;

impl Clock for WallClock {
    fn now(&mut self) -> f64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
    }
}

/// Virtual clock: returns 0.0, 1.0, 2.0, ... (+1.0 per call), the counting clock the exporter patches into `erploc.ift.time`.
#[derive(Default)]
pub struct TickClock {
    n: f64,
}

impl TickClock {
    pub fn new() -> TickClock {
        TickClock { n: 0.0 }
    }
}

impl Clock for TickClock {
    fn now(&mut self) -> f64 {
        let v = self.n;
        self.n += 1.0;
        v
    }
}

/// Search budget. `Wall` and `Ticks` are a deadline `t0 + b` on the given clock (they differ only in which clock `ift_det` uses);
/// `Iterations(n)` stops at the `(n+1)`-th tabu iteration (which is counted, as the aborted iteration is in Python), i.e. after
/// `n` neighbourhood evaluations; `Unlimited` never stops (a perturbed run with it never terminates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Wall(f64),
    Ticks(f64),
    Iterations(usize),
    Unlimited,
}

/// `max(1.0, abs(x))` of Python (`max` keeps the first argument unless the second is larger, so NaN gives 1.0).
fn max1(x: f64) -> f64 {
    if x.abs() > 1.0 {
        x.abs()
    } else {
        1.0
    }
}

/// `_pen_lt`.
pub(crate) fn pen_lt(a: f64, b: f64) -> bool {
    a < b - 1e-9 * max1(b)
}

/// `_pen_eq`: `abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))`.
pub(crate) fn pen_eq(a: f64, b: f64) -> bool {
    let mut m = 1.0;
    if a.abs() > m {
        m = a.abs();
    }
    if b.abs() > m {
        m = b.abs();
    }
    (a - b).abs() <= 1e-9 * m
}

/// `_better(a, b)`: key `a` strictly better than `b`: count, then penalty with a relative tolerance of 1e-9, then (negated) margin
/// with the same tolerance. Search-only; the strict lexicographic order is `Key3::lt`.
/// Not transitive (tolerance chains; an infinite penalty equals every penalty), mirrored from Python on purpose:
/// kept by decision 2026-10-13 because results within 1e-9 relative are unaffected and parity with `erploc/ift.py` is bit-exact.
pub fn better(a: &Key3, b: &Key3) -> bool {
    better_core(
        a.n_infeasible,
        b.n_infeasible,
        pen_eq(a.penalty_sum, b.penalty_sum),
        a.penalty_sum < b.penalty_sum,
        pen_lt(a.neg_margin, b.neg_margin),
    )
}

/// The decision structure of `better` on the three float predicates (all pure, so evaluating them eagerly changes nothing):
/// `pen_eq` = penalties equal within the band, `p_lt` = `a.penalty < b.penalty`, `m_lt` = `a.margin < b.margin - 1e-9 max(1, |b.margin|)`.
/// Float-free (Kani).
pub(crate) fn better_core(na: usize, nb: usize, pen_eq: bool, p_lt: bool, m_lt: bool) -> bool {
    if na != nb {
        return na < nb;
    }
    if !pen_eq {
        return p_lt;
    }
    m_lt
}

/// `np.argmin`: the first NaN if any, else the first minimum.
fn np_argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x < v[best] {
            best = i;
        }
    }
    best
}

/// `np.min` of floats: NaN propagates.
fn np_min(v: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        if x < m {
            m = x;
        }
    }
    m
}

/// `pick(INF, PEN, NM)`: lexicographic argmin of (count, penalty, margin) over a row-major matrix, the penalty with a relative tie
/// band of 1e-9. `None` when every entry is `BIG`. Returns the flat index (first minimum in row-major order) and its key.
pub fn pick(inf: &[i64], pen: &[f64], nm: &[f64]) -> Option<(usize, Key3)> {
    assert!(inf.len() == pen.len() && pen.len() == nm.len() && !inf.is_empty(), "pick: shapes");
    let mn = *inf.iter().min().expect("non-empty");
    if mn >= BIG {
        return None;
    }
    let p2: Vec<f64> = inf.iter().zip(pen).map(|(&c, &p)| if c == mn { p } else { f64::INFINITY }).collect();
    let pm = np_min(&p2);
    let band = pm + 1e-9 * max1(pm);
    let n2: Vec<f64> = p2.iter().zip(nm).map(|(&p, &m)| if p <= band { m } else { f64::INFINITY }).collect();
    let idx = np_argmin(&n2);
    Some((idx, Key3 { n_infeasible: mn as usize, penalty_sum: p2[idx], neg_margin: n2[idx] }))
}

/// `swap_move`: replace one selected marker by an unselected one (with `local`, half of the moves draw from the removed marker's
/// `near` list). The draw order is part of the contract: `integers(0, len)`, `random()` (only if `local`), then either
/// `integers(0, N)` until the candidate is unselected, or `integers(0, len(pool))`.
pub fn swap_move(ctx: &IftContext, sel: &[usize], rng: &mut dyn IftRng, local: bool) -> Vec<usize> {
    assert!(sel.len() < ctx.n, "swap_move: every candidate is selected");
    let i = rng.integers(0, sel.len());
    let r = sel[i];
    let mut pool: Vec<usize> = Vec::new();
    if local && rng.random() < 0.5 {
        pool = ctx.near[r].iter().copied().filter(|j| !sel.contains(j)).collect();
    }
    let j = if pool.is_empty() {
        loop {
            let j = rng.integers(0, ctx.n);
            if !sel.contains(&j) {
                break j;
            }
        }
    } else {
        pool[rng.integers(0, pool.len())]
    };
    let mut new = sel.to_vec();
    new[i] = j;
    new
}

/// Extra information of a run (Python `r['ift']`).
#[derive(Clone, Debug, PartialEq)]
pub struct IftInfo {
    pub t_construction: f64,
    pub iters: u64,
    pub phases: u64,
    pub fast_key_ok: bool,
    pub fast_key: Key3,
    pub exact_fast: bool,
    pub t_stall: usize,
    pub tenure: usize,
    pub perturb: bool,
    pub n_triples: Option<usize>,
    /// Key of every `pick` result in call order (construction, then tabu steps); not in Python, kept for the tie-equivalence checks.
    pub picks: Vec<Key3>,
}

/// Result of `run_ift` (Python `dict`).
#[derive(Clone, Debug, PartialEq)]
pub struct IftRun {
    /// Sorted.
    pub best_sel: Vec<usize>,
    /// The reference (`raw_key`) key of `best_sel`.
    pub best_key: Key3,
    pub feasible: f64,
    pub evals: u64,
    pub time: f64,
    /// `(evals, 1 - best count / K)` after every `note`.
    pub trace: Vec<(u64, f64)>,
    /// `(clock - t0, evals, 1 - best count / K)`.
    pub trace_time: Vec<(f64, u64, f64)>,
    pub info: IftInfo,
}

/// Internal early exit: the deadline (`Done`) or an evaluator error.
enum Stop {
    Done,
    Err(PlaceError),
}

impl From<PlaceError> for Stop {
    fn from(e: PlaceError) -> Stop {
        Stop::Err(e)
    }
}

/// State of one `run_ift` (the Python closure variables `st`, `t0`, `deadline`).
struct Run<'a> {
    c: &'a IftContext,
    k: usize,
    n: usize,
    clock: &'a mut dyn Clock,
    rng: &'a mut dyn IftRng,
    t0: f64,
    deadline: f64,
    max_iters: u64,
    t_stall: usize,
    tenure: usize,
    evals: u64,
    best: Option<Key3>,
    best_sel: Vec<usize>,
    trace: Vec<(u64, f64)>,
    trace_t: Vec<(f64, u64, f64)>,
    iters: u64,
    phases: u64,
    picks: Vec<Key3>,
}

impl Run<'_> {
    fn best(&self) -> Key3 {
        self.best.expect("best is set after the construction")
    }

    fn note(&mut self) {
        let f = 1.0 - self.best().n_infeasible as f64 / self.c.kc as f64;
        self.trace.push((self.evals, f));
        let t = self.clock.now() - self.t0;
        self.trace_t.push((t, self.evals, f));
    }

    fn consider(&mut self, sel: &[usize], key: Key3) {
        if self.best.map_or(true, |b| better(&key, &b)) {
            self.best = Some(key);
            let mut s = sel.to_vec();
            s.sort_unstable();
            self.best_sel = s;
        }
    }

    /// Best-improvement tabu search over the full swap neighbourhood; returns the phase's best (selection, key).
    fn tabu_phase(&mut self, ev: &mut Evaluator, s0: &[usize], key0: Key3) -> Result<(Vec<usize>, Key3), Stop> {
        let (k, n) = (self.k, self.n);
        let mut s = s0.to_vec();
        let (mut pb_sel, mut pb_key) = (s.clone(), key0);
        let mut tabu = vec![0usize; n];
        let (mut it, mut last_imp) = (0usize, 0usize);
        while it - last_imp < self.t_stall {
            it += 1;
            self.iters += 1;
            if self.clock.now() > self.deadline || self.iters > self.max_iters {
                return Err(Stop::Done);
            }
            let nb = ev.neighbourhood(self.c, &s)?;
            self.evals += (k * (n - k)) as u64;
            let gb = self.best();
            let gb0 = gb.n_infeasible as i64;
            let allowed_inf: Vec<i64> = (0..k * n)
                .map(|x| {
                    let (inf, pen, nm) = (nb.inf[x], nb.pen[x], nb.nm[x]);
                    let pl = pen_lt(pen, gb.penalty_sum);
                    let pe = (pen - gb.penalty_sum).abs() <= 1e-9 * max1(gb.penalty_sum);
                    let aspir = inf < gb0 || (inf == gb0 && (pl || (pe && nm < gb.neg_margin - 1e-9 * max1(gb.neg_margin))));
                    if tabu[x % n] <= it || aspir {
                        inf
                    } else {
                        BIG
                    }
                })
                .collect();
            let Some((idx, mkey)) = pick(&allowed_inf, &nb.pen, &nb.nm) else { break };
            self.picks.push(mkey);
            let (i, j) = (idx / n, idx % n);
            let r = s[i];
            s[i] = j;
            tabu[r] = it + self.tenure;
            self.consider(&s, mkey);
            if better(&mkey, &pb_key) {
                pb_sel = s.clone();
                pb_key = mkey;
                last_imp = it;
            }
            self.note();
        }
        Ok((pb_sel, pb_key))
    }

    fn search(&mut self, ev: &mut Evaluator, perturb: bool) -> Result<(), Stop> {
        let (home_sel, home_key) = (self.best_sel.clone(), self.best());
        let (_, hkey) = self.tabu_phase(ev, &home_sel, home_key)?;
        let mut hkey = hkey;
        self.phases = 1;
        if !perturb {
            return Ok(());
        }
        loop {
            let mut cand = self.best_sel.clone();
            for _ in 0..self.rng.integers(2, 4) {
                cand = swap_move(self.c, &cand, self.rng, true);
            }
            let ckey = ev.set_key(self.c, &cand)?;
            self.evals += 1;
            self.consider(&cand, ckey);
            let (_, key2) = self.tabu_phase(ev, &cand, ckey)?;
            self.phases += 1;
            if !better(&hkey, &key2) {
                hkey = key2;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_core(
    ev: &mut Evaluator,
    c: &IftContext,
    k: usize,
    budget: &Budget,
    clock: &mut dyn Clock,
    rng: &mut dyn IftRng,
    perturb: bool,
    (t_stall, tenure): (usize, usize),
) -> Result<IftRun, PlaceError> {
    let n = c.n;
    let t0 = clock.now();
    let (deadline, max_iters) = match *budget {
        Budget::Wall(b) | Budget::Ticks(b) => (t0 + b, u64::MAX),
        Budget::Iterations(m) => (f64::INFINITY, m as u64),
        Budget::Unlimited => (f64::INFINITY, u64::MAX),
    };
    let mut st = Run {
        c,
        k,
        n,
        clock,
        rng,
        t0,
        deadline,
        max_iters,
        t_stall,
        tenure,
        evals: 0,
        best: None,
        best_sel: Vec::new(),
        trace: Vec::new(),
        trace_t: Vec::new(),
        iters: 0,
        phases: 0,
        picks: Vec::new(),
    };
    // 1. greedy construction (exact occlusion-aware objective)
    let mut s: Vec<usize> = Vec::new();
    for _ in 0..k {
        let ak = ev.add_keys(c, &s)?;
        st.evals += (n - s.len()) as u64;
        let (j, key) = pick(&ak.inf, &ak.pen, &ak.nm).expect("a free candidate exists while |S| < k <= N");
        st.picks.push(key);
        s.push(j);
    }
    let key = ev.set_key(c, &s)?;
    st.evals += 1;
    st.consider(&s, key);
    st.note();
    let t_constr = st.clock.now() - st.t0;
    match st.search(ev, perturb) {
        Ok(()) | Err(Stop::Done) => {}
        Err(Stop::Err(e)) => return Err(e),
    }
    let elapsed = st.clock.now() - st.t0;
    st.note();
    let best = st.best();
    let raw = c.raw_key(&st.best_sel, k)?;
    let ok = raw.n_infeasible == best.n_infeasible
        && (raw.penalty_sum - best.penalty_sum).abs() <= 1e-6 * max1(raw.penalty_sum)
        && (raw.neg_margin - best.neg_margin).abs() <= 1e-6 * max1(raw.neg_margin);
    Ok(IftRun {
        best_sel: st.best_sel.clone(),
        best_key: raw,
        feasible: 1.0 - raw.n_infeasible as f64 / c.kc as f64,
        evals: st.evals,
        time: elapsed,
        trace: st.trace.clone(),
        trace_time: st.trace_t.clone(),
        info: IftInfo {
            t_construction: t_constr,
            iters: st.iters,
            phases: st.phases,
            fast_key_ok: ok,
            fast_key: best,
            exact_fast: ev.exact_fast(),
            t_stall,
            tenure,
            perturb,
            n_triples: None,
            picks: st.picks.clone(),
        },
    })
}

/// `run_ift(ctx, k, tbudget, seed, perturb, T_stall, tenure)`: greedy construction with the exact evaluator, then a tabu phase
/// (and, with `perturb`, repeated perturbation + tabu phases until the budget ends). The clock and the random source are
/// parameters; `clock.now()` is called exactly where Python calls `time.time()`.
///
/// Precondition `1 <= k <= N` (`k < N` too when `perturb`, because `swap_move` needs a free candidate); a violation, or a perturbed run
/// without a budget, is `PlaceError::InvalidInput`. The search proper needs `k < N`: a swap exchanges a selected candidate for a free one,
/// so with `k = N` the construction takes every candidate, no move exists and the run ends after one iteration.
#[allow(clippy::too_many_arguments)]
pub fn run_ift(
    ctx: &mut IftContext,
    k: usize,
    budget: &Budget,
    clock: &mut dyn Clock,
    rng: &mut dyn IftRng,
    perturb: bool,
    t_stall: usize,
    tenure: usize,
) -> Result<IftRun, PlaceError> {
    let bad = |m: String| Err(PlaceError::InvalidInput(format!("run_ift: {m}")));
    if k < 1 || k > ctx.n {
        return bad(format!("k = {k} outside 1..={}", ctx.n));
    }
    if perturb && *budget == Budget::Unlimited {
        return bad("a perturbed run needs a budget".into());
    }
    if perturb && k >= ctx.n {
        return bad(format!("a perturbed run needs k < N (k = {k}, N = {})", ctx.n));
    }
    let mut r = ctx.with_eval(k, |ev, c| run_core(ev, c, k, budget, clock, rng, perturb, (t_stall, tenure)))??;
    r.info.n_triples = ctx.n_triples();
    Ok(r)
}

/// A random source that must not be used (`perturb = false`).
struct NoRng;

impl IftRng for NoRng {
    fn integers(&mut self, _: usize, _: usize) -> usize {
        panic!("ift_det draws no random numbers")
    }
    fn random(&mut self) -> f64 {
        panic!("ift_det draws no random numbers")
    }
}

/// `ift_det`: no perturbation, `T_stall = 15`, `tenure = 7`. The clock is a `WallClock` for `Budget::Wall` and a `TickClock`
/// otherwise (then `time` fields count virtual seconds).
///
/// # Errors
/// `PlaceError::InvalidInput` unless `1 <= k <= N`.
pub fn ift_det(ctx: &mut IftContext, k: usize, budget: &Budget) -> Result<IftRun, PlaceError> {
    match budget {
        Budget::Wall(_) => run_ift(ctx, k, budget, &mut WallClock, &mut NoRng, false, 15, 7),
        _ => run_ift(ctx, k, budget, &mut TickClock::new(), &mut NoRng, false, 15, 7),
    }
}

/// One size of `ift_min_count` (Python `per_k` entry).
#[derive(Clone, Debug, PartialEq)]
pub struct PerK {
    pub k: usize,
    pub key: Key3,
    pub evals: u64,
    pub time: f64,
}

/// Result of `ift_min_count`.
#[derive(Clone, Debug, PartialEq)]
pub struct MinCount {
    pub k: usize,
    pub sel: Vec<usize>,
    pub key: Key3,
    pub found: bool,
    pub per_k: Vec<PerK>,
}

/// `ift_min_count`: `ift_det` for `k = 1, 2, ...` up to `min(k_max or spec.n_max, N)`, stopping at the first size with no
/// infeasible cell; otherwise the result of the last size with `found = false`.
pub fn ift_min_count(ctx: &mut IftContext, k_max: Option<usize>, budget: &Budget) -> Result<MinCount, PlaceError> {
    let k_max = k_max.unwrap_or(ctx.spec0().n_max).min(ctx.n);
    if k_max < 1 {
        return Err(PlaceError::InvalidInput("ift_min_count: k_max = 0 (no candidates, or n_max / k_max is 0)".into()));
    }
    let mut per_k = Vec::new();
    let mut last = None;
    for k in 1..=k_max {
        let r = ift_det(ctx, k, budget)?;
        per_k.push(PerK { k, key: r.best_key, evals: r.evals, time: r.time });
        if r.best_key.n_infeasible == 0 {
            return Ok(MinCount { k, sel: r.best_sel.clone(), key: r.best_key, found: true, per_k });
        }
        last = Some(r);
    }
    let r = last.expect("k_max >= 1");
    Ok(MinCount { k: k_max, sel: r.best_sel, key: r.best_key, found: false, per_k })
}

/// `optimize_ift`: tables, minimal count by the IFT k-loop, evaluation of that selection, survey triplet. The history holds one
/// `Ift` step per tried size, then `Survey`.
///
/// # Errors
/// `PlaceError::InvalidInput` for an empty candidate list, workspace polygon, headings or `z_cam`, or a spacing that is not positive
/// and finite.
pub fn optimize_ift(spec: &PlacementSpec, sigma_ts: f64, budget: &Budget) -> Result<PlacementResult, PlaceError> {
    if spec.candidates.is_empty() {
        return Err(PlaceError::InvalidInput("optimize_ift: no candidates".into()));
    }
    let mut ctx = IftContext::from_spec(spec)?;
    let mc = ift_min_count(&mut ctx, None, budget)?;
    let sel = mc.sel.clone();
    let stats = evaluate(&ctx.tb, &sel, spec)?;
    let (ids, score, n_skip) = choose_survey_triplet(&ctx.tb, spec, &sel, sigma_ts)?;
    let mut history: Vec<HistoryStep> =
        mc.per_k.iter().map(|p| HistoryStep::Ift { k: p.k, n_infeasible: p.key.n_infeasible, penalty: p.key.penalty_sum }).collect();
    history.push(HistoryStep::Survey { ids: ids.clone(), score, n_skip });
    Ok(PlacementResult { sel, stats, survey_ids: ids, history })
}

/// `brute_key`: the lexicographic minimum (`Key3::lt`, first of equals) of `raw_key` over all `k`-subsets (small instances only).
///
/// # Errors
/// `PlaceError::InvalidInput` when `k > N` (no subset exists).
pub fn brute_key(ctx: &IftContext, k: usize) -> Result<Key3, PlaceError> {
    if k > ctx.n {
        return Err(PlaceError::InvalidInput(format!("brute_key: subset size {k} exceeds the {} candidates", ctx.n)));
    }
    let mut best: Option<Key3> = None;
    for c in combinations(ctx.n, k) {
        let key = ctx.raw_key(&c, k)?;
        if best.map_or(true, |b| key.lt(&b)) {
            best = Some(key);
        }
    }
    best.ok_or_else(|| PlaceError::InvalidInput("brute_key: no subset".into()))
}

/// One line of the checks (Python `CheckResult`).
#[derive(Clone, Debug)]
pub struct CheckLine {
    pub name: String,
    pub passed: bool,
    pub value: String,
    pub criterion: String,
}

/// The four IFT checks of `erploc/ift.py run_self_checks` on the demo room: evaluator self-test (selections, swaps with
/// neighbourhoods; Rust's own random stream), `ift_det` k = 3 against brute force, and the minimal count against `brute_force`.
pub fn run_self_checks_ift() -> Result<Vec<CheckLine>, PlaceError> {
    let spec = crate::placement::demo_spec(10);
    let mut res = Vec::new();
    let mut ctx0 = IftContext::from_spec(&spec)?;
    let s = selftest(&mut ctx0, 200, 200, &mut Pcg64::new(1), 3, 5)?;
    res.push(CheckLine {
        name: "IFT FastEval == evaluate: selections".into(),
        passed: s.exact_fast && s.bad_sel == 0,
        value: format!("{}/{} equal (penalty rel err max {:.1e}, K={} N={})", s.n_sel - s.bad_sel, s.n_sel, s.err_sel, s.k, s.n),
        criterion: "all equal".into(),
    });
    res.push(CheckLine {
        name: "IFT FastEval == evaluate: swaps".into(),
        passed: s.bad_swap == 0,
        value: format!("{}/{} equal (penalty rel err max {:.1e})", s.n_swap - s.bad_swap, s.n_swap, s.err_swap),
        criterion: "all equal".into(),
    });
    let mut ctx = IftContext::from_spec(&spec)?;
    let r3 = ift_det(&mut ctx, 3, &Budget::Unlimited)?;
    let kb = brute_key(&ctx, 3)?;
    res.push(CheckLine {
        name: "3 markers: IFT vs brute force objective".into(),
        passed: !better(&kb, &r3.best_key),
        value: format!("IFT {:?} vs brute force {:?}", r3.best_key, kb),
        criterion: "(count, sum penalty, -margin) at least as good as brute force".into(),
    });
    let mc = ift_min_count(&mut ctx, None, &Budget::Unlimited)?;
    let sel_opt = brute_force(&ctx.tb, &spec, None)?;
    let st = evaluate(&ctx.tb, &mc.sel, &spec)?;
    let n_inf = st.feasible.iter().filter(|&&f| !f).count();
    res.push(CheckLine {
        name: "minimal count: IFT k-loop vs brute force".into(),
        passed: mc.found && mc.sel.len() == sel_opt.len() && n_inf == 0,
        value: format!("{} vs {} (infeasible {})", mc.sel.len(), sel_opt.len(), n_inf),
        criterion: "equal count and every cell feasible".into(),
    });
    Ok(res)
}

/// Result of a self-test (`dict` of `erploc.ift.selftest`): `n_swap`/`bad_swap` include the neighbourhood checks, as in Python.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfTest {
    pub exact_fast: bool,
    pub n_sel: usize,
    pub bad_sel: usize,
    pub err_sel: f64,
    pub n_swap: usize,
    pub bad_swap: usize,
    pub err_swap: f64,
    /// Candidates `N`.
    pub n: usize,
    /// Cells `K`.
    pub k: usize,
    /// Neighbourhood entries compared (included in `n_swap`) and mismatched (included in `bad_swap`).
    pub n_nbhd: usize,
    pub bad_nbhd: usize,
}

/// `e = max(|dpen| / max(1, |pen|), |dmargin|)` of the Python self-test; `NaN` propagates (a NaN `e` compares false with `<=`).
fn err3(f: &Key3, pen: f64, nm: f64) -> f64 {
    let a = (f.penalty_sum - pen).abs() / f.penalty_sum.abs().max(1.0);
    let b = (f.neg_margin - nm).abs();
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Python acceptance threshold of the self-tests (kept; the differential cases use tighter values).
pub const SELFTEST_TOL: f64 = 1e-6;

fn bad(count_equal: bool, e: f64) -> bool {
    !count_equal || e.is_nan() || e > SELFTEST_TOL
}

/// Common body of `erploc.ift.selftest` and `ift_lib.selftest`: `n_sel` selections (`set_key` vs `raw_key`), `n_swap` swaps x
/// 3 candidates (`add_keys` vs `raw_key`) and `n_nbhd` full neighbourhoods (every `(i, j)` vs `raw_key`). `size(t, rng)` gives the
/// size `k` of check `t` (also the evaluator size, as the Python), `rsel(k, rng)` a random selection of size `k`.
fn run_selftest(
    ctx: &mut IftContext,
    (n_sel, n_swap, n_nbhd): (usize, usize, usize),
    rng: &mut dyn IftRng,
    k_hi: usize,
    size: &mut dyn FnMut(usize, &mut dyn IftRng) -> usize,
    rsel: &mut dyn FnMut(&IftContext, usize, &mut dyn IftRng) -> Vec<usize>,
) -> Result<SelfTest, PlaceError> {
    let exact_fast = ctx.with_eval(k_hi, |e, _| e.exact_fast())?;
    let (mut bad_sel, mut err_sel) = (0usize, 0.0f64);
    for t in 0..n_sel {
        let k = size(t, rng);
        let s = rsel(ctx, k, rng);
        let f = ctx.raw_key(&s, k)?;
        let g = ctx.with_eval(k, |e, c| e.set_key(c, &s))??;
        let e = err3(&f, g.penalty_sum, g.neg_margin);
        err_sel = nan_max(err_sel, e);
        if bad(f.n_infeasible == g.n_infeasible, e) {
            bad_sel += 1;
        }
    }
    let (mut bad_swap, mut err_swap, mut n_swap_done) = (0usize, 0.0f64, 0usize);
    for t in 0..n_swap {
        let k = size(t, rng);
        let s = rsel(ctx, k, rng);
        let i = rng.integers(0, k);
        let sp: Vec<usize> = s[..i].iter().chain(&s[i + 1..]).copied().collect();
        let ak = ctx.with_eval(k, |e, c| e.add_keys(c, &sp))??;
        for j in choice(rng, ctx.n, 3) {
            if sp.contains(&j) {
                continue;
            }
            let mut s2 = sp.clone();
            s2.push(j);
            let f = ctx.raw_key(&s2, k)?;
            let e = err3(&f, ak.pen[j], ak.nm[j]);
            err_swap = nan_max(err_swap, e);
            n_swap_done += 1;
            if bad(f.n_infeasible as i64 == ak.inf[j], e) {
                bad_swap += 1;
            }
        }
    }
    let (mut bad_nb, mut n_nb) = (0usize, 0usize);
    for t in 0..n_nbhd {
        let k = size(t, rng);
        let s = rsel(ctx, k, rng);
        let nb = ctx.with_eval(k, |e, c| e.neighbourhood(c, &s))??;
        for i in 0..k {
            for j in 0..ctx.n {
                if s.contains(&j) {
                    debug_assert_eq!(nb.inf[i * nb.n + j], BIG);
                    continue;
                }
                let mut s2 = s.clone();
                s2[i] = j;
                let f = ctx.raw_key(&s2, k)?;
                let e = err3(&f, nb.pen[i * nb.n + j], nb.nm[i * nb.n + j]);
                n_nb += 1;
                if bad(f.n_infeasible as i64 == nb.inf[i * nb.n + j], e) {
                    bad_nb += 1;
                }
            }
        }
    }
    Ok(SelfTest {
        exact_fast,
        n_sel,
        bad_sel,
        err_sel,
        n_swap: n_swap_done + n_nb,
        bad_swap: bad_swap + bad_nb,
        err_swap,
        n: ctx.n,
        k: ctx.kc,
        n_nbhd: n_nb,
        bad_nbhd: bad_nb,
    })
}

/// Python `max(worst, e)` on floats: a NaN `e` never replaces the running value.
fn nan_max(worst: f64, e: f64) -> f64 {
    if e > worst {
        e
    } else {
        worst
    }
}

/// `erploc.ift.selftest`: `n_sel` selections of size `k in [k_lo, k_hi]` (any candidates), `n_swap` swaps x 3 candidates and 20
/// full neighbourhoods. The random stream is the given `rng`, not numpy's.
pub fn selftest(
    ctx: &mut IftContext,
    n_sel: usize,
    n_swap: usize,
    rng: &mut dyn IftRng,
    k_lo: usize,
    k_hi: usize,
) -> Result<SelfTest, PlaceError> {
    assert!(k_lo >= 1 && k_lo <= k_hi, "selftest: k range");
    // the draw order of Python: the size is drawn before the selection
    let mut size = |_t: usize, rng: &mut dyn IftRng| rng.integers(k_lo, k_hi + 1);
    let mut rsel = |c: &IftContext, k: usize, rng: &mut dyn IftRng| choice(rng, c.n, k);
    run_selftest(ctx, (n_sel, n_swap, 20), rng, k_hi, &mut size, &mut rsel)
}

/// `ift_lib.selftest` generator `rsel(k)`: with probability 0.7 draw `nw` in 0..=3 walls (otherwise 0), then `k - nw` stands.
pub fn rsel_site(ctx: &IftContext, k: usize, rng: &mut dyn IftRng) -> Vec<usize> {
    let walls: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Wall).collect();
    let stands: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Stand).collect();
    let nw = if rng.random() < 0.7 { rng.integers(0, 4) } else { 0 };
    let nw = nw.min(walls.len()).min(k);
    let mut out: Vec<usize> = choice(rng, walls.len(), nw).into_iter().map(|x| walls[x]).collect();
    out.extend(choice(rng, stands.len(), k - nw).into_iter().map(|x| stands[x]));
    out
}

/// `ift_lib.selftest` for a site context: k alternates 9 (even `t`) and 8 (odd `t`); `n_nbhd` neighbourhoods are checked in
/// addition (the Python site self-test has none; a full-size site makes every neighbourhood k x N reference evaluations).
pub fn selftest_site(
    ctx: &mut IftContext,
    n_sel: usize,
    n_swap: usize,
    n_nbhd: usize,
    rng: &mut dyn IftRng,
) -> Result<SelfTest, PlaceError> {
    let mut size = |t: usize, _rng: &mut dyn IftRng| if t % 2 == 1 { 8 } else { 9 };
    let mut rsel = |c: &IftContext, k: usize, rng: &mut dyn IftRng| rsel_site(c, k, rng);
    run_selftest(ctx, (n_sel, n_swap, n_nbhd), rng, 9, &mut size, &mut rsel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_is_distinct_and_in_range() {
        let mut rng = Pcg64::new(3);
        for n in 1..20 {
            for k in 0..=n {
                let c = choice(&mut rng, n, k);
                let mut d = c.clone();
                d.sort();
                d.dedup();
                assert_eq!((c.len(), d.len()), (k, k));
                assert!(c.iter().all(|&x| x < n));
            }
        }
    }

    #[test]
    fn integers_in_range() {
        let mut rng = Pcg64::new(4);
        let mut seen = [false; 5];
        for _ in 0..200 {
            seen[rng.integers(3, 8) - 3] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn err3_nan_propagates_and_counts_bad() {
        let f = Key3 { n_infeasible: 1, penalty_sum: 2.0, neg_margin: 0.5 };
        assert!(err3(&f, f64::NAN, 0.5).is_nan());
        assert!(bad(true, f64::NAN));
        assert!(!bad(true, 1e-9));
        assert!(bad(false, 0.0));
        assert_eq!(nan_max(1.0, f64::NAN), 1.0);
    }

    use crate::placement::demo_spec;

    fn k3(c: usize, p: f64, m: f64) -> Key3 {
        Key3 { n_infeasible: c, penalty_sum: p, neg_margin: m }
    }

    #[test]
    fn better_truth_table() {
        // count decides first, whatever the rest
        assert!(better(&k3(1, 1e9, 1e9), &k3(2, 0.0, -1e9)));
        assert!(!better(&k3(2, 0.0, -1e9), &k3(1, 1e9, 1e9)));
        // penalty: equal within 1e-9 * max(1, |a|, |b|) falls through to the margin; beyond it the penalty decides
        let b = k3(3, 100.0, 0.0);
        assert!(better(&k3(3, 100.0 - 0.99e-7, -5.0), &k3(3, 100.0, 5.0)), "inside the band the margin decides");
        assert!(!better(&k3(3, 100.0 - 0.99e-7, 6.0), &k3(3, 100.0, 5.0)), "inside the band a worse margin loses");
        assert!(better(&k3(3, 100.0 - 1.1e-7, 5.0), &b), "outside the band the smaller penalty wins even with a worse margin");
        assert!(!better(&k3(3, 100.0 + 1.1e-7, -5.0), &b));
        // small penalties: the band is 1e-9 absolute
        assert!(better(&k3(0, 0.0, 9.0), &k3(0, 2e-9, 0.0)));
        assert!(!better(&k3(0, 0.0, 9.0), &k3(0, 0.5e-9, 0.0)), "equal penalties: margin 9 is not below 0");
        // margin tolerance 1e-9 * max(1, |b2|): not better when inside the band
        assert!(!better(&k3(1, 1.0, 5.0 - 0.5e-9), &k3(1, 1.0, 5.0)));
        assert!(better(&k3(1, 1.0, 5.0 - 6e-9), &k3(1, 1.0, 5.0)));
        assert!(better(&k3(1, 1.0, 0.5 - 2e-9), &k3(1, 1.0, 0.5)));
        assert!(!better(&k3(1, 1.0, 0.5 - 0.5e-9), &k3(1, 1.0, 0.5)));
        // identical keys: never better
        assert!(!better(&b, &b));
        // NaN: never better
        let nan = f64::NAN;
        assert!(!better(&k3(1, nan, 0.0), &k3(1, 1.0, 1.0)));
        assert!(!better(&k3(1, 1.0, nan), &k3(1, 1.0, 1.0)));
        assert!(!better(&k3(1, 1.0, 0.0), &k3(1, nan, 1.0)));
        assert!(!better(&k3(1, 1.0, 0.0), &k3(1, 1.0, nan)));
    }

    /// Finding of the Kani harnesses: `better` is NOT transitive, not even acyclic. With penalties 1, 1 + 0.9e-9, 1 + 1.8e-9 (a ~ b and b ~ c inside the
    /// 1e-9 band, a < c outside it) and margins 0, -1, -2 (reversed order): better(b, a), better(c, b) by margin and better(a, c) by
    /// penalty, a 3-cycle. Python `_better` gives the same (checked with erploc.ift._better on the same triples: True True True).
    /// The search tolerates it: `better` only decides whether a plateau move / aspiration replaces the incumbent, while the best
    /// count `n_infeasible` is compared strictly first (an incumbent never loses a feasible point to a cycle) and every move is
    /// bounded by the budget; a cycle can at worst make the tabu phase wander on a penalty plateau of width 1e-9.
    #[test]
    fn better_is_not_transitive() {
        let (a, b, c) = (k3(3, 1.0, 0.0), k3(3, 1.0 + 0.9e-9, -1.0), k3(3, 1.0 + 1.8e-9, -2.0));
        assert!(better(&b, &a) && better(&c, &b) && better(&a, &c));
        assert!(!better(&a, &b) && !better(&b, &c) && !better(&c, &a));
    }

    /// Infinite penalties / margins (excluded from the Kani harnesses by their NaN check), expected values checked against Python `_better`.
    /// FINDING (same in Python): an infinite penalty is `pen_eq` to everything (`inf <= 1e-9 * inf`), so the margin decides.
    #[test]
    fn better_infinite_keys() {
        let inf = f64::INFINITY;
        // penalties
        assert!(!better(&k3(1, inf, 0.0), &k3(1, inf, 0.0)));
        assert!(!better(&k3(1, 1.0, 0.0), &k3(1, inf, 0.0)) && !better(&k3(1, inf, 0.0), &k3(1, 1.0, 0.0)));
        assert!(!better(&k3(1, -inf, 0.0), &k3(1, 1.0, 0.0)) && !better(&k3(1, 1.0, 0.0), &k3(1, -inf, 0.0)));
        assert!(better(&k3(1, 1.0, -5.0), &k3(1, inf, 0.0)) && better(&k3(1, inf, -5.0), &k3(1, 1.0, 0.0)), "the margin decides");
        // margins
        assert!(
            !better(&k3(1, 1.0, inf), &k3(1, 1.0, inf))
                && !better(&k3(1, 1.0, 0.0), &k3(1, 1.0, inf))
                && !better(&k3(1, 1.0, inf), &k3(1, 1.0, 0.0))
        );
        assert!(better(&k3(1, 1.0, -inf), &k3(1, 1.0, 0.0)));
        assert!(!better(&k3(1, 1.0, -inf), &k3(1, 1.0, -inf)));
    }

    #[test]
    fn better_penalty_band_edge() {
        // exactly at the edge |a - b| = 1e-9 * max(1, |a|, |b|) counts as equal (<=), so the margin decides
        let b: f64 = 8.0;
        let a = b - 1e-9 * b;
        assert!((a - b).abs() <= 1e-9 * 8.0);
        assert!(!better(&k3(0, a, 1.0), &k3(0, b, 0.0)));
        assert!(better(&k3(0, a, -1.0), &k3(0, b, 0.0)));
        let a2 = b - 3e-9 * b;
        assert!(better(&k3(0, a2, 1.0), &k3(0, b, 0.0)));
    }

    #[test]
    fn pick_first_minimum_row_major() {
        // two rows x three columns; the minimum count 1 occurs at flat 1, 4; equal penalty and margin: the first wins
        let inf = [2, 1, 3, 2, 1, 5];
        let pen = [0.0, 4.0, 0.0, 0.0, 4.0, 0.0];
        let nm = [0.0, -1.0, 0.0, 0.0, -1.0, 0.0];
        let (idx, key) = pick(&inf, &pen, &nm).unwrap();
        assert_eq!((idx, key), (1, k3(1, 4.0, -1.0)));
        // a strictly better margin later wins
        let nm2 = [0.0, -1.0, 0.0, 0.0, -2.0, 0.0];
        assert_eq!(pick(&inf, &pen, &nm2).unwrap().0, 4);
        // the smallest penalty among the minimal count, not among all
        let pen3 = [0.0, 4.0, 0.0, 0.0, 3.0, 0.0];
        assert_eq!(pick(&inf, &pen3, &nm).unwrap().0, 4);
    }

    #[test]
    fn pick_all_big_is_none() {
        assert!(pick(&[BIG, BIG, BIG], &[f64::INFINITY; 3], &[f64::INFINITY; 3]).is_none());
        assert!(pick(&[BIG, 7, BIG], &[0.0; 3], &[0.0; 3]).is_some());
    }

    #[test]
    fn pick_penalty_ties_use_margin() {
        let inf = [1, 1, 1, 1];
        // 100.0 and 100.0 + 5e-8 are within 1e-9 * 100 = 1e-7: the margin picks the later one; 100.0 + 2e-7 is outside
        let pen = [100.0, 100.0 + 5e-8, 100.0 + 2e-7, 100.0];
        let nm = [0.0, -3.0, -9.0, 1.0];
        let (idx, key) = pick(&inf, &pen, &nm).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(key, k3(1, 100.0 + 5e-8, -3.0));
        // the tie band is anchored at the minimum penalty (not at the previous entry)
        let pen2 = [100.0, 100.0 + 0.9e-7, 100.0 + 1.8e-7, 100.0];
        let nm2 = [0.0, 1.0, -9.0, 2.0];
        assert_eq!(pick(&inf, &pen2, &nm2).unwrap().0, 0);
    }

    fn demo_ctx() -> IftContext {
        IftContext::from_spec(&demo_spec(10)).unwrap()
    }

    fn rows(v: &[[f64; 4]]) -> Replay {
        Replay::new(v, &[])
    }

    #[test]
    fn swap_move_draw_order() {
        let ctx = demo_ctx();
        let sel = [2usize, 5, 7];
        let r = 5;
        let pool: Vec<usize> = ctx.near[r].iter().copied().filter(|j| !sel.contains(j)).collect();
        assert!(pool.len() >= 2, "the demo room has near candidates: {pool:?}");
        // (a) local = false: integers(0, 3), then integers(0, N) until an unselected candidate (2 is selected, 9 is not)
        let mut rp = rows(&[[0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 10.0, 2.0], [0.0, 0.0, 10.0, 9.0]]);
        assert_eq!(swap_move(&ctx, &sel, &mut rp, false), vec![2, 9, 7]);
        rp.assert_exhausted();
        // (b) local, random() < 0.5, non-empty pool: integers(0, 3), random(), integers(0, len(pool))
        let mut rp = rows(&[[0.0, 0.0, 3.0, 1.0], [1.0, 0.0, 0.0, 0.25], [0.0, 0.0, pool.len() as f64, 1.0]]);
        assert_eq!(swap_move(&ctx, &sel, &mut rp, true), vec![2, pool[1], 7]);
        rp.assert_exhausted();
        // (c) local, random() >= 0.5: the pool is not used, integers(0, N)
        let mut rp = rows(&[[0.0, 0.0, 3.0, 0.0], [1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 10.0, 4.0]]);
        assert_eq!(swap_move(&ctx, &sel, &mut rp, true), vec![4, 5, 7]);
        rp.assert_exhausted();
    }

    #[test]
    fn perturbed_run_with_k_equal_n_is_an_error() {
        let mut ctx = demo_ctx();
        let n = ctx.n;
        let e = run_ift(&mut ctx, n, &Budget::Iterations(5), &mut TickClock::new(), &mut Pcg64::new(2), true, 15, 7).unwrap_err();
        assert!(matches!(&e, PlaceError::InvalidInput(m) if m.contains("needs k < N")), "{e}");
    }

    #[test]
    #[should_panic(expected = "Replay")]
    fn swap_move_replay_mismatch_panics() {
        let ctx = demo_ctx();
        // local = true needs a random() draw after the index; the record has an integers(0, 10) there
        let mut rp = rows(&[[0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 10.0, 9.0]]);
        swap_move(&ctx, &[2, 5, 7], &mut rp, true);
    }

    #[test]
    fn swap_move_never_returns_member() {
        let ctx = demo_ctx();
        let mut rng = Pcg64::new(11);
        for t in 0..500 {
            let k = 1 + t % 8;
            let sel = choice(&mut rng, ctx.n, k);
            let new = swap_move(&ctx, &sel, &mut rng, t % 2 == 0);
            assert_eq!(new.len(), k);
            let changed: Vec<usize> = (0..k).filter(|&i| new[i] != sel[i]).collect();
            assert_eq!(changed.len(), 1, "exactly one position changes");
            assert!(!sel.contains(&new[changed[0]]));
            let mut d = new.clone();
            d.sort();
            d.dedup();
            assert_eq!(d.len(), k);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn tick_clock_budget_stops() {
        let mut ctx = demo_ctx();
        let mut clock = TickClock::new();
        assert_eq!((clock.now(), clock.now(), clock.now()), (0.0, 1.0, 2.0));
        let r = run_ift(&mut ctx, 3, &Budget::Ticks(50.0), &mut TickClock::new(), &mut Pcg64::new(5), true, 15, 7).unwrap();
        // deadline = 0 + 50: the first deadline check that returns > 50 returns 51, `elapsed` then reads 52
        assert_eq!(r.time, 52.0);
        assert!(r.info.phases >= 1 && r.info.iters >= 10);
        assert!(r.trace_time.windows(2).all(|w| w[0].0 < w[1].0), "virtual time increases");
        assert_eq!(r.trace_time.last().unwrap().0, 52.0 + 1.0, "the last note reads the clock once more");
        assert!(r.info.fast_key_ok);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn run_ift_k1_trivial() {
        let mut ctx = demo_ctx();
        let r = ift_det(&mut ctx, 1, &Budget::Unlimited).unwrap();
        assert_eq!(r.best_sel.len(), 1);
        let kb = brute_key(&ctx, 1).unwrap();
        assert!(!better(&kb, &r.best_key), "{kb:?} vs {:?}", r.best_key);
        assert!(r.info.fast_key_ok && r.info.exact_fast && !r.info.perturb);
        assert_eq!(r.info.n_triples, Some(0));
        assert_eq!(r.info.phases, 1);
        assert_eq!(r.feasible, 1.0 - r.best_key.n_infeasible as f64 / ctx.kc as f64);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn min_count_stops_at_first_feasible() {
        let mut ctx = demo_ctx();
        let mc = ift_min_count(&mut ctx, None, &Budget::Unlimited).unwrap();
        assert!(mc.found && mc.key.n_infeasible == 0);
        assert_eq!(mc.per_k.len(), mc.k);
        assert!(mc.per_k[..mc.k - 1].iter().all(|p| p.key.n_infeasible > 0));
        assert_eq!(mc.sel.len(), mc.k);
        let short = ift_min_count(&mut ctx, Some(mc.k - 1), &Budget::Unlimited).unwrap();
        assert!(!short.found && short.k == mc.k - 1 && short.key.n_infeasible > 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // > 400 s under Miri (killed at 420 s)
    fn iterations_budget() {
        let mut ctx = demo_ctx();
        let (k, n) = (3usize, ctx.n);
        let r = ift_det(&mut ctx, k, &Budget::Iterations(2)).unwrap();
        let construction = (0..k).map(|i| n - i).sum::<usize>() + 1;
        assert_eq!(r.info.iters, 3, "the aborted third iteration is counted, as in Python");
        assert_eq!(r.evals, (construction + 2 * k * (n - k)) as u64);
        // a perturbed run with an iteration budget terminates too
        let p = run_ift(&mut ctx, k, &Budget::Iterations(100), &mut TickClock::new(), &mut Pcg64::new(2), true, 15, 7).unwrap();
        assert_eq!(p.info.iters, 101);
        assert!(p.info.phases >= 2);
    }
}
