//! Generic Levenberg-Marquardt (Nielsen damping, manifold retract, Huber IRLS). Port of `erploc/lm.py`.

use erploc_linalg::{chol, lu, DMat, LinalgError};

/// Absolute gradient floor: a fully collapsed gradient counts as converged.
pub const GTOL: f64 = 1e-14;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    Linear,
    /// Python `loss='huber'` (default delta 1.5).
    Huber {
        delta: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// numpy.linalg.solve
    Lu,
    Cholesky,
    /// Columns [0, n_cam_cols) are camera blocks of width `cam_width`; A's camera–camera part must be block diagonal.
    /// Remaining columns (the marker part) are solved through the Schur complement with LU.
    Schur {
        n_cam_cols: usize,
        cam_width: usize,
    },
}

/// Internal normal builder: (x, unweighted r) -> (A, g).
pub(crate) type DenseNormal<'a, X> = dyn FnMut(&X, &[f64]) -> (DMat, Vec<f64>) + 'a;

/// Builds the weighted normal equations at `x`: given the UNWEIGHTED residual `r` (from `residual(x)`) and optional
/// sqrt weights `w` (Huber; `None` for Linear), return `(A, g)` with `A = Jwᵀ Jw`, `g = Jwᵀ (w ⊙ r)` (w = 1 if None).
pub type NormalFn<'a, X> = dyn FnMut(&X, &[f64], Option<&[f64]>) -> (DMat, Vec<f64>) + 'a;

#[derive(Clone, Debug)]
pub struct LmOptions {
    pub loss: Loss,
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub solver: Solver,
}

impl Default for LmOptions {
    fn default() -> Self {
        LmOptions { loss: Loss::Linear, max_iter: 50, ftol: 1e-12, xtol: 1e-12, solver: Solver::Lu }
    }
}

#[derive(Clone, Debug)]
pub struct LmResult<X> {
    pub x: X,
    /// Cost after the loss (linear: 0.5 sum r^2; Huber: the Huber cost).
    pub cost: f64,
    /// Attempts, rejected ones included.
    pub iters: usize,
    pub converged: bool,
    /// Final IRLS-weighted JᵀJ.
    pub info: DMat,
    /// Final residual before weighting.
    pub r: Vec<f64>,
}

/// IRLS sqrt weight: 1 if |r| <= delta, else sqrt(delta/|r|).
pub fn huber_sqrt_w(r: f64, delta: f64) -> f64 {
    let a = r.abs();
    if a > delta {
        (delta / a).sqrt()
    } else {
        1.0
    }
}

/// linear: 0.5 sum r^2. Huber: 0.5 sum_{|r|<=d} r^2 + sum_{|r|>d} d(|r| - 0.5 d) (small and large groups summed separately).
pub fn cost(r: &[f64], loss: Loss) -> f64 {
    match loss {
        Loss::Linear => 0.5 * r.iter().map(|v| v * v).sum::<f64>(),
        Loss::Huber { delta } => {
            let mut small = 0.0;
            let mut large = 0.0;
            for &v in r {
                let a = v.abs();
                if a <= delta {
                    small += v * v;
                } else {
                    large += delta * (a - 0.5 * delta);
                }
            }
            0.5 * small + large
        }
    }
}

/// Default retract: x + dx.
#[allow(clippy::ptr_arg)]
pub fn vec_retract(x: &Vec<f64>, dx: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), dx.len(), "vec_retract: length mismatch");
    x.iter().zip(dx).map(|(a, b)| a + b).collect()
}

/// Jwᵀ Jw. Python asserts that every entry is finite; the dense entry reports that as an `Err` instead (see
/// [`dense_normal_flagged`]), so a data-dependent overflow or a zero bearing is not a panic.
fn normal(jw: &DMat) -> DMat {
    jw.transpose().matmul(jw)
}

/// Weighted residual, weighted Jacobian (None weights for linear loss).
fn apply_w(r: &[f64], j: &DMat, loss: Loss) -> (Vec<f64>, DMat) {
    assert_eq!(r.len(), j.rows(), "levenberg_marquardt: residual/Jacobian row mismatch");
    match loss {
        Loss::Linear => (r.to_vec(), j.clone()),
        Loss::Huber { delta } => {
            let w: Vec<f64> = r.iter().map(|&v| huber_sqrt_w(v, delta)).collect();
            let rw = r.iter().zip(&w).map(|(a, b)| a * b).collect();
            let jw = DMat::from_fn(j.rows(), j.cols(), |i, c| w[i] * j[(i, c)]);
            (rw, jw)
        }
    }
}

pub(crate) fn solve(a: &DMat, g: &[f64], mu: f64, solver: Solver) -> Result<Vec<f64>, LinalgError> {
    let n = g.len();
    let shifted = || DMat::from_fn(n, n, |i, j| a[(i, j)] + if i == j { mu } else { 0.0 });
    let neg = || g.iter().map(|v| -v).collect::<Vec<f64>>();
    match solver {
        Solver::Lu => lu::solve(&shifted(), &neg()),
        Solver::Cholesky => {
            let l = chol::cholesky(&shifted())?;
            Ok(chol::cho_solve(&l, &neg()))
        }
        Solver::Schur { n_cam_cols, cam_width } => Ok(schur_solve(a, g, mu, n_cam_cols, cam_width)),
    }
}

/// Block counts of the Schur split: `(number of camera blocks, number of marker columns)`.
pub(crate) fn schur_split(n: usize, n_cam_cols: usize, cam_width: usize) -> (usize, usize) {
    (n_cam_cols / cam_width, n - n_cam_cols)
}

/// First column of camera block `f` in the Schur split.
pub(crate) fn schur_cam_origin(f: usize, cam_width: usize) -> usize {
    f * cam_width
}

/// `(A + mu I) dx = -g` through the Schur complement of the marker block. Any singular LU gives an all-NaN vector
/// (scipy `spsolve` warns and returns NaN), which the LM treats as a `NonFiniteStep`.
fn schur_solve(a: &DMat, g: &[f64], mu: f64, n_cam_cols: usize, cam_width: usize) -> Vec<f64> {
    let n = g.len();
    assert!(cam_width > 0 && n_cam_cols % cam_width == 0, "Schur: n_cam_cols must be a multiple of cam_width");
    assert!(n_cam_cols <= n, "Schur: n_cam_cols exceeds the system size");
    assert_eq!((a.rows(), a.cols()), (n, n), "Schur: A is not n x n");
    let (nc, nm) = schur_split(n, n_cam_cols, cam_width);
    let nan = || vec![f64::NAN; n];
    let diag = |i: usize, j: usize| if i == j { mu } else { 0.0 };
    for f in 0..nc {
        for h in 0..nc {
            if f != h {
                for i in 0..cam_width {
                    for j in 0..cam_width {
                        {
                            // NaN is allowed through: it must reach the NonFiniteStep path, not panic
                            let v = a[(f * cam_width + i, h * cam_width + j)];
                            assert!(v == 0.0 || v.is_nan(), "Schur: camera blocks not block diagonal");
                        }
                    }
                }
            }
        }
    }
    let mut s = DMat::from_fn(nm, nm, |i, j| a[(n_cam_cols + i, n_cam_cols + j)] + diag(i, j));
    let mut rhs: Vec<f64> = (0..nm).map(|i| -g[n_cam_cols + i]).collect();
    let mut blocks = Vec::with_capacity(nc);
    for f in 0..nc {
        let o = schur_cam_origin(f, cam_width);
        let u = DMat::from_fn(cam_width, cam_width, |i, j| a[(o + i, o + j)] + diag(i, j));
        let Ok(lf) = lu::lu(&u) else { return nan() };
        let w = DMat::from_fn(cam_width, nm, |i, j| a[(o + i, n_cam_cols + j)]);
        let gc: Vec<f64> = (0..cam_width).map(|i| g[o + i]).collect();
        let y = lf.solve(&gc);
        if nm > 0 {
            let x = lf.solve_mat(&w);
            let wt = w.transpose();
            let wx = wt.matmul(&x);
            let wy = wt.matvec(&y);
            for i in 0..nm {
                for j in 0..nm {
                    s[(i, j)] -= wx[(i, j)];
                }
                rhs[i] += wy[i];
            }
        }
        blocks.push((lf, w, gc));
    }
    let dm = if nm > 0 {
        match lu::solve(&s, &rhs) {
            Ok(v) => v,
            Err(_) => return nan(),
        }
    } else {
        Vec::new()
    };
    let mut dx = Vec::with_capacity(n);
    for (lf, w, gc) in &blocks {
        let wd = if nm > 0 { w.matvec(&dm) } else { vec![0.0; cam_width] };
        let b: Vec<f64> = (0..cam_width).map(|i| -gc[i] - wd[i]).collect();
        dx.extend(lf.solve(&b));
    }
    dx.extend(dm);
    dx
}

/// `np.max(np.abs(v))`: NaN propagates (unlike `f64::max`), so a NaN gradient never passes the gtol test.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m: f64, x| {
        let a = x.abs();
        if m.is_nan() || a.is_nan() {
            f64::NAN
        } else {
            m.max(a)
        }
    })
}

fn grad(jw: &DMat, rw: &[f64]) -> Vec<f64> {
    jw.transpose().matvec(rw)
}

/// What one pass of the LM loop body did, as seen by the loop controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub(crate) enum Tick {
    /// max|g| <= GTOL at the top of the pass: stop, converged, attempt NOT counted.
    GradConverged,
    /// dx not finite: stop, not converged (counted).
    NonFiniteStep,
    /// max|dx| <= xtol, or pred floor, or accepted with dcost <= ftol (state already updated): stop, converged (counted).
    Converged,
    /// rho > 0, state updated, continue (counted).
    Accepted,
    /// rho <= 0, mu *= nu, nu *= 2, continue (counted).
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LoopEnd {
    pub iters: usize,
    pub converged: bool,
}

/// The pure loop transition (Python's order): `Ok(next it)` continues, `Err(end)` stops.
pub(crate) fn advance(it: usize, t: Tick) -> Result<usize, LoopEnd> {
    match t {
        Tick::GradConverged => Err(LoopEnd { iters: it, converged: true }),
        Tick::NonFiniteStep => Err(LoopEnd { iters: it + 1, converged: false }),
        Tick::Converged => Err(LoopEnd { iters: it + 1, converged: true }),
        Tick::Accepted | Tick::Rejected => Ok(it + 1),
    }
}

/// The only loop of LM. It alone owns the iteration counter; each tick is at most one attempt. No floats here.
pub(crate) fn drive<E>(max_iter: usize, mut tick: impl FnMut() -> Result<Tick, E>) -> Result<LoopEnd, E> {
    let mut it = 0usize;
    while it < max_iter {
        match advance(it, tick()?) {
            Ok(n) => it = n,
            Err(end) => return Ok(end),
        }
    }
    Ok(LoopEnd { iters: it, converged: false })
}

/// Float state of the LM iteration; control flow lives in `drive`.
pub(crate) struct LmState<X> {
    x: X,
    r: Vec<f64>,
    a: DMat,
    g: Vec<f64>,
    cst: f64,
    mu: f64,
    nu: f64,
}

impl<X: Clone> LmState<X> {
    /// `normal(x, r)` returns the (A, g) pair at x for the unweighted residual r.
    pub(crate) fn new(residual: &mut dyn FnMut(&X) -> Vec<f64>, normal: &mut DenseNormal<'_, X>, x0: X, loss: Loss) -> Self {
        let x = x0;
        let r = residual(&x);
        let (a, g) = normal(&x, &r);
        let cst = cost(&r, loss);
        let n = a.rows();
        // np.max propagates NaN (then `NaN > 0` is false); f64::max would drop it.
        let diag_nan = (0..n).any(|i| a[(i, i)].is_nan());
        let max_diag = (0..n).fold(f64::NEG_INFINITY, |m, i| m.max(a[(i, i)]));
        let mu = if n > 0 && !diag_nan && max_diag > 0.0 { 1e-3 * max_diag } else { 1e-3 };
        LmState { x, r, a, g, cst, mu, nu: 2.0 }
    }

    /// One pass of the loop body (at most one attempt).
    pub(crate) fn tick(
        &mut self,
        residual: &mut dyn FnMut(&X) -> Vec<f64>,
        normal: &mut DenseNormal<'_, X>,
        retract: &dyn Fn(&X, &[f64]) -> X,
        opts: &LmOptions,
    ) -> Result<Tick, LinalgError> {
        let loss = opts.loss;
        if max_abs(&self.g) <= GTOL {
            return Ok(Tick::GradConverged);
        }
        let dx = solve(&self.a, &self.g, self.mu, opts.solver)?;
        if !dx.iter().all(|v| v.is_finite()) {
            return Ok(Tick::NonFiniteStep);
        }
        if max_abs(&dx) <= opts.xtol {
            return Ok(Tick::Converged);
        }
        let x_new = retract(&self.x, &dx);
        let r_new = residual(&x_new);
        let cost_new = cost(&r_new, loss);
        let pred = 0.5 * dx.iter().zip(&self.g).map(|(d, gi)| d * (self.mu * d - gi)).sum::<f64>();
        if pred <= 1e-14 * (self.cst.abs() + 1e-300) {
            return Ok(Tick::Converged);
        }
        let rho = (self.cst - cost_new) / pred;
        if rho > 0.0 {
            let dcost = self.cst - cost_new;
            self.x = x_new;
            self.r = r_new;
            self.cst = cost_new;
            let (a, g) = normal(&self.x, &self.r);
            self.a = a;
            self.g = g;
            // Python `max(1/3, x)` returns 1/3 for NaN x, like f64::max; `** 3` is libm pow, hence powf.
            self.mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powf(3.0));
            self.nu = 2.0;
            if dcost <= opts.ftol * (self.cst + opts.ftol) {
                return Ok(Tick::Converged);
            }
            Ok(Tick::Accepted)
        } else {
            self.mu *= self.nu;
            self.nu *= 2.0;
            Ok(Tick::Rejected)
        }
    }
}

/// Dense normal-equation builder from a Jacobian callback: apply the weights, `JᵀJ`, `Jᵀ rw`. The result of a non-finite `JᵀJ` is
/// passed on unchanged (the iteration then stops through `NonFiniteStep` or a rejected step); use [`dense_normal_flagged`] to learn of it.
pub(crate) fn dense_normal<'a, X>(jac: &'a mut dyn FnMut(&X) -> DMat, loss: Loss) -> impl FnMut(&X, &[f64]) -> (DMat, Vec<f64>) + 'a {
    move |x, r| {
        let j = jac(x);
        let (rw, jw) = apply_w(r, &j, loss);
        let a = normal(&jw);
        let g = grad(&jw, &rw);
        (a, g)
    }
}

/// [`dense_normal`] that sets `bad` whenever a built `JᵀJ` has a non-finite entry.
fn dense_normal_flagged<'a, X>(
    jac: &'a mut dyn FnMut(&X) -> DMat,
    loss: Loss,
    bad: &'a std::cell::Cell<bool>,
) -> impl FnMut(&X, &[f64]) -> (DMat, Vec<f64>) + 'a {
    let mut inner = dense_normal(jac, loss);
    move |x, r| {
        let (a, g) = inner(x, r);
        if !a.as_slice().iter().all(|v| v.is_finite()) {
            bad.set(true);
        }
        (a, g)
    }
}

pub fn levenberg_marquardt<X: Clone>(
    residual: &mut dyn FnMut(&X) -> Vec<f64>,
    jac: &mut dyn FnMut(&X) -> DMat,
    x0: X,
    retract: &dyn Fn(&X, &[f64]) -> X,
    opts: &LmOptions,
) -> Result<LmResult<X>, LinalgError> {
    let bad = std::cell::Cell::new(false);
    let res = {
        let mut nf = dense_normal_flagged(jac, opts.loss, &bad);
        lm_run(residual, &mut nf, x0, retract, opts)
    };
    // a non-finite JᵀJ (Python: AssertionError) is a data-dependent failure: report it, whatever the run did afterwards
    if bad.get() {
        return Err(LinalgError::NoConvergence);
    }
    res
}

/// LM driven by a normal-equation builder (see [`NormalFn`]). Unlike the dense entry it does not panic on a non-finite A:
/// like Python's sparse path, that gives a non-finite dx and stops through `NonFiniteStep`.
pub fn levenberg_marquardt_normal<X: Clone>(
    residual: &mut dyn FnMut(&X) -> Vec<f64>,
    normal: &mut NormalFn<'_, X>,
    x0: X,
    retract: &dyn Fn(&X, &[f64]) -> X,
    opts: &LmOptions,
) -> Result<LmResult<X>, LinalgError> {
    let loss = opts.loss;
    let mut nf = |x: &X, r: &[f64]| {
        let w: Option<Vec<f64>> = match loss {
            Loss::Linear => None,
            Loss::Huber { delta } => Some(r.iter().map(|&v| huber_sqrt_w(v, delta)).collect()),
        };
        normal(x, r, w.as_deref())
    };
    lm_run(residual, &mut nf, x0, retract, opts)
}

/// The single run loop behind both entry points: only `drive` loops.
fn lm_run<X: Clone>(
    residual: &mut dyn FnMut(&X) -> Vec<f64>,
    nf: &mut DenseNormal<'_, X>,
    x0: X,
    retract: &dyn Fn(&X, &[f64]) -> X,
    opts: &LmOptions,
) -> Result<LmResult<X>, LinalgError> {
    let mut st = LmState::new(residual, nf, x0, opts.loss);
    let end = drive(opts.max_iter, || st.tick(residual, nf, retract, opts))?;
    Ok(LmResult { x: st.x, cost: st.cst, iters: end.iters, converged: end.converged, info: st.a, r: st.r })
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_linalg::rng::Pcg64;
    use erploc_linalg::svd::lstsq;
    use erploc_testkit::max_abs_diff;
    use std::cell::Cell;

    fn lin_problem() -> (DMat, Vec<f64>) {
        let (m, n) = (40usize, 4usize);
        let t: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / m as f64).collect();
        let x = DMat::from_fn(m, n, |i, j| ((j as f64 + 1.0) * std::f64::consts::PI * t[i]).cos());
        let lin = x.matvec(&[1.0, -2.0, 0.5, 3.0]);
        let y = (0..m).map(|i| lin[i] + 0.05 * (17.0 * t[i]).sin()).collect();
        (x, y)
    }

    fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn linear_lstsq_exact() {
        let (x, y) = lin_problem();
        let x_exact = lstsq(&x, &y).unwrap();
        let re = sub(&x.matvec(&x_exact), &y);
        let cost_exact = 0.5 * re.iter().map(|v| v * v).sum::<f64>();
        let res = levenberg_marquardt(
            &mut |p: &Vec<f64>| sub(&x.matvec(p), &y),
            &mut |_| x.clone(),
            vec![0.0; 4],
            &vec_retract,
            &LmOptions::default(),
        )
        .unwrap();
        let e = max_abs_diff(&res.x, &x_exact);
        println!("linear_lstsq_exact: iters {} err {e:.2e}", res.iters);
        assert!(e < 1e-8 && res.iters <= 6 && res.converged, "e {e} iters {}", res.iters);
        assert!((res.cost / cost_exact - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nonlinear_exp() {
        let t = linspace(0.0, 2.0, 30);
        let yv: Vec<f64> = t.iter().map(|&s| 2.5 * (-0.8 * s).exp()).collect();
        let res = levenberg_marquardt(
            &mut |p: &Vec<f64>| t.iter().zip(&yv).map(|(&s, &y)| p[0] * (p[1] * s).exp() - y).collect(),
            &mut |p: &Vec<f64>| {
                DMat::from_fn(t.len(), 2, |i, c| {
                    let ex = (p[1] * t[i]).exp();
                    if c == 0 {
                        ex
                    } else {
                        p[0] * t[i] * ex
                    }
                })
            },
            vec![1.0, -0.1],
            &vec_retract,
            &LmOptions { max_iter: 100, ..Default::default() },
        )
        .unwrap();
        let e = max_abs_diff(&res.x, &[2.5, -0.8]);
        println!("nonlinear_exp: iters {} err {e:.2e}", res.iters);
        assert!(e < 1e-9 && res.converged, "e {e}");
    }

    #[test]
    fn huber_suppresses_outlier() {
        let mh = 12;
        let th = linspace(-1.0, 1.0, mh);
        let xh = DMat::from_fn(mh, 2, |i, c| if c == 0 { 1.0 } else { th[i] });
        let sig = 0.02;
        let y0 = xh.matvec(&[1.0, 0.5]);
        let mut y1 = y0.clone();
        y1[1] += 50.0 * sig;
        let solve_for = |yy: &[f64], loss: Loss| {
            levenberg_marquardt(
                &mut |p: &Vec<f64>| sub(&xh.matvec(p), yy).iter().map(|v| v / sig).collect(),
                &mut |_| xh.scale(1.0 / sig),
                vec![0.0; 2],
                &vec_retract,
                &LmOptions { loss, max_iter: 200, ..Default::default() },
            )
            .unwrap()
            .x
        };
        let shift = |loss: Loss| {
            let c = solve_for(&y0, loss);
            let o = solve_for(&y1, loss);
            let nrm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
            nrm(&sub(&o, &c)) / nrm(&c)
        };
        let h = shift(Loss::Huber { delta: 1.5 });
        let l = shift(Loss::Linear);
        assert!(h < 0.01, "huber {h}");
        assert!(l > 0.10, "linear {l}");
    }

    #[test]
    fn solvers_agree() {
        let mut rng = Pcg64::new(11);
        let (nb, bs, mm) = (6usize, 2usize, 60usize);
        let nn = nb * bs;
        let mut jd = DMat::zeros(mm, nn);
        for i in 0..mm {
            let b = i % nb;
            for k in 0..bs {
                jd[(i, b * bs + k)] = rng.normal();
            }
            jd[(i, ((b + 1) % nb) * bs)] = rng.normal();
        }
        let z: Vec<f64> = (0..nn).map(|_| rng.normal()).collect();
        let lin = jd.matvec(&z);
        let ys: Vec<f64> = lin.iter().map(|v| v + 0.01 * rng.normal()).collect();
        let run = |solver: Solver| {
            levenberg_marquardt(
                &mut |p: &Vec<f64>| sub(&jd.matvec(p), &ys),
                &mut |_| jd.clone(),
                vec![0.0; nn],
                &vec_retract,
                &LmOptions { solver, ..Default::default() },
            )
            .unwrap()
        };
        let (a, b) = (run(Solver::Lu), run(Solver::Cholesky));
        let e = max_abs_diff(&a.x, &b.x);
        assert!(e < 1e-10, "e {e}");
    }

    #[test]
    fn retract_hook_used() {
        let (x, y) = lin_problem();
        let x_exact = lstsq(&x, &y).unwrap();
        let s = [2.0, 3.0, 0.5, 1.5];
        let calls = Cell::new(0usize);
        let retract = |p: &Vec<f64>, dx: &[f64]| -> Vec<f64> {
            calls.set(calls.get() + 1);
            (0..4).map(|i| p[i] + s[i] * dx[i]).collect()
        };
        let xs = DMat::from_fn(40, 4, |i, j| x[(i, j)] * s[j]);
        let res = levenberg_marquardt(
            &mut |p: &Vec<f64>| sub(&x.matvec(p), &y),
            &mut |_| xs.clone(),
            vec![0.0; 4],
            &retract,
            &LmOptions::default(),
        )
        .unwrap();
        let e = max_abs_diff(&res.x, &x_exact);
        assert!(e < 1e-8 && calls.get() > 0, "e {e} calls {}", calls.get());
    }

    #[test]
    fn huber_weight_basic() {
        assert_eq!(huber_sqrt_w(0.5, 1.5), 1.0);
        assert_eq!(huber_sqrt_w(6.0, 1.5), 0.5);
        assert_eq!(huber_sqrt_w(-6.0, 1.5), 0.5);
    }

    #[test]
    fn zero_jacobian_gtol() {
        let res = levenberg_marquardt(
            &mut |_: &Vec<f64>| vec![1.0, 2.0, 3.0],
            &mut |_| DMat::zeros(3, 2),
            vec![0.0; 2],
            &vec_retract,
            &LmOptions::default(),
        )
        .unwrap();
        assert!(res.converged && res.iters == 0);
    }

    #[test]
    fn nan_residual_not_converged() {
        // Python: max|g| = nan fails the gtol test, dx is nan, the loop breaks with converged = false.
        let j = DMat::from_rows(2, 1, &[1.0, 1.0]);
        let mut res = |_: &Vec<f64>| vec![f64::NAN, 1.0];
        let mut jac = |_: &Vec<f64>| j.clone();
        let out = levenberg_marquardt(&mut res, &mut jac, vec![0.0], &vec_retract, &LmOptions::default()).unwrap();
        assert!(!out.converged);
        assert_eq!(out.iters, 1);
    }

    fn rand_blocks(seed: u64, nc: usize, cw: usize, nm: usize, mw: usize) -> (DMat, Vec<f64>) {
        let mut rng = Pcg64::new(seed);
        let n = nc * cw + nm * mw;
        let rows = 8 * nc * nm;
        let mut j = DMat::zeros(rows, n);
        for f in 0..nc {
            for m in 0..nm {
                let r0 = 8 * (f * nm + m);
                for i in 0..8 {
                    for k in 0..cw {
                        j[(r0 + i, f * cw + k)] = rng.normal();
                    }
                    for k in 0..mw {
                        j[(r0 + i, nc * cw + m * mw + k)] = rng.normal();
                    }
                }
            }
        }
        let a = j.transpose().matmul(&j);
        let g: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        (a, g)
    }

    #[test]
    fn schur_matches_lu() {
        // Each row block touches one camera, so JᵀJ is camera-block-diagonal by construction.
        for cw in [4usize, 6] {
            let (nc, nm, mw) = (5usize, 3usize, 6usize);
            let (a, g) = rand_blocks(31, nc, cw, nm, mw);
            for mu in [1e-3, 1.0] {
                let sch = Solver::Schur { n_cam_cols: nc * cw, cam_width: cw };
                let d1 = solve(&a, &g, mu, sch).unwrap();
                let d2 = solve(&a, &g, mu, Solver::Lu).unwrap();
                assert!(d1.iter().all(|v| v.is_finite()));
                let m = d2.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let e = max_abs_diff(&d1, &d2) / m;
                println!("schur_matches_lu cw {cw} mu {mu}: rel err {e:.2e}");
                assert!(e <= 1e-10, "cw {cw} mu {mu} e {e}");
            }
        }
    }

    #[test]
    fn schur_singular_gives_nan() {
        let n = 4 + 2;
        let mut a = DMat::zeros(n, n);
        a[(4, 4)] = 1.0;
        a[(5, 5)] = 1.0;
        let g = vec![1.0; n];
        let dx = solve(&a, &g, 0.0, Solver::Schur { n_cam_cols: 4, cam_width: 4 }).unwrap();
        assert_eq!(dx.len(), n);
        assert!(dx.iter().all(|v| v.is_nan()));
    }

    #[test]
    #[should_panic(expected = "multiple of cam_width")]
    fn schur_bad_width_panics() {
        let a = DMat::zeros(6, 6);
        let _ = solve(&a, &[0.0; 6], 1.0, Solver::Schur { n_cam_cols: 5, cam_width: 4 });
    }

    #[test]
    fn normal_entry_matches_dense() {
        let cases: Vec<(&str, Loss)> =
            vec![("lin", Loss::Linear), ("exp", Loss::Linear), ("rosen", Loss::Linear), ("huber", Loss::Huber { delta: 1.5 })];
        for (name, loss) in cases {
            // (residual, jac, x0, max_iter)
            type Res<'a> = Box<dyn Fn(&Vec<f64>) -> Vec<f64> + 'a>;
            type Jac<'a> = Box<dyn Fn(&Vec<f64>) -> DMat + 'a>;
            let (lx, ly) = lin_problem();
            let t = linspace(0.0, 2.0, 30);
            let yv: Vec<f64> = t.iter().map(|&s| 2.5 * (-0.8 * s).exp()).collect();
            let sig = 0.02;
            let mh = 12;
            let th = linspace(-1.0, 1.0, mh);
            let xh = DMat::from_fn(mh, 2, |i, c| if c == 0 { 1.0 } else { th[i] });
            let mut yh = xh.matvec(&[1.0, 0.5]);
            yh[1] += 50.0 * sig;
            let (res, jac, x0, mi): (Res, Jac, Vec<f64>, usize) = match name {
                "lin" => (Box::new(|p| sub(&lx.matvec(p), &ly)), Box::new(|_| lx.clone()), vec![0.0; 4], 50),
                "exp" => (
                    Box::new(|p| t.iter().zip(&yv).map(|(&s, &y)| p[0] * (p[1] * s).exp() - y).collect()),
                    Box::new(|p| {
                        DMat::from_fn(t.len(), 2, |i, c| {
                            let ex = (p[1] * t[i]).exp();
                            if c == 0 {
                                ex
                            } else {
                                p[0] * t[i] * ex
                            }
                        })
                    }),
                    vec![1.0, -0.1],
                    100,
                ),
                "rosen" => (
                    Box::new(|p| vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]]),
                    Box::new(|p| DMat::from_rows(2, 2, &[-20.0 * p[0], 10.0, -1.0, 0.0])),
                    vec![-1.2, 1.0],
                    200,
                ),
                _ => (
                    Box::new(|p| sub(&xh.matvec(p), &yh).iter().map(|v| v / sig).collect()),
                    Box::new(|_| xh.scale(1.0 / sig)),
                    vec![0.0; 2],
                    200,
                ),
            };
            let opts = LmOptions { loss, max_iter: mi, ..Default::default() };
            let d = levenberg_marquardt(&mut |p| res(p), &mut |p| jac(p), x0.clone(), &vec_retract, &opts).unwrap();
            let mut nf = |p: &Vec<f64>, r: &[f64], w: Option<&[f64]>| {
                let j = jac(p);
                let wv: Vec<f64> = w.map_or_else(|| vec![1.0; r.len()], |w| w.to_vec());
                let jw = DMat::from_fn(j.rows(), j.cols(), |i, c| wv[i] * j[(i, c)]);
                let rw: Vec<f64> = r.iter().zip(&wv).map(|(a, b)| a * b).collect();
                (jw.transpose().matmul(&jw), jw.transpose().matvec(&rw))
            };
            let n = levenberg_marquardt_normal(&mut |p| res(p), &mut nf, x0, &vec_retract, &opts).unwrap();
            assert!(d.x.iter().chain(&n.x).all(|v| v.is_finite()), "{name}");
            let e = max_abs_diff(&d.x, &n.x);
            println!("normal_entry_matches_dense {name}: iters {} vs {}, err {e:.2e}", d.iters, n.iters);
            assert_eq!((d.iters, d.converged), (n.iters, n.converged), "{name}");
            assert!(e <= 1e-12, "{name} e {e}");
        }
    }

    #[test]
    fn normal_entry_nonfinite_a_no_panic() {
        let mut res = |_: &Vec<f64>| vec![1.0];
        let mut nf = |_: &Vec<f64>, _: &[f64], _: Option<&[f64]>| (DMat::from_rows(1, 1, &[f64::NAN]), vec![1.0]);
        let out = levenberg_marquardt_normal(&mut res, &mut nf, vec![0.0], &vec_retract, &LmOptions::default()).unwrap();
        assert!(!out.converged);
        assert_eq!(out.iters, 1);
    }
}
