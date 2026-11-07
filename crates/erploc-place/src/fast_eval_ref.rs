//! Frozen copy of the original `FastEval::{base, pairs, neighbourhood}` and `min_margin`: the reference of the bit-exact
//! optimisation. The bodies are the original code with `self` renamed to `fe`; do not edit them. Only compiled for tests.
use crate::fast_eval::{bl_index, pair_key, unpair, AddKeys, Base, FastEval, Nbhd, Pairs, BIG};
use crate::npcompat::{bincount, nmax, nmin, np_sum, searchsorted_left, sum_weighted10, unique_sorted};
use crate::placement::PlaceError;
use crate::sigma10::{sigma10_one, Sigma10Counts};

/// `FastEval._min_margin`: for every candidate `j` the minimum over the cells in `mask` of the cell margin after adding `j`:
/// the affected pairs `(pj, pc)` carry the new margin `mn`, every other cell keeps its base margin `m0`; `+inf` for cells
/// outside `mask`. Transcription of the Python (stable ascending order of the masked base margins, NaN last; first finite
/// unaffected cell); the unit test compares it with a set-based literal transcription.
pub fn min_margin(m0: &[f64], pj: &[usize], pc: &[usize], mn: &[f64], mask: Option<&[bool]>, n: usize) -> Vec<f64> {
    assert!(!m0.is_empty(), "min_margin: no cells");
    assert!(pj.len() == pc.len() && pj.len() == mn.len(), "min_margin: pair array lengths");
    let mut out = vec![f64::INFINITY; n];
    let mm: Vec<f64> = match mask {
        None => m0.to_vec(),
        Some(mk) => m0.iter().zip(mk).map(|(&m, &b)| if b { m } else { f64::INFINITY }).collect(),
    };
    let mut cells_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..pj.len() {
        if mask.map_or(true, |mk| mk[pc[i]]) {
            out[pj[i]] = nmin(out[pj[i]], mn[i]); // np.minimum.at
            cells_of[pj[i]].push(pc[i]);
        }
    }
    for l in cells_of.iter_mut() {
        l.sort_unstable();
    }
    // np.argsort(kind='stable'): ascending, NaN last, ties by index
    let mut order: Vec<usize> = (0..mm.len()).collect();
    order.sort_by(|&a, &b| match (mm[a].is_nan(), mm[b].is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => mm[a].partial_cmp(&mm[b]).expect("non-NaN"),
    });
    let c0 = order[0];
    let first = mm[c0];
    for j in 0..n {
        let aff = &cells_of[j];
        if aff.is_empty() || aff.binary_search(&c0).is_err() {
            out[j] = pymin(out[j], first);
            continue;
        }
        for &c in &order {
            if !mm[c].is_finite() {
                break;
            }
            if aff.binary_search(&c).is_err() {
                out[j] = pymin(out[j], mm[c]);
                break;
            }
        }
    }
    out
}

/// `_base(S')`.
pub fn base(fe: &mut FastEval, sp: &[usize]) -> Result<Base, PlaceError> {
    let (kc, nn, n) = (fe.kc, fe.n, sp.len());
    let cap = fe.cap;
    let mut bl = vec![false; kc * nn];
    for &s in sp {
        if fe.is_stand[s] {
            for t in fe.occ.bs_ptr[s]..fe.occ.bs_ptr[s + 1] {
                bl[bl_index(fe.occ.bs_c[t] as usize, fe.occ.bs_t[t] as usize, nn)] = true;
            }
        }
    }
    let mut w = Vec::with_capacity(kc * n);
    let mut ip = Vec::with_capacity(kc * n * 10);
    for kk in 0..kc {
        for &s in sp {
            w.push(fe.v[kk * nn + s] && !bl[kk * nn + s]);
            ip.extend_from_slice(&fe.i10[(kk * nn + s) * 10..(kk * nn + s) * 10 + 10]);
        }
    }
    let mut isum = Vec::with_capacity(kc * 10);
    for kk in 0..kc {
        isum.extend_from_slice(&sum_weighted10(&w[kk * n..(kk + 1) * n], &ip[kk * n * 10..(kk + 1) * n * 10]));
    }
    let (mut feas, mut pen, mut marg) = (Vec::with_capacity(kc), Vec::with_capacity(kc), Vec::with_capacity(kc));
    for kk in 0..kc {
        let nv = w[kk * n..(kk + 1) * n].iter().filter(|&&b| b).count() as f64;
        let row = &isum[kk * 10..kk * 10 + 10];
        let sg = sig10(row, cap, &mut fe.amb)?;
        // _loo: max over the visible markers of the capped sigma of Isum - I_m; cap if none visible
        let mut loo = fe.cap;
        let mut first = true;
        for m in 0..n {
            if w[kk * n + m] {
                let s = sig10(&sub10(row, &ip[(kk * n + m) * 10..(kk * n + m) * 10 + 10]), cap, &mut fe.amb)?;
                loo = if first { s } else { nmax(loo, s) };
                first = false;
            }
        }
        let (f, p, mg) = fe.feas_pen(sg, nv, loo);
        feas.push(f);
        pen.push(p);
        marg.push(mg);
    }
    Ok(Base { bl, w, ip, isum, feas, pen, marg })
}

/// `_pairs(S', base, cell_mask)`: per affected pair (cell `c`, candidate `j` not in `S'`) the change of cell `c` when `j` is
/// added to `S'`. Affected = `j` visible at `c`, or `j`'s board blocks a currently effective marker of `S'` at `c`; every other
/// pair has change 0. `cell_mask` restricts the pairs to those cells. The information sum is updated sparsely
/// (`I(c) = I_S'(c) + w_j I_j(c)`); rows where `j`'s board blocks a marker are re-summed exactly (a subtraction would leave
/// a rounding residue where the true sum is the zero matrix, and the singularity test needs that exact zero).
pub fn pairs(fe: &mut FastEval, sp: &[usize], base: &Base, cell_mask: Option<&[bool]>) -> Result<Pairs, PlaceError> {
    let (kc, nn, n) = (fe.kc, fe.n, sp.len());
    let cap = fe.cap;
    let mut in_s = vec![false; nn];
    for &s in sp {
        in_s[s] = true;
    }
    let base_inf: Vec<i64> = base.feas.iter().map(|&f| !f as i64).collect();
    let o = &fe.occ;
    let mut keys: Vec<i64> = Vec::new();
    for i in 0..o.v_j.len() {
        let (j, c) = (o.v_j[i] as usize, o.v_c[i] as usize);
        if !in_s[j] && cell_mask.map_or(true, |m| m[c]) {
            keys.push((j * kc + c) as i64);
        }
    }
    // (blocker candidate j, cell c, position of the blocked marker in S')
    let mut blk: Vec<(usize, usize, usize)> = Vec::new();
    for (pos, &m) in sp.iter().enumerate() {
        for t in o.bt_ptr[m]..o.bt_ptr[m + 1] {
            let (bs, bc) = (o.bt_s[t] as usize, o.bt_c[t] as usize);
            if !in_s[bs] && base.w[bc * n + pos] && cell_mask.map_or(true, |mk| mk[bc]) {
                blk.push((bs, bc, pos));
                keys.push(pair_key(bs, bc, kc) as i64);
            }
        }
    }
    let uk = unique_sorted(&keys);
    let p = uk.len();
    if p == 0 {
        return Ok(Pairs { pj: vec![], pc: vec![], d_inf: vec![], d_pen: vec![], marg: vec![], n_resum: 0 });
    }
    let pj: Vec<usize> = uk.iter().map(|&u| unpair(u as usize, kc).0).collect();
    let pc: Vec<usize> = uk.iter().map(|&u| unpair(u as usize, kc).1).collect();
    let bidx: Vec<usize> = blk.iter().map(|&(j, c, _)| searchsorted_left(&uk, pair_key(j, c, kc) as i64)).collect();
    let wj: Vec<bool> = (0..p).map(|i| fe.v[pc[i] * nn + pj[i]] && !base.bl[pc[i] * nn + pj[i]]).collect();
    let ij = |i: usize| -> &[f64] {
        let at = (pc[i] * nn + pj[i]) * 10;
        &fe.i10[at..at + 10]
    };
    let mut isum = vec![0.0; p * 10];
    for i in 0..p {
        let wf = if wj[i] { 1.0 } else { 0.0 };
        for d in 0..10 {
            isum[i * 10 + d] = base.isum[pc[i] * 10 + d] + wf * ij(i)[d];
        }
    }
    let mut wn = vec![false; p * n];
    for i in 0..p {
        wn[i * n..(i + 1) * n].copy_from_slice(&base.w[pc[i] * n..(pc[i] + 1) * n]);
    }
    for (b, &(_, _, pos)) in blk.iter().enumerate() {
        wn[bidx[b] * n + pos] = false;
    }
    let ip_at = |c: usize, m: usize| -> &[f64] { &base.ip[(c * n + m) * 10..(c * n + m) * 10 + 10] };
    let rb: Vec<usize> = unique_sorted(&bidx.iter().map(|&b| b as i64).collect::<Vec<_>>()).iter().map(|&r| r as usize).collect();
    for &r in &rb {
        let e = sum_weighted10(&wn[r * n..(r + 1) * n], &base.ip[pc[r] * n * 10..(pc[r] + 1) * n * 10]);
        let wf = if wj[r] { 1.0 } else { 0.0 };
        for d in 0..10 {
            isum[r * 10 + d] = e[d] + wf * ij(r)[d];
        }
    }
    let (mut d_inf, mut d_pen, mut marg) = (Vec::with_capacity(p), Vec::with_capacity(p), Vec::with_capacity(p));
    for i in 0..p {
        let nv = (wn[i * n..(i + 1) * n].iter().filter(|&&b| b).count() + wj[i] as usize) as f64;
        let row = &isum[i * 10..i * 10 + 10];
        let sg = sig10(row, cap, &mut fe.amb)?;
        // leave-one-out over the effective markers: those of S' (wn) and j itself (wj)
        let mut loo = f64::NEG_INFINITY;
        for m in 0..n {
            if wn[i * n + m] {
                let s = sig10(&sub10(row, ip_at(pc[i], m)), cap, &mut fe.amb)?;
                loo = nmax(loo, s);
            }
        }
        if wj[i] {
            let s = sig10(&sub10(row, ij(i)), cap, &mut fe.amb)?;
            loo = nmax(loo, s);
        }
        if loo == f64::NEG_INFINITY {
            loo = fe.cap;
        }
        let (f, pn, mg) = fe.feas_pen(sg, nv, loo);
        d_inf.push(!f as i64 - base_inf[pc[i]]);
        d_pen.push(pn - base.pen[pc[i]]);
        marg.push(mg);
    }
    Ok(Pairs { pj, pc, d_inf, d_pen, marg, n_resum: rb.len() })
}

pub fn neighbourhood(fe: &mut FastEval, s: &[usize]) -> Result<Nbhd, PlaceError> {
    let (k, nn, kc) = (s.len(), fe.n, fe.kc);
    let bs = base(fe, s)?;
    let ps = pairs(fe, s, &bs, None)?;
    let dis: Vec<f64> = ps.d_inf.iter().map(|&d| d as f64).collect();
    let di_s = bincount(&ps.pj, &dis, nn);
    let dp_s = bincount(&ps.pj, &ps.d_pen, nn);
    let mut out = Nbhd { inf: Vec::with_capacity(k * nn), pen: Vec::with_capacity(k * nn), nm: Vec::with_capacity(k * nn), k, n: nn };
    for i in 0..k {
        let si = s[i];
        let sp: Vec<usize> = s[..i].iter().chain(&s[i + 1..]).copied().collect();
        // cells where S[i] is effective, plus (stand) the cells where its board blocks any sight line
        let mut a: Vec<bool> = (0..kc).map(|c| bs.w[c * k + i]).collect();
        if fe.is_stand[si] {
            for t in fe.occ.bs_ptr[si]..fe.occ.bs_ptr[si + 1] {
                a[fe.occ.bs_c[t] as usize] = true;
            }
        }
        let bp = base(fe, &sp)?;
        let p2 = pairs(fe, &sp, &bp, Some(&a))?;
        let m_s: Vec<usize> = (0..ps.pj.len()).filter(|&q| a[ps.pc[q]]).collect();
        let pj_m: Vec<usize> = m_s.iter().map(|&q| ps.pj[q]).collect();
        let di_m: Vec<f64> = m_s.iter().map(|&q| dis[q]).collect();
        let dp_m: Vec<f64> = m_s.iter().map(|&q| ps.d_pen[q]).collect();
        let (bi1, bp1) = (bincount(&pj_m, &di_m, nn), bincount(&pj_m, &dp_m, nn));
        let di2: Vec<f64> = p2.d_inf.iter().map(|&d| d as f64).collect();
        let (bi2, bp2) = (bincount(&p2.pj, &di2, nn), bincount(&p2.pj, &p2.d_pen, nn));
        let n_inf_p = bp.feas.iter().filter(|&&f| !f).count() as f64;
        let pen_p = np_sum(&bp.pen);
        let not_a: Vec<bool> = a.iter().map(|&x| !x).collect();
        let m1 = min_margin(&bs.marg, &ps.pj, &ps.pc, &ps.marg, Some(&not_a), nn);
        let m2 = min_margin(&bp.marg, &p2.pj, &p2.pc, &p2.marg, Some(&a), nn);
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
    }
    Ok(out)
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

/// Python builtin `min(a, b)` on floats: `b` replaces `a` only if `b < a`.
fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `add_keys` of the original code on top of the frozen `base` / `pairs`.
pub fn add_keys(fe: &mut FastEval, sp: &[usize]) -> Result<AddKeys, PlaceError> {
    let base = base(fe, sp)?;
    let pr = pairs(fe, sp, &base, None)?;
    let nn = fe.n;
    let n_inf0 = base.feas.iter().filter(|&&f| !f).count() as i64;
    let di: Vec<f64> = pr.d_inf.iter().map(|&d| d as f64).collect();
    let bi = bincount(&pr.pj, &di, nn);
    let bp = bincount(&pr.pj, &pr.d_pen, nn);
    let pen0 = np_sum(&base.pen);
    let mut inf: Vec<i64> = bi.iter().map(|&x| n_inf0 + x.round_ties_even() as i64).collect();
    let mut pen: Vec<f64> = bp.iter().map(|&x| pen0 + x).collect();
    let mut nm: Vec<f64> = min_margin(&base.marg, &pr.pj, &pr.pc, &pr.marg, None, nn).iter().map(|&x| -x).collect();
    for &s in sp {
        inf[s] = BIG;
        pen[s] = f64::INFINITY;
        nm[s] = f64::INFINITY;
    }
    Ok(AddKeys { inf, pen, nm })
}

/// Bit-equality of the optimised `FastEval` against the frozen code above.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_eval::{AddKeys, IftContext};
    use crate::placement::demo_spec;
    use crate::site::{CandKind, SiteTables, VisMode};
    use crate::zone::Crit;
    use erploc_linalg::rng::Pcg64;

    fn site(name: &str) -> SiteTables {
        {
            let _ = name;
            crate::site_synth::synthetic_site().unwrap()
        }
    }

    fn rsel(rng: &mut Pcg64, n: usize, size: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..size {
            let j = i + (rng.next_u64() % (n - i) as u64) as usize;
            pool.swap(i, j);
        }
        pool.truncate(size);
        pool
    }

    /// Stand-heavy selection (0..=3 walls) as the site self-test generator; any distinct selection on a room without stands.
    fn sel_of(rng: &mut Pcg64, ctx: &IftContext, size: usize) -> Vec<usize> {
        let stands: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Stand).collect();
        let walls: Vec<usize> = (0..ctx.n).filter(|&j| ctx.kinds[j] == CandKind::Wall).collect();
        if stands.is_empty() {
            return rsel(rng, ctx.n, size);
        }
        let nw = ((rng.next_u64() % 4) as usize).min(size).min(walls.len());
        let mut sel: Vec<usize> = rsel(rng, walls.len(), nw).iter().map(|&x| walls[x]).collect();
        sel.extend(rsel(rng, stands.len(), size - nw).iter().map(|&x| stands[x]));
        sel
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    pub(crate) fn assert_base_eq(a: &Base, b: &Base, tag: &str) {
        assert_eq!(a.bl, b.bl, "{tag}: bl");
        assert_eq!(a.w, b.w, "{tag}: w");
        assert_eq!(bits(&a.ip), bits(&b.ip), "{tag}: ip");
        assert_eq!(bits(&a.isum), bits(&b.isum), "{tag}: isum");
        assert_eq!(a.feas, b.feas, "{tag}: feas");
        assert_eq!(bits(&a.pen), bits(&b.pen), "{tag}: pen");
        assert_eq!(bits(&a.marg), bits(&b.marg), "{tag}: marg");
    }

    pub(crate) fn assert_pairs_eq(a: &Pairs, b: &Pairs, tag: &str) {
        assert_eq!((&a.pj, &a.pc, &a.d_inf, a.n_resum), (&b.pj, &b.pc, &b.d_inf, b.n_resum), "{tag}: pairs ints");
        assert_eq!(bits(&a.d_pen), bits(&b.d_pen), "{tag}: d_pen");
        assert_eq!(bits(&a.marg), bits(&b.marg), "{tag}: pair marg");
    }

    pub(crate) fn assert_nbhd_eq(a: &Nbhd, b: &Nbhd, tag: &str) {
        assert_eq!((a.k, a.n, &a.inf), (b.k, b.n, &b.inf), "{tag}: inf");
        assert_eq!(bits(&a.pen), bits(&b.pen), "{tag}: pen");
        assert_eq!(bits(&a.nm), bits(&b.nm), "{tag}: nm");
    }

    fn assert_ak_eq(a: &AddKeys, b: &AddKeys, tag: &str) {
        assert_eq!(a.inf, b.inf, "{tag}: ak inf");
        assert_eq!(bits(&a.pen), bits(&b.pen), "{tag}: ak pen");
        assert_eq!(bits(&a.nm), bits(&b.nm), "{tag}: ak nm");
    }

    /// One selection: `base`, `pairs` (no mask and two masks), `add_keys` of the selection minus its last element, `neighbourhood`.
    fn check_one(fe: &mut FastEval, sel: &[usize], rng: &mut Pcg64, tag: &str) {
        let b_new = fe.base(sel).unwrap();
        let b_ref = base(fe, sel).unwrap();
        assert_base_eq(&b_new, &b_ref, tag);
        let p_new = fe.pairs(sel, &b_new, None).unwrap();
        let p_ref = pairs(fe, sel, &b_ref, None).unwrap();
        assert_pairs_eq(&p_new, &p_ref, tag);
        let kc = fe.kc;
        let sparse: Vec<bool> = (0..kc).map(|_| rng.next_u64() % 5 == 0).collect();
        let first_half: Vec<bool> = (0..kc).map(|c| c < kc / 2).collect();
        let none: Vec<bool> = vec![false; kc];
        for (mi, m) in [&sparse, &first_half, &none].into_iter().enumerate() {
            let pn = fe.pairs(sel, &b_new, Some(m)).unwrap();
            let pr = pairs(fe, sel, &b_ref, Some(m)).unwrap();
            assert_pairs_eq(&pn, &pr, &format!("{tag} mask {mi}"));
        }
        let sp = &sel[..sel.len() - 1];
        let an = fe.add_keys(sp).unwrap();
        let ar = add_keys(fe, sp).unwrap();
        assert_ak_eq(&an, &ar, tag);
        let nn = fe.neighbourhood(sel).unwrap();
        let nr = neighbourhood(fe, sel).unwrap();
        assert_nbhd_eq(&nn, &nr, tag);
    }

    fn run_ctx(ctx: &mut IftContext, n_sel: usize, seed: u64, tag: &str) {
        let mut rng = Pcg64::new(seed);
        let mut fe = FastEval::new(ctx, 10).unwrap();
        for t in 0..n_sel {
            let size = 2 + t % 9; // k = 2..10
            let sel = sel_of(&mut rng, ctx, size);
            check_one(&mut fe, &sel, &mut rng, &format!("{tag} #{t} {sel:?}"));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn nbhd_bits_site_small() {
        let s = site("site_small.bin");
        let n = if cfg!(debug_assertions) { 40 } else { 200 };
        for mode in [VisMode::Centre, VisMode::Samples] {
            run_ctx(&mut IftContext::from_site(&s, mode, Crit::N), n, 71, &format!("small {mode:?}"));
        }
    }

    /// One small selection on the demo room: cheap enough for Miri, so the new index code (CSR walk, merge, fused sweeps) runs there.
    #[test]
    #[cfg_attr(miri, ignore)] // 695 s under Miri (passed once when run alone); the pairs / add_keys paths run under Miri in other tests
    fn nbhd_bits_demo_room_one_selection() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        let mut rng = Pcg64::new(74);
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        let sel = rsel(&mut rng, 10, 3);
        check_one(&mut fe, &sel, &mut rng, "demo one");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // demo room: fast, but the reference is O(10^5) float ops per call
    fn nbhd_bits_demo_room() {
        let mut ctx = IftContext::from_spec(&demo_spec(10)).unwrap();
        run_ctx(&mut ctx, 60, 72, "demo");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "full site: release only")]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn nbhd_bits_full_site() {
        let s = site("site.bin");
        for mode in [VisMode::Centre, VisMode::Samples] {
            run_ctx(&mut IftContext::from_site(&s, mode, Crit::N), 40, 73, &format!("full {mode:?}"));
        }
    }

    /// Two different selections evaluated in sequence on one `FastEval` give the same bits as on fresh instances (stale-scratch guard).
    /// There is no epoch/stamp array in the final design, so there is no wrap-around to force; the buffers are cleared per call.
    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn scratch_reuse_is_stateless() {
        let s = site("site_small.bin");
        for mode in [VisMode::Centre, VisMode::Samples] {
            let mut ctx = IftContext::from_site(&s, mode, Crit::N);
            let mut rng = Pcg64::new(81);
            let mut reused = FastEval::new(&mut ctx, 10).unwrap();
            let sels: Vec<Vec<usize>> = (0..12).map(|t| sel_of(&mut rng, &ctx, 2 + (t * 5) % 9)).collect();
            for (t, sel) in sels.iter().enumerate() {
                let a = reused.neighbourhood(sel).unwrap();
                let mut fresh = FastEval::new(&mut ctx, 10).unwrap();
                let b = fresh.neighbourhood(sel).unwrap();
                assert_nbhd_eq(&a, &b, &format!("{mode:?} #{t}"));
                // interleave the other entry points so every buffer has held a different shape
                let ak = reused.add_keys(&sel[..sel.len() - 1]).unwrap();
                let bk = fresh.add_keys(&sel[..sel.len() - 1]).unwrap();
                assert_ak_eq(&ak, &bk, &format!("{mode:?} #{t}"));
            }
        }
    }

    /// No limit on `n = |S|` (the weights are `bool`s, no bit mask): a 35-marker selection equals the frozen code.
    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn selection_larger_than_32_markers() {
        let s = site("site_small.bin");
        let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let mut rng = Pcg64::new(82);
        let mut fe = FastEval::new(&mut ctx, 35).unwrap();
        let sel = sel_of(&mut rng, &ctx, 35);
        let a = fe.neighbourhood(&sel).unwrap();
        let b = neighbourhood(&mut fe, &sel).unwrap();
        assert_nbhd_eq(&a, &b, "n = 35");
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures; threads
    fn parallel_feature_bits() {
        let s = site("site_small.bin");
        let mut ctx = IftContext::from_site(&s, VisMode::Samples, Crit::N);
        let mut rng = Pcg64::new(83);
        let mut fe = FastEval::new(&mut ctx, 10).unwrap();
        for t in 0..20 {
            let sel = sel_of(&mut rng, &ctx, 2 + t % 9);
            let a = fe.neighbourhood(&sel).unwrap(); // threaded (feature on)
            let b = neighbourhood(&mut fe, &sel).unwrap(); // frozen single-threaded code
            assert_nbhd_eq(&a, &b, &format!("parallel #{t}"));
        }
        assert!(fe.amb.total > 0);
    }
}
