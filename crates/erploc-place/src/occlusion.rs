//! Marker self-occlusion for the site model: stand boards and bases block sight lines to other markers.
//! Port of the Python stand-occlusion helpers that are not part of the reference package (`face_samples`, `_hit_board`,
//! `_hit_base`, `_blocker_masks`, `blocked_samples`, `_vis_pair_triples`, `blocked_triples`, `occ_tables_samples`,
//! `blocked_matrix`, `occ_tables`) and the centre-ray branch of `build_occ`. The raster half of `occ_geom` (`los_clear_samples`, `_march_clear`) is not ported.
//!
//! A stand marker is an opaque vertical board (width `2 * HALF_W` along `(-sin az, cos az, 0)`, `z` in `Z0..Z1`) on a base box
//! (`STAND_BASE`, an ASSUMED footprint). A stand's own parts never block its own marker; walls never block.
//! Dot products are `dot3` (the measured numpy association); `np.minimum/np.maximum` propagate NaN (`nmin`/`nmax`).
use crate::npcompat::{dot3, searchsorted_left};
use crate::placement::{nmax, nmin, Tables};
use crate::scene::Candidate;
use crate::site::CandKind;
pub use crate::visibility::{FRONT_GAP_STAND, FRONT_GAP_WALL};
use erploc_core::V3;

/// `r11_occlib`: half width and z range of the board.
pub const HALF_W: f64 = 0.255;
pub const Z0: f64 = 0.35;
pub const Z1: f64 = 0.85;

#[derive(Clone, Copy, Debug)]
pub struct StandBase {
    pub half_w: f64,
    pub depth: f64,
    pub z0: f64,
    pub z1: f64,
}

/// ASSUMED stand footprint (not surveyed): `occ_geom.STAND_BASE`.
pub const STAND_BASE: StandBase = StandBase { half_w: 0.255, depth: 0.30, z0: 0.0, z1: 0.35 };

/// Plan-view prefilter distance of `_vis_pair_triples` [m] (0.6 > base half diagonal + sample offset + margin).
const PREFILTER_D: f64 = 0.6;

/// `occ_geom.face_samples`: 4 tag corners then the 3x3 grid (`b` outer, `a` inner), `a`/`b` along `u = (-sin az, cos az, 0)` and
/// the vertical `(0, 0, 1)`; `(c + a*u) + b*v` as numpy evaluates it.
pub fn site_face_samples(c: &Candidate, a: f64) -> [V3; 13] {
    let h = a / 2.0;
    let t = a / 3.0;
    let mut ab = [(0.0, 0.0); 13];
    ab[..4].copy_from_slice(&[(-h, -h), (h, -h), (h, h), (-h, h)]);
    let mut k = 4;
    for b in [-t, 0.0, t] {
        for a_ in [-t, 0.0, t] {
            ab[k] = (a_, b);
            k += 1;
        }
    }
    let az = c.azimuth;
    let u = [-az.sin(), az.cos(), 0.0];
    let cc = [c.xy[0], c.xy[1], c.z];
    let v = [0.0, 0.0, 1.0];
    let mut out = [[0.0; 3]; 13];
    for (q, &(a_, b)) in ab.iter().enumerate() {
        for i in 0..3 {
            out[q][i] = (cc[i] + a_ * u[i]) + b * v[i];
        }
    }
    out
}

fn frame(s: &Candidate) -> (V3, V3) {
    let az = s.azimuth;
    ([az.cos(), az.sin(), 0.0], [-az.sin(), az.cos(), 0.0])
}

/// `_hit_board` for one segment `cell -> p` against the board of stand `s`.
pub fn hit_board(cell: &V3, p: &V3, s: &Candidate) -> bool {
    let (n, u) = frame(s);
    let c = [s.xy[0], s.xy[1], 0.5 * (Z0 + Z1)];
    let d = [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]];
    let den = dot3(&d, &n);
    let num = dot3(&[c[0] - cell[0], c[1] - cell[1], c[2] - cell[2]], &n);
    let t = num / den;
    let ok = den.abs() > 1e-12 && t > 1e-9 && t < 1.0 - 1e-6;
    if !ok {
        return false;
    }
    let h = [cell[0] + t * d[0], cell[1] + t * d[1], cell[2] + t * d[2]];
    dot3(&[h[0] - c[0], h[1] - c[1], h[2] - c[2]], &u).abs() <= HALF_W && h[2] >= Z0 && h[2] <= Z1
}

/// `_hit_base`: slab test of the open segment `cell + t*(p - cell)`, `t` in `(1e-9, 1 - 1e-6)`, against the base box of stand `s`.
pub fn hit_base(cell: &V3, p: &V3, s: &Candidate, base: &StandBase) -> bool {
    let (n, u) = frame(s);
    let d = [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]];
    let r = [cell[0] - s.xy[0], cell[1] - s.xy[1], cell[2] - 0.0];
    let q0 = [dot3(&r, &u), dot3(&r, &n), r[2]];
    let dv = [dot3(&d, &u), dot3(&d, &n), d[2]];
    let lo = [-base.half_w, -base.depth / 2.0, base.z0];
    let hi = [base.half_w, base.depth / 2.0, base.z1];
    let (mut t0, mut t1, mut alive) = (1e-9, 1.0 - 1e-6, true);
    for i in 0..3 {
        let par = dv[i].abs() < 1e-12;
        alive &= !(par && (q0[i] < lo[i] || q0[i] > hi[i]));
        let ta = (lo[i] - q0[i]) / dv[i];
        let tb = (hi[i] - q0[i]) / dv[i];
        let (tn, tf) = (nmin(ta, tb), nmax(ta, tb));
        if !par {
            t0 = nmax(t0, tn);
            t1 = nmin(t1, tf);
        }
    }
    alive && t0 <= t1
}

/// Board or base of stand `s` blocks the segment `cell -> p`.
fn blocks(cell: &V3, p: &V3, s: &Candidate) -> bool {
    hit_board(cell, p, s) | hit_base(cell, p, s, &STAND_BASE)
}

/// ANY of the samples is blocked by stand `s`.
fn any_sample_blocked(cell: &V3, ps: &[V3; 13], s: &Candidate) -> bool {
    ps.iter().fold(false, |h, p| h | blocks(cell, p, s))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccMode {
    /// Centre ray against boards only (`r11_occlib`, `ift_lib.build_occ` centre branch).
    Centre,
    /// Any of the 13 face samples against boards and bases (`occ_geom`).
    Samples,
}

/// `ift_lib.build_occ` dict: triples `(blocker stand s, target t, cell c)` for visible pairs, in CSR form by blocker and by target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Occ {
    pub v_c: Vec<u32>,
    pub v_j: Vec<u32>,
    pub n_triples: usize,
    pub bs_c: Vec<u32>,
    pub bs_t: Vec<u32>,
    pub bs_ptr: Vec<usize>,
    pub bt_s: Vec<u32>,
    pub bt_c: Vec<u32>,
    pub bt_ptr: Vec<usize>,
}

/// Stable sort of the triple indices by `keys` and the `np.searchsorted(sorted_key, arange(n + 1))` pointer array.
fn csr(keys: &[u32], n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&i| keys[i]); // stable
    let sorted: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
    let ptr = (0..=n as u32).map(|v| searchsorted_left(&sorted, v)).collect();
    (order, ptr)
}

fn stand_list(kinds: &[CandKind]) -> Vec<usize> {
    (0..kinds.len()).filter(|&s| kinds[s] == CandKind::Stand).collect()
}

/// `ift_lib.build_occ` / `occ_geom.blocked_triples(..., sel = all, vis)`. `vis` is `[k*n + j]`.
/// With no stands in `kinds` the result is an empty `Occ` (visible pairs only, zero triples, all-zero pointer arrays); the Python
/// `occ_geom.blocked_triples` raises in that case (`np.stack` of an empty list), so this is a deliberate difference.
///
/// # Panics
/// When `kinds.len() != cands.len()` or `vis.len() != cells.len() * cands.len()` (a shape error of the caller's arrays; the site
/// loader builds them consistently).
pub fn blocked_triples(cells: &[V3], cands: &[Candidate], kinds: &[CandKind], a: f64, vis: &[bool], mode: OccMode) -> Occ {
    let n = cands.len();
    assert_eq!(kinds.len(), n, "blocked_triples: kinds length");
    assert_eq!(vis.len(), cells.len() * n, "blocked_triples: vis length");
    let (mut v_c, mut v_j) = (Vec::new(), Vec::new());
    for k in 0..cells.len() {
        for j in 0..n {
            if vis[k * n + j] {
                v_c.push(k as u32);
                v_j.push(j as u32);
            }
        }
    }
    let (mut s_a, mut t_a, mut c_a) = (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());
    let stands = stand_list(kinds);
    match mode {
        OccMode::Samples => {
            let ps: Vec<[V3; 13]> = cands.iter().map(|c| site_face_samples(c, a)).collect();
            // segment in plan view (cell -> candidate xy) data per visible pair
            let seg0: Vec<[f64; 2]> = v_c.iter().map(|&k| [cells[k as usize][0], cells[k as usize][1]]).collect();
            let dv: Vec<[f64; 2]> = v_c
                .iter()
                .zip(&v_j)
                .map(|(&k, &j)| {
                    let (c, p) = (&cells[k as usize], &cands[j as usize]);
                    [p.xy[0] - c[0], p.xy[1] - c[1]]
                })
                .collect();
            let l2: Vec<f64> = dv.iter().map(|d| nmax(d[0] * d[0] + d[1] * d[1], 1e-18)).collect();
            for &s in &stands {
                let sp = cands[s].xy;
                for i in 0..v_c.len() {
                    let vj = v_j[i] as usize;
                    let num = (sp[0] - seg0[i][0]) * dv[i][0] + (sp[1] - seg0[i][1]) * dv[i][1];
                    let w = nmin(nmax(num / l2[i], 0.0), 1.0);
                    let (ex, ey) = ((seg0[i][0] + w * dv[i][0]) - sp[0], (seg0[i][1] + w * dv[i][1]) - sp[1]);
                    let d = (ex * ex + ey * ey).sqrt();
                    if !(d < PREFILTER_D && vj != s) {
                        continue;
                    }
                    if any_sample_blocked(&cells[v_c[i] as usize], &ps[vj], &cands[s]) {
                        s_a.push(s as u32);
                        t_a.push(vj as u32);
                        c_a.push(v_c[i]);
                    }
                }
            }
        }
        OccMode::Centre => {
            let p_all: Vec<V3> = cands.iter().map(|c| [c.xy[0], c.xy[1], c.z]).collect();
            for &s in &stands {
                for i in 0..v_c.len() {
                    let vj = v_j[i] as usize;
                    if vj != s && hit_board(&cells[v_c[i] as usize], &p_all[vj], &cands[s]) {
                        s_a.push(s as u32);
                        t_a.push(vj as u32);
                        c_a.push(v_c[i]);
                    }
                }
            }
        }
    }
    let (o1, bs_ptr) = csr(&s_a, n);
    let (o2, bt_ptr) = csr(&t_a, n);
    Occ {
        v_c,
        v_j,
        n_triples: s_a.len(),
        bs_c: o1.iter().map(|&i| c_a[i]).collect(),
        bs_t: o1.iter().map(|&i| t_a[i]).collect(),
        bs_ptr,
        bt_s: o2.iter().map(|&i| s_a[i]).collect(),
        bt_c: o2.iter().map(|&i| c_a[i]).collect(),
        bt_ptr,
    }
}

/// `blocked_samples` / `r11_occlib.blocked_matrix`: `(K, N)` row-major; only the columns `j` in `sel` are filled; a target is
/// blocked by another selected stand (`s != j`), over ALL cells (not only visible ones).
pub fn blocked_matrix(cells: &[V3], cands: &[Candidate], kinds: &[CandKind], sel: &[usize], a: f64, mode: OccMode) -> Vec<bool> {
    let n = cands.len();
    assert_eq!(kinds.len(), n, "blocked_matrix: kinds length");
    assert!(sel.iter().all(|&j| j < n), "blocked_matrix: selection index out of range");
    let mut b = vec![false; cells.len() * n];
    let stands: Vec<usize> = sel.iter().copied().filter(|&s| kinds[s] == CandKind::Stand).collect();
    for &j in sel {
        match mode {
            OccMode::Samples => {
                let ps = site_face_samples(&cands[j], a);
                for &s in stands.iter().filter(|&&s| s != j) {
                    for (k, c) in cells.iter().enumerate() {
                        if any_sample_blocked(c, &ps, &cands[s]) {
                            b[k * n + j] = true;
                        }
                    }
                }
            }
            OccMode::Centre => {
                let p = [cands[j].xy[0], cands[j].xy[1], cands[j].z];
                for &s in stands.iter().filter(|&&s| s != j) {
                    for (k, c) in cells.iter().enumerate() {
                        if hit_board(c, &p, &cands[s]) {
                            b[k * n + j] = true;
                        }
                    }
                }
            }
        }
    }
    b
}

/// `occ_tables` / `occ_tables_samples`: tables whose `vis` is `vis & !blocked` over every heading slice.
/// Returns `(tables, n_blocked_pairs, n_vis_before)` with `before = sum(V0[:, sel])` (heading 0) and `blocked = (B & V0).sum()`.
pub fn occ_tables(tb: &Tables, cands: &[Candidate], kinds: &[CandKind], sel: &[usize], a: f64, mode: OccMode) -> (Tables, usize, usize) {
    assert_eq!(tb.n, cands.len(), "occ_tables: tables/candidates mismatch");
    let (k, hd, n) = (tb.k, tb.hd, tb.n);
    let b = blocked_matrix(&tb.cells, cands, kinds, sel, a, mode);
    let v0 = |kk: usize, j: usize| tb.vis[(kk * hd) * n + j];
    let before = sel.iter().map(|&j| (0..k).filter(|&kk| v0(kk, j)).count()).sum();
    let blocked = (0..k).map(|kk| (0..n).filter(|&j| b[kk * n + j] && v0(kk, j)).count()).sum();
    let mut vis = tb.vis.clone();
    for kk in 0..k {
        for h in 0..hd {
            for j in 0..n {
                if b[kk * n + j] {
                    vis[(kk * hd + h) * n + j] = false;
                }
            }
        }
    }
    (Tables::from_raw(tb.cells.clone(), hd, n, vis, tb.info.clone()), blocked, before)
}

/// Sorted `(s, t, c)` triples of `occ` (blocker, target, cell) as the exporter checksums them: lexicographic order, each as
/// three `u32` little endian, FNV-1a 64.
pub fn triples_checksum(occ: &Occ) -> u64 {
    let mut tr: Vec<[u32; 3]> = Vec::with_capacity(occ.n_triples);
    for s in 0..occ.bs_ptr.len() - 1 {
        for i in occ.bs_ptr[s]..occ.bs_ptr[s + 1] {
            tr.push([s as u32, occ.bs_t[i], occ.bs_c[i]]);
        }
    }
    tr.sort_unstable();
    let mut bytes = Vec::with_capacity(tr.len() * 12);
    for t in &tr {
        for v in t {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    crate::site::fnv1a(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{demo_spec, precompute, precompute_typed, PlacementSpec};
    use crate::scene::{Obstacle, Workspace};
    use crate::site::{SiteTables, VisMode};
    use erploc_core::erp::ErpCamera;
    use erploc_linalg::rng::Pcg64;
    use std::f64::consts::PI;

    const S0: Candidate = Candidate { xy: [0.0, 0.0], z: 0.6, azimuth: 0.0 };

    #[test]
    fn board_blocks_between() {
        assert!(hit_board(&[-2.0, 0.0, 0.6], &[2.0, 0.0, 0.6], &S0));
        assert!(hit_board(&[2.0, 0.1, 0.5], &[-2.0, -0.1, 0.7], &S0));
        // target before the board, or cell behind it: no crossing
        assert!(!hit_board(&[-2.0, 0.0, 0.6], &[-1.0, 0.0, 0.6], &S0));
        assert!(!hit_board(&[1.0, 0.0, 0.6], &[2.0, 0.0, 0.6], &S0));
        // parallel to the plane (den = 0): never
        assert!(!hit_board(&[0.0, -2.0, 0.6], &[0.0, 2.0, 0.6], &S0));
    }

    #[test]
    fn board_end_not_blocking() {
        // a board at exactly the target position (t = 1) and at the cell (t = 0) does not block
        assert!(!hit_board(&[-2.0, 0.0, 0.6], &[0.0, 0.0, 0.6], &S0));
        assert!(!hit_board(&[0.0, 0.0, 0.6], &[2.0, 0.0, 0.6], &S0));
        // t just inside (1 - 1e-6): 1e-7 of the 2 m segment from the end blocks nothing, 1e-5 does
        let near = |dx: f64| hit_board(&[-2.0, 0.0, 0.6], &[dx, 0.0, 0.6], &S0);
        assert!(!near(1e-7) && near(1e-5));
    }

    #[test]
    fn board_edge_halfwidth() {
        let h = |y: f64, z: f64| hit_board(&[-2.0, y, z], &[2.0, y, z], &S0);
        assert!(h(HALF_W, 0.6) && h(-HALF_W, 0.6));
        assert!(!h(HALF_W + 1e-9, 0.6) && !h(-HALF_W - 1e-9, 0.6));
        assert!(h(0.0, Z0) && h(0.0, Z1));
        assert!(!h(0.0, Z0 - 1e-9) && !h(0.0, Z1 + 1e-9));
    }

    #[test]
    fn base_blocks_low_ray() {
        let (c, p) = ([-2.0, 0.0, 0.1], [2.0, 0.0, 0.1]);
        assert!(hit_base(&c, &p, &S0, &STAND_BASE));
        assert!(!hit_board(&c, &p, &S0)); // below the board
                                          // depth is 0.30: the box spans x in [-0.15, 0.15]; a segment ending before it does not hit
        assert!(!hit_base(&c, &[-0.2, 0.0, 0.1], &S0, &STAND_BASE));
        assert!(hit_base(&c, &[-0.1, 0.0, 0.1], &S0, &STAND_BASE));
    }

    #[test]
    fn base_does_not_block_high_ray() {
        let (c, p) = ([-2.0, 0.0, 0.6], [2.0, 0.0, 0.6]);
        assert!(!hit_base(&c, &p, &S0, &STAND_BASE));
        assert!(hit_board(&c, &p, &S0));
        let (c, p) = ([-2.0, 0.0, 0.9], [2.0, 0.0, 0.9]);
        assert!(!hit_base(&c, &p, &S0, &STAND_BASE) && !hit_board(&c, &p, &S0));
        // rising ray: z = 0.35 only at x = -2 + 4*(0.25/2.9), far outside the box
        assert!(!hit_base(&[-2.0, 0.0, 0.1], &[2.0, 0.0, 3.0], &S0, &STAND_BASE));
    }

    #[test]
    fn parallel_axis_skip() {
        let h = |y: f64, z: f64| hit_base(&[-2.0, y, z], &[2.0, y, z], &S0, &STAND_BASE);
        // dv_y = dv_z = 0 (both parallel): inside the slab hits, outside misses, and the slab faces themselves are inside
        // (0/0 = NaN in the slab arithmetic must not leak when the axis is skipped)
        assert!(h(0.0, 0.1));
        assert!(h(0.255, 0.1) && h(-0.255, 0.1));
        assert!(h(0.0, 0.0) && h(0.0, 0.35));
        assert!(!h(0.3, 0.1) && !h(-0.3, 0.1));
        assert!(!h(0.0, 0.4) && !h(0.0, -0.01));
        // |dv| < 1e-12 counts as parallel, outside the slab: miss; inside: hit
        let tiny = |y: f64| hit_base(&[-2.0, y, 0.1], &[2.0, y + 5e-13, 0.1], &S0, &STAND_BASE);
        assert!(tiny(0.0) && !tiny(0.3));
    }

    fn two_stands() -> (Vec<Candidate>, Vec<CandKind>) {
        let a = Candidate { xy: [0.0, 0.0], z: 0.6, azimuth: 0.0 };
        let b = Candidate { xy: [3.0, 0.0], z: 0.6, azimuth: PI };
        (vec![a, b], vec![CandKind::Stand, CandKind::Stand])
    }

    #[test]
    fn own_parts_never_block() {
        let (c, k) = two_stands();
        let cells = [[-3.0, 0.0, 0.6]];
        for mode in [OccMode::Centre, OccMode::Samples] {
            // only stand 0 selected: the ray to its own marker passes its own board / base, and nothing else is selected
            assert_eq!(blocked_matrix(&cells, &c, &k, &[0], 0.16, mode), vec![false, false], "{mode:?}");
            // both selected: stand 0 blocks marker 1, stand 1 (beyond) does not block marker 0
            assert_eq!(blocked_matrix(&cells, &c, &k, &[0, 1], 0.16, mode), vec![false, true], "{mode:?}");
        }
    }

    #[test]
    fn wall_never_blocks() {
        let (c, _) = two_stands();
        let k = [CandKind::Wall, CandKind::Stand];
        let cells = [[-3.0, 0.0, 0.6]];
        for mode in [OccMode::Centre, OccMode::Samples] {
            assert_eq!(blocked_matrix(&cells, &c, &k, &[0, 1], 0.16, mode), vec![false, false], "{mode:?}");
        }
        // triples: a wall marker as a target is visible-and-blocked by a stand, a wall as blocker never
        let k2 = [CandKind::Wall, CandKind::Wall];
        let o = blocked_triples(&cells, &c, &k2, 0.16, &[true, true], OccMode::Samples);
        assert_eq!(o.n_triples, 0);
    }

    #[test]
    fn samples_vs_centre() {
        let (c, k) = two_stands();
        // cell off to the side: the centre ray to marker 1 passes the board's edge at y = 0.35 (> HALF_W), the two corner
        // samples on the far side pass at y = 0.25 (inside); the grid samples pass outside
        let cells = [[-3.0, 0.7, 0.6]];
        assert_eq!(blocked_matrix(&cells, &c, &k, &[0, 1], 0.4, OccMode::Centre), vec![false, false]);
        assert_eq!(blocked_matrix(&cells, &c, &k, &[0, 1], 0.4, OccMode::Samples), vec![false, true]);
        let o = blocked_triples(&cells, &c, &k, 0.4, &[true, true], OccMode::Samples);
        assert_eq!((o.n_triples, o.bs_t.clone(), o.bs_ptr.clone()), (1, vec![1], vec![0, 1, 1]));
        assert_eq!((o.bt_s.clone(), o.bt_ptr.clone()), (vec![0], vec![0, 0, 1]));
        let o = blocked_triples(&cells, &c, &k, 0.4, &[true, true], OccMode::Centre);
        assert_eq!(o.n_triples, 0);
    }

    #[test]
    fn site_face_samples_layout() {
        let a = 0.4;
        let cd = Candidate { xy: [1.0, 2.0], z: 1.5, azimuth: 0.3 };
        let s = site_face_samples(&cd, a);
        assert_eq!(s.len(), 13);
        let (h, t) = (a / 2.0, a / 3.0);
        let u = [-0.3f64.sin(), 0.3f64.cos()];
        let want = |a_: f64, b: f64| [1.0 + a_ * u[0], 2.0 + a_ * u[1], 1.5 + b];
        let corners = [(-h, -h), (h, -h), (h, h), (-h, h)];
        for (k, &(a_, b)) in corners.iter().enumerate() {
            let w = want(a_, b);
            assert!(s[k].iter().zip(&w).all(|(x, y)| (x - y).abs() < 1e-15), "corner {k}");
        }
        let mut k = 4;
        for b in [-t, 0.0, t] {
            for a_ in [-t, 0.0, t] {
                let w = want(a_, b);
                assert!(s[k].iter().zip(&w).all(|(x, y)| (x - y).abs() < 1e-15), "grid {k}");
                k += 1;
            }
        }
        for q in &s {
            let d = ((q[0] - 1.0).powi(2) + (q[1] - 2.0).powi(2) + (q[2] - 1.5).powi(2)).sqrt();
            assert!(d <= 0.75 * a);
        }
        // az = 0: horizontal axis is +y
        let s0 = site_face_samples(&Candidate { xy: [0.0, 0.0], z: 1.0, azimuth: 0.0 }, a);
        assert!((s0[1][1] - h).abs() < 1e-15 && (s0[1][2] - (1.0 - h)).abs() < 1e-15 && s0[1][0].abs() < 1e-15);
    }

    fn pole_spec() -> PlacementSpec {
        let rect = |x0: f64, y0: f64, x1: f64, y1: f64, h: f64| Obstacle { poly: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]], height: h };
        let base = demo_spec(10);
        let m = |az: f64| Candidate { xy: [4.0, 0.0], z: 1.5, azimuth: az };
        PlacementSpec {
            ws: Workspace {
                poly: vec![[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                spacing: 0.5,
                z_cam: vec![1.2],
                headings: vec![0.0],
            },
            obstacles: vec![rect(3.94, -1.0, 3.96, 1.0, 2.0)],
            candidates: vec![m(PI), m(PI)],
            n_max: 2,
            cam_model: ErpCamera::new(6528, 3264, 1),
            ..base
        }
    }

    #[test]
    fn typed_precompute_all_wall_equals_precompute() {
        for spec in [demo_spec(10), pole_spec()] {
            let a = precompute(&spec).unwrap();
            let b = precompute_typed(&spec, &vec![CandKind::Wall; spec.candidates.len()]).unwrap();
            assert_eq!((a.k, a.hd, a.n), (b.k, b.hd, b.n));
            assert_eq!(a.vis, b.vis);
            assert!(a.cells.iter().zip(&b.cells).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())));
            assert!(a.info.iter().zip(&b.info).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }

    #[test]
    fn typed_precompute_stand_gap() {
        // the pole stands 0.05 m in front of the marker: the stand gap (0.02) sees it, the wall gap (0.07) stops before it
        let spec = pole_spec();
        let t = precompute_typed(&spec, &[CandKind::Stand, CandKind::Wall]).unwrap();
        assert_eq!((t.k, t.hd, t.n), (4, 1, 2));
        for k in 0..t.k {
            assert!(!t.vis_at(k, 0, 0), "stand-gap marker is blocked by the pole (cell {k})");
            assert!(t.vis_at(k, 0, 1), "wall-gap marker is not (cell {k})");
        }
        let w = precompute(&spec).unwrap();
        assert!((0..w.k).all(|k| w.vis_at(k, 0, 0) && w.vis_at(k, 0, 1)));
        // info does not depend on the gap
        assert_eq!(t.info, w.info);
    }

    fn small() -> SiteTables {
        crate::site_synth::synthetic_site().unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn small_scene_triples_reproduce_the_exporter() {
        let s = small();
        for (mode, vm, want) in [(OccMode::Centre, VisMode::Centre, 120usize), (OccMode::Samples, VisMode::Samples, 280)] {
            let i = vm.index();
            let o = blocked_triples(&s.cells, &s.cands, &s.kinds, s.spec0.a, &s.vis[i], mode);
            assert_eq!(o.n_triples, want, "{mode:?}");
            assert_eq!(o.n_triples as u64, s.refs.n_triples[i]);
            assert_eq!(triples_checksum(&o), s.refs.fnv_triples[i], "{mode:?} FNV");
            assert_eq!(o.v_c.len() as u64, s.refs.n_vis[i]);
            let n = s.cands.len();
            assert_eq!((o.bs_ptr.len(), o.bt_ptr.len()), (n + 1, n + 1));
            assert_eq!((o.bs_ptr[n], o.bt_ptr[n]), (want, want));
            // CSR arrays are sorted stably: within a blocker the order is (target-major pair order of generation)
            assert!(o.bs_ptr.windows(2).all(|w| w[0] <= w[1]) && o.bt_ptr.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn blocked_matrix_equals_triples() {
        let s = small();
        let (k, n) = (s.cells.len(), s.cands.len());
        let mut rng = Pcg64::new(20261011);
        for (mode, vm) in [(OccMode::Centre, VisMode::Centre), (OccMode::Samples, VisMode::Samples)] {
            let tb = s.tables(vm);
            let occ = blocked_triples(&s.cells, &s.cands, &s.kinds, s.spec0.a, &s.vis[vm.index()], mode);
            let mut any_blocked = 0usize;
            for case in 0..50 {
                let size = 6 + (rng.next_u64() % 4) as usize;
                let mut sel: Vec<usize> = Vec::new();
                while sel.len() < size {
                    let j = (rng.next_u64() % n as u64) as usize;
                    if !sel.contains(&j) {
                        sel.push(j);
                    }
                }
                let (tb2, nb, before) = occ_tables(&tb, &s.cands, &s.kinds, &sel, s.spec0.a, mode);
                // union of the triples of the selected stands whose target is selected
                let mut blocked = vec![false; k * n];
                for &st in sel.iter().filter(|&&j| s.kinds[j] == CandKind::Stand) {
                    for i in occ.bs_ptr[st]..occ.bs_ptr[st + 1] {
                        let t = occ.bs_t[i] as usize;
                        if sel.contains(&t) {
                            blocked[occ.bs_c[i] as usize * n + t] = true;
                        }
                    }
                }
                let mut removed = 0;
                for kk in 0..k {
                    for j in 0..n {
                        let want = tb.vis[kk * n + j] && !blocked[kk * n + j];
                        assert_eq!(tb2.vis[kk * n + j], want, "{mode:?} case {case} sel {sel:?} cell {kk} marker {j}");
                        removed += (tb.vis[kk * n + j] && !tb2.vis[kk * n + j]) as usize;
                    }
                }
                assert_eq!(nb, removed);
                assert_eq!(before, sel.iter().map(|&j| (0..k).filter(|&kk| tb.vis[kk * n + j]).count()).sum::<usize>());
                any_blocked += nb;
            }
            assert!(any_blocked > 0, "{mode:?}: the 50 selections never block anything (vacuous test)");
        }
    }
}
