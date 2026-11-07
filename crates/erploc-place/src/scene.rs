//! Port of `erploc/scene.py`: polygon geometry for the workspace, 2.5D obstacles and wall candidates.
//! Polygons are open vertex lists (the last vertex connects to the first); inside = even-odd rule.
//! Operation order follows the numpy code so results are bit-identical.

use crate::npcompat::{arange, col_max, col_min, np_sum, round_half_even_usize};
use crate::placement::PlaceError;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub poly: Vec<[f64; 2]>,
    pub height: f64,
}

#[derive(Clone, Debug)]
pub struct Workspace {
    pub poly: Vec<[f64; 2]>,
    pub spacing: f64,
    pub z_cam: Vec<f64>,
    pub headings: Vec<f64>,
}

/// Largest evaluation grid `cells` builds (points before the polygon test); a larger request is an error, not an allocation failure.
pub const MAX_GRID_POINTS: usize = 20_000_000;
/// Largest number of candidates `candidates_from_walls` produces.
pub const MAX_WALL_CANDIDATES: usize = 10_000_000;

fn poly_finite(poly: &[[f64; 2]]) -> bool {
    poly.iter().all(|p| p[0].is_finite() && p[1].is_finite())
}

impl Workspace {
    /// Checks what `precompute` and `cells` rely on: a polygon of at least three finite vertices, a finite positive spacing, and
    /// non-empty finite `z_cam` and `headings`.
    pub fn validate(&self) -> Result<(), PlaceError> {
        let bad = |m: &str| Err(PlaceError::InvalidInput(format!("workspace: {m}")));
        if self.poly.len() < 3 {
            return bad("the polygon needs at least 3 vertices");
        }
        if !poly_finite(&self.poly) {
            return bad("the polygon has a non-finite vertex");
        }
        if !(self.spacing.is_finite() && self.spacing > 0.0) {
            return bad("spacing must be finite and > 0");
        }
        if self.z_cam.is_empty() || !self.z_cam.iter().all(|z| z.is_finite()) {
            return bad("z_cam must be a non-empty list of finite heights");
        }
        if self.headings.is_empty() || !self.headings.iter().all(|h| h.is_finite()) {
            return bad("headings must be a non-empty list of finite angles");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub xy: [f64; 2],
    pub z: f64,
    pub azimuth: f64,
}

/// `DEFAULT_HEADINGS`: `pi*k/4.0`, k = 0..8.
pub fn default_headings() -> Vec<f64> {
    (0..8).map(|k| PI * k as f64 / 4.0).collect()
}

/// Even-odd rule, concave polygons allowed.
pub fn point_in_polygon(pts: &[[f64; 2]], poly: &[[f64; 2]]) -> Vec<bool> {
    let n = poly.len();
    let mut inside = vec![false; pts.len()];
    for i in 0..n {
        let [x1, y1] = poly[i];
        let [x2, y2] = poly[(i + 1) % n];
        for (ins, p) in inside.iter_mut().zip(pts) {
            let (x, y) = (p[0], p[1]);
            let cond = (y1 > y) != (y2 > y);
            let x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
            *ins ^= cond && (x < x_cross);
        }
    }
    inside
}

/// Signed area, positive = counter-clockwise. Sum is numpy's pairwise `np.sum`.
pub fn polygon_signed_area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    let terms: Vec<f64> = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
        })
        .collect();
    0.5 * np_sum(&terms)
}

/// Flat index of grid point `(iy, ix)`, y outer, x inner (`meshgrid` `'xy'` + `ravel`).
pub(crate) fn ravel_xy(iy: usize, ix: usize, nx: usize) -> usize {
    iy * nx + ix
}

/// Grid points inside the workspace and outside every obstacle; y outer, x inner (`meshgrid` `'xy'` + `ravel`).
///
/// # Errors
/// `PlaceError::InvalidInput` for a polygon with fewer than 3 or a non-finite vertex, a spacing that is not finite and positive, an
/// obstacle with a non-finite vertex, or a grid of more than [`MAX_GRID_POINTS`] points.
pub fn cells(ws: &Workspace, obstacles: &[Obstacle]) -> Result<Vec<[f64; 2]>, PlaceError> {
    let bad = |m: String| Err(PlaceError::InvalidInput(format!("cells: {m}")));
    if ws.poly.len() < 3 || !poly_finite(&ws.poly) {
        return bad("the workspace polygon needs at least 3 finite vertices".into());
    }
    if !(ws.spacing.is_finite() && ws.spacing > 0.0) {
        return bad(format!("spacing must be finite and > 0, got {}", ws.spacing));
    }
    if obstacles.iter().any(|o| !poly_finite(&o.poly)) {
        return bad("an obstacle has a non-finite vertex".into());
    }
    let poly = &ws.poly;
    let (x0, y0) = (col_min(poly, 0), col_min(poly, 1));
    let (x1, y1) = (col_max(poly, 0), col_max(poly, 1));
    let n_axis = |lo: f64, hi: f64| ((hi - (lo + ws.spacing / 2.0)) / ws.spacing).ceil().max(0.0);
    let (ax, ay) = (n_axis(x0, x1), n_axis(y0, y1));
    if !(ax.is_finite() && ay.is_finite()) || ax * ay > MAX_GRID_POINTS as f64 {
        return bad(format!("the grid would have {ax} x {ay} points (limit {MAX_GRID_POINTS}); increase the spacing"));
    }
    let xs = arange(x0 + ws.spacing / 2.0, x1, ws.spacing);
    let ys = arange(y0 + ws.spacing / 2.0, y1, ws.spacing);
    let nx = xs.len();
    let mut pts = Vec::with_capacity(nx * ys.len());
    for (iy, &y) in ys.iter().enumerate() {
        for (ix, &x) in xs.iter().enumerate() {
            debug_assert_eq!(pts.len(), ravel_xy(iy, ix, nx));
            pts.push([x, y]);
        }
    }
    let mut keep = point_in_polygon(&pts, poly);
    for ob in obstacles {
        let inob = point_in_polygon(&pts, &ob.poly);
        for (k, o) in keep.iter_mut().zip(&inob) {
            *k &= !*o;
        }
    }
    Ok(pts.iter().zip(&keep).filter(|(_, &k)| k).map(|(p, _)| *p).collect())
}

/// Parameter intervals of segment `a -> b` that lie inside `poly`, ascending (half-open side rule, see
/// the Python docstring).
pub fn segment_inside_intervals(a: [f64; 2], b: [f64; 2], poly: &[[f64; 2]]) -> Vec<(f64, f64)> {
    let d = [b[0] - a[0], b[1] - a[1]];
    let n = poly.len();
    let pos: Vec<bool> = poly
        .iter()
        .map(|p| {
            let v = [p[0] - a[0], p[1] - a[1]];
            d[0] * v[1] - d[1] * v[0] > 0.0
        })
        .collect();
    let mut ts: Vec<f64> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        if pos[i] == pos[j] {
            continue;
        }
        let (p, q) = (poly[i], poly[j]);
        let e = [q[0] - p[0], q[1] - p[1]];
        let den = d[0] * e[1] - d[1] * e[0];
        if den.abs() < 1e-15 {
            continue;
        }
        let w = [p[0] - a[0], p[1] - a[1]];
        let t = (w[0] * e[1] - w[1] * e[0]) / den;
        if 0.0 < t && t < 1.0 {
            ts.push(t);
        }
    }
    let mut inside = point_in_polygon(&[a], poly)[0];
    if ts.is_empty() {
        return if inside { vec![(0.0, 1.0)] } else { Vec::new() };
    }
    ts.sort_by(|x, y| x.partial_cmp(y).expect("segment_inside_intervals: NaN parameter"));
    let mut bounds = Vec::with_capacity(ts.len() + 2);
    bounds.push(0.0);
    bounds.extend_from_slice(&ts);
    bounds.push(1.0);
    let mut out = Vec::new();
    for k in 0..bounds.len() - 1 {
        if inside && bounds[k + 1] - bounds[k] > 1e-12 {
            out.push((bounds[k], bounds[k + 1]));
        }
        inside = !inside;
    }
    out
}

/// Number of candidates on a wall of `l/spacing` spacings: `max(1, round_half_even(.))`.
pub(crate) fn wall_count(x: f64) -> usize {
    round_half_even_usize(x).max(1)
}

/// Candidates spaced along every polygon edge, normal offset by `offset`; `inward` flips the outward normal.
///
/// # Errors
/// `PlaceError::InvalidInput` for a spacing that is not finite and positive, a non-finite `z` or `offset`, a non-finite vertex, or
/// more than [`MAX_WALL_CANDIDATES`] candidates.
pub fn candidates_from_walls(
    obstacles: &[Obstacle],
    spacing: f64,
    z: f64,
    offset: f64,
    inward: bool,
) -> Result<Vec<Candidate>, PlaceError> {
    let bad = |m: String| Err(PlaceError::InvalidInput(format!("candidates_from_walls: {m}")));
    if !(spacing.is_finite() && spacing > 0.0) {
        return bad(format!("spacing must be finite and > 0, got {spacing}"));
    }
    if !(z.is_finite() && offset.is_finite()) {
        return bad("z and offset must be finite".into());
    }
    if obstacles.iter().any(|o| !poly_finite(&o.poly)) {
        return bad("a polygon has a non-finite vertex".into());
    }
    let mut out = Vec::new();
    for ob in obstacles {
        let poly = &ob.poly;
        let ccw = polygon_signed_area(poly) > 0.0;
        let n_e = poly.len();
        for i in 0..n_e {
            let p = poly[i];
            let q = poly[(i + 1) % n_e];
            let e = [q[0] - p[0], q[1] - p[1]];
            let l = e[0].hypot(e[1]);
            if l <= 0.0 {
                continue;
            }
            let mut nrm = if ccw { [e[1] / l, -e[0] / l] } else { [-e[1] / l, e[0] / l] };
            if inward {
                nrm = [-nrm[0], -nrm[1]];
            }
            let want = l / spacing;
            if !want.is_finite() || want > MAX_WALL_CANDIDATES as f64 {
                return bad(format!("an edge of length {l} would get more than {MAX_WALL_CANDIDATES} candidates at spacing {spacing}"));
            }
            let m = wall_count(want);
            if out.len() + m > MAX_WALL_CANDIDATES {
                return bad(format!("more than {MAX_WALL_CANDIDATES} candidates"));
            }
            for k in 0..m {
                let t = (k as f64 + 0.5) / m as f64;
                let xy = [p[0] + t * e[0] + offset * nrm[0], p[1] + t * e[1] + offset * nrm[1]];
                out.push(Candidate { xy, z, azimuth: nrm[1].atan2(nrm[0]) });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_linalg::rng::Pcg64;

    const L_SHAPE: [[f64; 2]; 6] = [[0., 0.], [4., 0.], [4., 2.], [2., 2.], [2., 4.], [0., 4.]];
    const U_SHAPE: [[f64; 2]; 8] = [[0., 0.], [6., 0.], [6., 4.], [4., 4.], [4., 1.], [2., 1.], [2., 4.], [0., 4.]];
    const DIAMOND: [[f64; 2]; 4] = [[0., 0.], [1., -1.], [2., 0.], [1., 1.]];

    fn close(iv: &[(f64, f64)], want: &[(f64, f64)]) -> bool {
        iv.len() == want.len() && iv.iter().zip(want).all(|(g, w)| (g.0 - w.0).abs() < 1e-12 && (g.1 - w.1).abs() < 1e-12)
    }

    #[test]
    fn b2_l_shape_inside() {
        let pts = [[1., 1.], [3., 1.], [1., 3.], [3., 3.], [5., 1.], [2.5, 2.5]];
        assert_eq!(point_in_polygon(&pts, &L_SHAPE), vec![true, true, true, false, false, false]);
    }

    #[test]
    fn b2_two_crossings() {
        let iv = segment_inside_intervals([-1., 2.], [7., 2.], &U_SHAPE);
        assert!(close(&iv, &[(0.125, 0.375), (0.625, 0.875)]), "{iv:?}");
    }

    #[test]
    fn b2_outside_segment() {
        assert!(segment_inside_intervals([-1., 5.], [7., 5.], &U_SHAPE).is_empty());
    }

    #[test]
    fn b2_vertex_graze() {
        assert!(segment_inside_intervals([-1., 1.], [3., 1.], &DIAMOND).is_empty());
    }

    #[test]
    fn b2_start_inside() {
        let iv = segment_inside_intervals([1., -0.5], [1., 0.5], &DIAMOND);
        assert!(iv.len() == 1 && iv[0].1 == 1.0, "{iv:?}");
    }

    #[test]
    fn b2_vertex_cross() {
        let iv = segment_inside_intervals([1., 1.], [3., 3.], &L_SHAPE);
        assert!(close(&iv, &[(0.0, 0.5)]), "{iv:?}");
    }

    fn pt_seg_dist(pt: [f64; 2], p: [f64; 2], q: [f64; 2]) -> f64 {
        let pq = [q[0] - p[0], q[1] - p[1]];
        let denom = pq[0] * pq[0] + pq[1] * pq[1];
        if denom < 1e-30 {
            return (pt[0] - p[0]).hypot(pt[1] - p[1]);
        }
        let u = (((pt[0] - p[0]) * pq[0] + (pt[1] - p[1]) * pq[1]) / denom).clamp(0.0, 1.0);
        (pt[0] - (p[0] + u * pq[0])).hypot(pt[1] - (p[1] + u * pq[1]))
    }

    #[test]
    fn b2_random_segments_consistency() {
        let (mut compared, mut mismatches) = (0usize, 0usize);
        let mut rng = Pcg64::new(0);
        for fx in [&L_SHAPE[..], &DIAMOND[..]] {
            let lo0 = [col_min(fx, 0), col_min(fx, 1)];
            let hi0 = [col_max(fx, 0), col_max(fx, 1)];
            let mg = [0.25 * (hi0[0] - lo0[0]), 0.25 * (hi0[1] - lo0[1])];
            let lo = [lo0[0] - mg[0], lo0[1] - mg[1]];
            let hi = [hi0[0] + mg[0], hi0[1] + mg[1]];
            for _ in 0..300 {
                let mut pt2 = || [lo[0] + rng.uniform01() * (hi[0] - lo[0]), lo[1] + rng.uniform01() * (hi[1] - lo[1])];
                let pa = pt2();
                let pb = pt2();
                let iv = segment_inside_intervals(pa, pb, fx);
                let dd = [pb[0] - pa[0], pb[1] - pa[1]];
                for kk in 0..40 {
                    let t = (kk + 1) as f64 / 41.0;
                    let pt = [pa[0] + t * dd[0], pa[1] + t * dd[1]];
                    let md = (0..fx.len()).map(|i| pt_seg_dist(pt, fx[i], fx[(i + 1) % fx.len()])).fold(f64::INFINITY, f64::min);
                    if md <= 1e-9 {
                        continue;
                    }
                    compared += 1;
                    let want = point_in_polygon(&[pt], fx)[0];
                    let got = iv.iter().any(|&(t0, t1)| t0 <= t && t <= t1);
                    if got != want {
                        mismatches += 1;
                    }
                }
            }
        }
        println!("b2_random_segments_consistency: compared={compared} mismatches={mismatches}");
        assert!(compared > 10_000);
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn b2_axis_aligned_rect() {
        let rect = [[2., 1.], [4., 1.], [4., 3.], [2., 3.]];
        let iv = segment_inside_intervals([0.5, 2.0], [6.0, 2.0], &rect);
        assert!(close(&iv, &[(1.5 / 5.5, 3.5 / 5.5)]), "{iv:?}");
    }

    #[test]
    fn b2_collinear_edge() {
        let rect = [[2., 1.], [4., 1.], [4., 3.], [2., 3.]];
        let iv = segment_inside_intervals([0.5, 3.0], [6.0, 3.0], &rect);
        let tp = (3.0 - 0.5) / 5.5;
        let want = point_in_polygon(&[[3.0, 3.0]], &rect)[0];
        assert_eq!(iv.iter().any(|&(t0, t1)| t0 <= tp && tp <= t1), want);
    }

    fn room() -> Vec<[f64; 2]> {
        vec![[0., 0.], [8., 0.], [8., 6.], [0., 6.]]
    }

    #[test]
    fn b2_cell_grid() {
        let ws = Workspace { poly: room(), spacing: 0.5, z_cam: vec![1.2], headings: (0..4).map(|k| PI * k as f64 / 2.0).collect() };
        let ob = Obstacle { poly: vec![[3., 2.5], [5., 2.5], [5., 3.5], [3., 3.5]], height: 1.5 };
        let cs = cells(&ws, &[ob]).unwrap();
        assert_eq!(cs.len(), 184);
        // y outer, x inner (meshgrid 'xy' + ravel): the first row runs along x at y = 0.25.
        assert_eq!(cs[0], [0.25, 0.25]);
        assert_eq!(cs[1], [0.75, 0.25]);
        assert_eq!(cs[15], [7.75, 0.25]);
        assert_eq!(cs[16], [0.25, 0.75]);
        assert_eq!(cs[17], [0.75, 0.75]);
    }

    #[test]
    fn b2_wall_candidates() {
        let cands = candidates_from_walls(&[Obstacle { poly: room(), height: 3.0 }], 3.0, 1.5, 0.05, true).unwrap();
        assert_eq!(cands.len(), 10);
        for c in &cands {
            let n = [c.azimuth.cos(), c.azimuth.sin()];
            assert!(n[0] * (4.0 - c.xy[0]) + n[1] * (3.0 - c.xy[1]) > 0.0);
        }
    }

    #[test]
    fn headings_and_area() {
        let h = default_headings();
        assert_eq!(h.len(), 8);
        assert_eq!(h[1], PI * 1.0 / 4.0);
        assert_eq!(polygon_signed_area(&room()), 48.0);
        let cw: Vec<_> = room().into_iter().rev().collect();
        assert_eq!(polygon_signed_area(&cw), -48.0);
    }
}
