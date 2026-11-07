//! Zone statistics and the criteria N / K3 / Loo / Both.
//!
//! The zone of a feasible vector is its largest 4-connected component on the 1 m grid of the evaluation cells; ties go to the
//! component whose first cell comes first in C order of `(gi, gj)` (`scipy.ndimage.label` numbers components in raster order and
//! `np.argmax` takes the first maximum). Cells stacked on one grid point (several camera heights) share the grid value of the
//! LAST such cell (`G[gi, gj] = F`, numpy fancy assignment).
use crate::placement::PlacementSpec;
use erploc_core::V3;

/// Criteria of the Python reference: N = nominal, K3 = `k_min` 3, Loo = strict leave-one-out 50 mm, Both = both of those.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crit {
    N,
    K3,
    Loo,
    Both,
}

impl Crit {
    pub const ALL: [Crit; 4] = [Crit::N, Crit::K3, Crit::Loo, Crit::Both];
}

/// `rl.with_spec(ctx, **CRITS[c][0])`: the base specification with the criterion's overrides.
pub fn criterion_spec(base: &PlacementSpec, c: Crit) -> PlacementSpec {
    let mut s = base.clone();
    match c {
        Crit::N => {}
        Crit::K3 => s.k_min = 3,
        Crit::Loo => s.tau_loo = Some(0.05),
        Crit::Both => {
            s.k_min = 3;
            s.tau_loo = Some(0.05);
        }
    }
    s
}

/// 1 m grid of the evaluation cells.
#[derive(Clone, Debug)]
pub struct Grid {
    pub x0: f64,
    pub y0: f64,
    pub gi: Vec<usize>,
    pub gj: Vec<usize>,
    pub shape: (usize, usize),
    pub k: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneStats {
    pub zone: usize,
    pub total: usize,
    pub outside: usize,
    pub holes: usize,
    pub holes_nonfloor: usize,
    pub n_components: usize,
    pub mask: Vec<bool>,
    pub component_sizes: Vec<usize>,
}

/// `np.allclose(a, b)`: `|a - b| <= 1e-8 + 1e-5 |b|`.
fn allclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

impl Grid {
    /// # Panics
    /// When `cells` is empty, non-finite, or not on a 1 m grid (`np.allclose`, as in Python). The callers (`erploc site`, the site
    /// context) check the grid first with `cells_on_1m_grid`.
    pub fn new(cells: &[V3]) -> Grid {
        assert!(!cells.is_empty(), "Grid::new: no cells");
        assert!(cells.iter().all(|c| c[0].is_finite() && c[1].is_finite()), "Grid::new: non-finite cell");
        let x0 = cells.iter().fold(f64::INFINITY, |m, c| m.min(c[0]));
        let y0 = cells.iter().fold(f64::INFINITY, |m, c| m.min(c[1]));
        let mut gi = Vec::with_capacity(cells.len());
        let mut gj = Vec::with_capacity(cells.len());
        for c in cells {
            let (dx, dy) = (c[0] - x0, c[1] - y0);
            assert!(allclose(dx.round_ties_even(), dx) && allclose(dy.round_ties_even(), dy), "Grid::new: cells not on a 1 m grid");
            gi.push(dx.round_ties_even() as usize);
            gj.push(dy.round_ties_even() as usize);
        }
        let shape = (gi.iter().copied().max().unwrap() + 1, gj.iter().copied().max().unwrap() + 1);
        Grid { x0, y0, gi, gj, shape, k: cells.len() }
    }

    fn to_grid(&self, f: &[bool]) -> Vec<bool> {
        let mut g = vec![false; self.shape.0 * self.shape.1];
        for i in 0..self.k {
            g[self.gi[i] * self.shape.1 + self.gj[i]] = f[i];
        }
        g
    }

    /// `Grid.zone`: statistics of the largest 4-connected component of `feasible` (length `K`).
    pub fn zone(&self, feasible: &[bool]) -> ZoneStats {
        assert_eq!(feasible.len(), self.k, "Grid::zone: feasible length != K");
        let (h, w) = self.shape;
        let g = self.to_grid(feasible);
        // labels in raster order of the first cell of each component (scipy.ndimage.label, 4-connectivity)
        let mut lab = vec![0usize; h * w];
        let mut sizes: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for s in 0..h * w {
            if !g[s] || lab[s] != 0 {
                continue;
            }
            sizes.push(0);
            let l = sizes.len();
            lab[s] = l;
            stack.push(s);
            while let Some(p) = stack.pop() {
                sizes[l - 1] += 1;
                let (i, j) = (p / w, p % w);
                let mut push = |q: usize| {
                    if g[q] && lab[q] == 0 {
                        lab[q] = l;
                        stack.push(q);
                    }
                };
                if i > 0 {
                    push(p - w);
                }
                if i + 1 < h {
                    push(p + w);
                }
                if j > 0 {
                    push(p - 1);
                }
                if j + 1 < w {
                    push(p + 1);
                }
            }
        }
        let total = g.iter().filter(|&&v| v).count();
        if sizes.is_empty() {
            return ZoneStats {
                zone: 0,
                total: 0,
                outside: 0,
                holes: 0,
                holes_nonfloor: 0,
                n_components: 0,
                mask: vec![false; self.k],
                component_sizes: vec![],
            };
        }
        // np.argmax: first maximum
        let mut best = 0;
        for (l, &s) in sizes.iter().enumerate() {
            if s > sizes[best] {
                best = l;
            }
        }
        let zl = best + 1;
        let zone_g: Vec<bool> = lab.iter().map(|&l| l == zl).collect();
        // binary_fill_holes: background reachable from the array border through 4-connected background is outside
        let mut outside = vec![false; h * w];
        let mut st: Vec<usize> = Vec::new();
        for p in 0..h * w {
            let (i, j) = (p / w, p % w);
            if (i == 0 || j == 0 || i + 1 == h || j + 1 == w) && !zone_g[p] && !outside[p] {
                outside[p] = true;
                st.push(p);
            }
        }
        while let Some(p) = st.pop() {
            let (i, j) = (p / w, p % w);
            let mut nb = [usize::MAX; 4];
            if i > 0 {
                nb[0] = p - w;
            }
            if i + 1 < h {
                nb[1] = p + w;
            }
            if j > 0 {
                nb[2] = p - 1;
            }
            if j + 1 < w {
                nb[3] = p + 1;
            }
            for q in nb {
                if q != usize::MAX && !zone_g[q] && !outside[q] {
                    outside[q] = true;
                    st.push(q);
                }
            }
        }
        let mut cellmask = vec![false; h * w];
        for i in 0..self.k {
            cellmask[self.gi[i] * w + self.gj[i]] = true;
        }
        let enc = |p: usize| !zone_g[p] && !outside[p];
        let holes = (0..h * w).filter(|&p| enc(p) && cellmask[p] && !g[p]).count();
        let holes_nonfloor = (0..h * w).filter(|&p| enc(p) && !cellmask[p]).count();
        let zone = sizes[best];
        let mask = (0..self.k).map(|i| zone_g[self.gi[i] * w + self.gj[i]]).collect();
        let mut component_sizes = sizes.clone();
        component_sizes.sort_unstable_by(|a, b| b.cmp(a));
        ZoneStats { zone, total, outside: total - zone, holes, holes_nonfloor, n_components: sizes.len(), mask, component_sizes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::site::VisMode;

    /// Cells of a `h x w` lattice from a picture: `#` = evaluation cell (feasible), `.` = evaluation cell (infeasible), ` ` = no cell.
    fn pic(rows: &[&str]) -> (Vec<V3>, Vec<bool>) {
        let (mut cells, mut f) = (Vec::new(), Vec::new());
        for (i, r) in rows.iter().enumerate() {
            for (j, ch) in r.chars().enumerate() {
                if ch != ' ' {
                    cells.push([0.5 + i as f64, 0.5 + j as f64, 1.2]);
                    f.push(ch == '#');
                }
            }
        }
        (cells, f)
    }

    #[test]
    fn l_shape_with_hole() {
        let (cells, f) = pic(&["#####", "#...#", "#.###", "#####", "..#.."]);
        let g = Grid::new(&cells);
        let z = g.zone(&f);
        // one component of 17 cells; the 4 infeasible interior cells are evaluation cells enclosed by the outline, the bottom row's are not
        assert_eq!(z.n_components, 1);
        assert_eq!(z.zone, f.iter().filter(|&&v| v).count());
        assert_eq!(z.total, z.zone);
        assert_eq!(z.outside, 0);
        assert_eq!(z.component_sizes, vec![z.zone]);
        assert_eq!(z.holes, 4);
        assert_eq!(z.holes_nonfloor, 0);
        assert!(z.mask.iter().zip(&f).all(|(a, b)| a == b));
    }

    #[test]
    fn hole_not_a_cell_counts_as_nonfloor() {
        // 3x3 ring with a missing (no evaluation cell) centre
        let (cells, f) = pic(&["###", "# #", "###"]);
        let z = Grid::new(&cells).zone(&f);
        assert_eq!((z.zone, z.holes, z.holes_nonfloor), (8, 0, 1));
    }

    #[test]
    fn open_notch_is_not_a_hole() {
        let (cells, f) = pic(&["###", "#..", "###"]);
        let z = Grid::new(&cells).zone(&f);
        assert_eq!((z.zone, z.holes, z.holes_nonfloor), (7, 0, 0));
        // diagonal connection does not join (4-connectivity)
        let (cells, f) = pic(&["#.", ".#"]);
        let z = Grid::new(&cells).zone(&f);
        assert_eq!((z.n_components, z.zone, z.total, z.outside), (2, 1, 2, 1));
    }

    #[test]
    fn equal_components_first_in_c_order() {
        // two components of 2 cells; labels follow raster order of (gi, gj), the first wins
        let (cells, f) = pic(&["##.#", "...#"]);
        let z = Grid::new(&cells).zone(&f);
        assert_eq!(z.component_sizes, vec![2, 2]);
        let first: Vec<bool> = f.iter().enumerate().map(|(i, &v)| v && i < 2).collect();
        assert_eq!(z.mask, first);
        // shuffled cell order does not change which one wins
        let (mut c2, mut f2) = (cells.clone(), f.clone());
        c2.reverse();
        f2.reverse();
        let z2 = Grid::new(&c2).zone(&f2);
        let mut m2 = z2.mask.clone();
        m2.reverse();
        assert_eq!(m2, first);
    }

    #[test]
    fn stacked_layers_last_cell_wins() {
        // three heights on one grid point: the last cell of the point decides the grid value
        let cells: Vec<V3> = (0..3).flat_map(|l| (0..2).map(move |j| [0.5, 0.5 + j as f64, 1.2 + 0.6 * l as f64])).collect();
        let f = [true, true, false, true, true, false];
        let z = Grid::new(&cells).zone(&f);
        assert_eq!((z.zone, z.total), (1, 1));
        assert_eq!(z.mask, vec![true, false, true, false, true, false]);
    }

    #[test]
    fn empty_feasible_set() {
        let (cells, f) = pic(&["..", ".."]);
        let z = Grid::new(&cells).zone(&f);
        assert_eq!((z.zone, z.total, z.n_components), (0, 0, 0));
        assert!(z.mask.iter().all(|&m| !m) && z.component_sizes.is_empty());
    }

    #[test]
    #[should_panic(expected = "1 m grid")]
    fn off_grid_panics() {
        Grid::new(&[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]);
    }

    #[test]
    fn grid_tolerance_is_allclose() {
        // 1e-9 off the lattice is fine, 1e-3 is not
        let g = Grid::new(&[[0.0, 0.0, 1.0], [1.0 + 1e-9, 0.0, 1.0], [2.0, 1.0 - 1e-9, 1.0]]);
        assert_eq!((g.gi.clone(), g.gj.clone(), g.shape), (vec![0, 1, 2], vec![0, 0, 1], (3, 2)));
        assert!(std::panic::catch_unwind(|| Grid::new(&[[0.0, 0.0, 1.0], [1.001, 0.0, 1.0]])).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads fixtures (Miri isolation / too slow)
    fn criteria_specs() {
        let site = crate::site_synth::synthetic_site().unwrap();
        let b = &site.spec0;
        assert_eq!((b.k_min, b.tau_loo), (2, Some(0.5)));
        let n = criterion_spec(b, Crit::N);
        assert_eq!((n.k_min, n.tau_loo), (2, Some(0.5)));
        let r1 = criterion_spec(b, Crit::K3);
        assert_eq!((r1.k_min, r1.tau_loo), (3, Some(0.5)));
        let r2 = criterion_spec(b, Crit::Loo);
        assert_eq!((r2.k_min, r2.tau_loo), (2, Some(0.05)));
        let r3 = criterion_spec(b, Crit::Both);
        assert_eq!((r3.k_min, r3.tau_loo), (3, Some(0.05)));
        for s in [&n, &r1, &r2, &r3] {
            assert_eq!((s.tau, s.n_max, s.a, s.sigma_px), (b.tau, b.n_max, b.a, b.sigma_px));
        }
        let _ = VisMode::Samples;
    }
}
