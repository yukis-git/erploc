//! Site tables: loader of the `ERPSITE1` binary format (layout in `docs/site-tables.md`). A table is a set of evaluation cells, a set of
//! candidate markers (walls and free-standing stands), two visibility masks and the Fisher information of every visible pair. Tables are
//! produced by the user's own tooling or by `site_synth` (a fully synthetic room, see the `gen_site` example).
//! Two visibility modes share one information table: `Centre` (centre-ray raster test) and `Samples` (13 face samples).
//! Data-dependent failures (corrupt or truncated file) are `PlaceError::Site`, never a panic.
use crate::placement::{PlaceError, PlacementSpec, Tables};
use crate::scene::{default_headings, Candidate, Workspace};
use crate::visibility::VisParams;
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::V3;

const MAGIC: &[u8; 8] = b"ERPSITE1";
const VERSION: u32 = 1;
/// `PlacementSpec::n_max` of the site model. Not stored in the file: writers use it (and the default
/// headings, `lam_sign` 1 and a single camera height). The workspace polygon of `spec0` is a placeholder
/// (the cells are given explicitly); nothing in the site model reads it.
pub const N_MAX: usize = 10;

/// Upper-triangle order of the 4x4 camera block in `i10`.
pub const IU: [(usize, usize); 10] = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)];
/// Radius of the `near` lists \[m\].
pub const LOCAL_R: f64 = 6.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisMode {
    Centre,
    Samples,
}

impl VisMode {
    pub fn index(self) -> usize {
        match self {
            VisMode::Centre => 0,
            VisMode::Samples => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandKind {
    Wall,
    Stand,
}

/// Counts and checksums recorded by the exporter (indexed `[Centre, Samples]` where per mode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteRefs {
    pub n_vis: [u64; 2],
    pub n_triples: [u64; 2],
    /// FNV-1a over the sorted `(s, t, c)` triples, each as three `u32` little endian.
    pub fnv_triples: [u64; 2],
    /// Pairs visible by the centre ray but not by the 13 samples.
    pub n_centre_only: u64,
    /// Pairs with a non-zero information entry.
    pub n_nonzero_pairs: u64,
    /// FNV-1a over `near`: per candidate `u32` length then `u32` entries, little endian.
    pub fnv_near: u64,
    pub n_near_total: u64,
}

pub struct SiteTables {
    pub cells: Vec<V3>,
    pub cands: Vec<Candidate>,
    pub kinds: Vec<CandKind>,
    pub spec0: PlacementSpec,
    /// `[Centre, Samples]`, each `[k*n + j]`.
    pub vis: [Vec<bool>; 2],
    /// `[(k*n + j)*10 + t]`, `t` over `IU`; zero outside the stored pairs.
    pub i10: Vec<f64>,
    pub near: Vec<Vec<usize>>,
    pub refs: SiteRefs,
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Checksum of `near` as the exporter defines it.
pub fn near_checksum(near: &[Vec<usize>]) -> u64 {
    let mut bytes = Vec::new();
    for a in near {
        bytes.extend_from_slice(&(a.len() as u32).to_le_bytes());
        for &j in a {
            bytes.extend_from_slice(&(j as u32).to_le_bytes());
        }
    }
    fnv1a(&bytes)
}

/// `near[j] = nonzero((norm(P[:, None] - P[None]) <= LOCAL_R) & (arange != j))`: `np.linalg.norm` over 2 entries is
/// `sqrt(dx*dx + dy*dy)` (sequential sum), not `hypot`.
pub fn near_lists(cands: &[Candidate]) -> Vec<Vec<usize>> {
    let n = cands.len();
    (0..n)
        .map(|j| {
            (0..n)
                .filter(|&i| {
                    let dx = cands[j].xy[0] - cands[i].xy[0];
                    let dy = cands[j].xy[1] - cands[i].xy[1];
                    i != j && (dx * dx + dy * dy).sqrt() <= LOCAL_R
                })
                .collect()
        })
        .collect()
}

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T, PlaceError> {
    Err(PlaceError::Site(msg.into()))
}

pub(crate) struct Rd<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) p: usize,
}

impl<'a> Rd<'a> {
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], PlaceError> {
        match self.p.checked_add(n) {
            Some(e) if e <= self.b.len() => {
                let s = &self.b[self.p..e];
                self.p = e;
                Ok(s)
            }
            _ => err(format!("truncated file: need {n} bytes at offset {}, have {}", self.p, self.b.len())),
        }
    }
    pub(crate) fn u8(&mut self) -> Result<u8, PlaceError> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u32(&mut self) -> Result<u32, PlaceError> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(a))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, PlaceError> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }
    pub(crate) fn f64(&mut self) -> Result<f64, PlaceError> {
        Ok(f64::from_bits(self.u64()?))
    }
    pub(crate) fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
}

pub(crate) fn unpack_bits(bytes: &[u8], len: usize) -> Vec<bool> {
    (0..len).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
}

impl SiteTables {
    pub fn load(path: &std::path::Path) -> Result<SiteTables, PlaceError> {
        let bytes = std::fs::read(path).map_err(|e| PlaceError::Site(format!("{}: {e}", path.display())))?;
        SiteTables::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<SiteTables, PlaceError> {
        let mut r = Rd { b: bytes, p: 0 };
        if r.take(8)? != MAGIC {
            return err("bad magic (not an ERPSITE1 file)");
        }
        let version = r.u32()?;
        if version != VERSION {
            return err(format!("unsupported version {version} (expected {VERSION})"));
        }
        let nlen = r.u32()? as usize;
        r.take(nlen)?; // numpy version string, informational
        let a = r.f64()?;
        let (w, h) = (r.u32()? as usize, r.u32()? as usize);
        let sigma_px = r.f64()?;
        let noise = r.u32()?;
        let tau = r.f64()?;
        let tau_loo = r.f64()?;
        let k_min = r.u32()? as usize;
        let sigma_cap = r.f64()?;
        let w_k = r.f64()?;
        let psi_max = r.f64()?;
        let px_min = r.f64()?;
        let local_r = r.f64()?;
        let spacing = r.f64()?;
        let z_cam = r.f64()?;
        if local_r != LOCAL_R {
            return err(format!("LOCAL_R {local_r} != {LOCAL_R}"));
        }
        if h == 0 || h.checked_mul(2) != Some(w) {
            return err(format!("camera must be W = 2H (got {w} x {h})"));
        }
        let noise_model = match noise {
            0 => NoiseModel::Tangent,
            1 => NoiseModel::Pixel,
            v => return err(format!("unknown noise model {v}")),
        };
        for (name, v) in [
            ("A", a),
            ("sigma_px", sigma_px),
            ("tau", tau),
            ("tau_loo", tau_loo),
            ("sigma_cap", sigma_cap),
            ("w_k", w_k),
            ("psi_max", psi_max),
            ("px_min", px_min),
            ("spacing", spacing),
            ("z_cam", z_cam),
        ] {
            if !v.is_finite() {
                return err(format!("non-finite scalar {name}"));
            }
        }
        let (k, n) = (r.u32()? as usize, r.u32()? as usize);
        let kn = match k.checked_mul(n) {
            Some(v) => v,
            None => return err("K*N overflows"),
        };
        // every section is checked against the bytes left before anything is allocated
        if k.checked_mul(24).map_or(true, |v| v > r.remaining()) {
            return err("truncated file: cells");
        }
        let mut cells = Vec::with_capacity(k);
        for _ in 0..k {
            let c: V3 = [r.f64()?, r.f64()?, r.f64()?];
            if !c.iter().all(|x| x.is_finite()) {
                return err("non-finite cell coordinate");
            }
            cells.push(c);
        }
        if n.checked_mul(33).map_or(true, |v| v > r.remaining()) {
            return err("truncated file: candidates");
        }
        let mut cands = Vec::with_capacity(n);
        let mut kinds = Vec::with_capacity(n);
        for _ in 0..n {
            let (x, y, z, az) = (r.f64()?, r.f64()?, r.f64()?, r.f64()?);
            if ![x, y, z, az].iter().all(|v| v.is_finite()) {
                return err("non-finite candidate coordinate");
            }
            cands.push(Candidate { xy: [x, y], z, azimuth: az });
            kinds.push(match r.u8()? {
                0 => CandKind::Wall,
                1 => CandKind::Stand,
                v => return err(format!("unknown candidate kind {v}")),
            });
        }
        let nb = kn.div_ceil(8);
        let vis_c = unpack_bits(r.take(nb)?, kn);
        let vis_s = unpack_bits(r.take(nb)?, kn);
        if let Some(at) = (0..kn).find(|&at| vis_s[at] && !vis_c[at]) {
            return err(format!("samples-visible pair {at} is not centre-visible (samples must be a subset)"));
        }
        let n_pairs = r.u32()? as usize;
        if n_pairs > kn || n_pairs.checked_mul(88).map_or(true, |v| v > r.remaining()) {
            return err(format!("bad information pair count {n_pairs}"));
        }
        let mut i10 = vec![0.0; kn * 10];
        let mut seen = vec![false; kn];
        for _ in 0..n_pairs {
            let (pk, pj) = (r.u32()? as usize, r.u32()? as usize);
            if pk >= k || pj >= n {
                return err(format!("information index ({pk}, {pj}) out of range ({k}, {n})"));
            }
            let at = pk * n + pj;
            if seen[at] {
                return err(format!("duplicate information pair ({pk}, {pj})"));
            }
            seen[at] = true;
            for t in 0..10 {
                i10[at * 10 + t] = r.f64()?;
            }
        }
        // every visible pair must carry information
        for at in 0..kn {
            if (vis_c[at] || vis_s[at]) && !seen[at] {
                return err("visible pair without stored information");
            }
        }
        let mut u = [0u64; 10];
        for v in u.iter_mut() {
            *v = r.u64()?;
        }
        if r.remaining() != 0 {
            return err(format!("{} trailing bytes", r.remaining()));
        }
        // semantic consistency (a self-consistent file passes; a corrupted or hand-edited one is an error, not a panic later)
        if !(a > 0.0 && sigma_px > 0.0 && tau > 0.0 && tau_loo > 0.0 && sigma_cap > 0.0 && w_k > 0.0 && spacing > 0.0 && px_min >= 0.0) {
            return err("A, sigma_px, tau, tau_loo, sigma_cap, w_k, spacing must be positive and px_min non-negative");
        }
        if !(psi_max > 0.0 && psi_max < std::f64::consts::PI) {
            return err(format!("psi_max {psi_max} outside (0, pi)"));
        }
        if !(1..=N_MAX).contains(&k_min) {
            return err(format!("k_min {k_min} outside 1..={N_MAX}"));
        }
        if i10.iter().any(|v| !v.is_finite()) {
            return err("non-finite information entry");
        }
        let count = |m: &[bool]| m.iter().filter(|&&v| v).count() as u64;
        if [count(&vis_c), count(&vis_s)] != [u[0], u[3]] {
            return err(format!("visible pair counts {} / {} != recorded {} / {}", count(&vis_c), count(&vis_s), u[0], u[3]));
        }
        if count(&vis_c) - count(&vis_s) != u[6] {
            return err("centre-only pair count != recorded");
        }
        let nonzero = (0..kn).filter(|&at| i10[at * 10..at * 10 + 10].iter().any(|&v| v != 0.0)).count() as u64;
        if nonzero != u[7] {
            return err(format!("non-zero information pairs {nonzero} != recorded {}", u[7]));
        }
        let refs = SiteRefs {
            n_vis: [u[0], u[3]],
            n_triples: [u[1], u[4]],
            fnv_triples: [u[2], u[5]],
            n_centre_only: u[6],
            n_nonzero_pairs: u[7],
            fnv_near: u[8],
            n_near_total: u[9],
        };
        let spec0 = PlacementSpec {
            ws: Workspace {
                poly: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                spacing,
                z_cam: vec![z_cam],
                headings: default_headings(),
            },
            obstacles: Vec::new(),
            candidates: cands.clone(),
            n_max: N_MAX,
            a,
            cam_model: ErpCamera::new(w, h, 1),
            sigma_px,
            noise_model,
            vp: VisParams::new(psi_max, px_min),
            tau,
            k_min,
            tau_loo: Some(tau_loo),
            sigma_cap,
            w_k,
        };
        let near = near_lists(&cands);
        Ok(SiteTables { cells, cands, kinds, spec0, vis: [vis_c, vis_s], i10, near, refs })
    }

    /// Tables of one mode: `hd = 1`, the 10 stored entries expanded to the full symmetric 4x4 camera block.
    pub fn tables(&self, mode: VisMode) -> Tables {
        let (k, n) = (self.cells.len(), self.cands.len());
        let mut info = vec![0.0; k * n * 16];
        for at in 0..k * n {
            for (t, &(a, b)) in IU.iter().enumerate() {
                let v = self.i10[at * 10 + t];
                info[at * 16 + a * 4 + b] = v;
                info[at * 16 + b * 4 + a] = v;
            }
        }
        Tables::from_raw(self.cells.clone(), 1, n, self.vis[mode.index()].clone(), info)
    }

    /// The 10 upper-triangle entries of `I[k, j]` in `IU` order.
    pub fn i10_at(&self, k: usize, j: usize) -> &[f64] {
        let o = (k * self.cands.len() + j) * 10;
        &self.i10[o..o + 10]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::site_synth::{synthetic_site, synthetic_site_bytes};

    fn small_bytes() -> Vec<u8> {
        synthetic_site_bytes().unwrap()
    }

    fn small() -> SiteTables {
        synthetic_site().unwrap()
    }

    // the synthetic table (`site_synth`): 92 cells, 54 candidates, 16 walls / 38 stands
    const K: usize = 92;
    const N: usize = 54;

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn load_small_shapes() {
        let s = small();
        assert_eq!((s.cells.len(), s.cands.len(), s.kinds.len()), (K, N, N));
        assert_eq!(s.kinds.iter().filter(|&&t| t == CandKind::Wall).count(), 16);
        assert_eq!(s.kinds.iter().filter(|&&t| t == CandKind::Stand).count(), 38);
        let sp = &s.spec0;
        assert_eq!((sp.tau, sp.tau_loo, sp.k_min, sp.a, sp.sigma_px, sp.n_max), (0.05, Some(0.5), 2, 0.40, 0.3, 10));
        assert_eq!(sp.noise_model, NoiseModel::Tangent);
        assert_eq!(sp.candidates.len(), N);
        assert_eq!(sp.vp.px_min, 14.0);
        assert_eq!(sp.vp.psi_max, 70.0f64.to_radians());
        for mode in [VisMode::Centre, VisMode::Samples] {
            let tb = s.tables(mode);
            assert_eq!((tb.k, tb.hd, tb.n), (K, 1, N));
            assert!(tb.heading_independent());
            let nvis = s.vis[mode.index()].iter().filter(|&&v| v).count() as u64;
            assert_eq!(nvis, s.refs.n_vis[mode.index()]);
            assert_eq!(tb.vis.iter().filter(|&&v| v).count() as u64, nvis);
        }
        assert_eq!(s.refs.n_vis, [2028, 1962]);
        // I is zero exactly where the exporter says: non-zero pairs = stored non-zero count = centre-visible pairs
        let nz = (0..K * N).filter(|&at| s.i10[at * 10..at * 10 + 10].iter().any(|&v| v != 0.0)).count() as u64;
        assert_eq!(nz, s.refs.n_nonzero_pairs);
        assert_eq!(nz, s.refs.n_vis[0]);
        for at in 0..K * N {
            let any = s.i10[at * 10..at * 10 + 10].iter().any(|&v| v != 0.0);
            assert_eq!(any, s.vis[0][at], "pair {at}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn i10_expands_symmetric() {
        let s = small();
        for mode in [VisMode::Centre, VisMode::Samples] {
            let tb = s.tables(mode);
            for k in 0..K {
                for j in 0..N {
                    let m = tb.info_at(k, j);
                    for a in 0..4 {
                        for b in 0..4 {
                            assert_eq!(m[a * 4 + b].to_bits(), m[b * 4 + a].to_bits());
                        }
                    }
                    let i10 = s.i10_at(k, j);
                    for (t, &(a, b)) in IU.iter().enumerate() {
                        assert_eq!(m[a * 4 + b].to_bits(), i10[t].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn near_lists() {
        let s = small();
        assert_eq!(near_checksum(&s.near), s.refs.fnv_near);
        assert_eq!(s.near.iter().map(|a| a.len() as u64).sum::<u64>(), s.refs.n_near_total);
        for (j, a) in s.near.iter().enumerate() {
            assert!(!a.contains(&j) && a.windows(2).all(|w| w[0] < w[1]));
        }
        // a pair at distance exactly 6.0 is inside (<=), classified with sqrt of the sequential sum
        let mk = |x: f64, y: f64| Candidate { xy: [x, y], z: 0.0, azimuth: 0.0 };
        let c = [mk(0.0, 0.0), mk(6.0, 0.0), mk(0.0, 6.0), mk(3.6, 4.8), mk(6.0, 1e-9), mk(6.0, 1e-7)];
        let nl = super::near_lists(&c);
        // (6, 1e-9): 36 + 1e-18 rounds to 36 -> inside; (6, 1e-7): 36 + 1e-14 -> sqrt just above 6 -> outside
        assert_eq!(nl[0], vec![1, 2, 3, 4]);
        assert_eq!(nl[1], vec![0, 3, 4, 5]);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn corrupt_file() {
        let bytes = small_bytes();
        assert!(SiteTables::from_bytes(&bytes).is_ok());
        let is_site_err = |b: &[u8]| matches!(SiteTables::from_bytes(b), Err(PlaceError::Site(_)));
        // truncated at many places, including inside every section
        for cut in [0, 4, 8, 12, 40, 100, 1000, bytes.len() / 2, bytes.len() - 81, bytes.len() - 80, bytes.len() - 1] {
            assert!(is_site_err(&bytes[..cut]), "cut {cut}");
        }
        let mut b = bytes.clone();
        b[0] = b'X';
        assert!(is_site_err(&b));
        let mut b = bytes.clone();
        b[8] = 9; // version
        assert!(is_site_err(&b));
        // last stored pair: index k out of range (pair record = 88 bytes, then the 80-byte reference block)
        let mut b = bytes.clone();
        let o = bytes.len() - 80 - 88;
        b[o..o + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(is_site_err(&b));
        let mut b = bytes.clone();
        b[o + 4..o + 8].copy_from_slice(&(N as u32).to_le_bytes());
        assert!(is_site_err(&b));
        // trailing garbage
        let mut b = bytes.clone();
        b.push(0);
        assert!(is_site_err(&b));
        // missing file
        assert!(matches!(SiteTables::load(std::path::Path::new("/nonexistent/site.bin")), Err(PlaceError::Site(_))));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn modes_differ_only_in_vis() {
        let s = small();
        let (c, m) = (&s.vis[0], &s.vis[1]);
        let mut only = 0u64;
        for at in 0..K * N {
            assert!(!m[at] || c[at], "samples must be a subset of centre");
            only += (c[at] && !m[at]) as u64;
        }
        assert_eq!(only, s.refs.n_centre_only);
        assert_eq!(only, 66);
        let (a, b) = (s.tables(VisMode::Centre), s.tables(VisMode::Samples));
        assert_eq!(a.info, b.info);
        assert_eq!(a.cells, b.cells);
        assert_ne!(a.vis, b.vis);
    }

    /// Byte offsets of the header scalars and sections of a valid file (layout: docs/site-tables.md).
    struct Off {
        a: usize,
        sigma_px: usize,
        tau: usize,
        tau_loo: usize,
        k_min: usize,
        sigma_cap: usize,
        psi_max: usize,
        head: usize,
        mask_c: usize,
        mask_s: usize,
        pairs: usize,
        refs: usize,
    }

    fn offsets(b: &[u8]) -> Off {
        let nlen = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        let a = 16 + nlen;
        let (sigma_px, tau) = (a + 16, a + 28);
        let (tau_loo, k_min, sigma_cap) = (tau + 8, tau + 16, tau + 20);
        let psi_max = sigma_cap + 16;
        let kn_at = sigma_cap + 8 * 7;
        let (k, n) = (
            u32::from_le_bytes(b[kn_at..kn_at + 4].try_into().unwrap()) as usize,
            u32::from_le_bytes(b[kn_at + 4..kn_at + 8].try_into().unwrap()) as usize,
        );
        let mask_c = kn_at + 8 + k * 24 + n * 33;
        let nb = (k * n).div_ceil(8);
        Off {
            a,
            sigma_px,
            tau,
            tau_loo,
            k_min,
            sigma_cap,
            psi_max,
            head: kn_at + 8,
            mask_c,
            mask_s: mask_c + nb,
            pairs: mask_c + 2 * nb,
            refs: b.len() - 80,
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn offsets_helper_is_right() {
        // guards the offsets used by the corruption tests: the values read back are the known ones
        let b = small_bytes();
        let o = offsets(&b);
        let f = |at: usize| f64::from_le_bytes(b[at..at + 8].try_into().unwrap());
        assert_eq!((f(o.a), f(o.sigma_px), f(o.tau), f(o.tau_loo), f(o.sigma_cap)), (0.40, 0.3, 0.05, 0.5, 10.0));
        assert_eq!(u32::from_le_bytes(b[o.k_min..o.k_min + 4].try_into().unwrap()), 2);
        assert_eq!(f(o.psi_max), 70.0f64.to_radians());
        let npairs = u32::from_le_bytes(b[o.pairs..o.pairs + 4].try_into().unwrap()) as usize;
        assert_eq!(o.pairs + 4 + npairs * 88, o.refs);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn inconsistent_values_rejected() {
        let bytes = small_bytes();
        let o = offsets(&bytes);
        let msg = |b: &[u8]| match SiteTables::from_bytes(b) {
            Err(PlaceError::Site(m)) => m,
            other => panic!("expected PlaceError::Site, got {:?}", other.map(|_| "Ok")),
        };
        let put_f = |at: usize, v: f64| {
            let mut b = bytes.clone();
            b[at..at + 8].copy_from_slice(&v.to_le_bytes());
            b
        };
        for (at, name) in [(o.a, "A"), (o.sigma_px, "sigma_px"), (o.tau, "tau"), (o.tau_loo, "tau_loo"), (o.sigma_cap, "sigma_cap")] {
            for v in [0.0, -1.0] {
                assert!(msg(&put_f(at, v)).contains("positive"), "{name} = {v}");
            }
            assert!(msg(&put_f(at, f64::NAN)).contains("non-finite"), "{name} = NaN");
        }
        assert!(msg(&put_f(o.psi_max, 4.0)).contains("psi_max"));
        for v in [0u32, 11] {
            let mut b = bytes.clone();
            b[o.k_min..o.k_min + 4].copy_from_slice(&v.to_le_bytes());
            assert!(msg(&b).contains("k_min"), "k_min {v}");
        }
        // non-finite information entry: first stored pair, first value (pair record = u32 k, u32 j, 10 f64)
        let mut b = bytes.clone();
        b[o.pairs + 4 + 8..o.pairs + 4 + 16].copy_from_slice(&f64::INFINITY.to_le_bytes());
        assert!(msg(&b).contains("non-finite information"));
        // a bit set in the samples mask where the centre mask is 0
        let (cm, sm) = (&bytes[o.mask_c..o.mask_s], &bytes[o.mask_s..o.pairs]);
        let at = (0..K * N).find(|&i| cm[i / 8] >> (i % 8) & 1 == 0).unwrap();
        let mut b = bytes.clone();
        b[o.mask_s + at / 8] |= 1 << (at % 8);
        assert!(msg(&b).contains("subset"), "{}", msg(&b));
        // popcount != recorded n_vis: clear one centre bit that is also clear in samples ... first set bit of the samples mask
        let at = (0..K * N).find(|&i| sm[i / 8] >> (i % 8) & 1 == 1).unwrap();
        let mut b = bytes.clone();
        b[o.mask_c + at / 8] &= !(1 << (at % 8)); // centre loses a pair the samples mask still has: subset violation is reported first
        assert!(msg(&b).contains("subset"));
        let at = (0..K * N).find(|&i| cm[i / 8] >> (i % 8) & 1 == 1 && sm[i / 8] >> (i % 8) & 1 == 0).unwrap();
        let mut b = bytes.clone();
        b[o.mask_c + at / 8] &= !(1 << (at % 8)); // a centre-only pair removed: n_vis[0] and n_centre_only no longer match
        assert!(msg(&b).contains("counts"), "{}", msg(&b));
        // recorded non-zero pair count
        let mut b = bytes.clone();
        b[o.refs + 7 * 8] ^= 1;
        assert!(msg(&b).contains("non-zero information pairs"));
    }

    /// Never-panics claim of the loader: every prefix and a byte-flip sweep of the synthetic table return `Ok` or `PlaceError::Site`.
    #[test]
    #[cfg_attr(miri, ignore)] // too slow under Miri
    fn no_panic_on_any_prefix_or_byte_flip() {
        let bytes = small_bytes();
        // every prefix in release; in debug builds (about 1 ms per parse) every prefix of the header and of the last 400 bytes
        // and then every 17th
        let step = if cfg!(debug_assertions) { 17 } else { 1 };
        for cut in 0..bytes.len() {
            if cut > 2000 && cut + 400 < bytes.len() && cut % step != 0 {
                continue;
            }
            assert!(matches!(SiteTables::from_bytes(&bytes[..cut]), Err(PlaceError::Site(_))), "prefix {cut} must be an error");
        }
        let o = offsets(&bytes);
        // every bit of the scalar header and of the reference block; the body (cells, candidates, masks, information) (a parse of the full file costs ~5 ms in debug builds):
        // release every 61st byte with all 8 bits, debug every 211th byte with one bit
        let (stride, bits) = if cfg!(debug_assertions) { (211, 1) } else { (61, 8) };
        let header_end = o.head;
        let positions = (0..header_end)
            .map(|p| (p, 8))
            .chain((o.refs..bytes.len()).map(|p| (p, 8)))
            .chain((header_end..o.refs).step_by(stride).map(|p| (p, bits)));
        let mut ok = 0usize;
        let mut b = bytes.clone();
        for (p, nb) in positions {
            for bit in 0..nb {
                let bit = (bit + p) % 8;
                b[p] ^= 1 << bit;
                match SiteTables::from_bytes(&b) {
                    Ok(_) => ok += 1,
                    Err(PlaceError::Site(_)) => {}
                    Err(e) => panic!("unexpected error {e:?}"),
                }
                b[p] ^= 1 << bit;
            }
        }
        eprintln!("byte-flip sweep: {ok} flips still gave a valid file (information values and ignored header fields)");
    }
}
