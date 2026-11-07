//! Fully synthetic site tables in the `ERPSITE1` format.
//!
//! [`synthetic_site_bytes`] builds a small rectangular room (14 m x 10 m, one box-shaped obstacle, wall markers all round and a grid
//! of free-standing stand markers) and writes it as a site table. Nothing in it comes from a measurement: the cells, the
//! candidates, the visibility masks and the information blocks are computed from this geometry with the library's own visibility
//! and Fisher-information code. The `gen_site` example writes the bytes to a file, which is the intended way to obtain a table for
//! `erploc site --tables`.
//!
//! The table is self-consistent in the sense the loader checks (samples-visible pairs are a subset of the centre-visible pairs,
//! information is stored for every centre-visible pair, the recorded counts and checksums match), so [`SiteTables::from_bytes`] accepts it.
use crate::occlusion::{blocked_triples, triples_checksum, OccMode};
use crate::placement::{precompute_typed, PlaceError, PlacementSpec};
use crate::scene::{candidates_from_walls, Candidate, Obstacle, Workspace};
use crate::site::{near_checksum, near_lists, CandKind, SiteTables, IU, LOCAL_R};
use crate::visibility::{visible_batch, VisParams, FRONT_GAP_STAND, FRONT_GAP_WALL};
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::markers::upright_pose;
use erploc_core::V3;

const ROOM: [[f64; 2]; 4] = [[0., 0.], [14., 0.], [14., 10.], [0., 10.]];
const WORK: [[f64; 2]; 4] = [[1., 1.], [13., 1.], [13., 9.], [1., 9.]];
const BOX: [[f64; 2]; 4] = [[6., 4.], [8., 4.], [8., 6.], [6., 6.]];
/// Marker side [m].
const A: f64 = 0.40;
/// Height of the marker centre on a stand [m] (inside the board, `occlusion::Z0..Z1`).
const STAND_Z: f64 = 0.6;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn pack_bits(m: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; m.len().div_ceil(8)];
    for (i, &v) in m.iter().enumerate() {
        out[i / 8] |= (v as u8) << (i % 8);
    }
    out
}

fn synth_spec() -> (PlacementSpec, Vec<CandKind>) {
    let mut candidates: Vec<Candidate> =
        candidates_from_walls(&[Obstacle { poly: ROOM.to_vec(), height: 3.0 }], 3.0, 1.5, 0.05, true).expect("synthetic room walls");
    let mut kinds = vec![CandKind::Wall; candidates.len()];
    // free-standing stands on a coarse grid, each facing one of four directions in turn; none inside the box
    let mut q = 0;
    for &y in &[2.0, 3.5, 5.0, 6.5, 8.0, 9.0] {
        for &x in &[2.0, 3.5, 5.0, 9.0, 10.5, 12.0] {
            candidates.push(Candidate { xy: [x, y], z: STAND_Z, azimuth: std::f64::consts::FRAC_PI_2 * (q % 4) as f64 });
            kinds.push(CandKind::Stand);
            q += 1;
        }
    }
    for &(x, y) in &[(7.0, 2.0), (7.0, 8.0)] {
        candidates.push(Candidate { xy: [x, y], z: STAND_Z, azimuth: std::f64::consts::FRAC_PI_2 * (q % 4) as f64 });
        kinds.push(CandKind::Stand);
        q += 1;
    }
    let spec = PlacementSpec {
        ws: Workspace { poly: WORK.to_vec(), spacing: 1.0, z_cam: vec![1.2], headings: vec![0.0] },
        obstacles: vec![Obstacle { poly: BOX.to_vec(), height: 2.5 }],
        candidates,
        n_max: 10,
        a: A,
        cam_model: ErpCamera::new(6528, 3264, 1),
        sigma_px: 0.3,
        noise_model: NoiseModel::Tangent,
        vp: VisParams::new(70.0_f64.to_radians(), 14.0),
        tau: 0.05,
        k_min: 2,
        tau_loo: Some(0.5),
        sigma_cap: 10.0,
        w_k: 1.0,
    };
    (spec, kinds)
}

/// The bytes of the synthetic site table (format `ERPSITE1`, documented in `docs/site-tables.md`). Deterministic.
pub fn synthetic_site_bytes() -> Result<Vec<u8>, PlaceError> {
    let (spec, kinds) = synth_spec();
    let tb = precompute_typed(&spec, &kinds)?;
    let (k, n) = (tb.k, tb.n);
    let mks: Vec<_> =
        spec.candidates.iter().enumerate().map(|(j, c)| upright_pose(j as i64, &[c.xy[0], c.xy[1], c.z], c.azimuth)).collect();
    let gaps: Vec<f64> = kinds.iter().map(|t| if *t == CandKind::Stand { FRONT_GAP_STAND } else { FRONT_GAP_WALL }).collect();
    // centre ray: the legacy test; samples: the 13-sample test of `precompute_typed`, restricted to the centre-visible pairs
    let vis_c = visible_batch(&tb.cells, &spec.ws.headings, &mks, spec.a, &spec.cam_model, &spec.obstacles, &spec.vp, false, Some(&gaps));
    let vis_s: Vec<bool> = tb.vis.iter().zip(&vis_c).map(|(&s, &c)| s && c).collect();

    let mut b = Vec::new();
    b.extend_from_slice(b"ERPSITE1");
    put_u32(&mut b, 1);
    let tag = b"synthetic";
    put_u32(&mut b, tag.len() as u32);
    b.extend_from_slice(tag);
    put_f64(&mut b, spec.a);
    put_u32(&mut b, spec.cam_model.w() as u32);
    put_u32(&mut b, spec.cam_model.h() as u32);
    put_f64(&mut b, spec.sigma_px);
    put_u32(&mut b, 0); // tangent noise
    put_f64(&mut b, spec.tau);
    put_f64(&mut b, spec.tau_loo_eff());
    put_u32(&mut b, spec.k_min as u32);
    put_f64(&mut b, spec.sigma_cap);
    put_f64(&mut b, spec.w_k);
    put_f64(&mut b, spec.vp.psi_max);
    put_f64(&mut b, spec.vp.px_min);
    put_f64(&mut b, LOCAL_R);
    put_f64(&mut b, spec.ws.spacing);
    put_f64(&mut b, spec.ws.z_cam[0]);
    put_u32(&mut b, k as u32);
    put_u32(&mut b, n as u32);
    for c in &tb.cells {
        for v in c {
            put_f64(&mut b, *v);
        }
    }
    for (c, kind) in spec.candidates.iter().zip(&kinds) {
        for v in [c.xy[0], c.xy[1], c.z, c.azimuth] {
            put_f64(&mut b, v);
        }
        b.push((*kind == CandKind::Stand) as u8);
    }
    b.extend_from_slice(&pack_bits(&vis_c));
    b.extend_from_slice(&pack_bits(&vis_s));
    // information for every centre-visible pair, upper triangle in `IU` order
    let pairs: Vec<usize> = (0..k * n).filter(|&at| vis_c[at]).collect();
    put_u32(&mut b, pairs.len() as u32);
    for &at in &pairs {
        put_u32(&mut b, (at / n) as u32);
        put_u32(&mut b, (at % n) as u32);
        let m = &tb.info[at * 16..at * 16 + 16];
        for &(r, c) in IU.iter() {
            put_f64(&mut b, m[r * 4 + c]);
        }
    }
    // reference block: counts and checksums, computed here from the same data
    let cells: Vec<V3> = tb.cells.clone();
    let occ_c = blocked_triples(&cells, &spec.candidates, &kinds, spec.a, &vis_c, OccMode::Centre);
    let occ_s = blocked_triples(&cells, &spec.candidates, &kinds, spec.a, &vis_s, OccMode::Samples);
    let count = |m: &[bool]| m.iter().filter(|&&v| v).count() as u64;
    let near = near_lists(&spec.candidates);
    let nonzero = pairs.iter().filter(|&&at| tb.info[at * 16..at * 16 + 16].iter().any(|&v| v != 0.0)).count() as u64;
    for v in [
        count(&vis_c),
        occ_c.n_triples as u64,
        triples_checksum(&occ_c),
        count(&vis_s),
        occ_s.n_triples as u64,
        triples_checksum(&occ_s),
        count(&vis_c) - count(&vis_s),
        nonzero,
        near_checksum(&near),
        near.iter().map(|a| a.len() as u64).sum::<u64>(),
    ] {
        put_u64(&mut b, v);
    }
    Ok(b)
}

/// [`synthetic_site_bytes`] parsed by the ordinary loader.
pub fn synthetic_site() -> Result<SiteTables, PlaceError> {
    SiteTables::from_bytes(&synthetic_site_bytes()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::site::VisMode;

    #[test]
    fn loads_and_is_deterministic() {
        let a = synthetic_site_bytes().unwrap();
        assert_eq!(a, synthetic_site_bytes().unwrap());
        let s = SiteTables::from_bytes(&a).unwrap();
        assert!(s.cells.len() > 60 && s.cands.len() > 30);
        assert!(s.kinds.contains(&CandKind::Wall) && s.kinds.contains(&CandKind::Stand));
        assert!(s.refs.n_vis[0] > s.refs.n_vis[1] || s.refs.n_centre_only == 0);
        assert!(s.refs.n_triples[1] > 0, "the stands must block something");
        assert!(a.len() < 400_000, "{}", a.len());
        // regression pin of the generator output (counts of this geometry, not measurements)
        assert_eq!((s.cells.len(), s.cands.len(), s.kinds.iter().filter(|&&t| t == CandKind::Stand).count()), (92, 54, 38));
        assert_eq!(s.refs.n_vis, [2028, 1962]);
        assert_eq!((s.refs.n_triples, s.refs.n_centre_only), ([120, 280], 66));
        for m in [VisMode::Centre, VisMode::Samples] {
            assert!(s.tables(m).heading_independent());
        }
    }
}
