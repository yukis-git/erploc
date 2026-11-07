//! Fisher information, CRLB, Schur marginalisation and gauge nullspace. Port of `erploc/fisher.py`.

use crate::corners::{row_select_ms, RowSel};
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::{jac_marker, predict_marker, Axis, CamPose};
use erploc_core::markers::MarkerPose;
use erploc_linalg::eig::{eigh, eigvalsh};
use erploc_linalg::lu;
use erploc_linalg::svd::{pinv, PINV_RCOND};
use erploc_linalg::{DMat, LinalgError};

pub const SIGMA_RTOL: f64 = 1e-12;
pub const NULL_RTOL: f64 = 1e-9;

/// Residual-scale diagonal for the 8 residuals (4 corners x (lam, phi)).
pub fn dvec(phi: &[f64; 4], cam_model: &ErpCamera, nm: NoiseModel) -> [f64; 8] {
    let s = cam_model.s();
    let mut dv = [0.0; 8];
    for k in 0..4 {
        dv[2 * k] = match nm {
            NoiseModel::Tangent => phi[k].cos() / s,
            NoiseModel::Pixel => 1.0 / s,
        };
        dv[2 * k + 1] = 1.0 / s;
    }
    dv
}

#[derive(Clone, Debug)]
pub struct MarkerInfo {
    pub icc: DMat,
    pub icm: DMat,
    pub imm: DMat,
}

/// Rows `sel.idx[..n]` of `j`, each scaled by `dv[idx] * w` (Python `dvw[:, None] * J[idx]`).
fn scale_rows(dv: &[f64; 8], sel: &RowSel, j: &DMat) -> DMat {
    DMat::from_fn(sel.n, j.cols(), |r, c| (dv[sel.idx[r]] * sel.w[r]) * j.as_slice()[sel.idx[r] * j.cols() + c])
}

#[allow(clippy::too_many_arguments)]
pub fn info_marker(
    cam: &CamPose,
    mk: &MarkerPose,
    a: f64,
    cam_model: &ErpCamera,
    sigma_px: f64,
    nm: NoiseModel,
    phi_obs: Option<&[f64; 4]>,
    dof: &[Axis],
) -> MarkerInfo {
    info_marker_masked(cam, mk, a, cam_model, sigma_px, nm, phi_obs, dof, &[true; 4], &[1.0; 4])
}

/// [`info_marker`] with the per-corner mask and sigma scale (Python `corner_mask` / `corner_sigma_scale`): the 2 rows
/// of every invalid corner are dropped and the remaining rows are whitened by `1 / scale`, the same row selection and
/// weights as `resect` / `bundle_adjust`. All-valid and scale 1 is bit-identical to the unmasked information.
/// Panics unless the scales are finite and > 0.
#[allow(clippy::too_many_arguments)]
pub fn info_marker_masked(
    cam: &CamPose,
    mk: &MarkerPose,
    a: f64,
    cam_model: &ErpCamera,
    sigma_px: f64,
    nm: NoiseModel,
    phi_obs: Option<&[f64; 4]>,
    dof: &[Axis],
    corner_mask: &[bool; 4],
    corner_sigma_scale: &[f64; 4],
) -> MarkerInfo {
    let (j_cam, j_mk) = jac_marker(cam, mk, a, dof);
    let phi: [f64; 4] = match phi_obs {
        Some(p) => *p,
        None => {
            let lp = predict_marker(cam, mk, a);
            [lp[0][1], lp[1][1], lp[2][1], lp[3][1]]
        }
    };
    let dv = dvec(&phi, cam_model, nm);
    let sel = row_select_ms(corner_mask, corner_sigma_scale);
    let ac = scale_rows(&dv, &sel, &j_cam);
    let am = scale_rows(&dv, &sel, &j_mk);
    let w = 1.0 / (sigma_px * sigma_px);
    let act = ac.transpose();
    MarkerInfo { icc: act.matmul(&ac).scale(w), icm: act.matmul(&am).scale(w), imm: am.transpose().matmul(&am).scale(w) }
}

pub fn info_camera(
    cam: &CamPose,
    markers: &[MarkerPose],
    a: f64,
    cam_model: &ErpCamera,
    sigma_px: f64,
    nm: NoiseModel,
    dof: &[Axis],
) -> DMat {
    let n = 3 + dof.len();
    let mut i = DMat::zeros(n, n);
    for mk in markers {
        i = i.add(&info_marker(cam, mk, a, cam_model, sigma_px, nm, None, dof).icc);
    }
    i
}

/// `sqrt(tr(I^-1[0:3,0:3]))`; `inf` if singular (or NaN input).
///
/// # Errors
/// `LinalgError::Shape` unless `i` is square and at least 3 x 3.
pub fn sigma_pos(i: &DMat, rtol: f64) -> Result<f64, LinalgError> {
    if i.rows() != i.cols() || i.rows() < 3 {
        return Err(LinalgError::Shape);
    }
    let w = eigvalsh(i)?;
    let n = w.len();
    // np.maximum would propagate NaN, f64::max does not; equivalent only because eigvalsh returns all-NaN for non-finite
    // input, so `w[0] > ..` is false either way and the result is inf.
    // Deliberate difference from Python: an all-NaN matrix makes
    // `np.linalg.eigvalsh` raise `LinAlgError` ("Eigenvalues did not converge") there, while here it yields `inf`
    // (the callers then cap it). A single NaN entry gives the cap in both. The case needs non-finite tables and cannot occur
    // with finite inputs.
    let ok = w[0] > rtol * w[n - 1].max(0.0);
    if !ok {
        return Ok(f64::INFINITY);
    }
    let cov = lu::inv(i)?;
    let s = cov.as_slice();
    let nc = cov.cols();
    let tr = s[0] + s[nc + 1] + s[2 * nc + 2];
    // np.maximum(tr, 0) propagates NaN; f64::max would turn it into 0
    Ok(if tr < 0.0 { 0.0 } else { tr }.sqrt())
}

pub fn sigma_pos_batch(is: &[DMat], rtol: f64) -> Result<Vec<f64>, LinalgError> {
    is.iter().map(|i| sigma_pos(i, rtol)).collect()
}

pub fn crlb(i: &DMat) -> Result<DMat, LinalgError> {
    lu::inv(i)
}

/// `S = Imm - sum_f Icm_f^T Icc_f^-1 Icm_f`, with a pinv fallback for singular frames.
pub fn schur_markers(icc: &[DMat], icm: &[DMat], imm: &DMat) -> Result<DMat, LinalgError> {
    assert_eq!(icc.len(), icm.len(), "schur_markers: icc/icm frame count mismatch");
    let mut s = imm.clone();
    for (c, m) in icc.iter().zip(icm) {
        let x = match lu::solve_mat(c, m) {
            Ok(x) => x,
            Err(LinalgError::Singular) => pinv(c, PINV_RCOND)?.matmul(m),
            Err(e) => return Err(e),
        };
        s = s.sub(&m.transpose().matmul(&x));
    }
    Ok(s)
}

/// `(dim, basis (n x dim))` of eigenvalues with `|w| <= rtol * max|w|`, ascending.
pub fn nullspace(i: &DMat, rtol: f64) -> Result<(usize, DMat), LinalgError> {
    let n = i.rows();
    let sym = i.add(&i.transpose()).scale(0.5);
    let e = eigh(&sym)?;
    let mx = e.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let thr = rtol * mx.max(0.0);
    let sel: Vec<usize> = (0..n).filter(|&k| e.values[k].abs() <= thr).collect();
    let mut basis = DMat::zeros(n, sel.len());
    for (c, &k) in sel.iter().enumerate() {
        basis.set_column(c, &e.vectors.column(k));
    }
    Ok((sel.len(), basis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use erploc_core::markers::facing_pose;
    use erploc_linalg::rng::Pcg64;

    const A: f64 = 0.16;
    const SIG: f64 = 0.3;
    const Z: [Axis; 1] = [Axis::Z];
    const XYZ: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn cam_model() -> ErpCamera {
        ErpCamera::new(6528, 3264, 1)
    }
    fn scene() -> (CamPose, MarkerPose) {
        let cam = CamPose::euler([0.2, -0.4, 1.2], 0.7, 0.0, 0.0);
        let mk = facing_pose(0, &[2.6, 1.1, 1.5], &cam.c, 22f64.to_radians(), 0.8, -0.3);
        (cam, mk)
    }
    fn im(cam: &CamPose, mk: &MarkerPose, dof: &[Axis]) -> MarkerInfo {
        info_marker(cam, mk, A, &cam_model(), SIG, NoiseModel::Tangent, None, dof)
    }
    fn block(m: &MarkerInfo) -> DMat {
        let n = m.icc.rows();
        let k = m.imm.rows();
        DMat::from_fn(n + k, n + k, |r, c| match (r < n, c < n) {
            (true, true) => m.icc.as_slice()[r * n + c],
            (true, false) => m.icm.as_slice()[r * k + c - n],
            (false, true) => m.icm.as_slice()[c * k + r - n],
            (false, false) => m.imm.as_slice()[(r - n) * k + c - n],
        })
    }
    fn asym(m: &DMat) -> f64 {
        m.sub(&m.transpose()).max_abs()
    }
    fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }

    #[test]
    fn info_symmetric_psd() {
        let (cam, mk) = scene();
        let m = block(&im(&cam, &mk, &Z));
        let e = asym(&m);
        let w = eigvalsh(&m).unwrap();
        println!("asym {e:.3e} lmin/lmax {:.3e}", w[0] / w[9]);
        assert!(e < 1e-9);
        assert!(w[0] > -1e-12 * w[9]);
    }

    #[test]
    fn gauge_nullspace_dim4() {
        let (cam, mk) = scene();
        let m = block(&im(&cam, &mk, &Z));
        let (dim, _) = nullspace(&m, 1e-9).unwrap();
        let w = eigvalsh(&m).unwrap();
        let ez = [0.0, 0.0, 1.0];
        let mut gs: Vec<[f64; 10]> = vec![];
        for t in 0..3 {
            let mut g = [0.0; 10];
            g[t] = 1.0;
            g[4 + t] = 1.0;
            gs.push(g);
        }
        let mut g = [0.0; 10];
        let cc = cross(&ez, &cam.c);
        let cp = cross(&ez, &mk.p);
        g[0..3].copy_from_slice(&cc);
        g[3] = 1.0;
        g[4..7].copy_from_slice(&cp);
        g[7..10].copy_from_slice(&mk.r.0[2]);
        gs.push(g);
        let mut q = 0.0f64;
        for g in &gs {
            let mg = m.matvec(g);
            let num: f64 = g.iter().zip(&mg).map(|(a, b)| a * b).sum();
            let den: f64 = g.iter().map(|a| a * a).sum();
            q = q.max(num / (w[9] * den));
        }
        println!("dim {dim} rayleigh {q:.3e}");
        assert_eq!(dim, 4);
        assert!(q < 1e-12);
    }

    #[test]
    fn sigma_pos_definition() {
        let (cam, mk) = scene();
        let icc = im(&cam, &mk, &Z).icc;
        let c = crlb(&icc).unwrap();
        let cs = c.as_slice();
        let want = (cs[0] + cs[5] + cs[10]).sqrt();
        let e = (sigma_pos(&icc, SIGMA_RTOL).unwrap() - want).abs();
        println!("sigma_pos err {e:.3e}");
        assert!(e < 1e-12);
        let b = sigma_pos_batch(&[icc, DMat::zeros(4, 4)], SIGMA_RTOL).unwrap();
        assert!(b[0].is_finite() && b[1].is_infinite() && b.len() == 2);
    }

    #[test]
    fn schur_is_marginalisation() {
        let (cam, mk) = scene();
        let cam2 = CamPose::euler([-1.0, 0.8, 1.3], -1.1, 0.0, 0.0);
        let i1 = im(&cam, &mk, &Z);
        let i2 = im(&cam2, &mk, &Z);
        let imm = i1.imm.add(&i2.imm).add(&DMat::identity(6).scale(1e4));
        let s = schur_markers(&[i1.icc.clone(), i2.icc.clone()], &[i1.icm.clone(), i2.icm.clone()], &imm).unwrap();
        let mut f = DMat::zeros(14, 14);
        let mut put = |r0: usize, c0: usize, m: &DMat| {
            for r in 0..m.rows() {
                for c in 0..m.cols() {
                    f[(r0 + r, c0 + c)] = m.as_slice()[r * m.cols() + c];
                }
            }
        };
        put(0, 0, &i1.icc);
        put(4, 4, &i2.icc);
        put(8, 8, &imm);
        put(0, 8, &i1.icm);
        put(8, 0, &i1.icm.transpose());
        put(4, 8, &i2.icm);
        put(8, 4, &i2.icm.transpose());
        let fi = lu::inv(&f).unwrap();
        let sub = DMat::from_fn(6, 6, |r, c| fi.as_slice()[(8 + r) * 14 + 8 + c]);
        let s2 = lu::inv(&sub).unwrap();
        assert!(finite(&s) && finite(&s2), "schur result not finite");
        let e = s.sub(&s2).max_abs() / s.max_abs();
        println!("schur rel {e:.3e}");
        assert!(e < 1e-8);
    }

    #[test]
    fn schur_pinv_fallback() {
        let mut icc = DMat::zeros(4, 4);
        icc[(0, 0)] = 1.0;
        let icm = DMat::from_fn(4, 6, |_, _| 1.0);
        let s = schur_markers(&[icc], &[icm], &DMat::identity(6).scale(2.0)).unwrap();
        assert!(s.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn info_6dof_shape() {
        let (cam, mk) = scene();
        let m = im(&cam, &mk, &XYZ);
        assert_eq!((m.icc.rows(), m.icc.cols()), (6, 6));
        assert_eq!((m.icm.rows(), m.icm.cols()), (6, 6));
        assert_eq!((m.imm.rows(), m.imm.cols()), (6, 6));
        let e6 = asym(&m.icc);
        let w = eigvalsh(&m.icc).unwrap();
        assert!(e6 < 1e-6 && w[0] > -1e-12 * w[5]);
        let ic = info_camera(&cam, &[mk], A, &cam_model(), SIG, NoiseModel::Tangent, &XYZ);
        assert!(finite(&ic) && finite(&m.icc));
        let e_ic = ic.sub(&m.icc).max_abs();
        let c = crlb(&m.icc).unwrap();
        let cs = c.as_slice();
        let e_sp = (sigma_pos(&m.icc, SIGMA_RTOL).unwrap() - (cs[0] + cs[7] + cs[14]).sqrt()).abs();
        println!("6dof asym {e6:.3e} lmin/lmax {:.3e} info_camera {e_ic:.3e} sigma {e_sp:.3e}", w[0] / w[5]);
        assert!(e_ic < 1e-9 && e_sp < 1e-9);
        let b = sigma_pos_batch(&[m.icc.clone(), DMat::zeros(6, 6)], SIGMA_RTOL).unwrap();
        assert!(b[0].is_finite() && b[1].is_infinite());
    }

    /// `DMat::max_abs` ignores NaN, so every compared matrix is checked for finiteness first.
    fn finite(m: &DMat) -> bool {
        m.as_slice().iter().all(|v| v.is_finite())
    }

    #[test]
    fn orientation_invariance() {
        let mut g = Pcg64::new(20260912);
        let c = [0.3, -0.5, 1.4];
        let mk = facing_pose(0, &[2.8, 1.3, 1.6], &c, 28f64.to_radians(), 0.6, -0.2);
        let cm = cam_model();
        let i0 = info_camera(&CamPose::euler(c, 0.0, 0.0, 0.0), &[mk], A, &cm, SIG, NoiseModel::Tangent, &XYZ);
        let mut worst = 0.0f64;
        for _ in 0..20 {
            let yaw = g.uniform(-std::f64::consts::PI, std::f64::consts::PI);
            let roll = g.uniform(-std::f64::consts::PI, std::f64::consts::PI);
            let pitch = g.uniform(-1.4, 1.4);
            let i = info_camera(&CamPose::euler(c, yaw, roll, pitch), &[mk], A, &cm, SIG, NoiseModel::Tangent, &XYZ);
            assert!(finite(&i) && finite(&i0), "info not finite");
            worst = worst.max(i.sub(&i0).max_abs() / i0.max_abs());
        }
        println!("orientation worst {worst:.3e}");
        assert!(worst < 1e-9);
    }

    /// All-valid / scale 1 is bit-identical, corner infos add up, a single corner's scale s divides its info by s^2.
    #[test]
    fn masked_info_properties() {
        let (cam, mk) = scene();
        for dof in [&Z[..], &XYZ[..]] {
            let full = im(&cam, &mk, dof);
            let same = info_marker_masked(&cam, &mk, A, &cam_model(), SIG, NoiseModel::Tangent, None, dof, &[true; 4], &[1.0; 4]);
            assert_eq!(full.icc.as_slice(), same.icc.as_slice());
            assert_eq!(full.icm.as_slice(), same.icm.as_slice());
            assert_eq!(full.imm.as_slice(), same.imm.as_slice());
            let one = |k: usize, sc: f64| {
                let mut m = [false; 4];
                m[k] = true;
                let mut s = [1.0; 4];
                s[k] = sc;
                info_marker_masked(&cam, &mk, A, &cam_model(), SIG, NoiseModel::Tangent, None, dof, &m, &s)
            };
            let mut sum = DMat::zeros(full.icc.rows(), full.icc.cols());
            for k in 0..4 {
                sum = sum.add(&one(k, 1.0).icc);
            }
            assert!(sum.sub(&full.icc).max_abs() / full.icc.max_abs() < 1e-12);
            let (a1, a2) = (one(2, 1.0).imm, one(2, 2.0).imm);
            assert!(a1.scale(0.25).sub(&a2).max_abs() / a1.max_abs() < 1e-14);
            let none = info_marker_masked(&cam, &mk, A, &cam_model(), SIG, NoiseModel::Tangent, None, dof, &[false; 4], &[1.0; 4]);
            assert!(none.icc.as_slice().iter().chain(none.icm.as_slice()).chain(none.imm.as_slice()).all(|&v| v == 0.0));
        }
    }
}
