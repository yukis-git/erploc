//! `resect` gives an `Err` for a `corner_sigma_scale` that is zero, negative or not finite; it does not panic.
use erploc_core::erp::{ErpCamera, NoiseModel};
use erploc_core::jacobian::CamPose;
use erploc_core::markers::upright_pose;
use erploc_est::lm::Loss;
use erploc_est::resection::{resect, CamDof, ResectError, ResectOptions};
use erploc_est::synth::{render, Observation, SynthConfig};
use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};

#[test]
fn resect_rejects_bad_corner_sigma_scale() {
    let cam = ErpCamera::new(4096, 2048, 1);
    let mut markers = BTreeMap::new();
    for (i, (x, y, az)) in
        [(3.0, 0.0, PI), (0.0, 3.0, -FRAC_PI_2), (-3.0, 0.0, 0.0), (0.0, -3.0, FRAC_PI_2), (2.0, 2.5, -2.3)].into_iter().enumerate()
    {
        markers.insert(i as i64, upright_pose(i as i64, &[x, y, 1.0], az));
    }
    let truth = CamPose::euler([0.1, 0.2, 1.3], 0.2, 0.0, 0.0);
    let cfg = SynthConfig { cam, a: 0.2, sigma_px: 0.3, noise_model: NoiseModel::Tangent, seed: 0 };
    let obs: Vec<Observation> = render(&[truth], &markers.values().copied().collect::<Vec<_>>(), &cfg, None).remove(0);
    assert!(!obs.is_empty());
    let opts = ResectOptions { sigma_px: 0.3, noise_model: NoiseModel::Tangent, loss: Loss::Linear, dof: CamDof::Yaw, x0: None };
    assert!(resect(&obs, &markers, 0.2, &cam, 0.0, 0.0, &opts).is_ok());
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut o = obs.clone();
        o[0].corner_sigma_scale = [1.0, bad, 1.0, 1.0];
        match resect(&o, &markers, 0.2, &cam, 0.0, 0.0, &opts) {
            Err(ResectError::InvalidInput(m)) => assert!(m.contains("corner_sigma_scale"), "{m}"),
            other => panic!("scale {bad}: expected InvalidInput, got {other:?}"),
        }
        // a masked corner counts too
        let mut o = obs.clone();
        o[0].corner_mask[1] = false;
        o[0].corner_sigma_scale = [1.0, bad, 1.0, 1.0];
        assert!(matches!(resect(&o, &markers, 0.2, &cam, 0.0, 0.0, &opts), Err(ResectError::InvalidInput(_))), "masked, scale {bad}");
    }
}
