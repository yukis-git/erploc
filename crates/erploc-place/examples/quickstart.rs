//! Place markers in a small room, render the corners a camera would see, and recover the camera pose.
//!
//! `cargo run --release -p erploc-place --example quickstart`
use erploc_core::jacobian::CamPose;
use erploc_est::lm::Loss;
use erploc_est::resection::{resect, CamDof, ResectOptions};
use erploc_est::synth::{add_noise, render, SynthConfig};
use erploc_linalg::rng::Pcg64;
use erploc_place::placement::{demo_spec, markers_from_candidates, optimize};
use std::collections::BTreeMap;

fn main() {
    // A made-up 8 m x 6 m room with a box in the middle (`demo_spec`): 16 wall candidates, at most 10 markers.
    let spec = demo_spec(10);
    let plan = optimize(&spec, 0.005).expect("placement");
    let infeasible = plan.stats.feasible.iter().filter(|f| !**f).count();
    println!(
        "chosen candidates {:?}, survey {:?}, infeasible cells {infeasible} of {}",
        plan.sel,
        plan.survey_ids,
        plan.stats.feasible.len()
    );

    // The chosen markers as a map id -> pose, and one camera frame rendered from them (corner pixels, 0.3 px noise).
    let all = markers_from_candidates(&spec.candidates);
    let chosen: Vec<_> = plan.sel.iter().map(|&j| all[j]).collect();
    let map: BTreeMap<_, _> = chosen.iter().map(|m| (m.id, *m)).collect();
    let truth = CamPose::euler([2.0, 1.5, 1.2], 0.8, 0.0, 0.0);
    let cfg = SynthConfig { cam: spec.cam_model, a: spec.a, sigma_px: spec.sigma_px, noise_model: spec.noise_model, seed: 1 };
    let clean = render(&[truth], &chosen, &cfg, None);
    let noisy = add_noise(&clean, &cfg, &mut Pcg64::new(1));

    // Single-frame resection (position and yaw; roll and pitch known to be zero).
    let opts = ResectOptions { x0: None, sigma_px: spec.sigma_px, noise_model: spec.noise_model, loss: Loss::Linear, dof: CamDof::Yaw };
    let fit = resect(&noisy[0], &map, spec.a, &spec.cam_model, 0.0, 0.0, &opts).expect("resection");
    let err = (0..3).map(|i| (fit.cam.c[i] - truth.c[i]).powi(2)).sum::<f64>().sqrt();
    println!(
        "{} markers seen, position error {:.2} mm, yaw error {:.4} deg",
        fit.n_markers,
        err * 1e3,
        (fit.cam.yaw - truth.yaw).to_degrees()
    );
}
