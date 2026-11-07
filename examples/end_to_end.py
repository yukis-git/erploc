"""End to end on a synthetic room: place markers, render 360 degree frames, detect with eqar, localise with erploc.

    pip install numpy
    pip install eqar --no-index --find-links https://github.com/yukis-git/eqar/releases/expanded_assets/v0.1.0
    pip install erploc --no-index --find-links <where the erploc wheel is>
    python examples/end_to_end.py

Everything is synthetic: the room is a made-up 8 m x 6 m rectangle, the frames are rendered here with numpy (flat tags pasted
by inverse mapping, then blur and noise), nothing is a measurement. The true poses are known, so the errors are real errors of
this synthetic setup, not of a camera in a room.
"""

import time

import numpy as np

import eqar
import erploc
from synth_scene import observe, render_erp, rot_zyx, upright_marker

A = 0.20  # printed tag side [m]
W = 6528  # frame width [px]
DICT = "4x4_50"
SIGMA_PX = 0.1  # corner noise assumed by the estimators [px]; eqar's own synthetic accuracy is about 0.02 px at this scale
ROOM = [[0, 0], [8, 0], [8, 6], [0, 6]]
WORKSPACE = [[0.8, 0.8], [7.2, 0.8], [7.2, 5.2], [0.8, 5.2]]
CAMERA = erploc.Camera(W, lam_sign=-1)  # -1: columns grow to the right for a body frame with x forward, y left, z up


def main():
    rng = np.random.default_rng(2026)
    print(f"erploc {erploc.__version__}, eqar {eqar.__version__}")

    # 1. choose marker positions
    cand = erploc.wall_candidates(ROOM, 1.5, z=1.5)
    scene = erploc.Scene(WORKSPACE, cand, camera=CAMERA, a=A, sigma_px=SIGMA_PX, px_min=24.0, spacing=0.5, z_cam=[1.2])
    t0 = time.perf_counter()
    plan = erploc.place_markers(scene, tau=0.02, k_min=3)
    print(f"placement: {plan.count} of {scene.n_candidates} candidates, feasible cells {plan.feasible_fraction:.0%} "
          f"of {scene.n_cells}, worst sigma_pos {plan.worst_sigma_pos * 1000:.1f} mm, survey ids {plan.survey_ids.tolist()} "
          f"({time.perf_counter() - t0:.1f} s)")
    markers = {int(j): upright_marker(c[:3], c[3]) for j, c in zip(plan.chosen, plan.chosen_candidates)}

    # 2. camera poses (yaw only: roll and pitch are known to be zero), frames, detections
    poses = [(np.array([x, y, 1.2]), yaw) for (x, y, yaw) in
             [(1.6, 1.4, 0.4), (3.0, 1.8, 1.0), (4.4, 2.6, 1.9), (5.8, 3.2, 2.7), (4.6, 4.4, -2.4), (2.8, 3.6, -1.2), (1.8, 4.2, -0.3)]]
    frames = []
    t_render = t_detect = 0.0
    for c, yaw in poses:
        t0 = time.perf_counter()
        img = render_erp(W, markers, A, c, rot_zyx(yaw), eqar.generate_marker, DICT, lam_sign=-1, blur_sd=0.8, noise_sd=3.0,
                         seed=int(rng.integers(1 << 30)))
        t_render += time.perf_counter() - t0
        t0 = time.perf_counter()
        det = eqar.detect(img, DICT, format="gray")
        t_detect += time.perf_counter() - t0
        frames.append(erploc.observations_from_eqar(det))
    seen = [len(f) for f in frames]
    n_expected = [len(observe(markers, A, c, rot_zyx(yaw), W, lam_sign=-1)) for c, yaw in poses]
    print(f"{len(poses)} frames of {W}x{W // 2}: markers detected per frame {seen} (facing the camera: {n_expected}); "
          f"render {t_render / len(poses):.2f} s, eqar.detect {t_detect / len(poses) * 1000:.0f} ms per frame")

    # 3. resection per frame with the true marker poses
    print("\nresection (marker map known):")
    print("  frame  markers  position error [mm]  predicted sigma_pos [mm]  yaw error [deg]  rms residual [px]")
    pos_err = []
    for k, ((c, yaw), obs) in enumerate(zip(poses, frames)):
        r = erploc.resection(obs, markers, CAMERA, A, sigma_px=SIGMA_PX)
        e = np.linalg.norm(r.position - c)
        pos_err.append(e)
        print(f"  {k:5d}  {r.n_markers:7d}  {e * 1000:19.2f}  {r.sigma_pos * 1000:24.2f}  "
              f"{np.degrees(abs(r.yaw - yaw)):15.4f}  {r.rms_px:17.3f}")
    print(f"  median position error {np.median(pos_err) * 1000:.2f} mm, max {np.max(pos_err) * 1000:.2f} mm")

    # 4. bundle adjustment: only the survey markers have known positions (5 mm survey noise)
    survey = {int(i): (markers[int(i)][0] + rng.normal(0.0, 0.005, 3), 0.005) for i in plan.survey_ids}
    init = erploc.init_bundle(frames, CAMERA, A, survey, sigma_px=SIGMA_PX)
    res = erploc.bundle_adjust(frames, CAMERA, A, survey, sigma_px=SIGMA_PX, loss="huber", x0=init)
    cam_err = [np.linalg.norm(cam.position - c) for cam, (c, _) in zip(res.cameras, poses)]
    mk_err = [np.linalg.norm(res.markers[i][0] - markers[i][0]) for i in markers if i not in survey]
    mk_pred = [np.sqrt(np.trace(res.cov_markers[i][:3, :3])) for i in markers if i not in survey]
    survey_err = [np.linalg.norm(survey[i][0] - markers[i][0]) for i in survey]
    print(f"\nbundle adjustment: {len(frames)} frames, {len(markers)} markers, {len(survey)} surveyed "
          f"(survey error {np.mean(survey_err) * 1000:.1f} mm mean), converged={res.converged}, rms {res.rms_px:.3f} px")
    print(f"  camera position error: median {np.median(cam_err) * 1000:.2f} mm, max {np.max(cam_err) * 1000:.2f} mm")
    print(f"  unsurveyed marker position error: median {np.median(mk_err) * 1000:.2f} mm, max {np.max(mk_err) * 1000:.2f} mm "
          f"(predicted sigma: median {np.median(mk_pred) * 1000:.2f} mm)")


if __name__ == "__main__":
    main()
