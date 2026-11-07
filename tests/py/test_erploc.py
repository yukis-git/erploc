"""Behaviour tests of the Python package (numpy and pytest only)."""

import threading
import time

import numpy as np
import pytest

import erploc
import synth_scene as synth

A = 0.16
W = 4096
CAM = erploc.Camera(W)
ROOM = [[0, 0], [8, 0], [8, 6], [0, 6]]


@pytest.fixture(scope="module")
def room():
    cand, markers = synth.wall_markers(erploc, ROOM, 2.0, a=A)
    return cand, markers


def observations(markers, c, rot, rng=None, sigma=0.0, **kw):
    return [erploc.Observation(i, uv, **kw) for i, uv in synth.observe(markers, A, c, rot, W, rng, sigma)]


# ------------------------------------------------------------------------------------------------ resection
def test_resection_recovers_truth_noise_free(room):
    _, markers = room
    c, yaw = np.array([3.3, 2.1, 1.2]), 0.7
    r = erploc.resection(observations(markers, c, synth.rot_zyx(yaw)), markers, CAM, A)
    assert r.converged and r.dof == "yaw" and r.n_markers == len(markers)
    assert np.abs(r.position - c).max() < 1e-6
    assert abs(r.yaw - yaw) < 1e-8
    assert r.rms_px < 1e-6


def test_resection_noisy_error_matches_predicted_sigma(room):
    """RMS position error over many noisy frames is within 25 % of the predicted sigma_pos."""
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    rng = np.random.default_rng(7)
    errs, preds = [], []
    for _ in range(150):
        r = erploc.resection(observations(markers, c, rot, rng, 0.3), markers, CAM, A, sigma_px=0.3, noise_model="pixel")
        errs.append(np.sum((r.position - c) ** 2))
        preds.append(r.sigma_pos**2)
    ratio = np.sqrt(np.mean(errs) / np.mean(preds))
    assert 0.8 < ratio < 1.25, ratio


def test_resection_with_the_other_column_direction(room):
    """lam_sign = -1: columns grow the other way; the same scene projected with it is recovered with it (and not with +1)."""
    _, markers = room
    c, yaw = np.array([3.3, 2.1, 1.2]), 0.7
    obs = [erploc.Observation(i, uv) for i, uv in synth.observe(markers, A, c, synth.rot_zyx(yaw), W, lam_sign=-1)]
    r = erploc.resection(obs, markers, erploc.Camera(W, lam_sign=-1), A)
    assert np.abs(r.position - c).max() < 1e-6 and abs(r.yaw - yaw) < 1e-8
    try:  # the mirrored model does not explain the frame: no pose, or a poor fit
        wrong = erploc.resection(obs, markers, CAM, A)
    except erploc.NumericalError:
        return
    assert np.abs(wrong.position - c).max() > 1e-3 or wrong.rms_px > 1.0


def test_resection_covariance_is_sane(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    r = erploc.resection(observations(markers, c, rot), markers, CAM, A, sigma_px=0.3)
    assert r.cov.shape == (4, 4) and r.info.shape == (4, 4)
    assert np.allclose(r.cov, r.cov.T, rtol=1e-9, atol=1e-18)
    assert np.linalg.eigvalsh(r.cov).min() > 0
    assert np.allclose(r.cov @ r.info, np.eye(4), atol=1e-6)
    assert r.sigma_pos == pytest.approx(np.sqrt(np.trace(r.cov[:3, :3])), rel=1e-6)
    assert 1e-5 < r.sigma_pos < 0.05
    # fewer markers: larger sigma; the pixel noise scales sigma linearly
    few = [o for o in observations(markers, c, rot) if o.id < 3]
    assert erploc.resection(few, markers, CAM, A, sigma_px=0.3).sigma_pos > r.sigma_pos
    r2 = erploc.resection(observations(markers, c, rot), markers, CAM, A, sigma_px=0.6)
    assert r2.sigma_pos == pytest.approx(2 * r.sigma_pos, rel=1e-6)


def test_resection_6dof(room):
    _, markers = room
    c = np.array([3.3, 2.1, 1.2])
    yaw, roll, pitch = 0.7, 0.05, -0.04
    rot = synth.rot_zyx(yaw, roll, pitch)
    r = erploc.resection(observations(markers, c, rot), markers, CAM, A, dof="full")
    assert r.dof == "full" and r.cov.shape == (6, 6) and r.info.shape == (6, 6)
    assert np.abs(r.position - c).max() < 1e-6
    assert np.abs(r.pose.rotation - rot).max() < 1e-8
    assert (r.pose.yaw, r.pose.roll, r.pose.pitch) == pytest.approx((yaw, roll, pitch), abs=1e-8)
    # with a start pose
    r2 = erploc.resection(observations(markers, c, rot), markers, CAM, A, dof="full", x0=erploc.Pose(c + 0.2, yaw + 0.1))
    assert np.abs(r2.position - c).max() < 1e-6
    # yaw-only with known roll / pitch
    r3 = erploc.resection(observations(markers, c, rot), markers, CAM, A, roll=roll, pitch=pitch)
    assert np.abs(r3.position - c).max() < 1e-6 and abs(r3.yaw - yaw) < 1e-8


def test_resection_per_corner_mask_and_scale(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    full = observations(markers, c, rot)
    base = erploc.resection(full, markers, CAM, A, sigma_px=0.3)

    def with_mask(garbage):
        out = []
        for o in full:
            valid = np.array([False, True, True, True]) if o.id % 2 == 0 else None
            uv = o.corners.copy()
            if valid is not None:
                uv[0] = garbage
            out.append(erploc.Observation(o.id, uv, corner_valid=valid))
        return out

    nan_res = erploc.resection(with_mask(np.nan), markers, CAM, A, sigma_px=0.3)
    junk_res = erploc.resection(with_mask([123.0, 456.0]), markers, CAM, A, sigma_px=0.3)
    # the invalid corners are never read: the value there cannot matter
    assert np.array_equal(nan_res.position, junk_res.position) and nan_res.sigma_pos == junk_res.sigma_pos
    assert np.abs(nan_res.position - c).max() < 1e-6
    assert nan_res.sigma_pos > base.sigma_pos
    # a larger sigma scale on all corners of one marker lowers its weight: sigma_pos grows
    scaled = [erploc.Observation(o.id, o.corners, corner_sigma_scale=[4, 4, 4, 4] if o.id == 0 else None) for o in full]
    assert erploc.resection(scaled, markers, CAM, A, sigma_px=0.3).sigma_pos > base.sigma_pos
    # all corners of every marker invalid: nothing to estimate from
    none = [erploc.Observation(o.id, o.corners, corner_valid=[0, 0, 0, 0]) for o in full]
    with pytest.raises(erploc.NumericalError):
        erploc.resection(none, markers, CAM, A)


def test_huber_loss_resists_an_outlier(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    obs = observations(markers, c, rot)
    bad = list(obs)
    bad[0] = erploc.Observation(obs[0].id, obs[0].corners + np.array([[25.0, -18.0]] * 4))
    plain = erploc.resection(bad, markers, CAM, A)
    robust = erploc.resection(bad, markers, CAM, A, loss="huber", huber_delta=1.5)
    assert np.linalg.norm(robust.position - c) < np.linalg.norm(plain.position - c)


# ------------------------------------------------------------------------------------------------ bundle adjustment
def trajectory(n):
    t = np.linspace(0.0, 1.0, n)
    return [(np.array([2.0 + 4.0 * s, 1.5 + 3.0 * s * (1 - s) * 2 + 0.5, 1.2]), 0.3 + 2.0 * s) for s in t]


def test_bundle_adjust_with_survey_markers(room):
    _, markers = room
    poses = trajectory(8)
    frames = [observations(markers, c, synth.rot_zyx(yaw)) for c, yaw in poses]
    survey = {i: (markers[i][0], 0.005) for i in (0, 4, 8, 11)}

    init = erploc.init_bundle(frames, CAM, A, survey)
    assert isinstance(init, erploc.BundleState) and len(init.cameras) == 8 and set(init.markers) == set(markers)

    res = erploc.bundle_adjust(frames, CAM, A, survey, sigma_px=0.3)
    assert res.converged and res.rms_px < 1e-5
    for (c, yaw), cam in zip(poses, res.cameras):
        assert np.abs(cam.position - c).max() < 1e-6 and abs(cam.yaw - yaw) < 1e-7
    for i, (p, r) in markers.items():
        mp, mr = res.markers[i]
        assert np.abs(mp - p).max() < 1e-6 and np.abs(mr - r).max() < 1e-6
    # covariances: positive definite blocks, gauge fixed by the surveyed markers
    assert res.null_dim == 4  # x, y, z and yaw of the whole network: fixed only by the survey priors
    assert res.cov is not None and res.info.shape == res.cov.shape
    for i, cv in res.cov_markers.items():
        assert cv.shape == (6, 6) and np.linalg.eigvalsh(cv).min() > 0
    for cv in res.cov_cams:
        assert cv.shape == (4, 4) and np.linalg.eigvalsh(cv).min() > 0
    # the same result from an explicit start
    again = erploc.bundle_adjust(frames, CAM, A, survey, x0=res.state)
    assert again.converged and again.rms_px < 1e-5


def test_bundle_adjust_corrects_a_perturbed_map(room):
    """Noisy survey positions and detections: the unsurveyed markers are recovered to a few millimetres."""
    _, markers = room
    rng = np.random.default_rng(3)
    poses = trajectory(10)
    frames = [observations(markers, c, synth.rot_zyx(yaw), rng, 0.3) for c, yaw in poses]
    survey = {i: (markers[i][0] + rng.normal(0, 0.005, 3), 0.005) for i in (0, 4, 8, 11)}
    res = erploc.bundle_adjust(frames, CAM, A, survey, sigma_px=0.3, loss="huber")
    assert res.converged
    err = [np.linalg.norm(res.markers[i][0] - markers[i][0]) for i in markers if i not in survey]
    assert max(err) < 0.02, max(err)
    assert 0.3 < res.rms_px < 0.6  # about sqrt(2) * 0.3 px: the corner residuals are 2-D


def test_bundle_adjust_6dof(room):
    _, markers = room
    poses = trajectory(8)
    rolls = [(0.02 * np.sin(k), -0.03 * np.cos(k)) for k in range(8)]
    frames = [observations(markers, c, synth.rot_zyx(yaw, *rp)) for (c, yaw), rp in zip(poses, rolls)]
    survey = {i: (markers[i][0], 0.005) for i in (0, 4, 8, 11)}
    res = erploc.bundle_adjust(frames, CAM, A, survey, dof="full", rp=rolls)
    assert res.converged and res.rms_px < 1e-5
    for (c, yaw), rp, cam in zip(poses, rolls, res.cameras):
        assert np.abs(cam.rotation - synth.rot_zyx(yaw, *rp)).max() < 1e-7
    assert all(cv.shape == (6, 6) for cv in res.cov_cams)


# ------------------------------------------------------------------------------------------------ fisher
def test_fisher_camera_matches_resection_information(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    info = erploc.fisher_camera(erploc.Pose(c, 0.7), markers, CAM, A, sigma_px=0.3)
    r = erploc.resection(observations(markers, c, rot), markers, CAM, A, sigma_px=0.3)
    assert info.shape == (4, 4)
    assert np.allclose(info, r.info, rtol=1e-6, atol=1e-6 * np.abs(info).max())
    assert erploc.sigma_pos(info) == pytest.approx(r.sigma_pos, rel=1e-9)
    assert erploc.fisher_camera(erploc.Pose(c, 0.7), markers, CAM, A, dof="full").shape == (6, 6)
    assert erploc.sigma_pos(np.zeros((4, 4))) == float("inf")


# ------------------------------------------------------------------------------------------------ placement
def small_scene(**kw):
    cand = erploc.wall_candidates([[0, 0], [6, 0], [6, 5], [0, 5]], 1.5, z=1.5)
    return erploc.Scene(
        [[0.5, 0.5], [5.5, 0.5], [5.5, 4.5], [0.5, 4.5]],
        cand,
        [([[2.5, 2.0], [3.5, 2.0], [3.5, 3.0], [2.5, 3.0]], 1.5)],
        camera=erploc.Camera(6528, 3264),
        a=0.16,
        spacing=0.5,
        z_cam=[1.2],
        headings=[0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
        **kw,
    )


def test_place_markers_returns_feasible_layout():
    scene = small_scene()
    p = erploc.place_markers(scene, tau=0.05, k_min=2)
    assert p.method == "greedy"
    assert p.feasible_fraction == 1.0
    assert 2 <= p.count <= scene.n_candidates
    assert len(set(p.chosen.tolist())) == p.count and p.chosen.max() < scene.n_candidates
    assert p.worst_sigma_pos <= 0.05
    assert p.cells.shape == (scene.n_cells, 3) and p.sigma.shape == (scene.n_cells,)
    assert p.feasible.all() and (p.n_vis >= 2).all() and (p.margin >= 0).all()
    assert set(p.survey_ids.tolist()) <= set(p.chosen.tolist()) and len(p.survey_ids) == min(3, p.count)
    assert np.allclose(p.chosen_candidates, scene.candidates[p.chosen])


def test_place_markers_ift_not_worse_than_greedy():
    scene = small_scene()
    g = erploc.place_markers(scene, tau=0.05)
    i = erploc.place_markers(scene, tau=0.05, method="ift")
    assert i.method == "ift" and i.feasible_fraction == 1.0
    assert i.count <= g.count


def test_place_markers_reports_infeasible_when_too_few_allowed():
    scene = small_scene()
    p = erploc.place_markers(scene, tau=0.05, n_max=1)
    assert p.count == 1 and p.feasible_fraction == 0.0  # k_min = 2 cannot be met with one marker
    q = erploc.place_markers(scene, tau=1e-6)
    assert q.feasible_fraction < 1.0


def test_evaluate_agrees_with_place_markers():
    scene = small_scene()
    for kw in ({"tau": 0.05}, {"tau": 0.02, "k_min": 3}, {"tau": 0.05, "tau_loo": 0.2, "n_max": 6}):
        p = erploc.place_markers(scene, **kw)
        ev_kw = {k: v for k, v in kw.items() if k != "n_max"}
        e = erploc.evaluate(scene, p.chosen, **ev_kw)
        assert e.method == "evaluate" and len(e.survey_ids) == 0
        for name in ("sigma", "n_vis", "sigma_loo", "feasible", "margin", "cells"):
            assert np.array_equal(getattr(p, name), getattr(e, name)), name
        assert e.feasible_fraction == p.feasible_fraction and e.worst_sigma_pos == p.worst_sigma_pos
    # a worse layout is worse
    one = erploc.evaluate(scene, [0, 1], tau=0.05)
    full = erploc.evaluate(scene, np.arange(scene.n_candidates), tau=0.05)
    assert full.feasible_fraction >= one.feasible_fraction and full.worst_sigma_pos <= one.worst_sigma_pos


def test_placed_layout_localises_in_the_same_scene():
    """A layout from place_markers gives resections whose error is of the size evaluate() predicts."""
    scene = small_scene()
    p = erploc.place_markers(scene, tau=0.05)
    cand = scene.candidates[p.chosen]
    markers = {int(j): synth.upright_marker(c[:3], c[3]) for j, c in zip(p.chosen, cand)}
    rng = np.random.default_rng(5)
    errs, preds = [], []
    cams = [(x, y) for x in (1.0, 2.0, 4.5) for y in (1.0, 4.0)]
    width = 6528
    for x, y in cams:
        c, rot = np.array([x, y, 1.2]), synth.rot_zyx(0.4)
        obs = [erploc.Observation(i, uv) for i, uv in synth.observe(markers, 0.16, c, rot, width, rng, 0.3)]
        if len(obs) < 2:
            continue
        r = erploc.resection(obs, markers, erploc.Camera(width), 0.16, sigma_px=0.3)
        errs.append(np.linalg.norm(r.position - c))
        preds.append(r.sigma_pos)
    assert errs and max(errs) < 0.05  # tau


# ------------------------------------------------------------------------------------------------ eqar bridge
class FakeMarker:
    """Attribute layout of eqar.Marker (duck typing: erploc does not import eqar)."""

    def __init__(self, id, corners, valid=None, scale=None):
        self.id = id
        self.corners = np.asarray(corners, np.float32)
        if valid is not None:
            self.corner_valid = np.asarray(valid, bool)
        if scale is not None:
            self.corner_sigma_scale = np.asarray(scale, np.float32)


def test_observations_from_eqar_keeps_mask_and_scale(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    raw = synth.observe(markers, A, c, rot, W)
    objs = [FakeMarker(i, uv, [True, False, True, True], [1, 1, 2, 1]) if i % 2 == 0 else FakeMarker(i, uv) for i, uv in raw]
    obs = erploc.observations_from_eqar(objs)
    assert len(obs) == len(raw)
    assert obs[0].corner_valid == [True, False, True, True]
    assert list(obs[0].corner_sigma_scale) == [1, 1, 2, 1]
    assert obs[1].corner_valid == [True] * 4 and list(obs[1].corner_sigma_scale) == [1] * 4
    # dicts (Marker.to_dict layout) give the same observations; ok=False and id < 0 are dropped
    dicts = [{"id": o.id, "corners": o.corners.tolist(), "corner_valid": o.corner_valid, "corner_sigma_scale": o.corner_sigma_scale.tolist()} for o in obs]
    dicts += [{"id": 99, "corners": np.zeros((4, 2)), "ok": False}, {"id": -1, "corners": np.zeros((4, 2))}]
    again = erploc.observations_from_eqar(dicts)
    assert [o.id for o in again] == [o.id for o in obs]
    # the estimator accepts the eqar-like objects directly and gives the same answer
    a = erploc.resection(objs, markers, CAM, A)
    b = erploc.resection(obs, markers, CAM, A)
    assert np.array_equal(a.position, b.position)
    assert np.abs(a.position - c).max() < 1e-3  # float32 corners: about 1e-4 px


def test_predict_corners_matches_the_numpy_projection(room):
    _, markers = room
    c, yaw = np.array([3.3, 2.1, 1.2]), 0.7
    for sign in (1, -1):
        got = erploc.predict_corners(erploc.Pose(c, yaw), markers, erploc.Camera(W, lam_sign=sign), A)
        assert set(got) == set(markers)
        for i, (p, r) in markers.items():
            want = synth.project(synth.marker_corners_world(p, r, A), c, synth.rot_zyx(yaw), W, sign)
            assert np.abs(got[i] - want).max() < 1e-6
    with pytest.raises(ValueError):
        erploc.predict_corners(erploc.Pose(c, yaw), markers, CAM, 0.0)


def test_upright_marker_matches_the_placement_convention(room):
    cand, markers = room
    p, r = erploc.upright_marker(cand[0, :3], cand[0, 3])
    assert np.array_equal(p, markers[0][0]) and np.abs(r - markers[0][1]).max() < 1e-15
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) > 0
    for bad in ((np.zeros(2), 0.0), (np.array([np.nan, 0, 0]), 0.0), (np.zeros(3), float("inf"))):
        with pytest.raises(ValueError):
            erploc.upright_marker(*bad)


# ------------------------------------------------------------------------------------------------ errors
def good_args(room):
    _, markers = room
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7)
    return observations(markers, c, rot), markers


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o, m: (o, {}, CAM, A),  # no markers
        lambda o, m: (o, m, CAM, 0.0),  # a <= 0
        lambda o, m: (o, m, CAM, float("nan")),
        lambda o, m: (o + o[:1], m, CAM, A),  # duplicate observation id
        lambda o, m: ([erploc.Observation(0, [[np.inf, 0]] * 4)], m, CAM, A),
        lambda o, m: ([erploc.Observation(0, [[1e12, 0]] * 4)], m, CAM, A),
        lambda o, m: (o, {0: (np.zeros(3), np.eye(3) * 2)}, CAM, A),  # not a rotation
        lambda o, m: (o, {0: (np.array([np.nan, 0, 0]), np.eye(3))}, CAM, A),
        lambda o, m: (o, {0: (np.zeros(2), np.eye(3))}, CAM, A),
        lambda o, m: (o, {-1: (np.zeros(3), np.eye(3))}, CAM, A),
        lambda o, m: (o, m, "camera", A),
        lambda o, m: ("observations", m, CAM, A),
        lambda o, m: (5, m, CAM, A),
    ],
)
def test_resection_bad_input_raises(room, mutate):
    o, m = good_args(room)
    args = mutate(o, m)
    with pytest.raises((ValueError, TypeError, erploc.NumericalError)):
        erploc.resection(*args)


@pytest.mark.parametrize(
    "kw",
    [{"sigma_px": 0.0}, {"sigma_px": -1.0}, {"noise_model": "x"}, {"loss": "l1"}, {"dof": "6"}, {"loss": "huber", "huber_delta": 0.0}, {"roll": float("nan")}],
)
def test_resection_bad_options_raise(room, kw):
    o, m = good_args(room)
    with pytest.raises(ValueError):
        erploc.resection(o, m, CAM, A, **kw)


def test_resection_unknown_marker_and_start_on_corner(room):
    o, m = good_args(room)
    with pytest.raises(erploc.NumericalError):
        erploc.resection(o, {1000: m[0]}, CAM, A)
    corner = synth.marker_corners_world(*m[0], A)[2]
    with pytest.raises(ValueError):
        erploc.resection(o, m, CAM, A, x0=erploc.Pose(corner))


def test_camera_and_pose_validation():
    for args in ((100, 40), (0,), (4096, 2048, 0), (4096, 2048, 2)):
        with pytest.raises(ValueError):
            erploc.Camera(*args)
    with pytest.raises(OverflowError):
        erploc.Camera(-4)
    assert erploc.Camera(4096).height == 2048
    with pytest.raises(ValueError):
        erploc.Pose([0, 0, 0], rotation=np.eye(3) * 2)
    with pytest.raises(ValueError):
        erploc.Pose([0, 0])
    with pytest.raises(ValueError):
        erploc.Observation(0, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        erploc.Observation(0, np.zeros((4, 2)), corner_valid=[2, 0, 0, 0])


def test_bundle_adjust_bad_input_raises(room):
    _, markers = room
    frames = [observations(markers, np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7))]
    survey = {i: (markers[i][0], 0.005) for i in (0, 4)}
    with pytest.raises(ValueError):
        erploc.bundle_adjust([], CAM, A, survey)
    with pytest.raises(ValueError):
        erploc.bundle_adjust(frames, CAM, A, {0: (markers[0][0], -1.0), 4: (markers[4][0], 0.005)})
    with pytest.raises(ValueError):
        erploc.bundle_adjust(frames, CAM, A, {0: (markers[0][0], 0.005), 77: (markers[4][0], 0.005)})  # 77 not observed
    with pytest.raises(ValueError):
        erploc.bundle_adjust(frames, CAM, A, survey, rp=[[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        erploc.bundle_adjust(frames, CAM, A, survey, max_iter=0)
    with pytest.raises(ValueError):
        erploc.bundle_adjust(frames, CAM, A, {0: markers[0][0]})  # not a pair
    with pytest.raises(erploc.NumericalError):
        erploc.bundle_adjust(frames, CAM, A, {0: (markers[0][0], 0.005)})  # one surveyed marker
    with pytest.raises(erploc.NumericalError):
        erploc.init_bundle(frames, CAM, A, {0: (markers[0][0], 0.005)})  # no seed frame (needs two surveyed markers)


def test_placement_bad_input_raises():
    scene = small_scene()
    for kw in ({"tau": 0.0}, {"tau": -1}, {"tau": float("nan")}, {"tau": 0.05, "method": "sa"}, {"tau": 0.05, "n_max": 0},
               {"tau": 0.05, "n_max": 10**6}, {"tau": 0.05, "budget_ms": 0}, {"tau": 0.05, "sigma_ts": 0.0}, {"tau": 0.05, "sigma_cap": 0.0},
               {"tau": 0.05, "tau_loo": -1.0}):
        with pytest.raises(ValueError):
            erploc.place_markers(scene, **kw)
    for chosen in ([0, 0], [1000], [-1], [0.5], [float("nan")], [[0, 1]]):
        with pytest.raises(ValueError):
            erploc.evaluate(scene, chosen, tau=0.05)
    with pytest.raises(TypeError):
        erploc.place_markers("scene", tau=0.05)
    cand = scene.candidates
    poly = [[0.5, 0.5], [5.5, 0.5], [5.5, 4.5], [0.5, 4.5]]
    bad_scenes = [
        dict(workspace=poly[:2], candidates=cand),  # 2 vertices
        dict(workspace=poly, candidates=cand[:, :3]),
        dict(workspace=poly, candidates=cand[:0]),
        dict(workspace=poly, candidates=cand, spacing=0.0),
        dict(workspace=poly, candidates=cand, spacing=1e-9),  # too many cells
        dict(workspace=poly, candidates=cand, a=0.0),
        dict(workspace=poly, candidates=cand, noise_model="x"),
        dict(workspace=poly, candidates=cand, z_cam=[]),
        dict(workspace=[[100, 100], [100.1, 100], [100.1, 100.1]], candidates=cand),  # no cell inside
        dict(workspace=poly, candidates=np.where(np.arange(cand.size).reshape(cand.shape) == 0, np.nan, cand)),
    ]
    for d in bad_scenes:
        d.setdefault("a", 0.16)
        with pytest.raises(ValueError):
            erploc.Scene(camera=erploc.Camera(6528), **d)
    with pytest.raises(ValueError):
        erploc.wall_candidates(ROOM, 0.0)
    with pytest.raises(ValueError):
        erploc.wall_candidates(ROOM, 1e-7)
    with pytest.raises(ValueError):
        erploc.wall_candidates(ROOM[:2], 1.0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # numpy: a 1-element array passed where a float is expected
def test_random_garbage_never_panics(room):
    """Whatever is passed, the outcome is a normal exception (never pyo3's PanicException, a BaseException)."""
    _, markers = room
    rng = np.random.default_rng(11)
    ok = (ValueError, TypeError, OverflowError, erploc.NumericalError)

    def junk():
        kind = rng.integers(0, 8)
        if kind == 0:
            return None
        if kind == 1:
            return float(rng.choice([np.nan, np.inf, -np.inf, 0.0, -1.0, 1e300, 1e-300]))
        if kind == 2:
            return int(rng.integers(-5, 10**12))
        if kind == 3:
            return rng.normal(size=tuple(rng.integers(0, 5, size=rng.integers(0, 4)))) * 10 ** rng.integers(-5, 12)
        if kind == 4:
            return [junk() for _ in range(rng.integers(0, 4))]
        if kind == 5:
            return {int(rng.integers(-2, 4)): (junk(), junk())}
        if kind == 6:
            return "x"
        return object()

    good_obs, _ = good_args(room)
    calls = [
        lambda: erploc.resection([junk() for _ in range(2)], junk(), junk(), junk()),
        lambda: erploc.resection(good_obs, markers, CAM, A, sigma_px=junk(), roll=junk(), pitch=junk(), x0=junk()),
        lambda: erploc.resection([erploc.Observation(5, junk(), junk(), junk())], markers, CAM, A),
        lambda: erploc.bundle_adjust(junk(), junk(), junk(), junk(), rp=junk(), x0=junk()),
        lambda: erploc.bundle_adjust([good_obs], CAM, A, {0: (junk(), junk()), 4: (junk(), junk())}, rp=junk()),
        lambda: erploc.init_bundle([good_obs], CAM, A, {0: (markers[0][0], junk()), 4: (markers[4][0], 0.01)}, refine_rounds=junk()),
        lambda: erploc.fisher_camera(junk(), junk(), junk(), junk()),
        lambda: erploc.sigma_pos(junk()),
        lambda: erploc.Scene(junk(), junk(), junk(), camera=junk(), a=junk()),
        lambda: erploc.wall_candidates(junk(), junk(), z=junk(), offset=junk()),
        lambda: erploc.evaluate(small_scene(), junk(), junk()),
        lambda: erploc.observations_from_eqar(junk()),
        lambda: erploc.Pose(junk(), junk(), junk(), junk(), junk()),
    ]
    scene = small_scene()
    calls.append(lambda: erploc.place_markers(scene, tau=junk(), k_min=int(rng.integers(0, 4)), n_max=junk(), sigma_cap=junk(), w_k=junk()))
    for _ in range(400):
        call = calls[rng.integers(0, len(calls))]
        try:
            call()
        except ok:
            pass


def test_random_valid_looking_input_never_panics(room):
    """Well-formed but arbitrary detections (random pixels, masks, scales, starts): a result or NumericalError, nothing else."""
    _, markers = room
    rng = np.random.default_rng(12)
    survey = {i: (markers[i][0], 0.01) for i in (0, 4, 8)}
    for _ in range(150):
        frame = [
            erploc.Observation(
                int(i),
                rng.uniform([-W, -W / 2], [2 * W, W], (4, 2)) if rng.random() < 0.5 else rng.uniform([0, 0], [W, W / 2], (4, 2)),
                corner_valid=rng.random(4) < 0.7,
                corner_sigma_scale=10 ** rng.uniform(-2, 2, 4),
            )
            for i in rng.choice(len(markers), int(rng.integers(1, 6)), replace=False)
        ]
        dof = "full" if rng.random() < 0.5 else "yaw"
        try:
            erploc.resection(frame, markers, CAM, A, dof=dof, loss="huber" if rng.random() < 0.5 else "linear")
        except erploc.NumericalError:
            pass
        try:
            erploc.bundle_adjust([frame, frame], CAM, A, survey, dof=dof, max_iter=5)
        except (erploc.NumericalError, ValueError):
            pass


# ------------------------------------------------------------------------------------------------ GIL
def test_computation_releases_the_gil():
    """While one thread computes, a pure-Python thread keeps running (it would stall for the whole call if the GIL were held)."""
    cand = erploc.wall_candidates([[0, 0], [8, 0], [8, 6], [0, 6]], 0.75, z=1.5)
    scene = erploc.Scene([[0.5, 0.5], [7.5, 0.5], [7.5, 5.5], [0.5, 5.5]], cand, camera=erploc.Camera(6528), a=0.16, spacing=0.4)
    done = threading.Event()
    elapsed = []

    def work():
        t0 = time.perf_counter()
        erploc.place_markers(scene, tau=0.05, n_max=2)
        elapsed.append(time.perf_counter() - t0)
        done.set()

    th = threading.Thread(target=work)
    th.start()
    ticks, last, gap = 0, time.perf_counter(), 0.0
    while not done.wait(0.002):
        now = time.perf_counter()
        ticks += 1
        gap = max(gap, now - last)
        last = now
    th.join()
    gap = max(gap, time.perf_counter() - last)
    # independent of the timer granularity (15 ms on some systems): with the GIL held the loop would not run at all until the call
    # returned (no ticks, one gap as long as the call); with it released there are ticks and no gap near the call's length
    assert elapsed[0] > 0.2, "the workload is too small to show the effect"
    assert ticks >= 5, (ticks, elapsed[0])
    assert gap < 0.5 * elapsed[0], (gap, elapsed[0])


# ------------------------------------------------------------------------------------------------ integer and size limits
def test_negative_or_oversized_integers_raise_overflow_error():
    for bad in (lambda: erploc.Camera(-1), lambda: erploc.Camera(2**70), lambda: erploc.Camera(4096, -2048)):
        with pytest.raises(OverflowError):
            bad()


def test_bundle_adjust_too_large_for_the_dense_hessian_is_a_value_error():
    corners = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    frame = [erploc.Observation(i, corners) for i in range(4)]
    # 1100 yaw frames and 4 markers are 4424 unknowns: a dense Hessian of 19.6 million entries, over the limit of 16 million
    with pytest.raises(ValueError, match="dense Hessian"):
        erploc.bundle_adjust([frame] * 1100, erploc.Camera(4096), 0.2, {0: ([3.0, 0.0, 1.0], 0.005)})
