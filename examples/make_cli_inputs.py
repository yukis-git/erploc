"""Writes the synthetic input files in examples/cli/ (numpy only):  python examples/make_cli_inputs.py

Everything is generated from a made-up 8 m x 6 m room; nothing comes from a real site. Conventions of the file format:
angles in radians, rotations row-major, pixel coordinates as in `erploc.Camera` (pixel centres at integers).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth_scene as synth  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "cli")
A, W = 0.16, 4096
ROOM = [[0, 0], [8, 0], [8, 6], [0, 6]]


def wall_markers():
    """16 markers every 2 m along the walls, facing inward, 1.5 m high."""
    out = {}
    k = 0
    for (x0, y0), (x1, y1) in zip(ROOM, ROOM[1:] + ROOM[:1]):
        e = np.array([x1 - x0, y1 - y0], float)
        length = np.hypot(*e)
        n = int(round(length / 2.0))
        inward = np.array([-e[1], e[0]]) / length  # room is counter-clockwise: left normal points inward
        for i in range(n):
            xy = np.array([x0, y0]) + e * (i + 0.5) / n + 0.05 * inward
            out[k] = synth.upright_marker([xy[0], xy[1], 1.5], np.arctan2(inward[1], inward[0]))
            k += 1
    return out


def marker_json(markers):
    return [{"id": i, "p": np.round(p, 6).tolist(), "r": np.round(r, 9).tolist()} for i, (p, r) in markers.items()]


def detections(markers, c, rot, rng, sigma):
    return [{"id": i, "corners": np.round(uv, 4).tolist()} for i, uv in synth.observe(markers, A, c, rot, W, rng, sigma)]


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261108)
    markers = wall_markers()
    camera = {"w": W, "h": W // 2, "model": 1}

    c, yaw = np.array([3.3, 2.1, 1.2]), 0.7
    dump("resect.json", {
        "camera": camera, "a": A, "sigma_px": 0.3, "noise_model": "tangent", "dof": "yaw", "loss": "huber",
        "markers": marker_json(markers), "detections": detections(markers, c, synth.rot_zyx(yaw), rng, 0.3),
    })

    poses = [(np.array([2.0 + 0.5 * k, 1.5 + 0.4 * k, 1.2]), 0.3 + 0.35 * k) for k in range(6)]
    dump("ba.json", {
        "camera": camera, "a": A, "sigma_px": 0.3, "noise_model": "tangent", "dof": "yaw", "loss": "huber", "max_iter": 50,
        "rp": [[0.0, 0.0]] * len(poses),
        "surveyed": [{"id": i, "p": np.round(markers[i][0] + rng.normal(0, 0.003, 3), 6).tolist(), "sigma_ts": 0.005} for i in (0, 4, 8, 12)],
        "frames": [detections(markers, cc, synth.rot_zyx(yy), rng, 0.3) for cc, yy in poses],
    })

    dump("place.json", {
        "workspace": {"poly": [[0.5, 0.5], [5.5, 0.5], [5.5, 4.5], [0.5, 4.5]], "spacing": 0.5, "z_cam": [1.2],
                      "headings": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]},
        "obstacles": [{"poly": [[2.5, 2.0], [3.5, 2.0], [3.5, 3.0], [2.5, 3.0]], "height": 1.5}],
        "candidates": [
            {"xy": [x, y], "z": 1.5, "azimuth": az}
            for (x, y, az) in (
                [(1.5 * (i + 0.5), 0.05, np.pi / 2) for i in range(4)]
                + [(5.95, 1.25 * (i + 0.5), np.pi) for i in range(4)]
                + [(6 - 1.5 * (i + 0.5), 4.95, -np.pi / 2) for i in range(4)]
                + [(0.05, 5 - 1.25 * (i + 0.5), 0.0) for i in range(4)]
            )
        ],
        "n_max": 16, "a": A, "camera": {"w": 6528, "h": 3264, "model": 1}, "sigma_px": 0.3, "noise_model": "tangent",
        "vis": {"psi_max": float(np.radians(70.0)), "px_min": 12.0}, "tau": 0.05, "k_min": 2,
    })

    dump("site_query.json", {"mode": "centre", "crit": "N", "action": {"min_count": {"k_max": None}}})


if __name__ == "__main__":
    main()
