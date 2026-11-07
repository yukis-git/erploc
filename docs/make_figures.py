"""Writes the figures of the README from synthetic data:  python docs/make_figures.py   (needs matplotlib)

    docs/placement.png      layout chosen for a made-up room and the worst-case sigma_pos of every evaluation cell
    docs/error_ellipses.png  1-sigma position error ellipses of single-frame resections, with Monte Carlo errors for two cameras
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

import erploc  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
import synth_scene as synth  # noqa: E402

A, W = 0.16, 6528
ROOM = [[0, 0], [9, 0], [9, 6.5], [0, 6.5]]
WORK = [[0.7, 0.7], [8.3, 0.7], [8.3, 5.8], [0.7, 5.8]]
BOX = ([[3.6, 2.6], [5.4, 2.6], [5.4, 3.9], [3.6, 3.9]], 1.6)
INK, MUTED = "#222222", "#777777"


def setup_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(-0.4, 6.9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.add_patch(Polygon(ROOM, closed=True, fill=False, edgecolor=INK, linewidth=1.4))
    ax.add_patch(Polygon(BOX[0], closed=True, facecolor="#cccccc", edgecolor=INK, linewidth=1.0, zorder=3))


def placement_figure():
    cand = erploc.wall_candidates(ROOM, 1.25, z=1.5)
    scene = erploc.Scene(WORK, cand, [BOX], camera=erploc.Camera(W), a=A, spacing=0.4, z_cam=[1.2])
    plan = erploc.place_markers(scene, tau=0.02, k_min=2, tau_loo=0.1)
    fig, ax = plt.subplots(figsize=(6.6, 4.9), dpi=110)
    setup_axes(ax)
    sc = ax.scatter(plan.cells[:, 0], plan.cells[:, 1], c=plan.sigma * 1000, s=26, marker="s", cmap="viridis_r", zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("worst-case sigma_pos of the cell [mm]")
    ax.scatter(cand[:, 0], cand[:, 1], s=14, facecolors="none", edgecolors=MUTED, linewidths=0.8, zorder=4, label="candidate")
    ch = plan.chosen_candidates
    surveyed = np.isin(plan.chosen, plan.survey_ids)
    for (x, y, z, az), s in zip(ch, surveyed):
        ax.annotate("", xy=(x + 0.55 * np.cos(az), y + 0.55 * np.sin(az)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b" if s else INK, lw=1.6), zorder=6)
    ax.scatter(ch[~surveyed, 0], ch[~surveyed, 1], s=46, color=INK, zorder=5, label=f"chosen ({plan.count})")
    ax.scatter(ch[surveyed, 0], ch[surveyed, 1], s=46, color="#c0392b", zorder=5, label="chosen, to survey")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.09))
    ax.set_title(f"synthetic 9 m x 6.5 m room, box obstacle; feasible cells {plan.feasible_fraction:.0%}, "
                 f"worst sigma_pos {plan.worst_sigma_pos * 1000:.1f} mm", fontsize=8.5, pad=22)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "placement.png"))
    plt.close(fig)
    return plan, scene


def blocked(c, p):
    """Does the sight line from the camera c to the marker p pass through the box (which is higher than both ends)?"""
    (x0, y0), (x1, y1) = BOX[0][0], BOX[0][2]
    t0, t1 = 0.0, 1.0
    for lo, hi, a, b in ((x0, x1, c[0], p[0]), (y0, y1, c[1], p[1])):
        d = b - a
        if abs(d) < 1e-12:
            if not lo <= a <= hi:
                return False
            continue
        ta, tb = sorted(((lo - a) / d, (hi - a) / d))
        t0, t1 = max(t0, ta), min(t1, tb)
    return t0 < t1


def ellipse_figure(plan, scene):
    rng = np.random.default_rng(4)
    cand = scene.candidates[plan.chosen]
    markers = {int(j): synth.upright_marker(c[:3], c[3]) for j, c in zip(plan.chosen, cand)}
    cam = erploc.Camera(W)
    fig, ax = plt.subplots(figsize=(6.6, 4.9), dpi=110)
    setup_axes(ax)
    pts = [(x, y) for x in np.linspace(1.2, 7.8, 6) for y in np.linspace(1.2, 5.3, 4)]
    scale = 300.0  # ellipses drawn 300 x larger than the 1-sigma error
    shown = 0
    for x, y in pts:
        c, rot = np.array([x, y, 1.2]), synth.rot_zyx(0.5)
        visible = {i: m for i, m in markers.items() if not blocked(c, m[0])}
        obs = [erploc.Observation(i, uv) for i, uv in synth.observe(visible, A, c, rot, W)]
        if len(obs) < 2 or (3.6 - 0.2 < x < 5.6 and 2.4 < y < 4.1):
            continue
        r = erploc.resection(obs, markers, cam, A, sigma_px=0.3)
        w, v = np.linalg.eigh(r.cov[:2, :2])
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ax.add_patch(Ellipse((x, y), 2 * scale * np.sqrt(w[1]), 2 * scale * np.sqrt(w[0]), angle=ang, facecolor="#2a6f97", alpha=0.35,
                             edgecolor="#2a6f97", zorder=5))
        ax.plot([x], [y], ".", color=INK, ms=3, zorder=6)
        shown += 1
        if (x, y) in {pts[7], pts[16]}:  # two cameras with Monte Carlo errors
            errs = []
            for _ in range(300):
                o = [erploc.Observation(i, uv) for i, uv in synth.observe(visible, A, c, rot, W, rng, 0.3)]
                errs.append(erploc.resection(o, markers, cam, A, sigma_px=0.3).position[:2] - c[:2])
            errs = np.array(errs)
            ax.scatter(x + scale * errs[:, 0], y + scale * errs[:, 1], s=2.5, color="#c0392b", alpha=0.6, zorder=7)
    ch = plan.chosen_candidates
    ax.scatter(ch[:, 0], ch[:, 1], s=40, color=INK, zorder=8)
    ax.set_title(f"1-sigma position error ellipses of single frames (drawn {scale:.0f}x), 0.3 px corner noise;\n"
                 "red: 300 simulated errors of two of the cameras (same scale)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "error_ellipses.png"))
    plt.close(fig)
    return shown


if __name__ == "__main__":
    p, s = placement_figure()
    n = ellipse_figure(p, s)
    for f in ("placement.png", "error_ellipses.png"):
        print(f, os.path.getsize(os.path.join(HERE, f)) // 1024, "KB")
