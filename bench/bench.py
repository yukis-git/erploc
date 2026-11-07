"""Reproducible benchmark on synthetic rooms (numpy only):  python bench/bench.py [--quick] [--seed N]

Part A: wall-clock time of the estimators and of the placement search (median of repeated calls, in one process).
Part B: placement quality on random synthetic rooms: the IFT search and the greedy search against simple baselines
        (random layouts, evenly spaced layouts) and, on small rooms, against exhaustive enumeration.

All rooms are generated from the seed; nothing is a measurement of a real place. Output is Markdown.
"""

import argparse
import itertools
import os
import platform
import subprocess
import sys
import time

import numpy as np

import erploc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import synth_scene as synth  # noqa: E402

A = 0.16
TAU, K_MIN, TAU_LOO = 0.03, 2, 0.1


def machine():
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return cpu


def timeit(fn, repeat, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts), np.min(ts), np.max(ts)


def fmt_t(t):
    return f"{t * 1e3:.2f} ms" if t < 1.0 else f"{t:.2f} s"


# ------------------------------------------------------------------------------------------------ rooms
def random_room(rng, big=False):
    w, d = (rng.uniform(8, 11), rng.uniform(6, 8)) if big else (rng.uniform(5, 8), rng.uniform(4, 6))
    n_obs = int(rng.integers(0, 3))
    obstacles = []
    for _ in range(n_obs):
        bw, bd = rng.uniform(0.8, 1.6, 2)
        cx, cy = rng.uniform(1.2 + bw / 2, w - 1.2 - bw / 2), rng.uniform(1.2 + bd / 2, d - 1.2 - bd / 2)
        poly = [[cx - bw / 2, cy - bd / 2], [cx + bw / 2, cy - bd / 2], [cx + bw / 2, cy + bd / 2], [cx - bw / 2, cy + bd / 2]]
        obstacles.append((poly, float(rng.uniform(1.2, 1.8))))
    room = [[0, 0], [w, 0], [w, d], [0, d]]
    work = [[0.6, 0.6], [w - 0.6, 0.6], [w - 0.6, d - 0.6], [0.6, d - 0.6]]
    return room, work, obstacles


def make_scene(rng, big=False, spacing_cand=2.0, spacing_cell=0.7, headings=4, camera_w=6528):
    room, work, obstacles = random_room(rng, big)
    cand = erploc.wall_candidates(room, spacing_cand, z=1.5)
    scene = erploc.Scene(work, cand, obstacles, camera=erploc.Camera(camera_w), a=A, spacing=spacing_cell,
                         headings=np.linspace(0, 2 * np.pi, headings, endpoint=False))
    return scene


# ------------------------------------------------------------------------------------------------ part A
def part_a(seed):
    rng = np.random.default_rng(seed)
    room = [[0, 0], [8, 0], [8, 6], [0, 6]]
    cand = erploc.wall_candidates(room, 2.0, z=1.5)
    markers = {i: synth.upright_marker(c[:3], c[3]) for i, c in enumerate(cand)}
    cam = erploc.Camera(4096)
    c, rot = np.array([3.3, 2.1, 1.2]), synth.rot_zyx(0.7, 0.03, -0.02)
    obs = [erploc.Observation(i, uv) for i, uv in synth.observe(markers, A, c, rot, 4096, rng, 0.3)]
    rows = []
    rows.append((f"resection, yaw (4 dof), {len(obs)} markers", timeit(lambda: erploc.resection(obs, markers, cam, A, roll=0.03, pitch=-0.02), 200)))
    rows.append((f"resection, full (6 dof), {len(obs)} markers", timeit(lambda: erploc.resection(obs, markers, cam, A, dof="full"), 200)))
    rows.append(("resection, full, Huber loss", timeit(lambda: erploc.resection(obs, markers, cam, A, dof="full", loss="huber"), 200)))
    pose = erploc.Pose(c, 0.7)
    rows.append((f"fisher_camera, {len(markers)} markers", timeit(lambda: erploc.fisher_camera(pose, markers, cam, A), 500)))
    for nf in (10, 40):
        frames = []
        for k in range(nf):
            s = k / (nf - 1)
            cc = np.array([2.0 + 4.0 * s, 1.5 + 6.0 * s * (1 - s) + 0.5, 1.2])
            frames.append([erploc.Observation(i, uv) for i, uv in synth.observe(markers, A, cc, synth.rot_zyx(0.3 + 2.0 * s), 4096, rng, 0.3)])
        survey = {i: (markers[i][0], 0.005) for i in (0, 4, 8, 11)}
        rows.append((f"init_bundle, {nf} frames, {len(markers)} markers", timeit(lambda: erploc.init_bundle(frames, cam, A, survey), 20 if nf < 20 else 5)))
        rows.append((f"bundle_adjust (incl. init), {nf} frames, {len(markers)} markers", timeit(lambda: erploc.bundle_adjust(frames, cam, A, survey), 20 if nf < 20 else 5)))
    out = ["| operation | median | min | max |", "|---|---|---|---|"]
    for name, (med, lo, hi) in rows:
        out.append(f"| {name} | {fmt_t(med)} | {fmt_t(lo)} | {fmt_t(hi)} |")
    return out


def part_a_place(seed):
    rows = []
    for label, big in (("small room", False), ("large room", True)):
        rng = np.random.default_rng(seed + (7 if big else 3))
        scene = make_scene(rng, big)
        t_eval = timeit(lambda: erploc.evaluate(scene, [0, 1, 2], TAU, k_min=K_MIN, tau_loo=TAU_LOO), 30)
        t_g = timeit(lambda: erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO), 3, warmup=0)
        t_i = timeit(lambda: erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO, method="ift"), 3, warmup=0)
        g = erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO)
        i = erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO, method="ift")
        rows.append(f"| {label}: {scene.n_candidates} candidates, {scene.n_cells} cells | evaluate (tables cached) | {fmt_t(t_eval[0])} | |")
        rows.append(f"| | place_markers greedy -> {g.count} markers | {fmt_t(t_g[0])} | min {fmt_t(t_g[1])}, max {fmt_t(t_g[2])} |")
        rows.append(f"| | place_markers ift -> {i.count} markers | {fmt_t(t_i[0])} | min {fmt_t(t_i[1])}, max {fmt_t(t_i[2])} |")
    return ["| scene | operation | median | spread |", "|---|---|---|---|"] + rows


# ------------------------------------------------------------------------------------------------ part B
def evenly_spaced(n, k):
    """k candidates spread evenly over the perimeter ordering of the candidates (the order wall_candidates returns)."""
    return np.unique(np.round(np.linspace(0, n, k, endpoint=False) + n / (2 * k)).astype(int) % n)


def stats_of(p):
    return p.feasible_fraction, p.worst_sigma_pos


def part_b(seed, n_rooms, n_random, exhaustive_limit, big):
    rng = np.random.default_rng(seed + (101 if big else 0))
    recs = []
    for r in range(n_rooms):
        scene = make_scene(rng, big=big, spacing_cand=1.5 if big else 2.0)
        n = scene.n_candidates
        t0 = time.perf_counter()
        g = erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO)
        tg = time.perf_counter() - t0
        t0 = time.perf_counter()
        i = erploc.place_markers(scene, TAU, k_min=K_MIN, tau_loo=TAU_LOO, method="ift")
        ti = time.perf_counter() - t0
        rec = dict(room=r, n=n, cells=scene.n_cells, g_count=g.count, g_ff=g.feasible_fraction, g_t=tg, i_count=i.count, i_ff=i.feasible_fraction,
                   i_t=ti, g_w=g.worst_sigma_pos, i_w=i.worst_sigma_pos)
        k = i.count
        # equal-count baselines
        rf, rw, rfull = [], [], 0
        for _ in range(n_random):
            sel = rng.choice(n, k, replace=False)
            e = erploc.evaluate(scene, sel, TAU, k_min=K_MIN, tau_loo=TAU_LOO)
            rf.append(e.feasible_fraction)
            rw.append(e.worst_sigma_pos)
            rfull += e.feasible_fraction == 1.0
        rec.update(rand_ff_mean=float(np.mean(rf)), rand_full=rfull / n_random)
        ev = erploc.evaluate(scene, evenly_spaced(n, k), TAU, k_min=K_MIN, tau_loo=TAU_LOO)
        rec.update(even_ff=ev.feasible_fraction, even_n=len(evenly_spaced(n, k)))
        # smallest k at which any of n_random random layouts is fully feasible (capped at n)
        kr = None
        for kk in range(1, n + 1):
            if any(erploc.evaluate(scene, rng.choice(n, kk, replace=False), TAU, k_min=K_MIN, tau_loo=TAU_LOO).feasible_fraction == 1.0
                   for _ in range(n_random)):
                kr = kk
                break
        rec["rand_min_k"] = kr
        # exhaustive: smallest count with a fully feasible layout, and the best worst-case sigma at the IFT count
        total = sum(_comb(n, kk) for kk in range(1, i.count + 1))
        if total <= exhaustive_limit:
            kmin, best_w = None, {}
            for kk in range(1, i.count + 1):
                ws = []
                full = False
                for sel in itertools.combinations(range(n), kk):
                    e = erploc.evaluate(scene, list(sel), TAU, k_min=K_MIN, tau_loo=TAU_LOO)
                    if e.feasible_fraction == 1.0:
                        full = True
                        ws.append(e.worst_sigma_pos)
                if full:
                    kmin, best_w[kk] = kk, min(ws)
                    break
            rec.update(ex_min_k=kmin, ex_w=best_w.get(kmin))
        else:
            rec.update(ex_min_k=None, ex_w=None)
        recs.append(rec)
        print(f"  room {r + 1}/{n_rooms} done", file=sys.stderr)
    return recs


def _comb(n, k):
    from math import comb
    return comb(n, k)


def summarize_b(recs):
    out = ["| room | cand. | cells | greedy count (time) | IFT count (time) | random, same count: mean feasible / any full | evenly spaced: feasible | smallest count of 200 random draws | exhaustive min count |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        out.append(f"| {r['room'] + 1} | {r['n']} | {r['cells']} | {r['g_count']} ({r['g_t']:.2f} s) | {r['i_count']} ({r['i_t']:.2f} s) | "
                   f"{r['rand_ff_mean']:.0%} / {r['rand_full']:.0%} | {r['even_ff']:.0%} | {r['rand_min_k']} | {r['ex_min_k'] if r['ex_min_k'] else 'not run'} |")
    ex = [r for r in recs if r["ex_min_k"]]
    lines = ["", f"Rooms: {len(recs)}." + (f" Exhaustive enumeration was run on {len(ex)} of them (those with at most the stated number of subsets up to the IFT count)." if ex else "")]
    if ex:
        lines.append(f"- IFT count equals the exhaustive minimum count in {sum(r['i_count'] == r['ex_min_k'] for r in ex)} of {len(ex)}; "
                     f"greedy in {sum(r['g_count'] == r['ex_min_k'] for r in ex)} of {len(ex)}.")
        ri = [r["i_w"] / r["ex_w"] for r in ex if r["i_count"] == r["ex_min_k"]]
        rg = [r["g_w"] / r["ex_w"] for r in ex if r["g_count"] == r["ex_min_k"]]
        if ri:
            lines.append(f"- Worst-case sigma_pos of the IFT layout over the best layout of the same size (exhaustive min-max), where the counts agree: "
                         f"median {np.median(ri):.2f}, max {np.max(ri):.2f} (n={len(ri)}).")
        if rg:
            lines.append(f"- The same for greedy: median {np.median(rg):.2f}, max {np.max(rg):.2f} (n={len(rg)}).")
    lines.append(f"- Markers used, summed over the rooms: IFT {sum(r['i_count'] for r in recs)}, greedy {sum(r['g_count'] for r in recs)}, "
                 f"random layouts (smallest count at which any of the draws was fully feasible; all candidates if none was) "
                 f"{sum(r['rand_min_k'] or r['n'] for r in recs)}.")
    lines.append(f"- Mean feasible fraction of equally many random markers: {np.mean([r['rand_ff_mean'] for r in recs]):.0%}; "
                 f"of evenly spaced ones: {np.mean([r['even_ff'] for r in recs]):.0%}.")
    return out + lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    rooms, big_rooms, draws, limit = (4, 1, 40, 3000) if args.quick else (24, 8, 200, 12000)
    load0 = os.getloadavg()
    print("# erploc benchmark\n")
    print(f"- date: {time.strftime('%Y-%m-%d %H:%M')}; seed {args.seed}; python {platform.python_version()}, numpy {np.__version__}, erploc {erploc.__version__}")
    print(f"- machine: {machine()}, {os.cpu_count()} logical CPUs, {platform.platform()}")
    print(f"- load average before: {load0[0]:.2f} {load0[1]:.2f} {load0[2]:.2f} (1/5/15 min)")
    print(f"- conditions: placement tau={TAU} m, k_min={K_MIN}, tau_loo={TAU_LOO} m, tag side {A} m, 6528x3264 camera, 4 headings, cells every 0.7 m, 2 m wall spacing\n")
    print("## A. Time\n\nSingle process, release build, one thread. Estimators on an 8 m x 6 m room with 14 markers, 4096 px wide frames, 0.3 px noise.\n")
    print("\n".join(part_a(args.seed)))
    print("\nPlacement search on random synthetic rooms (the search is deterministic; times include building the score tables on the first call).\n")
    print("\n".join(part_a_place(args.seed)))
    print("\n## B. Placement quality on synthetic rooms\n")
    print(f"Small rooms (5-8 m x 4-6 m, 0-2 box obstacles, candidates every 2 m, {draws} random draws per count, exhaustive enumeration up to {limit} subsets):\n")
    print("\n".join(summarize_b(part_b(args.seed, rooms, draws, limit, False))))
    print(f"\nLarger rooms (8-11 m x 6-8 m, candidates every 1.5 m, no exhaustive enumeration):\n")
    print("\n".join(summarize_b(part_b(args.seed, big_rooms, draws, 0, True))))
    load1 = os.getloadavg()
    print(f"\n- load average after: {load1[0]:.2f} {load1[1]:.2f} {load1[2]:.2f}")


if __name__ == "__main__":
    main()
