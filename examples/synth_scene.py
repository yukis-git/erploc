"""Synthetic scenes for the tests and the examples (numpy only; `render_erp` needs eqar.generate_marker): markers on the walls of a room, an equirectangular
projection of their corners, and a noise model. The camera/ERP convention is that of erploc:

    d = R^T (X - c);  lam = atan2(d_y, d_x), phi = atan2(d_z, hypot(d_x, d_y))
    u = lam_sign * lam / s + (W - 1) / 2,  v = -phi / s + (H - 1) / 2,  s = 2 pi / W
"""

import numpy as np

TL_TR_BR_BL = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])


def rot_zyx(yaw, roll=0.0, pitch=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll): camera body to world."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def upright_marker(position, azimuth):
    """Marker pose (p, R): normal (cos az, sin az, 0), y up; columns of R are the marker x, y and normal axes."""
    n = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, n)
    return np.asarray(position, float), np.column_stack([x, y, n])


def marker_corners_world(p, r, a):
    q = np.zeros((4, 3))
    q[:, :2] = 0.5 * a * TL_TR_BR_BL
    return p + q @ r.T


def project(corners_world, c, rot, width, lam_sign=1):
    """ERP pixels (n, 2) of world points."""
    d = (np.asarray(corners_world) - c) @ rot
    lam = np.arctan2(d[:, 1], d[:, 0])
    phi = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    s = 2.0 * np.pi / width
    return np.column_stack([lam_sign * lam / s + (width - 1) / 2.0, -phi / s + (width / 2 - 1) / 2.0])


def wall_markers(erploc, polygon, spacing, z=1.5, a=0.16, first_id=0):
    """Markers along the walls of a polygon: (candidates (m, 4), {id: (p, R)})."""
    cand = erploc.wall_candidates(polygon, spacing, z=z)
    markers = {first_id + i: upright_marker(c[:3], c[3]) for i, c in enumerate(cand)}
    return cand, markers


def observe(markers, a, c, rot, camera_width, rng=None, sigma_px=0.0, ids=None, lam_sign=1):
    """Observations of the markers a camera at (c, rot) sees from the front: list of (id, corners (4, 2)).

    Markers behind their own plane (seen from the back) are skipped; corner noise is isotropic Gaussian in pixels."""
    out = []
    for mid, (p, r) in markers.items():
        if ids is not None and mid not in ids:
            continue
        if np.dot(r[:, 2], c - p) <= 0.0:
            continue
        uv = project(marker_corners_world(p, r, a), c, rot, camera_width, lam_sign)
        if rng is not None and sigma_px > 0.0:
            uv = uv + rng.normal(0.0, sigma_px, uv.shape)
        out.append((mid, uv))
    return out


def render_erp(width, markers, a, c, rot, generate_marker, dictionary="4x4_50", lam_sign=-1, background=205, dark=40, bright=220,
               supersample=3, noise_sd=0.0, blur_sd=0.0, seed=0):
    """A grey equirectangular frame (width/2, width) of flat square markers (uint8).

    Each marker is the printed tag (side `a`, black border included) with a quiet zone of 1.5 cells, on the plane through
    `markers[id] = (p, R)` with normal R[:, 2]; only the side facing the camera is painted. `generate_marker(id, dictionary, size)`
    is `eqar.generate_marker` (a square (size, size) image of the tag, 0 = black cell). Painting is by inverse mapping of every
    pixel (supersampled) onto the marker plane, nearest marker first. `lam_sign=-1` gives an image in which the markers are not
    mirrored for the usual body frame (x forward, y left, z up, image columns increasing to the right).
    """
    h = width // 2
    ncell = int(dictionary[0]) + 2
    s = 2.0 * np.pi / width
    img = np.full((h, width), float(background))
    order = sorted(markers, key=lambda i: -np.linalg.norm(markers[i][0] - c))  # far first, near overwrites
    for mid in order:
        p, r = markers[mid]
        n = r[:, 2]
        if np.dot(n, c - p) <= 0.0:
            continue
        grid = generate_marker(mid, dictionary, size=ncell) > 0  # True = white cell
        cell = a / ncell
        half_q = a / 2 + 1.5 * cell
        # window around the marker
        d_w = p - c
        d_b = rot.T @ d_w
        lam0, phi0 = np.arctan2(d_b[1], d_b[0]), np.arctan2(d_b[2], np.hypot(d_b[0], d_b[1]))
        theta = np.arctan(np.sqrt(2.0) * half_q / np.linalg.norm(d_w)) * 1.5 + 2 * s
        v0, u0 = (h - 1) / 2.0, (width - 1) / 2.0
        r0 = int(max(0, np.floor(v0 - (phi0 + theta) / s) - 1))
        r1 = int(min(h - 1, np.ceil(v0 - (phi0 - theta) / s) + 1))
        cl = np.cos(min(np.pi / 2 - 1e-3, abs(phi0) + theta))
        dlam = min(np.pi, theta / max(cl, 1e-3))
        c_mid = lam0 / (lam_sign * s) + u0
        ncols = int(np.ceil(dlam / s)) + 2
        cols = np.arange(int(np.floor(c_mid)) - ncols, int(np.floor(c_mid)) + ncols + 1) if 2 * ncols + 1 < width else np.arange(width)
        Y, X = np.meshgrid(np.arange(r0, r1 + 1), cols, indexing="ij")
        acc = np.zeros(X.shape)
        hit = np.zeros(X.shape)
        for sy in range(supersample):
            for sx in range(supersample):
                fx = X + ((sx + 0.5) / supersample - 0.5)
                fy = Y + ((sy + 0.5) / supersample - 0.5)
                lam = lam_sign * s * (fx - u0)
                phi = -s * (fy - v0)
                db = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
                dw = db @ rot.T
                den = dw @ n
                ok = den < -1e-12  # ray runs against the normal: it hits the front side
                t = np.where(ok, (n @ (p - c)) / np.where(ok, den, 1.0), 0.0)
                q = dw * t[..., None] - (p - c)
                qx, qy = q @ r[:, 0], q @ r[:, 1]
                inside = ok & (t > 0) & (np.abs(qx) <= half_q) & (np.abs(qy) <= half_q)
                col = np.clip(np.floor((qx + a / 2) / cell).astype(int), 0, ncell - 1)
                row = np.clip(np.floor((a / 2 - qy) / cell).astype(int), 0, ncell - 1)
                in_tag = (np.abs(qx) <= a / 2) & (np.abs(qy) <= a / 2)
                val = np.where(in_tag, np.where(grid[row, col], bright, dark), bright)
                acc += np.where(inside, val, 0.0)
                hit += inside
        xi = X % width
        old = img[Y, xi]
        new = (acc + (supersample**2 - hit) * old) / supersample**2
        img[Y, xi] = np.where(hit > 0, new, old)
    if blur_sd > 0:
        rad = int(np.ceil(3 * blur_sd))
        k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / blur_sd) ** 2)
        k /= k.sum()
        horiz = sum(w * np.roll(img, -j, axis=1) for j, w in zip(range(-rad, rad + 1), k))
        padded = np.pad(horiz, ((rad, rad), (0, 0)), mode="edge")
        img = sum(w * padded[rad + j : rad + j + h] for j, w in zip(range(-rad, rad + 1), k))
    if noise_sd > 0:
        img = img + np.random.default_rng(seed).normal(0.0, noise_sd, img.shape)
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)
