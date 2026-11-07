#!/usr/bin/env python3
"""Write fixtures/cli/*.json: CLI input files and `*.expected.json` Python outputs for the erploc-cli round-trip tests.
The cases are converted from the records of fixtures/est.txt and fixtures/place.txt (the same inputs and Python outputs the
existing differential tests use; no Python erploc code runs here, so the output is a pure function of the two fixtures).
Any python3 works: the script uses only the standard library.
JSON floats are Python reprs (round-trip exact); non-finite values are written as null."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, '..', 'fixtures')
OUT = os.path.join(FX, 'cli')


def load(name):
    recs = {}
    with open(os.path.join(FX, name)) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            t = line.split()
            r, c = int(t[2]), int(t[3])
            v = [float(x) for x in t[4:]]
            assert len(v) == r * c
            recs[(t[0], t[1])] = [v[i * c:(i + 1) * c] for i in range(r)]
    return recs


def fin(x):
    return x if math.isfinite(x) else None


def rows(m):
    return [[fin(x) for x in r] for r in m]


def flat(m):
    return [fin(x) for r in m for x in r]


def scalar(m):
    assert len(m) == 1 and len(m[0]) == 1
    return m[0][0]


def rot(v):
    return [v[0:3], v[3:6], v[6:9]]


def dump(path, obj):
    with open(os.path.join(OUT, path), 'w') as f:
        f.write(json.dumps(obj, allow_nan=False, separators=(',', ':')) + '\n')


CAMERA = {'w': 6528, 'h': 3264, 'model': 1}


def markers_of(mk):
    return [{'id': int(r[0]), 'p': r[1:4], 'r': rot(r[4:13])} for r in mk]


def detections(est, nm, frame):
    """Detector marker entries of one frame; masks and scales from the `cm` record when the case has one (ms_ cases)."""
    obs = [r for r in est[('obs', nm)] if int(r[0]) == frame]
    cm = {(int(r[0]), int(r[1])): r for r in est.get(('cm', nm), [])}
    out = []
    for r in obs:
        d = {'id': int(r[1]), 'corners': [[fin(r[2 + 2 * k]), fin(r[3 + 2 * k])] for k in range(4)]}
        c = cm.get((int(r[0]), int(r[1])))
        if c is not None:
            d['corner_valid'] = [int(x) for x in c[2:6]]
            d['corner_sigma_scale'] = c[6:10]
        out.append(d)
    return out


def resect_case(est, name, nm, dof):
    par = est[('par', nm)][0]
    a, sg, pixel, huber, roll, pitch, has_x0 = par[:7]
    inp = {'camera': CAMERA, 'a': a, 'sigma_px': sg, 'noise_model': 'pixel' if pixel == 1.0 else 'tangent',
           'loss': 'huber' if huber == 1.0 else 'linear', 'huber_delta': 1.5, 'dof': dof,
           'roll': roll, 'pitch': pitch}
    if has_x0 == 1.0:
        x = est[('x0', nm)][0]
        inp['x0'] = {'c': x[0:3], 'yaw': x[3], 'roll': x[4], 'pitch': x[5]}
    inp['markers'] = markers_of(est[('mk', nm)])
    inp['detections'] = detections(est, nm, 0)
    exp = {'source': nm, 'c': est[('c', nm)][0], 'cov': rows(est[('cov', nm)]), 'info': rows(est[('info', nm)]),
           'rms_px': scalar(est[('rms', nm)]), 'n_markers': int(scalar(est[('nmk', nm)])),
           'converged': scalar(est[('conv', nm)]) == 1.0}
    if dof == 'yaw':
        exp['yaw'] = scalar(est[('yaw', nm)])
    else:
        exp['r'] = rot(est[('R', nm)][0])
    dump(name + '.json', inp)
    dump(name + '.expected.json', exp)


def ba_case(est, name, nm):
    par = est[('par', nm)][0]
    a, sg, pixel, full, huber, delta = par[:6]
    nf = int(scalar(est[('nf', nm)]))
    inp = {'camera': CAMERA, 'a': a, 'sigma_px': sg, 'noise_model': 'pixel' if pixel == 1.0 else 'tangent',
           'dof': 'full' if full == 1.0 else 'yaw', 'loss': 'huber' if huber == 1.0 else 'linear', 'huber_delta': delta,
           'max_iter': 50, 'rp': est[('rp', nm)],
           'surveyed': [{'id': int(r[0]), 'p': r[1:4], 'sigma_ts': r[4]} for r in est[('sv', nm)]],
           'frames': [detections(est, nm, f) for f in range(nf)]}
    exp = {'source': nm, 'dof': inp['dof'], 'cams': rows(est[('cams', nm)]), 'mk': rows(est[('mk', nm)]),
           'covm': rows(est[('covm', nm)]), 'covc': rows(est[('covc', nm)]), 'rms_px': scalar(est[('rms', nm)]),
           'gauge': est[('gauge', nm)][0], 'converged': scalar(est[('conv', nm)]) == 1.0}
    dump(name + '.json', inp)
    dump(name + '.expected.json', exp)


def place_case(pl, name, spec, case):
    sc = pl[('sc', spec)][0]
    ob = {}
    for r in pl[('ob', spec)]:
        ob.setdefault(int(r[0]), {'poly': [], 'height': r[3]})['poly'].append([r[1], r[2]])
    inp = {'workspace': {'poly': pl[('wspoly', spec)], 'spacing': scalar(pl[('ws', spec)]), 'z_cam': pl[('zc', spec)][0],
                         'headings': pl[('hd', spec)][0]},
           'obstacles': [ob[i] for i in sorted(ob)],
           'candidates': [{'xy': r[0:2], 'z': r[2], 'azimuth': r[3]} for r in pl[('cand', spec)]],
           'n_max': int(sc[0]), 'a': sc[1], 'camera': {'w': int(sc[2]), 'h': int(sc[3]), 'model': int(sc[4])},
           'sigma_px': sc[5], 'noise_model': 'tangent' if sc[6] == 0.0 else 'pixel',
           'vis': {'psi_max': sc[7], 'px_min': sc[8], 'eps': sc[9],
                   'sectors': [dict(zip(('lam_min', 'lam_max', 'phi_min', 'phi_max'), r)) for r in pl.get(('sec', spec), [])]},
           'tau': sc[10], 'k_min': int(sc[11]), 'tau_loo': fin(sc[12]), 'sigma_cap': sc[13], 'w_k': sc[14]}
    k3 = pl[('key3', case)][0]
    exp = {'source': case, 'sigma_ts': scalar(pl[('ts', case)]), 'sel': [int(x) for x in pl[('sel', case)][0]],
           'survey_ids': [int(x) for x in pl[('ids', case)][0]], 'key3': {'n_infeasible': int(k3[0]), 'penalty_sum': k3[1], 'neg_margin': k3[2]},
           'score': scalar(pl[('score', case)]), 'n_skip': int(scalar(pl[('nskip', case)])),
           'csv': ''.join(chr(int(x)) for x in pl[('csv', case)][0])}
    dump(name + '.json', inp)
    dump(name + '.expected.json', exp)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith('.json'):
            os.remove(os.path.join(OUT, f))
    est = load('est.txt')
    pl = load('place.txt')
    resect_case(est, 'resect_yaw_0', 'rs_4_noisy0', 'yaw')
    resect_case(est, 'resect_full_0', 'rs_6_noisy0', 'full')
    resect_case(est, 'resect_huber_0', 'rs_4_noisy1', 'yaw')
    resect_case(est, 'resect_masked2_0', 'ms_rs_4_2', 'yaw')
    resect_case(est, 'resect_zerovalid_0', 'ms_rs_6_1', 'full')
    resect_case(est, 'resect_pixel_0', 'rs_4_noisy2', 'yaw')
    ba_case(est, 'ba_yaw_0', 'ba_noisy40')
    ba_case(est, 'ba_full_0', 'ba_full25')
    ba_case(est, 'ba_pixel_0', 'ba_pixel_yaw')
    place_case(pl, 'place_demo_0', 'pl_spec_demo', 'pl_survey_opt_demo_ts5')


main()
