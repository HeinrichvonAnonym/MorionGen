#!/usr/bin/env python
"""Differential defect injection (config/e_criterion_domain_20261105.json, 'injection'): human kin30 motions with and
without a defect of known size through the full E pipeline; per-frame residual, envelope verdict and foot skate under
R0/R4/R5.

Writes develop_logs/test_runs/20261105_e_domain/inject/<id>.json and <id>.npz; work directories on the data disk.
OpenSim python.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
SPEC = json.loads((ROOT/'config/e_criterion_domain_20261105.json').read_text())['injection']
OUT = ROOT/'develop_logs/test_runs/20261105_e_domain/inject'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_domain_20261105/inject')
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
CROP_S, RAMP_S, RETURN_S = 6., .1, .3
SLIDE_MAX_S = SPEC['pre_result_amendment']['slide_max_s']
UPPER = ('Spine', 'Spine3', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow', 'left_wrist')


def bases():
    rng = np.random.default_rng(20261105)
    rows = json.loads((VAL/'selection.json').read_text())['trials']
    walk = sorted([r for r in rows if r['class'] == 'walk'], key=lambda r: -r['seconds'])[:12]
    squat = [r for r in rows if r['class'] == 'squat' and r['seconds'] >= 8]
    squat = [squat[k] for k in sorted(rng.choice(len(squat), min(12, len(squat)), replace=False))]
    return walk+squat


def jobs():
    out = []
    for r in bases():
        base = {'base': r['id'], 'class': r['class'], 'run': r['sources']['kin30'], 'mass_kg': r['mass_kg']}
        out.append({**base, 'id': f"{r['id']}__control", 'defect': 'control', 'size': 0})
        for defect, sizes in SPEC['defects'].items():
            for s in sizes:
                out.append({**base, 'id': f"{r['id']}__{defect}_{s:g}", 'defect': defect, 'size': s})
    return out


def ramp(n, fps, a, b):
    """Raised-cosine window: 0 outside [a, b), rising and falling over RAMP_S inside it."""
    w = np.zeros(n); k = max(1, int(round(RAMP_S*fps)))
    w[a:b] = 1
    up = .5-.5*np.cos(np.pi*np.arange(1, k+1)/(k+1))
    w[a:a+k] = up[:max(0, min(k, b-a))]; w[max(a, b-k):b] = up[::-1][-min(k, b-a):]
    return w


def heading(joints, names):
    left = joints[:, names.index('left_hip')]-joints[:, names.index('right_hip')]
    left[:, 1] = 0; left /= np.linalg.norm(left, axis=1, keepdims=True)+1e-9
    fwd = np.cross(left, [0., 1., 0.])
    return left, fwd


def rotate(v, axis, angle):
    axis = axis/np.linalg.norm(axis, axis=-1, keepdims=True)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return v*c+np.cross(axis, v)*s+axis*np.sum(axis*v, -1, keepdims=True)*(1-c)


def stance(joints, names, fps):
    foot = joints[:, names.index('right_foot')]
    h = foot[:, 1]; v = np.linalg.norm(np.gradient(foot[:, [0, 2]], 1/fps, axis=0), axis=1)
    low = (h <= np.percentile(h, 5)+.03) & (v <= .2)
    edges = np.diff(np.r_[0, low.astype(int), 0]); runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    runs = [(a, b) for a, b in runs if (b-a)/fps >= .3]
    return max(runs, key=lambda x: x[1]-x[0]) if runs else None


def inject(joints, names, fps, defect, size):
    """Modified Y-up joints, fps, and the analysis window [t0, t1) in seconds (None: whole clip)."""
    j = joints.copy(); n = len(j); a, b = n//3, 2*n//3
    if defect == 'control':
        return j, fps, None
    if defect in ('float_window_m', 'penetrate_window_m'):
        j[..., 1] += (1 if defect.startswith('float') else -1)*size*ramp(n, fps, a, b)[:, None]
        return j, fps, (a/fps, b/fps)
    if defect == 'float_whole_m':
        j[..., 1] += size
        return j, fps, None
    if defect == 'lean_deg':
        left, _ = heading(j, names); pivot = j[:, names.index('pelvis')]
        angle = np.deg2rad(size)*ramp(n, fps, a, b)
        for name in UPPER:
            k = names.index(name)
            j[:, k] = pivot+rotate(j[:, k]-pivot, left, angle)
        return j, fps, (a/fps, b/fps)
    if defect == 'slide_m_s':
        st = stance(j, names, fps)
        if st is None:
            return None, fps, None
        s0, s1 = st; s1 = min(s1, s0+int(round(SLIDE_MAX_S*fps))); k = int(round(RETURN_S*fps)); _, fwd = heading(j, names)
        d = np.zeros(n); d[s0:s1] = size*np.arange(s1-s0)/fps
        back = np.arange(1, k+1); e = min(n, s1+k)
        d[s1:e] = d[s1-1]*(.5+.5*np.cos(np.pi*back[:e-s1]/k))
        for name, gain in (('right_ankle', 1.), ('right_foot', 1.), ('right_knee', .5)):
            j[:, names.index(name)] += gain*d[:, None]*fwd
        return j, fps, (s0/fps, e/fps)
    if defect == 'time_compression':
        return j, fps*size, None
    raise ValueError(defect)


def job(p):
    os.nice(19)
    import biomech
    import biomech_sources
    import e_rules
    out = OUT/f"{p['id']}.json"
    if out.exists():
        return 'skip'
    row = dict(p)
    try:
        joints, names, fps = biomech_sources.load(p['run'])
        keep = int(min(len(joints), CROP_S*fps)); c0 = (len(joints)-keep)//2; joints = joints[c0:c0+keep]
        mod, fps2, window = inject(joints, names, fps, p['defect'], p['size'])
        if mod is None:
            row.update(status='skipped', reason='no right-foot stance of >= 0.3 s')
            out.write_text(json.dumps(row, indent=1))
            return 'skipped'
        biomech.load = lambda _run: (mod, names, fps2)
        work = DISK/'work'/p['id']
        receipt = biomech.prepare(Path(p['run']), work, p['mass_kg'], free_yaw=True)
        biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
        times, _, _, kin = e_rules.kinematics(work)
        arrays = {'times': times}
        for rule in e_rules.RULES:
            res = e_rules.evaluate(times, kin, p['mass_kg'], rule)
            bad, st = e_rules.infeasible(res, rule)
            contact = np.stack([res['masks'][s][k] for s in ('r', 'l') for k in ('heel', 'toe')], 1)
            fast = np.stack([res['speed'][s][k] > e_rules.SKATE_M_S for s in ('r', 'l') for k in ('heel', 'toe')], 1)
            arrays.update({f'{rule}_residual_bw': res['residual_bw'], f'{rule}_infeasible': bad, f'{rule}_contact': contact,
                           f'{rule}_skate': fast & contact, **{f'{rule}_state_{k}': v for k, v in st.items()}})
        np.savez_compressed(OUT/f"{p['id']}.npz", **arrays)
        row.update(fps=fps2, frames=len(mod), window_s=window, ik=receipt['ik_marker_error'], status='success')
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=6); a.add_argument('--only', nargs='*')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = jobs()
    (OUT.parent/'inject_jobs.json').write_text(json.dumps(todo, indent=1))
    if args.only:
        todo = [p for p in todo if p['id'] in args.only]
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(job, todo):
            counts[s] = counts.get(s, 0)+1
            print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
