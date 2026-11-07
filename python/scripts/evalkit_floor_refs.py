#!/usr/bin/env python
"""Per-method flat-foot landmark references through the evalkit front end (evalkit.fit: upright calibration, feet
levelled on the given floor). Same standing poses as config/e_v2_floor_reference_20261106.json (e_v2_floor_poses.py,
one per side and variant, each method's own body model on its own floor), held for 3 s at 30 fps; reference_m is the
median landmark height over the scored frames. Writes config/evalkit_floor_reference_20261107.json with the same
layout. OpenSim python.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen')]
POSES = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/floor_audit/poses')
WORK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/evalkit_20261107/floor_refs')
OLD = ROOT/'config/e_v2_floor_reference_20261106.json'
NEW = ROOT/'config/evalkit_floor_reference_20261107.json'
HOLD_S, FPS, MASS, FLOOR = 3., 30., 75., 0.
LANDMARK_BODY = {'heel': 'calcn', 'toe': 'toes'}


def measure(pose):
    os.nice(19)
    import opensim as osim
    osim.Logger.setLevelString('error')
    import biomech
    from evalkit import fit
    with np.load(POSES/f'{pose}.npz', allow_pickle=False) as m:
        joints, names = m['joints'], [str(n) for n in m['joint_names']]
    run = WORK/pose/'input'; run.mkdir(parents=True, exist_ok=True)
    np.savez(run/'motion.npz', joints=np.repeat(joints[None], int(HOLD_S*FPS), 0), joint_names=np.array(names), fps=FPS)
    f = fit.fit(run, WORK/pose/'work', MASS, FLOOR)
    kin = biomech.body_kinematics(osim.Model(str(WORK/pose/'work/model.osim')), f['labels'], f['coords'])
    sel = fit.scored(f['times'])
    ref = {s: {k: float(np.median(kin['origin'][sel, kin['names'].index(f'{b}_{s}'), 1])) for k, b in LANDMARK_BODY.items()} for s in 'rl'}
    return pose, {'reference_m': ref, 'feet': {s: {k: v for k, v in r.items() if k != 'body_frame_m'} for s, r in f['receipt']['feet'].items()},
                  'ik_marker_error': f['receipt']['ik_marker_error']}


def main():
    from evalkit.fit import PREPARE
    old = json.loads(OLD.read_text())
    previous = json.loads(NEW.read_text()) if NEW.exists() else None
    poses = sorted(p.stem for p in POSES.glob('*@*.npz'))
    with ProcessPoolExecutor(6, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        got = dict(p.map(measure, poses))
    side = lambda tag: {s: got[f'{tag}@{s}']['reference_m'][s] for s in 'lr'}
    detail = lambda tag: {s: {k: got[f'{tag}@{s}'][k] for k in ('feet', 'ik_marker_error')} for s in 'lr'}
    methods = {}
    for variant, rec in old['methods'].items():
        if variant == 'chois':
            methods['chois'] = {**{k: v for k, v in rec.items() if k != 'windows'},
                                'windows': {w: {**{k: v for k, v in r.items() if k not in ('reference_m', 'ik_marker_error')},
                                                'reference_m': side(f'chois_w{w}'), 'fit': detail(f'chois_w{w}'),
                                                'e_v2_reference_m': r['reference_m']} for w, r in rec['windows'].items()}}
        else:
            methods[variant] = {**{k: v for k, v in rec.items() if k not in ('reference_m', 'ik_marker_error')},
                                'reference_m': side(variant), 'fit': detail(variant), 'e_v2_reference_m': rec['reference_m']}
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'fit_prepare': PREPARE,
           'definition': ('reference_m[side][landmark]: median height of the landmark (heel = calcn origin, toe = toes origin) over the scored '
                          'frames of a 3 s hold of the method\'s flat-foot standing pose (the e_v2 poses, that side\'s foot on the method\'s '
                          'floor), through evalkit.fit (upright calibration, feet levelled on the floor); world_floor_m as in e_v2'),
           'scripts': ['python/scripts/e_v2_floor_poses.py', 'python/scripts/evalkit_floor_refs.py'], 'poses': str(POSES),
           'methods': methods, 'run_mapping': old['run_mapping'], 'no_floor_reference': old['no_floor_reference']}
    NEW.write_text(json.dumps(out, indent=1, ensure_ascii=False)+'\n')
    for variant, rec in methods.items():
        for tag, r in ([(f'chois_w{w}', x) for w, x in rec['windows'].items()] if variant == 'chois' else [(variant, rec)]):
            d = {f'{s}_{k}': round(r['reference_m'][s][k]-r['e_v2_reference_m'][s][k], 4) for s in 'lr' for k in LANDMARK_BODY}
            print(f'{tag:22s} new-e_v2 (m) {d}')
    if previous:
        flat = lambda f: {(v, w): r['reference_m'] for v, rec in f['methods'].items()
                          for w, r in (rec['windows'].items() if v == 'chois' else [('', rec)])}
        a, b = flat(previous), flat(out)
        diff = [abs(b[t][s][k]-a[t][s][k]) for t in a if t in b for s in 'lr' for k in LANDMARK_BODY]
        print(json.dumps({'previous_fit_prepare': previous.get('fit_prepare'), 'max_abs_change_vs_previous_m': max(diff)}))


if __name__ == '__main__':
    main()
