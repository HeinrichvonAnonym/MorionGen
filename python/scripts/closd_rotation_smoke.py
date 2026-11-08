#!/usr/bin/env python
"""CLoSD rigid-body orientation smoke (design notes section 14 item 3; contract method_capability.closd).

  closd_rotation_smoke.py generate [--gpu 0]   one CLoSD run with the current code under the generation GPU lock
  closd_rotation_smoke.py check RUN [RUN ...]  native_physics.npz: rotations_xyzw present, shapes, quaternion norms, and
                                               bone vectors R_parent^T (p_child - p_parent) constant over frames

The run uses the runtime-reveal launcher with the arguments of the 10-29 supplement runs (paused clock, seed 1, the
same goal event, goal ignored); nothing of the generation queues is touched.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
OUT = ROOT/'develop_logs/test_runs/20261109_closd_rotation_smoke'
EVENTS = ROOT/'develop_logs/test_runs/20261028_supplement/closd_matrix/event_3.json'
PARENT = {'L_Hip': 'Pelvis', 'L_Knee': 'L_Hip', 'L_Ankle': 'L_Knee', 'L_Toe': 'L_Ankle',
          'R_Hip': 'Pelvis', 'R_Knee': 'R_Hip', 'R_Ankle': 'R_Knee', 'R_Toe': 'R_Ankle',
          'Torso': 'Pelvis', 'Spine': 'Torso', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
          'L_Thorax': 'Chest', 'L_Shoulder': 'L_Thorax', 'L_Elbow': 'L_Shoulder', 'L_Wrist': 'L_Elbow', 'L_Hand': 'L_Wrist',
          'R_Thorax': 'Chest', 'R_Shoulder': 'R_Thorax', 'R_Elbow': 'R_Shoulder', 'R_Wrist': 'R_Elbow', 'R_Hand': 'R_Wrist'}


def generate(args):
    from motiogen.config import data_root, load_config
    from motiogen.runtime import gpu_lock
    cfg = load_config()
    python = data_root(cfg)/'envs/motiogen-closd/bin/python'
    run = OUT/f"{time.strftime('%Y%m%d_%H%M%S')}_closd_runtime_reveal_smoke"
    command = [str(python), str(ROOT/'python/scripts/run_simulation_runtime_reveal.py'), '--run-dir', str(run),
               '--events', str(EVENTS), '--seed', '1', '--clock', 'paused', '--ignore']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu)}
    t0 = time.time()
    with gpu_lock({**cfg, 'gpu_index': str(args.gpu)}):
        waited = time.time()-t0
        code = subprocess.call(command, env=env, stdout=open(OUT/'generate.log', 'w'), stderr=subprocess.STDOUT)
    rec = {'run': str(run.relative_to(ROOT)), 'command': command[1:], 'gpu': args.gpu, 'lock_wait_s': round(waited, 1),
           'wall_s': round(time.time()-t0-waited, 1), 'exit_code': code}
    (OUT/'generate.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def quat_xyzw_to_matrix(q):
    x, y, z, w = np.moveaxis(q/np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)


def check_run(run):
    run = Path(run); f = run/'native_physics.npz'
    out = {'run': str(run.relative_to(ROOT) if run.is_absolute() else run), 'native_physics': f.exists()}
    if not f.exists():
        return out
    m = np.load(f)
    out['arrays'] = {k: list(m[k].shape) for k in m.files}
    if 'rotations_xyzw' not in m.files:
        return {**out, 'rotations_xyzw': False}
    q, p, names = m['rotations_xyzw'], m['joints'], [str(n) for n in m['body_names']]
    norm = np.linalg.norm(q, axis=-1)
    R = quat_xyzw_to_matrix(q)
    bones = {}
    for child, parent in PARENT.items():
        if child not in names or parent not in names:
            continue
        c, a = names.index(child), names.index(parent)
        v = np.einsum('tji,tj->ti', R[:, a], p[:, c]-p[:, a])
        bones[f'{parent}-{child}'] = {'length_m': float(np.linalg.norm(v, axis=-1).mean()),
                                      'max_dev_from_median_mm': float(1e3*np.linalg.norm(v-np.median(v, 0), axis=-1).max())}
    worst = max(bones.items(), key=lambda kv: kv[1]['max_dev_from_median_mm'])
    out.update(rotations_xyzw=True, frames=int(q.shape[0]), bodies=len(names),
               quat_norm_min=float(norm.min()), quat_norm_max=float(norm.max()), bones_checked=len(bones),
               bone_vector_max_dev_mm=worst[1]['max_dev_from_median_mm'], worst_bone=worst[0],
               bone_vector_median_max_dev_mm=float(np.median([b['max_dev_from_median_mm'] for b in bones.values()])),
               finite=bool(np.isfinite(q).all() and np.isfinite(p).all()), bones=bones)
    return out


def reproducibility(runs):
    """pairwise array equality of runs with rotations; contact force spread of the summed vertical force."""
    files = {r: np.load(Path(r)/'native_physics.npz') for r in runs if (Path(r)/'native_physics.npz').exists()}
    files = {r: m for r, m in files.items() if 'rotations_xyzw' in m.files}
    out = []
    keys = list(files)
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = files[keys[i]], files[keys[j]]
            fa, fb = a['contact_forces'][..., 2].sum(1), b['contact_forces'][..., 2].sum(1)
            d = np.abs(fa-fb)
            w = lambda x: np.convolve(x, np.ones(15)/15, 'valid')
            out.append({'runs': [keys[i], keys[j]], 'equal': {k: bool(np.array_equal(a[k], b[k])) for k in a.files},
                        'vertical_force_abs_diff_N': {'median': float(np.median(d)), 'p95': float(np.percentile(d, 95)), 'max': float(d.max()),
                                                      'window_0p5s_max': float(np.abs(w(fa)-w(fb)).max()),
                                                      'run_mean': [float(fa.mean()), float(fb.mean())]}})
    return out


def check(args):
    res = [check_run(r) for r in args.runs]
    rep = reproducibility(args.runs)
    (OUT/'check.json').write_text(json.dumps({'runs': res, 'reproducibility': rep}, indent=1))
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != 'bones'}))
    for r in rep:
        print(json.dumps(r))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='step', required=True)
    g = sub.add_parser('generate'); g.add_argument('--gpu', default='0')
    c = sub.add_parser('check'); c.add_argument('runs', nargs='+')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'generate': generate, 'check': check}[args.step](args)


if __name__ == '__main__':
    main()
