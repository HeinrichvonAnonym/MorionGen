#!/usr/bin/env python
"""E v2 rescoring (config/e_v2_protocol_20261106.json): finished E runs of e4 / x3 / m1 / e1 re-evaluated from their
coordinates.mot (no IK) with contact_v2 (each method's floor reference, config/e_v2_floor_reference_20261106.json) under
R0, R4 and R5 with the registered v2 envelope; REBA r3 rescored through a shadow directory per rule.

--set <name> writes develop_logs/test_runs/20261106_e_v2/rescore/<set>/<id>.json (finished runs are skipped) and per-frame
arrays on the data disk. Refuses to run before the protocol is registered (--plumbing: one run, no tau, scratch output).
OpenSim python.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
OUT = ROOT/'develop_logs/test_runs/20261106_e_v2/rescore'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/rescore')
FLOOR = json.loads((ROOT/'config/e_v2_floor_reference_20261106.json').read_text())
PROTOCOL = ROOT/'config/e_v2_protocol_20261106.json'
KEYS = ('id', 'method', 'task', 'seed', 'arm', 'condition', 'offset_m', 'valid')


def chois_window(source):
    scenario = json.loads((Path(source)/'scenario.json').read_text())
    if scenario.get('native_window_index') is not None:
        return str(scenario['native_window_index'])
    log = (Path(source)/'stdout.log').read_text(errors='replace')
    sequences = set(re.findall(r'sub\d+_[a-z]+_\d+', log))
    windows = [w for w, rec in FLOOR['methods']['chois']['windows'].items() if rec['sequence'] in sequences]
    if len(windows) != 1:
        raise ValueError(f'CHOIS window not determined from {source}: {sorted(sequences)}')
    return windows[0]


def floor_reference(set_name, metrics, row):
    """(variant key, reference_m, world_floor_m) by the run_mapping of the floor reference config."""
    method = row['method']
    if method == 'ardy':
        if set_name == 'e4':
            correction = metrics['condition']['correction']
        else:
            correction = json.loads((Path(metrics['run'])/'config.json').read_text())['_intervention']['correction']
        key = 'ardy__correction' if correction else 'ardy'
    elif method == 'kimodo':
        post = metrics['condition']['correction'] if set_name == 'e4' else row['arm'] in ('O', 'TO')
        key = 'kimodo__postprocess' if post else 'kimodo'
    elif method == 'chois':
        w = chois_window(metrics['source'])
        rec = FLOOR['methods']['chois']['windows'][w]
        return f'chois_w{w}', rec['reference_m'], rec['world_floor_m']
    elif method in ('motionlcm', 'omnicontrol'):
        key = 'humanml3d'
    else:
        key = method
    rec = FLOOR['methods'][key]
    return key, rec['reference_m'], rec['world_floor_m']


def reba(frames, bad_at, risk):
    hi = frames['reba_bounds_r3'][:, 1]
    return {'ungated': float(np.mean(hi < risk)), 'gated_envelope': float(np.mean(~bad_at & (hi < risk)))}


def job(args):
    set_name, path, out_root, disk_root, tau = args
    os.nice(19)
    import e_rules
    import ergo_frames
    import run_ergo
    metrics = json.loads(Path(path).read_text())
    out = Path(out_root)/set_name/f"{metrics['id']}.json"
    if out.exists():
        return 'skip'
    row = {k: metrics.get(k) for k in KEYS}
    info = metrics.get('m1') or metrics.get('m2')
    if info:
        row.update(method='ardy', info=info, **{k: info[k] for k in ('arm', 'task', 'offset_m', 'seed')})
    row['method'] = row['method'] or metrics['id'].split('_')[0]
    try:
        work = Path(metrics['work']); mass = float(metrics.get('scaling', {}).get('mass_kg', 75.))
        source = metrics['run'] if info else metrics.get('source')
        key, reference, world_floor = floor_reference(set_name, metrics, row)
        row['floor_reference'] = {'variant': key, 'reference_m': reference, 'world_floor_m': world_floor}
        free = run_ergo.object_free(set_name, metrics)
        risk = ergo_frames.CONV['risk_threshold_REBA']
        times, _, _, kin = e_rules.kinematics(work)
        dt = float(np.median(np.diff(times))); sel = np.zeros(len(times), bool); sel[int(.5/dt):len(times)-int(.5/dt)] = True
        row['rules'] = {}; frames_out = {'times': times, 'sel': sel}
        ref0 = None
        for rule in e_rules.RULES:
            x = e_rules.evaluate(times, kin, mass, rule, reference=reference, world_floor=world_floor)
            ref0 = ref0 or x['masks']
            s = e_rules.summary(x, rule, sel, ref0, tau=tau[rule] if tau else None)
            if not tau:
                s.pop('envelope_infeasible_share')
            st = e_rules.states(x['masks'], x['dt'])
            frames_out.update({f'{rule}_residual_bw': x['residual_bw'], f'{rule}_lowest_elevation': x['lowest_elevation'],
                               **{f'{rule}_state_{k}': m for k, m in st.items()},
                               **{f'{rule}_contact_{sd}_{k}': x['masks'][sd][k] for sd in 'rl' for k in ('heel', 'toe')}})
            if tau:
                bad, _ = e_rules.infeasible(x, rule, tau[rule])
                shadow = e_rules.write_shadow(work, Path(disk_root)/set_name/metrics['id']/rule, times, x)
                ergo_frames.score_run(shadow, source, free, mass, method=row['method'])
                frames = np.load(shadow/'ergo_frames.npz')
                idx = np.clip(np.searchsorted(times, frames['times']-1e-6), 0, len(times)-1)
                s['reba_r3_certain_acceptable'] = reba(frames, bad[idx], risk)
                frames_out[f'{rule}_infeasible'] = bad
            row['rules'][rule] = s
        (Path(disk_root)/set_name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(disk_root)/set_name/f"{metrics['id']}_frames.npz", **frames_out)
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    import run_ergo
    a = argparse.ArgumentParser(); a.add_argument('--set', required=True, choices=sorted(run_ergo.SETS))
    a.add_argument('--workers', type=int, default=4); a.add_argument('--only', nargs='*')
    a.add_argument('--plumbing', action='store_true', help='no tau, scratch output; for testing before registration')
    args = a.parse_args()
    files = {f.parent.name: f for f in run_ergo.SETS[args.set].glob('*/metrics.json')}
    if args.set in run_ergo.OVERRIDES:
        files.update({f.parent.name: f for f in run_ergo.OVERRIDES[args.set].glob('*/metrics.json')})
    files = [files[k] for k in sorted(files)]
    if args.only:
        files = [f for f in files if f.parent.name in args.only]
    if args.plumbing:
        out_root, disk_root, tau = Path('/tmp/e_v2_rescore_plumbing'), Path('/tmp/e_v2_rescore_plumbing/disk'), None
    else:
        if not PROTOCOL.exists():
            raise SystemExit('register config/e_v2_protocol_20261106.json first')
        spec = json.loads(PROTOCOL.read_text())
        tau = spec['tau_envelope_bw']
        if not tau:
            raise SystemExit('protocol has no frozen tau yet')
        out_root, disk_root = OUT, DISK
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(job, [(args.set, str(f), str(out_root), str(disk_root), tau) for f in files]):
            counts[s] = counts.get(s, 0)+1
    print(json.dumps({'set': args.set, 'runs': len(files), **counts}))


if __name__ == '__main__':
    main()
