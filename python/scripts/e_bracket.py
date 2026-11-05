#!/usr/bin/env python
"""Rule bracket (config/e_criterion_domain_20261105.json, 'bracket'): finished E runs of e4 / x3 / m1 / e1 re-evaluated
under R0, R4 and R5 from their coordinates.mot (no inverse dynamics); R4 and R5 also rescored for REBA r3 through a
shadow directory (links to model.osim and coordinates.mot, a dynamics.npz of that rule).

--set <name> writes develop_logs/test_runs/20261105_e_domain/bracket/<set>/<id>.json; finished runs are skipped.
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
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
OUT = ROOT/'develop_logs/test_runs/20261105_e_domain/bracket'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_domain_20261105/bracket')
ERGO = ROOT/'develop_logs/test_runs/20261105_ergo'
KEYS = ('id', 'method', 'task', 'seed', 'arm', 'condition', 'offset_m', 'valid')


def reba(frames, bad_at, risk):
    hi = frames['reba_bounds_r3'][:, 1]
    return {'ungated': float(np.mean(hi < risk)), 'gated_fixed': float(np.mean(frames['feasible'] & (hi < risk))),
            'gated_envelope': float(np.mean(~bad_at & (hi < risk)))}


def job(args):
    set_name, path = args
    os.nice(19)
    import e_rules
    import ergo_frames
    import run_ergo
    metrics = json.loads(Path(path).read_text())
    out = OUT/set_name/f"{metrics['id']}.json"
    if out.exists():
        return 'skip'
    row = {k: metrics.get(k) for k in KEYS}
    info = metrics.get('m1') or metrics.get('m2')
    if info:
        row.update(method='ardy', info=info, **{k: info[k] for k in ('arm', 'task', 'offset_m', 'seed')})
    try:
        work = Path(metrics['work']); mass = float(metrics.get('scaling', {}).get('mass_kg', 75.))
        source = metrics['run'] if info else metrics.get('source')
        method = row['method'] or metrics['id'].split('_')[0]
        free = run_ergo.object_free(set_name, metrics)
        risk = ergo_frames.CONV['risk_threshold_REBA']
        times, _, _, kin = e_rules.kinematics(work)
        dt = float(np.median(np.diff(times))); sel = np.zeros(len(times), bool); sel[int(.5/dt):len(times)-int(.5/dt)] = True
        res = {r: e_rules.evaluate(times, kin, mass, r) for r in e_rules.RULES}
        row['rules'] = {}
        for rule, x in res.items():
            s = e_rules.summary(x, rule, sel, res['R0']['masks'])
            bad, _ = e_rules.infeasible(x, rule)
            if rule == 'R0':
                frames = np.load(work/'ergo_frames.npz')
                s['r0_matches_original'] = float(abs(s['residual_force_rms_bw']-metrics['metrics']['residual_force_rms_bw']))
            else:
                shadow = e_rules.write_shadow(work, DISK/set_name/metrics['id']/rule, times, x)
                ergo_frames.score_run(shadow, source, free, mass, method=method)
                frames = np.load(shadow/'ergo_frames.npz')
            idx = np.clip(np.searchsorted(times, frames['times']-1e-6), 0, len(times)-1)
            s['reba_r3_certain_acceptable'] = reba(frames, bad[idx], risk)
            row['rules'][rule] = s
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    import run_ergo
    a = argparse.ArgumentParser(); a.add_argument('--set', required=True, choices=sorted(run_ergo.SETS))
    a.add_argument('--workers', type=int, default=6); a.add_argument('--only', nargs='*')
    args = a.parse_args()
    (OUT/args.set).mkdir(parents=True, exist_ok=True)
    files = {f.parent.name: f for f in run_ergo.SETS[args.set].glob('*/metrics.json')}
    if args.set in run_ergo.OVERRIDES:
        files.update({f.parent.name: f for f in run_ergo.OVERRIDES[args.set].glob('*/metrics.json')})
    files = [files[k] for k in sorted(files)]
    if args.only:
        files = [f for f in files if f.parent.name in args.only]
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(job, [(args.set, str(f)) for f in files]):
            counts[s] = counts.get(s, 0)+1
    print(json.dumps({'set': args.set, 'runs': len(files), **counts}))


if __name__ == '__main__':
    main()
