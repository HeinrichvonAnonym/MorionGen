#!/usr/bin/env python
"""Experiment-1 jerk panels on the common 0.1 s step (jerk_0p1s, boundary_jerk_0p1s), from the archived paused-clock runs.

The archived native matrix stores per-frame jerk only. This recomputes the same panels with the step used by every later
experiment; pairing, equal task weighting and the clustered bootstrap are those of the archive, which the per-frame
reproduction check confirms before anything is written.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.native_analysis import clustered_interval, common_step_boundary_jerk, common_step_jerk
from motiogen.runtime import json_write

SRC = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
OUT = ROOT/'develop_logs/test_runs/20261103_native_step_jerk'
METHODS = ('motionlcm', 'omnicontrol', 'chois', 'ardy', 'primal')
PANELS = [('motionlcm', 'control_residual', None), ('omnicontrol', 'spatial', None), ('omnicontrol', 'realism', None),
          ('omnicontrol', 'realism', {'spatial': 0}), ('omnicontrol', 'realism', {'spatial': 1}), ('chois', 'guidance', None),
          ('ardy', 'correction', None), ('primal', 'inertialization', None), ('primal', 'reprojection', None), ('primal', 'ground_snap', None)]
NATIVE = {'jerk_0p1s': 'jerk_rms', 'boundary_jerk_0p1s': 'boundary_jerk_rms'}


def motion(row):
    if row['method'] == 'primal':
        c = row['condition']
        path = ROOT/row['run']/'factorial'/f'ground{c["ground_snap"]}_reproject{c["reprojection"]}_inertialize{c["inertialization"]}_canonical.npz'
    else:
        path = ROOT/row['run']/'motion.npz'
    return np.load(path)


def values(row):
    m = motion(row); joints = np.asarray(m['joints'], float); fps = float(m['fps'])
    out = {'jerk_0p1s': common_step_jerk(joints, fps)}
    if row['method'] == 'ardy':
        out['boundary_jerk_0p1s'] = common_step_boundary_jerk(joints, fps, np.asarray(m['chunk_boundaries'], int))
    elif row['method'] == 'primal':
        out['boundary_jerk_0p1s'] = common_step_boundary_jerk(joints, fps, np.arange(14, len(joints), 14))
    return out


def level(rows, metric, fixed):
    t = collections.defaultdict(list)
    for r in rows:
        if all(r['condition'].get(k) == v for k, v in fixed.items()) and r['v'].get(metric) is not None:
            t[str(r['task'])].append(r['v'][metric])
    return float(np.mean([np.mean(v) for v in t.values()]))


def paired(rows, metric, factor, fixed=None, interaction=False):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['task'], r['seed']].append(r)
    vals, tasks = [], []
    for (task, seed), rs in groups.items():
        sel = [r for r in rs if all(r['condition'][k] == v for k, v in (fixed or {}).items())]
        if interaction:
            cell = {(r['condition']['realism'], r['condition']['spatial']): r['v'].get(metric) for r in sel}
            if len(cell) != 4 or None in cell.values():
                continue
            vals.append(cell[1, 1]-cell[0, 1]-cell[1, 0]+cell[0, 0]); tasks.append(task); continue
        on = [r['v'].get(metric) for r in sel if r['condition'][factor] == 1]
        off = [r['v'].get(metric) for r in sel if r['condition'][factor] == 0]
        if not on or len(on) != len(off) or any(v is None for v in on+off):
            continue
        vals.append(np.mean(on)-np.mean(off)); tasks.append(task)
    ci = clustered_interval(vals, tasks)
    return {'v': ci['mean_equal_group_weight'], 'lo': ci['ci95'][0], 'hi': ci['ci95'][1], 'n': len(vals), 'g': len(set(tasks))}


def panel(rows, method, factor, fixed, metric):
    base = dict(fixed or {})
    return {'method': method, 'factor': factor, 'fixed': fixed, 'metric': metric,
            'levels': [level(rows, metric, {**base, factor: 0}), level(rows, metric, {**base, factor: 1})],
            'delta': paired(rows, metric, factor, fixed)}


def main():
    archive = json.loads(SRC.read_text())
    rows = [r for r in archive['rows'] if r['clock'] == 'paused' and r['method'] in METHODS]
    for r in rows:
        r['v'] = {**{k: r['values'].get(k) for k in NATIVE.values()}, **values(r)}
    effects = {(e['method'], tuple(e['factors']), e['metric']): e for e in archive.get('effects', []) if e.get('clock') == 'paused'}
    check = []
    for method, factor, fixed in PANELS:
        for metric in ('jerk_rms', 'boundary_jerk_rms'):
            if fixed is None and (method, (factor,), metric) in effects:
                ours = paired([r for r in rows if r['method'] == method], metric, factor)
                arch = effects[method, (factor,), metric]
                check.append({'method': method, 'factor': factor, 'metric': metric, 'ours': ours['v'], 'archive': arch['mean_equal_group_weight'],
                              'reproduced': abs(ours['v']-arch['mean_equal_group_weight']) < 1e-8})
    if not check or not all(c['reproduced'] for c in check):
        raise SystemExit(json.dumps(check, indent=1))
    panels = []
    for method, factor, fixed in PANELS:
        mr = [r for r in rows if r['method'] == method]
        for metric in (('jerk_0p1s', 'boundary_jerk_0p1s') if method in ('ardy', 'primal') else ('jerk_0p1s',)):
            panels.append(panel(mr, method, factor, fixed, metric))
    om = [r for r in rows if r['method'] == 'omnicontrol']
    inter = {'method': 'omnicontrol', 'factor': 'realism x spatial', 'metric': 'jerk_0p1s', 'delta': paired(om, 'jerk_0p1s', None, interaction=True)}
    OUT.mkdir(parents=True, exist_ok=True)
    json_write(OUT/'panels.json', {'source': str(SRC.relative_to(ROOT)), 'step_seconds': .1, 'reproduction_check': check,
                                   'panels': panels, 'interaction': inter,
                                   'rows': [{'method': r['method'], 'task': r['task'], 'seed': r['seed'], 'condition': r['condition'],
                                             **{k: r['v'][k] for k in r['v']}} for r in rows]})
    print(json.dumps({'checked': len(check), 'panels': [(p['method'], p['factor'], p['fixed'], p['metric'], [round(x, 2) for x in p['levels']],
                                                         round(p['delta']['v'], 2), [round(p['delta']['lo'], 2), round(p['delta']['hi'], 2)]) for p in panels],
                      'interaction': inter['delta']}, indent=0))


if __name__ == '__main__':
    main()
