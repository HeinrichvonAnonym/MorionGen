#!/usr/bin/env python
"""Registered analysis of the AddBiomechanics estimator calibration (config/val_a_addb_20261105.json).

Per variant and class: median and IQR over trials of V1-V4; the gate (walk and run V1, walk/run/jump V3);
tau_frame (99th percentile of per-frame residual, classes weighted equally, sts excluded) and tau_run (95th
percentile of per-trial residual RMS). Writes develop_logs/test_runs/20261105_val_a/analysis.json.
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
SPEC = json.loads((ROOT/'config/val_a_addb_20261105.json').read_text())
THRESHOLD_CLASSES = ('stand', 'walk', 'run', 'squat', 'jump', 'wholebody')
METRICS = ('V1_r', 'V1_rmse_bw', 'V1_bias_bw', 'V2_r', 'V2_rmse_bw', 'V3_balanced_accuracy_20N', 'V3_balanced_accuracy_50N',
           'V4_residual_rms_bw', 'V4_share_above_5pct', 'V4_residual_p95_bw', 'measured_vertical_mean_bw')


def weighted_quantile(values, weights, q):
    order = np.argsort(values); v, w = values[order], weights[order]
    c = np.cumsum(w)/w.sum()
    return float(v[np.searchsorted(c, q)])


def summary(xs):
    xs = np.array([x for x in xs if x is not None], float)
    if not len(xs):
        return None
    return {'median': float(np.median(xs)), 'q25': float(np.percentile(xs, 25)), 'q75': float(np.percentile(xs, 75)), 'n': int(len(xs))}


def variant(name):
    rows = [json.loads(f.read_text()) for f in sorted((LOG/'runs'/name).glob('*.json'))]
    ok = [r for r in rows if r['status'] == 'success']
    by = collections.defaultdict(list)
    for r in ok:
        by[r['class']].append(r)
    classes = {c: {'trials': len(v), **{m: summary([r.get(m) for r in v]) for m in METRICS},
                   'mass_header_over_dynamics': summary([r['mass_header_kg']/r['mass_kg'] for r in v])} for c, v in sorted(by.items())}
    values, weights, rms = [], [], []
    for c in THRESHOLD_CLASSES:
        frames = []
        for r in by.get(c, []):
            f = np.load(LOG/'runs'/name/f"{r['id']}_frames.npz")
            frames.append(f['residual_bw'][f['keep']])
        if frames:
            x = np.concatenate(frames); values.append(x); weights.append(np.full(len(x), 1/len(x)))
            rms += [r['V4_residual_rms_bw'] for r in by[c] if r.get('V4_residual_rms_bw') is not None]
    tau = None
    if values:
        v, w = np.concatenate(values), np.concatenate(weights)
        tau = {'tau_frame_bw': weighted_quantile(v, w, .99), 'tau_run_bw': float(np.percentile(rms, 95)),
               'frame_share_above_5pct_bw_equal_class_weight': float(np.sum(w*(v > .05))/w.sum()),
               'per_class_p99_bw': {c: float(np.percentile(x, 99)) for c, x in zip([c for c in THRESHOLD_CLASSES if by.get(c)], values)}}
    med = lambda c, m: (classes.get(c, {}).get(m) or {}).get('median')
    checks = {f'{c} V1 r >= 0.9': (med(c, 'V1_r') is not None and med(c, 'V1_r') >= .9) for c in ('walk', 'run')}
    checks.update({f'{c} V1 RMSE <= 0.15 BW': (med(c, 'V1_rmse_bw') is not None and med(c, 'V1_rmse_bw') <= .15) for c in ('walk', 'run')})
    checks.update({f'{c} V3 >= 0.85': (med(c, 'V3_balanced_accuracy_20N') is not None and med(c, 'V3_balanced_accuracy_20N') >= .85)
                   for c in ('walk', 'run', 'jump')})
    return {'runs': len(rows), 'success': len(ok), 'failed': [r['id'] for r in rows if r['status'] != 'success'], 'classes': classes,
            'threshold': tau, 'gate': {'checks': checks, 'pass': bool(all(checks.values()))}}


def main():
    result = {'protocol': SPEC['protocol'], 'variants': {}}
    for name in ('kin30', 'dyn30', 'kin_native'):
        if (LOG/'runs'/name).exists():
            result['variants'][name] = variant(name)
    (LOG/'analysis.json').write_text(json.dumps(result, indent=1))
    for name, v in result['variants'].items():
        print(f"== {name}: {v['success']}/{v['runs']} success; gate pass {v['gate']['pass']}", v['gate']['checks'])
        for c, s in v['classes'].items():
            g = lambda m: '   -  ' if s[m] is None else f"{s[m]['median']:6.3f}"
            print(f"  {c:9s} n={s['trials']:2d} V1 r {g('V1_r')} rmse {g('V1_rmse_bw')} bias {g('V1_bias_bw')} | V2 r {g('V2_r')} rmse {g('V2_rmse_bw')}"
                  f" | V3 {g('V3_balanced_accuracy_20N')} | resid rms {g('V4_residual_rms_bw')} >5% {g('V4_share_above_5pct')} p95 {g('V4_residual_p95_bw')}"
                  f" | meas Fy {g('measured_vertical_mean_bw')}")
        print('  threshold', json.dumps(v['threshold']))


if __name__ == '__main__':
    main()
