#!/usr/bin/env python
"""Analysis of P1 (config/p1_20261103.json): sampling-layer guidance against the P0 arms of the same runs."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_next_round as nr
from analyze_p0 import METRICS, rows
from motiogen.native_analysis import holm
from motiogen.runtime import json_write
from plan_p0 import OUT as P0_OUT
from plan_p1 import OUT, SPEC_PATH

COMPARATORS = ['G0', 'G4', 'W0g', 'G4W']
EQUIVALENCE = .05


def mechanism(p1_rows):
    """x_t / clean-prediction gradient ratio (S0) and per-step erasure of the clean-prediction edit (X0, V0, X4)."""
    ratios, erasure = [], {}
    for r in p1_rows:
        task = json.loads((Path(r['run'])/'intervention.json').read_text())['task']
        trace = task['sampling_guidance'].get('first_window_trace', [])
        if r['arm'] == 'S0':
            ratios += [s['xt_gradient_norm']/s['clean_gradient_norm'] for s in trace if s['clean_gradient_norm'] > 0]
        elif trace:
            free = trace[0]['loss_m2']
            if free > 1e-8:
                erasure.setdefault(r['arm'], []).append([s['loss_m2']/free for s in trace])
    return {'S0_gradient_ratio': {'steps': len(ratios), 'median': float(np.median(ratios)) if ratios else None,
                                  'p95': float(np.percentile(ratios, 95)) if ratios else None,
                                  'max': float(np.max(ratios)) if ratios else None},
            'pre_edit_error_ratio': {a: {'runs': len(v), 'per_step_median': np.median(v, axis=0).tolist(),
                                         'last_step_min': float(np.min(np.array(v)[:, -1]))} for a, v in erasure.items()},
            'scope': 'first guided window of each run; erasure is the squared goal error of the clean prediction before '
                     'each edit relative to the first step (1 = edit erased); X4 starts near zero error, so few X4 runs qualify'}


def main():
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    offsets = spec['offsets_m']
    p1_rows = rows(OUT)
    table = {(r['arm'], r['offset_m'], r['task'], r['seed']): r['values'] for r in p1_rows}
    table.update({(r['arm'], r['offset_m'], r['task'], r['seed']): r['values'] for r in rows(P0_OUT) if r['arm'] in COMPARATORS})
    arms = list(spec['arms'])+COMPARATORS

    def paired(on, off, chosen, metric, shift=0.):
        deltas, clusters = [], []
        for (a, o, task, seed), values in sorted(table.items()):
            base = table.get((off, o, task, seed))
            if a == on and o in chosen and base is not None and metric in values and metric in base:
                deltas.append(values[metric]-base[metric]-shift); clusters.append(task)
        return deltas, clusters

    def cell(arm, offset, metric):
        by = {}
        for (a, o, task, _), values in table.items():
            if a == arm and o == offset and metric in values:
                by.setdefault(task, []).append(values[metric])
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    result = {'p1_runs': len(p1_rows), 'expected_p1_runs': spec['runs'], 'runs_with_comparators': len(table)}
    result['cells'] = {f'{a}|d{o:g}': {m: cell(a, o, m) for m in METRICS} for a in arms for o in offsets}
    upper = nr.test(*paired('S0', 'G0', offsets, 'goal_error_m', EQUIVALENCE))
    lower = nr.test(*paired('S0', 'G0', offsets, 'goal_error_m', -EQUIVALENCE))
    l1 = nr.test(*paired('S0', 'G0', offsets, 'goal_error_m'))
    l1.update(margin=EQUIVALENCE, upper_shifted=upper, lower_shifted=lower,
              sign_flip_p=max(upper['sign_flip_p'], lower['sign_flip_p']))
    g4 = [v['jerk_0p1s'] for (a, *_), v in table.items() if a == 'G4' and 'jerk_0p1s' in v]
    margin = .1*float(np.mean(g4)) if g4 else 0.
    l4 = nr.test(*paired('X4', 'G4', offsets, 'jerk_0p1s', margin))
    l4.update(margin=margin, mean_difference=l4['mean']+margin)
    hypotheses = {'L1': l1, 'L2': nr.test(*paired('X0', 'G0', offsets, 'jerk_0p1s')),
                  'L3': nr.test(*paired('X0', 'W0g', offsets, 'jerk_0p1s')), 'L4': l4}
    side = {'L1': lambda h: -EQUIVALENCE < h['ci95'][0] and h['ci95'][1] < EQUIVALENCE,
            'L2': lambda h: h['ci95'][0] > 0, 'L3': lambda h: h['ci95'][0] > 0 or h['ci95'][1] < 0,
            'L4': lambda h: h['ci95'][1] < 0}
    for key, p in zip(hypotheses, holm([h['sign_flip_p'] for h in hypotheses.values()])):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and side[key](hypotheses[key]))
    accurate = {str(o): (result['cells'][f'X0|d{o:g}']['goal_error_m'] or 1.) <= .05 for o in offsets}
    hypotheses['L2']['X0_accurate_per_offset'] = accurate
    hypotheses['L2']['supported'] = bool(hypotheses['L2']['supported'] and all(accurate.values()))
    result['primary'] = hypotheses
    result['secondary'] = {f'{on}-{off}': {'pooled': {m: nr.test(*paired(on, off, offsets, m)) for m in METRICS},
                                           'per_offset': {str(o): {m: nr.test(*paired(on, off, [o], m)) for m in METRICS} for o in offsets}}
                           for on, off in [('S0', 'G0'), ('X0', 'G0'), ('V0', 'G0'), ('V0', 'X0'), ('X0', 'W0g'), ('V0', 'W0g'),
                                           ('X4', 'G4'), ('X4', 'G4W')]}
    result['mechanism'] = mechanism(p1_rows)
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'p1_runs': result['p1_runs'], 'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')}
                                                                for k, v in hypotheses.items()}, 'mechanism': result['mechanism']['S0_gradient_ratio']}))


if __name__ == '__main__':
    main()
