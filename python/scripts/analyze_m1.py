#!/usr/bin/env python
"""Analysis of M1 (G1), M1-Pareto and V3-ARDY (config/m1_20261031.json).

Inputs are the pilot and batch queue receipts; pilot runs reused by the batch
enter once. Clusters are tasks; seeds and offsets are paired repetitions.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import ardy_row, test
from motiogen.config import ROOT
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261031_m1'
M1_ARMS = ['C0', 'C1', 'C2', 'C3a', 'C3b', 'C5a', 'C5b', 'XFh']
METRICS = ['end_waypoint_error_m', 'waypoint_error_m', 'applied_root_displacement_m', 'boundary_jerk_0p1s', 'jerk_0p1s',
           'boundary_jerk_native', 'trunk_inclination', 'right_upper_arm_elevation', 'foot_sliding_5cm_10cms',
           'history_consistency_32_m', 'success_5cm', 'wall_seconds']
G1_OFFSETS = [.1, .25, .5]
EPSILON = .05


def load_rows():
    found = {}
    for stage in ('pilot', 'batch', 'continuation'):
        plan_path, state_path = OUT/f'{stage}_plan.json', OUT/f'{stage}_queue/state.json'
        if not plan_path.exists() or not state_path.exists():
            continue
        plan = json.loads(plan_path.read_text()); state = json.loads(state_path.read_text())['jobs']
        for row in plan.get('superseded', []):
            found.pop(row['id'], None)
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') == 'success':
                found[job['id']] = (job, Path(row['run_dir']), row['elapsed_seconds'])
    rows = []
    for job_id, (job, run, elapsed) in sorted(found.items()):
        values = ardy_row(job, run); values['wall_seconds'] = float(elapsed)
        info = job['m1']
        rows.append({'id': job_id, 'block': info['block'], 'arm': info['arm'], 'epsilon': info['epsilon_m'],
                     'offset': info['offset_m'], 'task': info['task'], 'seed': job['seed'], 'values': values})
    return rows


def index(rows):
    table = {}
    for r in rows:
        table[(r['arm'], r['epsilon'], r['offset'], r['task'], r['seed'])] = r['values']
    return table


def cell(table, arm, epsilon, offset, metric):
    by_task = collections.defaultdict(list)
    for (a, e, o, task, _), values in table.items():
        if a == arm and e == epsilon and o == offset and metric in values:
            by_task[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by_task.values()])) if by_task else None


def paired(table, on, off, offsets, metric, epsilon=EPSILON, off_epsilon=None):
    """on/off: arm name or dict offset -> arm name. One difference per task-seed-offset."""
    deltas, clusters = [], []
    for (a, e, o, task, seed), values in sorted(table.items()):
        if o not in offsets:
            continue
        arm_on = on[o] if isinstance(on, dict) else on
        if a != arm_on or e != epsilon or metric not in values:
            continue
        arm_off = off[o] if isinstance(off, dict) else off
        base = table.get((arm_off, EPSILON if arm_off == 'C0' else (off_epsilon or epsilon), o, task, seed))
        if base is not None and metric in base:
            deltas.append(values[metric]-base[metric]); clusters.append(task)
    return deltas, clusters


def g1(table, offsets):
    choice = {}
    for o in offsets:
        cells = {a: {'end_waypoint_error_m': cell(table, a, EPSILON, o, 'end_waypoint_error_m'),
                     'boundary_jerk_0p1s': cell(table, a, EPSILON, o, 'boundary_jerk_0p1s')} for a in M1_ARMS}
        accurate = sorted([a for a in M1_ARMS if cells[a]['end_waypoint_error_m'] is not None
                           and cells[a]['end_waypoint_error_m'] <= EPSILON], key=lambda a: cells[a]['boundary_jerk_0p1s'])
        row = {'cells': cells, 'accurate_arms': accurate, 'best': accurate[0] if accurate else None}
        if len(accurate) > 1:
            row['runner_up'] = accurate[1]
            row['best_minus_runner_up_boundary_jerk'] = test(*paired(table, accurate[0], accurate[1], [o], 'boundary_jerk_0p1s'))
        choice[str(o)] = row
    result = {'per_offset': choice}
    best = {o: choice[str(o)]['best'] for o in G1_OFFSETS if choice[str(o)]['best']}
    no_accurate = [o for o in offsets if o <= .5 and choice[str(o)]['best'] is None]
    result['offsets_without_accurate_arm'] = no_accurate
    if len(best) == len(G1_OFFSETS):
        primary = test(*paired(table, best, 'C0', G1_OFFSETS, 'boundary_jerk_0p1s'))
        base = np.mean([cell(table, 'C0', EPSILON, o, 'boundary_jerk_0p1s') for o in G1_OFFSETS])
        primary.update(c0_mean=float(base), relative_increase=primary['mean']/float(base), best_arm_per_offset=best)
        primary['cost_remains'] = bool(primary.get('excludes_zero') and primary['ci95'][0] > 0 and primary['relative_increase'] >= .2)
        result['primary'] = primary
    result['passed'] = bool(no_accurate) or bool(result.get('primary', {}).get('cost_remains'))
    winners = {choice[str(o)]['best'] for o in offsets}
    result['single_arm_best_at_every_offset'] = len(winners) == 1 and None not in winners
    return result


def v3(table):
    pairs = {'history': {'inertialization': 'C2', 'crossfade': 'XFh', 'lowpass': 'LPh'},
             'output_only': {'inertialization': 'C2o', 'crossfade': 'XFo', 'lowpass': 'LPo'}}
    offsets = [.25, 1.]
    result = {}
    for mode, arms in pairs.items():
        effect = {}
        for name, arm in arms.items():
            reduction = cell_mean_diff(table, arm, 'C1', offsets, 'boundary_jerk_0p1s')
            effect[name] = {
                'boundary_jerk_reduction_vs_C1': None if reduction is None else -reduction,
                'end_error_regression_vs_C1': cell_mean_diff(table, arm, 'C1', offsets, 'end_waypoint_error_m'),
                'boundary_jerk_minus_C1': test(*paired(table, arm, 'C1', offsets, 'boundary_jerk_0p1s')),
                'end_error_minus_C1': test(*paired(table, arm, 'C1', offsets, 'end_waypoint_error_m')),
                'history_consistency_minus_C1': test(*paired(table, arm, 'C1', offsets, 'history_consistency_32_m')),
                'per_offset': {str(o): {m: cell(table, arm, EPSILON, o, m) for m in ('boundary_jerk_0p1s', 'end_waypoint_error_m', 'jerk_0p1s')}
                               for o in offsets}}
        inert = effect['inertialization']
        verdict = {}
        if any(e[k] is None for e in effect.values() for k in ('boundary_jerk_reduction_vs_C1', 'end_error_regression_vs_C1')):
            result[mode] = {'effects': effect, 'stop_rule': 'incomplete'}
            continue
        for name in ('crossfade', 'lowpass'):
            e = effect[name]
            jerk_ok = abs(e['boundary_jerk_reduction_vs_C1']-inert['boundary_jerk_reduction_vs_C1']) <= .1*abs(inert['boundary_jerk_reduction_vs_C1'])
            error_ok = abs(e['end_error_regression_vs_C1']-inert['end_error_regression_vs_C1']) <= .1*abs(inert['end_error_regression_vs_C1'])
            verdict[name] = {'jerk_within_10pct': bool(jerk_ok), 'error_within_10pct': bool(error_ok),
                             'minus_inertialization_boundary_jerk': test(*paired(table, arms[name], arms['inertialization'], offsets, 'boundary_jerk_0p1s')),
                             'minus_inertialization_end_error': test(*paired(table, arms[name], arms['inertialization'], offsets, 'end_waypoint_error_m'))}
        generic = any(v['jerk_within_10pct'] and v['error_within_10pct'] for v in verdict.values())
        result[mode] = {'effects': effect, 'generic_vs_inertialization': verdict,
                        'stop_rule': 'generic' if generic else 'tuned'}
    return result


def cell_mean_diff(table, arm, base, offsets, metric):
    deltas, clusters = paired(table, arm, base, offsets, metric)
    by = collections.defaultdict(list)
    for d, c in zip(deltas, clusters):
        by[c].append(d)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def main():
    rows = load_rows()
    table = index(rows)
    offsets = sorted({r['offset'] for r in rows if r['block'] == 'm1'})
    result = {'runs': dict(collections.Counter(r['block'] for r in rows)), 'expected_runs': {'m1': 3456, 'par': 1728, 'v3': 576}}
    result['cells'] = {f'{a}|e{e:g}|d{o:g}': {m: cell(table, a, e, o, m) for m in METRICS}
                       for a, e, o in sorted({(r['arm'], r['epsilon'], r['offset']) for r in rows})}
    result['G1'] = g1(table, offsets)
    result['arms_vs_C0'] = {a: {str(o): {m: test(*paired(table, a, 'C0', [o], m)) for m in ('boundary_jerk_0p1s', 'end_waypoint_error_m')}
                                for o in offsets} for a in M1_ARMS if a != 'C0'}
    result['arms_vs_C1_pooled'] = {a: {m: test(*paired(table, a, 'C1', offsets, m)) for m in METRICS if m != 'success_5cm'}
                                   for a in M1_ARMS if a != 'C1'}
    pareto = {}
    for arm in ('C1', 'C2', 'C5a', 'XFh'):
        for e in (.02, .05, .1):
            for o in (.1, .5, 1.):
                pareto[f'{arm}|e{e:g}|d{o:g}'] = {m: cell(table, arm, e, o, m) for m in ('end_waypoint_error_m', 'boundary_jerk_0p1s', 'jerk_0p1s')}
    result['pareto'] = pareto
    result['V3'] = v3(table)
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'G1_passed': result['G1']['passed'],
                      'best': {o: v['best'] for o, v in result['G1']['per_offset'].items()},
                      'primary': {k: result['G1'].get('primary', {}).get(k) for k in ('mean', 'ci95', 'relative_increase', 'sign_flip_p')},
                      'V3': {m: v['stop_rule'] for m, v in result['V3'].items()}}))


if __name__ == '__main__':
    main()
