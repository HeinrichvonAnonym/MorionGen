#!/usr/bin/env python
"""Pilot audit (--audit) and analysis of M2: output-only correction with output-only crossfade (config/m2_20261101.json)."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import ardy_row, test
from audit_next_round_pilot import heading_error
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.native_analysis import holm
from motiogen.runtime import json_write
from plan_m2 import OUT, SPEC_PATH

SPEC = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
ARMS = list(SPEC['arms']); OFFSETS = SPEC['offsets_m']; EPSILON = SPEC['epsilon_m']
MARGIN = .05
CORRECTED = [a for a in ARMS if a != 'C0']
CROSSFADED = [a for a, v in SPEC['arms'].items() if v.get('boundary') == 'crossfade']
METRICS = ['end_waypoint_error_m', 'boundary_jerk_0p1s', 'jerk_0p1s', 'boundary_jerk_native', 'trunk_inclination',
           'right_upper_arm_elevation', 'foot_sliding_5cm_10cms', 'history_consistency_32_m', 'applied_root_displacement_m', 'wall_seconds']


def audit():
    plan = json.loads((OUT/'pilot_plan.json').read_text()); state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
    checks = []; heads = collections.defaultdict(dict)

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {}); info = job['m2']; arm = info['arm']
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason')); continue
        run = Path(row['run_dir']); motion = load_motion(run/'motion.npz'); joints = np.asarray(motion['joints'])
        values = ardy_row(job, run)
        check('finite_outputs', np.isfinite(joints).all() and all(np.isfinite(v) for v in values.values()), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], job=job['id'], elapsed=row['elapsed_seconds'])
        check('frame0_heading_body_forward', heading_error(values['frame0_heading_deg']) <= 20., job=job['id'], heading=values['frame0_heading_deg'])
        intervention = json.loads((run/'intervention.json').read_text()); task = intervention['task']; calls = intervention['calls']
        windows = len(task['windows'])
        check('constraint_guidance_as_registered', intervention.get('observed_denoiser_guidance') == [2., 2.], job=job['id'])
        check('reference_pass_executed', calls.get('native_reference_pass', 0) == windows, job=job['id'])
        registered = SPEC['arms'][arm]
        check('correction_as_registered', calls.get('native_motion_correction', 0) == (windows if arm in CORRECTED else 0), job=job['id'])
        check('crossfade_as_registered', calls.get('native_boundary_crossfade', 0) == (windows-1 if arm in CROSSFADED else 0), job=job['id'])
        check('writeback_receipt', task['correction_writeback'] == registered.get('correction_writeback', True)
              and task['boundary'] == registered.get('boundary', 'none')
              and (arm not in CROSSFADED or task['boundary_writeback'] == registered['boundary_writeback'])
              and not task.get('boundary_repin') and not task.get('boundary_prefix_frames'), job=job['id'])
        if arm in CORRECTED:
            check('corrected_end_within_margin', values['end_waypoint_error_m'] <= EPSILON+.005, job=job['id'], end_error=values['end_waypoint_error_m'])
            first = int(min(b for b in np.asarray(motion['chunk_boundaries'], int) if b > 0))
            heads[info['offset_m']][arm] = joints[:first]
    for offset, arms in heads.items():
        reference = arms.get('W0')
        for arm, head in arms.items():
            check('first_window_identical_to_W0', reference is not None and head.shape == reference.shape and np.array_equal(head, reference),
                  offset=offset, arm=arm)
    passed = bool(checks) and all(c['passed'] for c in checks)
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']], 'checks': len(checks),
                                        'criteria': SPEC['pilot']['admission']})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']][:10]}))


def load_table():
    found = {}
    for stage in ('pilot', 'batch', *sorted(p.name[:-len('_plan.json')] for p in OUT.glob('continuation*_plan.json'))):
        if not (OUT/f'{stage}_queue/state.json').exists():
            continue
        plan = json.loads((OUT/f'{stage}_plan.json').read_text()); state = json.loads((OUT/f'{stage}_queue/state.json').read_text())['jobs']
        for row in plan.get('superseded', []):
            found.pop(row['id'], None)
        for item in plan.get('reused', []):
            found[item['id']] = (item['job'], Path(item['run']), None)
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') == 'success':
                found[job['id']] = (job, Path(row['run_dir']), row['elapsed_seconds'])
    table = {}
    for job, run, elapsed in found.values():
        values = ardy_row(job, run)
        if elapsed is not None:
            values['wall_seconds'] = float(elapsed)
        info = job['m2']
        table[(info['arm'], info['offset_m'], info['task'], job['seed'])] = values
    return table


def paired(table, on, off, offsets, metric, shift=0.):
    deltas, clusters = [], []
    for (a, o, task, seed), values in sorted(table.items()):
        base = table.get((off, o, task, seed))
        if a == on and o in offsets and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]-shift); clusters.append(task)
    return deltas, clusters


def cell(table, arm, offset, metric):
    by = collections.defaultdict(list)
    for (a, o, task, _), values in table.items():
        if a == arm and o == offset and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def pareto(cells, offset):
    keys = ['end_waypoint_error_m', 'boundary_jerk_0p1s', 'foot_sliding_5cm_10cms', 'jerk_0p1s']
    points = {a: [cells[f'{a}|d{offset:g}'][k] for k in keys] for a in ARMS}
    dominated = {a: [b for b in ARMS if b != a and all(x <= y for x, y in zip(points[b], points[a])) and points[b] != points[a]] for a in ARMS}
    return {'metrics': keys, 'non_dominated': [a for a in ARMS if not dominated[a]], 'dominated_by': dominated}


def analyze():
    table = load_table()
    result = {'runs': len(table), 'expected_runs': SPEC['runs']}
    result['cells'] = {f'{a}|d{o:g}': {m: cell(table, a, o, m) for m in METRICS} for a in ARMS for o in OFFSETS}
    hypotheses = {'P1': test(*paired(table, 'W0X', 'W0', OFFSETS, 'boundary_jerk_0p1s')),
                  'P2': test(*paired(table, 'W0X', 'W0', OFFSETS, 'foot_sliding_5cm_10cms', MARGIN)),
                  'P3': test(*paired(table, 'XFh', 'C0', OFFSETS, 'foot_sliding_5cm_10cms'))}
    hypotheses['P2']['margin'] = MARGIN; hypotheses['P2']['mean_difference'] = hypotheses['P2']['mean']+MARGIN
    adjusted = holm([hypotheses[k]['sign_flip_p'] for k in ('P1', 'P2', 'P3')])
    side = {'P1': lambda ci: ci[1] < 0, 'P2': lambda ci: ci[1] < 0, 'P3': lambda ci: ci[0] > 0}
    for key, p in zip(('P1', 'P2', 'P3'), adjusted):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and side[key](hypotheses[key]['ci95']))
    accurate = {str(o): result['cells'][f'W0X|d{o:g}']['end_waypoint_error_m'] <= EPSILON for o in OFFSETS}
    result['primary'] = {'hypotheses': hypotheses, 'W0X_accurate_per_offset': accurate,
                         'W0X_adopted': bool(hypotheses['P1']['supported'] and hypotheses['P2']['supported'] and all(accurate.values())),
                         'smoothing_foot_cost_confirmed': hypotheses['P3']['supported']}
    secondary = {}
    for on, off in [('W0X', 'C0'), ('W0X', 'XFo'), ('W0X', 'XFh'), ('W0', 'C0'), ('XFo', 'C0')]:
        secondary[f'{on}-{off}'] = {'pooled': {m: test(*paired(table, on, off, OFFSETS, m)) for m in METRICS if m != 'wall_seconds'},
                                    'per_offset': {str(o): {m: test(*paired(table, on, off, [o], m))
                                                            for m in ('boundary_jerk_0p1s', 'jerk_0p1s', 'foot_sliding_5cm_10cms', 'trunk_inclination')}
                                                   for o in OFFSETS}}
    c0 = {m: float(np.mean([result['cells'][f'C0|d{o:g}'][m] for o in OFFSETS])) for m in ('boundary_jerk_0p1s', 'jerk_0p1s', 'foot_sliding_5cm_10cms')}
    secondary['W0X-C0']['relative_to_C0'] = {m: secondary['W0X-C0']['pooled'][m]['mean']/c0[m] for m in c0}
    result['secondary'] = secondary
    result['pareto'] = {str(o): pareto(result['cells'], o) for o in OFFSETS}
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'adopted': result['primary']['W0X_adopted'],
                      'confirmed_foot_cost': result['primary']['smoothing_foot_cost_confirmed'],
                      'hypotheses': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')} for k, v in hypotheses.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    audit() if parser.parse_args().audit else analyze()
