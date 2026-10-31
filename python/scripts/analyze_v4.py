#!/usr/bin/env python
"""Pilot audit (--audit) and analysis of V4: correction writeback x history length (config/v4_20261101.json)."""
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
from motiogen.runtime import json_write
from plan_v4 import HISTORY, OFFSETS, OUT, m1_runs

METRICS = ['end_waypoint_error_m', 'boundary_jerk_0p1s', 'jerk_0p1s', 'boundary_jerk_native', 'trunk_inclination',
           'right_upper_arm_elevation', 'foot_sliding_5cm_10cms', 'history_consistency_32_m', 'applied_root_displacement_m', 'wall_seconds']


def audit():
    plan = json.loads((OUT/'pilot_plan.json').read_text()); state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
    m1 = m1_runs(); checks = []; consistency = {}

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {}); info = job['v4']
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason')); continue
        run = Path(row['run_dir']); joints = np.asarray(load_motion(run/'motion.npz')['joints'])
        values = ardy_row(job, run)
        check('finite_outputs', np.isfinite(joints).all() and all(np.isfinite(v) for v in values.values()), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], job=job['id'], elapsed=row['elapsed_seconds'])
        check('frame0_heading_body_forward', heading_error(values['frame0_heading_deg']) <= 20., job=job['id'], heading=values['frame0_heading_deg'])
        intervention = json.loads((run/'intervention.json').read_text())
        check('constraint_guidance_as_registered', intervention.get('observed_denoiser_guidance') == [2., 2.], job=job['id'])
        check('reference_pass_executed', intervention['calls'].get('native_reference_pass', 0) == len(intervention['task']['windows']), job=job['id'])
        check('writeback_receipt', intervention['task']['correction_writeback'] == (info['arm'] != 'W0'), job=job['id'])
        if info['arm'] != 'C0':
            check('corrected_end_within_margin', values['end_waypoint_error_m'] <= .055, job=job['id'], end_error=values['end_waypoint_error_m'])
            consistency[(info['arm'], info['history_frames'], info['offset_m'])] = values['history_consistency_32_m']
        if info.get('m1_job'):
            legacy = np.asarray(load_motion(Path(m1[info['m1_job']])/'motion.npz')['joints'])
            same = legacy.shape == joints.shape
            check('m1_replay_bit_identical', same and np.array_equal(legacy, joints), job=job['id'], m1=info['m1_job'],
                  max_abs_diff=float(np.abs(legacy-joints).max()) if same else None)
    for h in HISTORY:
        for o in OFFSETS:
            a, b = consistency.get(('W0', h, o)), consistency.get(('W1', h, o))
            check('output_only_history_differs', a is not None and b is not None and a != b, history=h, offset=o, w0=a, w1=b)
    passed = bool(checks) and all(c['passed'] for c in checks)
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']], 'checks': len(checks),
                                        'criteria': json.loads((ROOT/'config/v4_20261101.json').read_text())['pilot']['admission']})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']][:10]}))


def load_table():
    found = {}
    for stage in ('pilot', 'batch', 'continuation'):
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
        info = job['v4']
        table[(info['arm'], info['history_frames'], info['offset_m'], info['task'], job['seed'])] = values
    return table


def paired(table, on, off, history, offsets, metric):
    deltas, clusters = [], []
    for (a, h, o, task, seed), values in sorted(table.items()):
        base = table.get((off, h, o, task, seed))
        if a == on and h == history and o in offsets and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]); clusters.append(task)
    return deltas, clusters


def interaction(table, metric):
    deltas, clusters = [], []
    for (a, h, o, task, seed), values in sorted(table.items()):
        if a != 'W0' or h != 160:
            continue
        rows = [table.get(k) for k in [('W1', 160, o, task, seed), ('W0', 32, o, task, seed), ('W1', 32, o, task, seed)]]
        if all(r is not None and metric in r for r in rows) and metric in values:
            deltas.append((values[metric]-rows[0][metric])-(rows[1][metric]-rows[2][metric])); clusters.append(task)
    return deltas, clusters


def cell(table, arm, history, offset, metric):
    by = collections.defaultdict(list)
    for (a, h, o, task, _), values in table.items():
        if a == arm and h == history and o == offset and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def analyze():
    table = load_table()
    result = {'runs': len(table), 'expected_runs': 1296}
    result['cells'] = {f'{a}|h{h}|d{o:g}': {m: cell(table, a, h, o, m) for m in METRICS}
                       for a in ('C0', 'W1', 'W0') for h in HISTORY for o in OFFSETS}
    primary = test(*paired(table, 'W0', 'W1', 32, [.1, .5], 'boundary_jerk_0p1s'))
    base = np.mean([cell(table, 'W1', 32, o, 'boundary_jerk_0p1s') for o in (.1, .5)])
    primary.update(w1_mean=float(base), relative=primary['mean']/float(base) if primary['mean'] is not None else None)
    if primary.get('excludes_zero'):
        primary['decision_for_M2'] = 'writeback' if primary['mean'] > 0 else 'output_only'
    else:
        primary['decision_for_M2'] = 'both'
    result['primary'] = primary
    result['contrasts'] = {f'{on}-{off}|h{h}|d{o:g}': {m: test(*paired(table, on, off, h, [o], m)) for m in METRICS if m != 'wall_seconds'}
                           for on, off in [('W0', 'W1'), ('W1', 'C0'), ('W0', 'C0')] for h in HISTORY for o in OFFSETS}
    result['history_interaction'] = {m: test(*interaction(table, m)) for m in ('boundary_jerk_0p1s', 'history_consistency_32_m', 'foot_sliding_5cm_10cms', 'jerk_0p1s')}
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'primary': {k: primary.get(k) for k in ('mean', 'ci95', 'sign_flip_p', 'relative', 'decision_for_M2')}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    audit() if parser.parse_args().audit else analyze()
