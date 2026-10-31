#!/usr/bin/env python
"""Pilot audit (--audit) and analysis of the V3-PRIMAL omega sweep (config/v3_primal_20261031.json)."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import test
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.native_analysis import boundary_kinematics, common_step_jerk, foot_sensitivity
from motiogen.runtime import json_write
from plan_v3_primal import OMEGAS, OUT, VARIANTS

CONTRACT = json.loads((ROOT/'config/native_metric_contract.json').read_text())
METRICS = ['boundary_jerk_rms', 'boundary_velocity_jump', 'boundary_position_jump', 'jerk_0p1s',
           'foot_sliding_5cm_10cms', 'foot_penetration_fraction', 'wall_seconds']


def variant_values(folder, record):
    motion = load_motion(folder/f"{record['variant']}_canonical.npz"); joints = np.asarray(motion['joints']); fps = float(motion['fps'])
    values = boundary_kinematics(joints, fps, np.arange(14, len(joints), 14))
    values['jerk_0p1s'] = common_step_jerk(joints, fps); values['wall_seconds'] = float(record['seconds'])
    feet = foot_sensitivity(motion, CONTRACT)
    if feet['status'] == 'measured':
        values['foot_penetration_fraction'] = feet['floor_penetration_fraction']
        f = next(r for r in feet['rows'] if r['height_m'] == .05 and r['speed_m_s'] == .1)
        if f['sliding_fraction_given_candidate'] is not None:
            values['foot_sliding_5cm_10cms'] = f['sliding_fraction_given_candidate']
    return joints, values


def audit():
    plan = json.loads((OUT/'pilot_plan.json').read_text()); state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
    checks = []

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {})
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason')); continue
        folder = Path(row['run_dir'])/'factorial'
        records = json.loads((folder/'timing.json').read_text())
        check('six_conditions', sorted(r['variant'] for r in records) == sorted(VARIANTS), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], elapsed=row['elapsed_seconds'])
        for record in records:
            joints, values = variant_values(folder, record)
            check('finite', np.isfinite(joints).all() and all(np.isfinite(v) for v in values.values()), variant=record['variant'])
            calls = record['native_branch_calls'].get('inertialization', 0)
            if record['variant'] == 'inertialize0':
                check('off_has_no_inertialization', calls == 0, calls=calls)
            elif record['inertialization_omega'] != 10.:
                check('swept_replaces_every_call', record['swept_inertialization_calls'] == calls > 0, variant=record['variant'],
                      swept=record['swept_inertialization_calls'], native=calls)
            else:
                legacy = np.asarray(load_motion(Path(job['legacy_matrix_run_dir'])/'factorial/ground1_reproject1_inertialize1_canonical.npz')['joints'])
                same = legacy.shape == joints.shape
                check('omega10_bit_identical_to_matrix', same and np.array_equal(legacy, joints),
                      max_abs_diff=float(np.abs(legacy-joints).max()) if same else None)
    passed = bool(checks) and all(c['passed'] for c in checks)
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']], 'checks': len(checks),
                                        'criteria': json.loads((ROOT/'config/v3_primal_20261031.json').read_text())['pilot']['admission']})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']][:10]}))


def analyze():
    table = {}
    for stage in ('pilot', 'batch'):
        if not (OUT/f'{stage}_queue/state.json').exists():
            continue
        plan = json.loads((OUT/f'{stage}_plan.json').read_text()); state = json.loads((OUT/f'{stage}_queue/state.json').read_text())['jobs']
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') != 'success':
                continue
            folder = Path(row['run_dir'])/'factorial'
            for record in json.loads((folder/'timing.json').read_text()):
                table[(job['task_id'], job['seed'], record['variant'])] = variant_values(folder, record)[1]

    def cell(variant, metric):
        by = collections.defaultdict(list)
        for (task, _, v), values in table.items():
            if v == variant and metric in values:
                by[task].append(values[metric])
        return float(np.mean([np.mean(x) for x in by.values()])) if by else None

    def paired(on, off, metric):
        deltas, clusters = [], []
        for (task, seed, v), values in sorted(table.items()):
            base = table.get((task, seed, off))
            if v == on and base is not None and metric in values and metric in base:
                deltas.append(values[metric]-base[metric]); clusters.append(task)
        return deltas, clusters
    result = {'scenes': len({(t, s) for t, s, _ in table}), 'cells': {v: {m: cell(v, m) for m in METRICS} for v in VARIANTS}}
    result['vs_off'] = {v: {m: test(*paired(v, 'inertialize0', m)) for m in METRICS} for v in VARIANTS if v != 'inertialize0'}
    result['vs_omega10'] = {v: {m: test(*paired(v, 'omega10', m)) for m in METRICS} for v in VARIANTS if v != 'omega10'}
    best_boundary = min(result['cells'][v]['boundary_jerk_rms'] for v in VARIANTS)
    best_jerk = min(result['cells'][v]['jerk_0p1s'] for v in VARIANTS)
    native = result['cells']['omega10']
    result['near_optimal'] = {'boundary_jerk_ratio_to_best': native['boundary_jerk_rms']/best_boundary,
                              'jerk_ratio_to_best': native['jerk_0p1s']/best_jerk,
                              'omega10_near_optimal': bool(native['boundary_jerk_rms'] <= 1.1*best_boundary and native['jerk_0p1s'] <= 1.1*best_jerk),
                              'best_boundary_variant': min(VARIANTS, key=lambda v: result['cells'][v]['boundary_jerk_rms']),
                              'best_jerk_variant': min(VARIANTS, key=lambda v: result['cells'][v]['jerk_0p1s'])}
    ardy_path = ROOT/'develop_logs/test_runs/20261031_m1/analysis.json'
    if ardy_path.exists():
        ardy = json.loads(ardy_path.read_text())['V3']
        readings = {m: v['stop_rule'] for m, v in ardy.items()}
        generic = all(r == 'generic' for r in readings.values()) and result['near_optimal']['omega10_near_optimal']
        result['combined_V3'] = {'ardy': readings, 'primal_omega10_near_optimal': result['near_optimal']['omega10_near_optimal'],
                                 'reading': 'generic' if generic else 'tuned'}
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'scenes': result['scenes'], 'near_optimal': result['near_optimal'], 'combined': result.get('combined_V3')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    audit() if parser.parse_args().audit else analyze()
