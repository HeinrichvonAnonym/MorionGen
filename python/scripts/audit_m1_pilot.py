#!/usr/bin/env python
"""Admission audit for the M1 / M1-Pareto / V3-ARDY pilot (criteria in config/m1_20261031.json)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import ardy_row
from audit_next_round_pilot import heading_error
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261031_m1'
M0C = ROOT/'develop_logs/test_runs/20261030_next_round/m0_amend2/pilot_queue/state.json'


def main():
    plan = json.loads((OUT/'pilot_plan.json').read_text())
    state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
    m0c = json.loads(M0C.read_text())['jobs']
    checks, rows = [], []

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {}); info = job['m1']
        record = {'id': job['id'], 'status': row.get('status'), 'elapsed_seconds': row.get('elapsed_seconds'), **info}
        rows.append(record)
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason'))
            continue
        run = Path(row['run_dir'])
        motion = load_motion(run/'motion.npz'); joints = np.asarray(motion['joints'])
        values = ardy_row(job, run)
        record.update({k: values[k] for k in ('end_waypoint_error_m', 'boundary_jerk_0p1s', 'frame0_heading_deg')})
        check('finite_outputs', np.isfinite(joints).all() and all(np.isfinite(v) for v in values.values()), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], job=job['id'],
              elapsed=row['elapsed_seconds'], estimate=job['estimated_seconds'])
        check('frame0_heading_body_forward', heading_error(values['frame0_heading_deg']) <= 20., job=job['id'],
              heading_deg=values['frame0_heading_deg'])
        intervention = json.loads((run/'intervention.json').read_text())
        condition = intervention['condition']; windows = intervention['task']['windows']; calls = intervention['calls']
        check('constraint_guidance_as_registered',
              intervention.get('observed_denoiser_guidance') == [2., float(condition.get('constraint_cfg', 2.))], job=job['id'],
              observed=intervention.get('observed_denoiser_guidance'))
        check('reference_pass_executed', calls.get('native_reference_pass', 0) == len(windows), job=job['id'])
        required = [float(np.linalg.norm(np.subtract(w['targets_xz'][0], w['generated_at_targets_xz'][0]))) for w in windows]
        ratio = float(np.median(required))/condition['offset_m']
        check('offset_realised_within_20pct', .8 <= ratio <= 1.2, job=job['id'], median_ratio=ratio)
        boundary = condition.get('boundary', 'none')
        if boundary != 'none':
            check('boundary_on_every_later_window', calls.get('native_boundary_'+boundary, 0) == len(windows)-1,
                  job=job['id'], applied=calls.get('native_boundary_'+boundary, 0))
        if condition.get('renoise_fraction'):
            renoise = intervention.get('renoise', {})
            check('renoise_on_every_window', calls.get('native_renoise_pass', 0) == len(windows), job=job['id'])
            check('renoise_token_round_trip', renoise.get('roundtrip_root_max_m', 1.) <= 1e-4, job=job['id'],
                  roundtrip=renoise.get('roundtrip_root_max_m'))
        if info['arm'] in ('C1', 'XFh'):
            check('corrected_end_within_epsilon', values['end_waypoint_error_m'] <= info['epsilon_m']+.005, job=job['id'],
                  end_error=values['end_waypoint_error_m'], epsilon=info['epsilon_m'])
        if info['arm'] in ('C0', 'C1') and info['epsilon_m'] == .05 and info['offset_m'] in (.1, .25, .5, 1.):
            key = f"m0c_task{info['task']}_seed{job['seed']}_d{info['offset_m']:g}_c{1 if info['arm'] == 'C1' else 0}"
            legacy = np.asarray(load_motion(Path(m0c[key]['run_dir'])/'motion.npz')['joints'])
            same = legacy.shape == joints.shape
            check('m0c_replay_bit_identical', same and np.array_equal(legacy, joints), job=job['id'], m0c=key,
                  max_abs_diff=float(np.abs(legacy-joints).max()) if same else None)
    passed = all(c['passed'] for c in checks) and len(rows) == len(plan['jobs'])
    criteria = json.loads((ROOT/'config/m1_20261031.json').read_text())['pilot']['admission']
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']],
                                        'checks': len(checks), 'rows': rows, 'criteria': criteria})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']][:10]},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
