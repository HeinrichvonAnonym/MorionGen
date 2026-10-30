#!/usr/bin/env python
"""Admission audit for the 20261030 next-round pilots (criteria in config/next_round_20261030.json)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.native_analysis import facing_heading
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261030_next_round'
BODY_HEADING_DEG = -90.


def heading_error(heading):
    return abs((heading-BODY_HEADING_DEG+180) % 360-180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['m0', 'v1'], required=True)
    parser.add_argument('--amendment', help='registered revision folder suffix, e.g. amend1')
    args = parser.parse_args()
    folder = OUT/(args.block+(f'_{args.amendment}' if args.amendment else ''))
    plan = json.loads((folder/'pilot_plan.json').read_text())
    state = json.loads((folder/'pilot_queue/state.json').read_text())['jobs']
    checks, rows = [], []

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {})
        record = {'id': job['id'], 'status': row.get('status'), 'elapsed_seconds': row.get('elapsed_seconds')}
        rows.append(record)
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason'))
            continue
        run = Path(row['run_dir'])
        motion = load_motion(run/'motion.npz')
        joints = np.asarray(motion['joints'])
        check('finite_motion', np.isfinite(joints).all(), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], job=job['id'],
              elapsed=row['elapsed_seconds'], estimate=job['estimated_seconds'])
        _, heading = facing_heading(joints, motion['joint_names'])
        record['frame0_heading_deg'] = heading
        check('frame0_heading_body_forward', heading_error(heading) <= 20., job=job['id'], heading_deg=heading)
        if 'legacy_run_dir' in job:
            legacy = np.asarray(load_motion(Path(job['legacy_run_dir'])/'motion.npz')['joints'])
            same = legacy.shape == joints.shape
            check('legacy_replay_bit_identical', same and np.array_equal(legacy, joints),
                  job=job['id'], max_abs_diff=float(np.abs(legacy-joints).max()) if same else None)
            continue
        intervention = json.loads((run/'intervention.json').read_text())
        if job['method'] == 'ardy':
            task = intervention['task']; condition = intervention['condition']; windows = task['windows']
            check('ardy_constraint_guidance_observed',
                  intervention['calls'].get('denoiser_with_text_and_spatial_guidance', 0) > 0, job=job['id'])
            if condition.get('waypoints') == 'dense':
                check('dense_mask_covers_horizon', all(len(w['local_frames']) == 8 for w in windows), job=job['id'],
                      frames_per_window=sorted({len(w['local_frames']) for w in windows}))
            if condition.get('target_mode') == 'relative_offset':
                required = [float(np.linalg.norm(np.subtract(w['targets_xz'][0], w['generated_at_targets_xz'][0])))
                            for w in windows]
                ratio = float(np.median(required))/condition['offset_m']
                record['median_required_over_delta'] = ratio
                check('offset_realised_within_20pct', .8 <= ratio <= 1.2, job=job['id'], median_ratio=ratio)
                check('reference_pass_executed',
                      intervention['calls'].get('native_reference_pass', 0) == len(windows), job=job['id'])
        else:
            spec = job['scene']['constraints'][0]; meta = job['scene']['next_round_task']
            frame = int(round(spec['time_seconds']*float(motion['fps'])))
            names = [str(n) for n in motion['joint_names']]; ix = {n: i for i, n in enumerate(names)}
            target = np.asarray(spec['position_m']); wrist = spec['joint']
            other = ('left' if wrist.startswith('right') else 'right')+'_wrist'
            side = wrist.split('_')[0]
            shoulder, elbow = joints[0, ix[side+'_shoulder']], joints[0, ix[side+'_elbow']]
            arm = np.linalg.norm(shoulder-elbow)+np.linalg.norm(elbow-joints[0, ix[wrist]])
            record.update(tier=meta['tier'], wrist_error_m=float(np.linalg.norm(joints[frame, ix[wrist]]-target)),
                          other_wrist_distance_m=float(np.linalg.norm(joints[frame, ix[other]]-target)),
                          realised_shoulder_distance_ratio=float(np.linalg.norm(target-shoulder)/arm),
                          nominal_shoulder_distance_ratio=meta['nominal_shoulder_distance_ratio'])
            if meta['tier'] == 'reachable':
                check('reachable_tier_within_arm', record['realised_shoulder_distance_ratio'] < 1., job=job['id'],
                      ratio=record['realised_shoulder_distance_ratio'])
    passed = all(c['passed'] for c in checks) and len(rows) == len(plan['jobs'])
    criteria = json.loads((ROOT/'config/next_round_20261030.json').read_text())['pilot']['admission']
    json_write(folder/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']],
                                           'checks': len(checks), 'rows': rows, 'criteria': criteria})
    print(json.dumps({'block': args.block, 'passed': passed, 'checks': len(checks),
                      'failed': [c for c in checks if not c['passed']][:10]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
