#!/usr/bin/env python
"""P0 development smoke: does native ARDY root conditioning move the window toward the target?

M2 receipts show the C0 window end moving 1-4 mm toward a 0.1-1.0 m offset target relative to the
same-noise unconstrained pass. Arms E/T/R vary how the in-window constraint is given (frames carrying
it, constraint guidance weight); arms G* place fixed goals at 4/8/12/16 s and expose each one as a
future constraint up to 0/1/2/4 s ahead (the official demo's future window). G* need a task without a
prompt update (offset tasks 0-7). Not analyzed as a registered result.
"""
import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motiogen import task_geometry as tg
from motiogen.runtime import json_write
from motiogen.config import ROOT
from plan_m1 import condition as m1_condition
from plan_next_round import LEGACY_ARDY, ardy_job

OUT = ROOT/'develop_logs/test_runs/20261102_p0'
ARMS = {'E': ('end', 2.), 'E5': ('end', 10.), 'T': ('end_token', 2.), 'T5': ('end_token', 10.),
        'R': ('ramp', 2.), 'R5': ('ramp', 10.)}
GOAL_ARMS = {'G0': (0., False), 'G1': (1., False), 'G2': (2., False), 'G4': (4., False),
             'W0g': (0., True), 'G4W': (4., True)}
GOAL_SECONDS = [4., 8., 12., 16.]


def condition(arm, offset):
    if arm in GOAL_ARMS:
        lookahead, corrected = GOAL_ARMS[arm]
        base = m1_condition('C1' if corrected else 'C0', .05, offset)
        if corrected:
            base['correction_writeback'] = False
        return {**base, 'target_mode': 'goal_lookahead', 'goal_seconds': GOAL_SECONDS, 'lookahead_seconds': lookahead}
    frames, weight = ARMS[arm]
    return {**m1_condition('C0', .05, offset), 'constraint_cfg': weight, 'constraint_frames': frames}


def jobs_for(task_ids, seeds, arms, offsets):
    template = copy.deepcopy(json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['jobs'][0])
    jobs = []
    for task in tg.ardy_offset_tasks():
        if task['id'] not in task_ids:
            continue
        path = {'direction_xz': task['direction_xz'], 'lateral_xz': task['lateral_xz'],
                'speed_before_m_s': 0., 'speed_after_m_s': 0., 'switch_seconds': 5.}
        update = None if task['update_seconds'] is None else (task['update_seconds'], task['aligned_update'])
        for seed in seeds:
            block = []
            for arm in arms:
                for offset in offsets:
                    job = ardy_job(template, f"p0_{arm}_task{task['id']}_seed{seed}_d{offset:g}", 500+task['id'],
                                   seed, task['aligned_prompt'], update, path, condition(arm, offset), 90)
                    job['p0'] = {'arm': arm, 'offset_m': offset, 'task': task['id']}
                    job['environment'] = {'PYTHONNOUSERSITE': '1'}
                    block.append(job)
            random.Random(1000*(600+task['id'])+seed).shuffle(block); jobs.extend(block)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--tasks', default='10')
    parser.add_argument('--seeds', default='20')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--offsets', default='0.5,1')
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    folder = OUT/f'smoke_{args.tag}'
    destination = folder/'smoke_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a smoke plan; use a new tag')
    jobs = jobs_for({int(t) for t in args.tasks.split(',')}, [int(s) for s in args.seeds.split(',')],
                    args.arms.split(','), [float(o) for o in args.offsets.split(',')])
    identity = json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['identity_files']
    json_write(destination, {'devices': [int(d) for d in args.devices.split(',')], 'deadline': '2026-11-04T12:00:00Z',
                             'report_reserve_seconds': 0, 'jobs': jobs, 'reused': [], 'identity_files': identity,
                             'stage': 'smoke', 'scope': 'development smoke, not analyzed',
                             'registered_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
