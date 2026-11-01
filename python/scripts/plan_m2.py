#!/usr/bin/env python
"""Register M2 queue plans (config/m2_20261101.json): boundary term inside MotionCorrection.

smoke: unregistered development runs under 20261101_m2/smoke_<tag>/ (offset task 10, seed 20);
pilot: offset task 10, seed 23, every cell; batch: requires a passed pilot audit;
continuation: re-queues batch jobs without success.  Registered plans are never replaced.
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
from motiogen.config import ROOT
from motiogen.runtime import checksum, json_write
from plan_m1 import condition as m1_condition
from plan_next_round import LEGACY_ARDY, ardy_job

SPEC_PATH = ROOT/'config/m2_20261101.json'
OUT = ROOT/'develop_logs/test_runs/20261101_m2'
ARMS = {'C0': ('C0', {}), 'W1': ('C1', {}), 'W0': ('C1', {'correction_writeback': False}),
        'XFh': ('XFh', {}),
        'B2': ('C1', {'correction_writeback': False, 'boundary_prefix_frames': 2}),
        'B4': ('C1', {'correction_writeback': False, 'boundary_prefix_frames': 4}),
        'B8': ('C1', {'correction_writeback': False, 'boundary_prefix_frames': 8}),
        'B4h': ('C1', {'boundary_prefix_frames': 4}),
        'XFo': ('XFo', {}), 'XFhR': ('XFh', {'boundary_repin': True}), 'XFoR': ('XFo', {'boundary_repin': True}),
        'W0X': ('XFo', {'correction_writeback': False}),
        'W0XR': ('XFo', {'correction_writeback': False, 'boundary_repin': True})}
OFFSETS = [.1, .5, 1.]
SMOKE_TASK, SMOKE_SEED = 10, 20
PILOT_TASK, PILOT_SEED = 10, 23


def condition(arm, offset):
    base, extra = ARMS[arm]
    return {**m1_condition(base, .05, offset), **extra}


def jobs_for(stage, seeds, arms=None, offsets=OFFSETS):
    template = copy.deepcopy(json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['jobs'][0])
    jobs = []
    for task in tg.ardy_offset_tasks():
        if stage in ('smoke', 'pilot') and task['id'] != (SMOKE_TASK if stage == 'smoke' else PILOT_TASK):
            continue
        path = {'direction_xz': task['direction_xz'], 'lateral_xz': task['lateral_xz'],
                'speed_before_m_s': 0., 'speed_after_m_s': 0., 'switch_seconds': 5.}
        update = None if task['update_seconds'] is None else (task['update_seconds'], task['aligned_update'])
        for seed in seeds:
            block = []
            for arm in arms or ARMS:
                for offset in offsets:
                    job = ardy_job(template, f"m2_{arm}_task{task['id']}_seed{seed}_d{offset:g}", 500+task['id'],
                                   seed, task['aligned_prompt'], update, path, condition(arm, offset), 90)
                    job['m2'] = {'arm': arm, 'offset_m': offset, 'task': task['id']}
                    job['environment'] = {'PYTHONNOUSERSITE': '1'}
                    block.append(job)
            random.Random(1000*(500+task['id'])+seed).shuffle(block); jobs.extend(block)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'batch', 'continuation'], required=True)
    parser.add_argument('--tag', help='smoke folder suffix')
    parser.add_argument('--only', help='smoke: comma-separated arms')
    parser.add_argument('--offsets', help='smoke: comma-separated offsets')
    parser.add_argument('--name', help='plan name for a later continuation, e.g. continuation2')
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(',')]
    if args.stage == 'smoke':
        folder = OUT/f'smoke_{args.tag}'
        destination = folder/'smoke_plan.json'
        if destination.exists():
            raise ValueError('Refusing to replace a smoke plan; use a new tag')
        offsets = [float(o) for o in args.offsets.split(',')] if args.offsets else [1.]
        jobs = jobs_for('smoke', [SMOKE_SEED], args.only.split(',') if args.only else None, offsets)
        identity = json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['identity_files']
        json_write(destination, {'devices': devices, 'deadline': '2026-11-03T12:00:00Z', 'report_reserve_seconds': 0,
                                 'jobs': jobs, 'reused': [], 'identity_files': identity, 'stage': 'smoke',
                                 'scope': 'development smoke, not analyzed', 'registered_at': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'plan': str(destination), 'jobs': len(jobs)}))
        return
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    name = args.name or args.stage
    if not name.startswith(args.stage):
        raise ValueError('Plan name must start with the stage name')
    destination = OUT/f'{name}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    jobs = jobs_for(args.stage, [PILOT_SEED] if args.stage == 'pilot' else spec['seeds'])
    reused = []
    if args.stage == 'continuation':
        registered = {j['id'] for j in json.loads((OUT/'batch_plan.json').read_text(encoding='utf-8'))['jobs']}
        done = set()
        for path in [OUT/'batch_queue/state.json', *sorted(OUT.glob('continuation*_queue/state.json'))]:
            state = json.loads(path.read_text(encoding='utf-8'))['jobs']
            if any(v.get('status') in ('pending', 'running') for v in state.values()):
                raise ValueError(f'Continuation requires finished queues ({path.parent.name})')
            done |= {k for k, v in state.items() if v.get('status') == 'success'}
        jobs = [j for j in jobs if j['id'] in registered and j['id'] not in done]
    elif args.stage == 'batch':
        audit = json.loads((OUT/'pilot_audit.json').read_text(encoding='utf-8'))
        if not audit.get('passed'):
            raise ValueError('Batch admission requires a passed pilot audit')
        pilot = json.loads((OUT/'pilot_queue/state.json').read_text(encoding='utf-8'))['jobs']
        keep = []
        for job in jobs:
            row = pilot.get(job['id'], {})
            if row.get('status') == 'success':
                reused.append({'id': job['id'], 'run': row['run_dir'], 'job': job,
                               'motion_sha256': checksum(Path(row['run_dir'])/'motion.npz')})
            else:
                keep.append(job)
        jobs = keep
    identity = json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['identity_files']
    plan = {'devices': devices, 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'identity_files': identity+[str(SPEC_PATH)],
            'specification_sha256': checksum(SPEC_PATH), 'registered_at': datetime.now(timezone.utc).isoformat(),
            'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
