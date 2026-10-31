#!/usr/bin/env python
"""Register M1 (equal-accuracy arms), M1-Pareto and V3-ARDY queue plans (config/m1_20261031.json).

smoke: unregistered development run under 20261031_m1/smoke_<tag>/;
pilot: offset task 10, seed 20, every condition; batch: requires a passed pilot audit.
Registered plans are never replaced.
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
from plan_next_round import LEGACY_ARDY, ardy_job

SPEC_PATH = ROOT/'config/m1_20261031.json'
OUT = ROOT/'develop_logs/test_runs/20261031_m1'
PILOT_TASK, PILOT_SEED = 10, 20
ARMS = {'C0': {'correction': 0}, 'C1': {'correction': 1},
        'C2': {'correction': 1, 'boundary': 'inertialization', 'boundary_writeback': True},
        'C3a': {'correction': 0, 'constraint_cfg': 4.}, 'C3b': {'correction': 0, 'constraint_cfg': 8.},
        'C5a': {'correction': 1, 'renoise_fraction': .2}, 'C5b': {'correction': 1, 'renoise_fraction': .5},
        'C2o': {'correction': 1, 'boundary': 'inertialization', 'boundary_writeback': False},
        'XFh': {'correction': 1, 'boundary': 'crossfade', 'boundary_writeback': True},
        'XFo': {'correction': 1, 'boundary': 'crossfade', 'boundary_writeback': False},
        'LPh': {'correction': 1, 'boundary': 'lowpass', 'boundary_writeback': True},
        'LPo': {'correction': 1, 'boundary': 'lowpass', 'boundary_writeback': False},
        'C5d': {'correction': 1, 'renoise_fraction': .1}}
SMOKE_GRIDS = [('dev', ['C5d'], [.05], [1.])]
GRIDS = [('m1', ['C0', 'C1', 'C2', 'C3a', 'C3b', 'C5a', 'C5b', 'XFh'], [.05], [.05, .1, .25, .5, .75, 1.]),
         ('par', ['C1', 'C2', 'C5a', 'XFh'], [.02, .1], [.1, .5, 1.]),
         ('v3', ['C2o', 'XFo', 'LPh', 'LPo'], [.05], [.25, 1.])]


def condition(arm, epsilon, offset):
    spec = ARMS[arm]
    row = {'history_frames': 32, 'correction': spec['correction'], 'text': 'aligned', 'waypoints': 'sparse',
           'target_mode': 'relative_offset', 'offset_m': offset}
    if spec['correction']:
        row['root_margin_m'] = round(epsilon-.01, 6)
    for key in ('constraint_cfg', 'boundary', 'boundary_writeback', 'renoise_fraction'):
        if key in spec:
            row[key] = spec[key]
    return row


def jobs_for(stage, seeds):
    template = copy.deepcopy(json.loads((LEGACY_ARDY/'plan.json').read_text())['jobs'][0])
    tasks = tg.ardy_offset_tasks()
    jobs = []
    for task in tasks:
        if stage in ('pilot', 'smoke') and task['id'] != PILOT_TASK:
            continue
        path = {'direction_xz': task['direction_xz'], 'lateral_xz': task['lateral_xz'],
                'speed_before_m_s': 0., 'speed_after_m_s': 0., 'switch_seconds': 5.}
        update = None if task['update_seconds'] is None else (task['update_seconds'], task['aligned_update'])
        for seed in seeds:
            block = []
            for name, arms, epsilons, offsets in GRIDS+(SMOKE_GRIDS if stage == 'smoke' else []):
                for arm in arms:
                    for epsilon in epsilons:
                        for offset in offsets:
                            if ARMS[arm]['correction'] == 0 and epsilon != .05:
                                continue
                            job = ardy_job(template, f"m1_{name}_{arm}_e{epsilon:g}_task{task['id']}_seed{seed}_d{offset:g}",
                                           200+task['id'], seed, task['aligned_prompt'], update, path,
                                           condition(arm, epsilon, offset), 90)
                            job['m1'] = {'block': name, 'arm': arm, 'epsilon_m': epsilon, 'offset_m': offset, 'task': task['id']}
                            block.append(job)
            random.Random(1000*(200+task['id'])+seed).shuffle(block); jobs.extend(block)
    return jobs, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'batch'], required=True)
    parser.add_argument('--tag', help='smoke folder suffix')
    parser.add_argument('--only', help='smoke: comma-separated arms')
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    folder = OUT/(f'smoke_{args.tag}' if args.stage == 'smoke' else '')
    destination = folder/f'{args.stage}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    seeds = [PILOT_SEED] if args.stage in ('pilot', 'smoke') else spec['seeds']
    jobs, tasks = jobs_for(args.stage, seeds)
    if args.stage == 'smoke':
        keep = set(args.only.split(',')) if args.only else set(ARMS)
        first = {}
        for job in sorted(jobs, key=lambda j: (j['m1']['epsilon_m'] != .05, abs(j['m1']['offset_m']-1.))):
            if job['m1']['arm'] in keep:
                first.setdefault(job['m1']['arm'], job)
        jobs = list(first.values())
    reused = []
    if args.stage == 'batch':
        audit = json.loads((OUT/'pilot_audit.json').read_text())
        if not audit.get('passed'):
            raise ValueError('Batch admission requires a passed pilot audit')
        state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
        keep = []
        for job in jobs:
            row = state.get(job['id'])
            if row and row.get('status') == 'success':
                reused.append({'id': job['id'], 'run': row['run_dir'], 'motion_sha256': checksum(Path(row['run_dir'])/'motion.npz')})
            else:
                keep.append(job)
        jobs = keep
    for job in jobs:
        job['environment'] = {'PYTHONNOUSERSITE': '1'}
    identity = json.loads((LEGACY_ARDY/'plan.json').read_text())['identity_files']
    plan = {'devices': [0, 1], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'tasks': {'m1': tasks}, 'identity_files': identity+[str(SPEC_PATH)],
            'specification_sha256': checksum(SPEC_PATH), 'registered_at': datetime.now(timezone.utc).isoformat(),
            'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
