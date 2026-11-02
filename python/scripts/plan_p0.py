#!/usr/bin/env python
"""P0 queue plans (config/p0_20261102.json): native ARDY conditioning with goal lookahead.

smoke: unregistered development runs under 20261102_p0/smoke_<tag>/. Arms E/T/R vary how the
in-window constraint is given (frames carrying it, constraint guidance weight); goal arms place fixed
goals at 4/8/12/16 s exposed as future constraints up to 0/1/2/4 s ahead (the official demo's future
window), optionally with output-only MotionCorrection on the goal window.
pilot: offset task 10, seed 26, every registered cell; audit: pilot admission (pilot_audit.json);
batch: requires a passed pilot audit; continuation: re-queues batch jobs without success.
Registered plans are never replaced.
"""
import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motiogen import task_geometry as tg
from motiogen.runtime import checksum, json_write
from motiogen.config import ROOT
from plan_m1 import condition as m1_condition
from plan_next_round import LEGACY_ARDY, ardy_job

SPEC_PATH = ROOT/'config/p0_20261102.json'
OUT = ROOT/'develop_logs/test_runs/20261102_p0'
ARMS = {'E': ('end', 2.), 'E5': ('end', 10.), 'T': ('end_token', 2.), 'T5': ('end_token', 10.),
        'R': ('ramp', 2.), 'R5': ('ramp', 10.)}
GOAL_ARMS = {'G0': (0., False), 'G1': (1., False), 'G2': (2., False), 'G4': (4., False),
             'W0g': (0., True), 'G4W': (4., True)}
GOAL_SECONDS = [4., 8., 12., 16.]
PILOT_TASK, PILOT_SEED = 10, 26
ESTIMATE_SECONDS = 19.


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
        if task_ids is not None and task['id'] not in task_ids:
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


def audit():
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    plan = json.loads((OUT/'pilot_plan.json').read_text(encoding='utf-8'))
    state = json.loads((OUT/'pilot_queue/state.json').read_text(encoding='utf-8'))['jobs']
    checks, runs = [], {}

    def check(name, ok, detail=None):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})

    for job in plan['jobs']:
        row = state.get(job['id'], {})
        check(f"{job['id']} success", row.get('status') == 'success', row.get('reason'))
        if row.get('status') != 'success':
            continue
        run = Path(row['run_dir']); runs[(job['p0']['arm'], job['p0']['offset_m'])] = run
        check(f"{job['id']} time", row['elapsed_seconds'] <= 3*ESTIMATE_SECONDS, row['elapsed_seconds'])
        motion = np.load(run/'motion.npz')
        check(f"{job['id']} finite", bool(np.isfinite(motion['joints']).all()) and len(motion['joints']) == 400, len(motion['joints']))
        evidence = json.loads((run/'intervention.json').read_text())
        task, calls = evidence['task'], evidence['calls']
        check(f"{job['id']} goals", [g['frame'] for g in task['goals']] == [80, 160, 240, 320], [g['frame'] for g in task['goals']])
        check(f"{job['id']} one reference run", calls.get('native_reference_run') == 1, calls.get('native_reference_run'))
        lookahead = task['lookahead_frames']
        frames_ok = all(w['num_frames'] <= 200
                        and (w['num_frames'] > w['history_frames']+8) == any(g >= w['frame']+8 for g in w['visible_goal_frames'])
                        and all(w['frame'] <= g < w['frame']+8+lookahead for g in w['visible_goal_frames']) for w in task['windows'])
        check(f"{job['id']} window budget and visibility", frames_ok)
        corrected = spec['arms'][job['p0']['arm']]['correction']
        check(f"{job['id']} corrections", calls.get('native_motion_correction', 0) == (4 if corrected else 0), calls.get('native_motion_correction', 0))
    for base, arm in (('G0', 'W0g'), ('G4', 'G4W')):
        for offset in spec['offsets_m']:
            if (base, offset) in runs and (arm, offset) in runs:
                a = np.load(runs[(base, offset)]/'motion.npz')['joints'][:80]
                b = np.load(runs[(arm, offset)]/'motion.npz')['joints'][:80]
                check(f'{base} = {arm} before the first goal window d{offset:g}', float(np.abs(a-b).max()) <= 1e-6, float(np.abs(a-b).max()))
    passed = all(c['passed'] for c in checks) and len(runs) == len(plan['jobs'])
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'checks': checks, 'audited_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'audit', 'batch', 'continuation'], required=True)
    parser.add_argument('--tag', help='smoke folder suffix')
    parser.add_argument('--tasks', default='10', help='smoke tasks')
    parser.add_argument('--seeds', default='20', help='smoke seeds')
    parser.add_argument('--arms', default=','.join(ARMS), help='smoke arms')
    parser.add_argument('--offsets', default='0.5,1', help='smoke offsets')
    parser.add_argument('--name', help='plan name for a later continuation, e.g. continuation2')
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(',')]
    identity = json.loads((LEGACY_ARDY/'plan.json').read_text(encoding='utf-8'))['identity_files']
    if args.stage == 'audit':
        audit()
        return
    if args.stage == 'smoke':
        destination = OUT/f'smoke_{args.tag}'/'smoke_plan.json'
        if destination.exists():
            raise ValueError('Refusing to replace a smoke plan; use a new tag')
        jobs = jobs_for({int(t) for t in args.tasks.split(',')}, [int(s) for s in args.seeds.split(',')],
                        args.arms.split(','), [float(o) for o in args.offsets.split(',')])
        json_write(destination, {'devices': devices, 'deadline': '2026-11-04T12:00:00Z', 'report_reserve_seconds': 0,
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
    if args.stage == 'pilot':
        jobs = jobs_for({PILOT_TASK}, [PILOT_SEED], list(spec['arms']), spec['offsets_m'])
    else:
        jobs = jobs_for(None, spec['seeds'], list(spec['arms']), spec['offsets_m'])
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
        if not json.loads((OUT/'pilot_audit.json').read_text(encoding='utf-8')).get('passed'):
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
    plan = {'devices': devices, 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'identity_files': identity+[str(SPEC_PATH)],
            'specification_sha256': checksum(SPEC_PATH), 'registered_at': datetime.now(timezone.utc).isoformat(),
            'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
