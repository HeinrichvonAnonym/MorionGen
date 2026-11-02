#!/usr/bin/env python
"""P1 queue plans (config/p1_20261103.json): sampling-layer guidance on the P0 goal task.

Registered arms come from the specification. Smoke arms take a scale after '@' (e.g. S0@1);
S/U guide x_t through the denoiser, X/V guide the clean prediction:
  S0, X0  P0 G0 plus sampling guidance in the window holding each goal (G0 / W0g information)
  U0, V0  sampling guidance only, no native goal constraint
  S4, X4  P0 G4 plus sampling guidance in the goal window (G4W counterpart)
Jobs share the P0 task, scene seed and seed of the same id suffix, so P1 runs pair with the P0 arms.
smoke: unregistered development runs under 20261103_p1/smoke_<tag>/; pilot: task 10, seed 26;
audit: pilot admission; batch: requires a passed audit; continuation: re-queues batch jobs without success.
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
from plan_next_round import LEGACY_ARDY, ardy_job
from plan_p0 import OUT as P0_OUT, condition as p0_condition

SPEC_PATH = ROOT/'config/p1_20261103.json'
OUT = ROOT/'develop_logs/test_runs/20261103_p1'
BASES = {'S0': ('G0', 'xt', True), 'U0': ('G0', 'xt', False), 'S4': ('G4', 'xt', True),
         'X0': ('G0', 'x0', True), 'V0': ('G0', 'x0', False), 'X4': ('G4', 'x0', True)}
PILOT_TASK, PILOT_SEED = 10, 26
ESTIMATE_SECONDS = 25.


def smoke_arm(arm):
    name, scale = arm.split('@')
    base, space, constraints = BASES[name]
    return {'base': base, 'sampling_guidance': {'space': space, 'scale': float(scale), 'iterations': 1},
            'goal_constraints': constraints}


def condition(arm, offset):
    base = p0_condition(arm['base'], offset)
    base['sampling_guidance'] = dict(arm['sampling_guidance'])
    if not arm.get('goal_constraints', True):
        base['goal_constraints'] = False
    return base


def jobs_for(task_ids, seeds, arms, offsets):
    """arms: {id label: arm definition}."""
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
            for label, arm in arms.items():
                for offset in offsets:
                    job = ardy_job(template, f"p1_{label}_task{task['id']}_seed{seed}_d{offset:g}", 500+task['id'],
                                   seed, task['aligned_prompt'], update, path, condition(arm, offset), 90)
                    job['p1'] = {'arm': label, 'offset_m': offset, 'task': task['id'], 'base': arm['base']}
                    job['environment'] = {'PYTHONNOUSERSITE': '1'}
                    block.append(job)
            random.Random(1000*(700+task['id'])+seed).shuffle(block); jobs.extend(block)
    return jobs


def p0_run(arm, task, seed, offset):
    """Successful P0 run of the same task, seed and offset, from any P0 stage."""
    job_id = f'p0_{arm}_task{task}_seed{seed}_d{offset:g}'
    for name in ['pilot', 'batch', *sorted(p.name[:-len('_plan.json')] for p in P0_OUT.glob('continuation*_plan.json'))]:
        path = P0_OUT/f'{name}_queue/state.json'
        if path.exists():
            row = json.loads(path.read_text(encoding='utf-8'))['jobs'].get(job_id, {})
            if row.get('status') == 'success':
                return Path(row['run_dir'])
    return None


def audit():
    plan = json.loads((OUT/'pilot_plan.json').read_text(encoding='utf-8'))
    state = json.loads((OUT/'pilot_queue/state.json').read_text(encoding='utf-8'))['jobs']
    checks, runs = [], 0

    def check(name, ok, detail=None):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})

    for job in plan['jobs']:
        row = state.get(job['id'], {})
        check(f"{job['id']} success", row.get('status') == 'success', row.get('reason'))
        if row.get('status') != 'success':
            continue
        runs += 1; run = Path(row['run_dir']); info = job['p1']
        check(f"{job['id']} time", row['elapsed_seconds'] <= 3*ESTIMATE_SECONDS, row['elapsed_seconds'])
        joints = np.load(run/'motion.npz')['joints']
        check(f"{job['id']} finite", bool(np.isfinite(joints).all()) and len(joints) == 400, len(joints))
        evidence = json.loads((run/'intervention.json').read_text())
        task, calls = evidence['task'], evidence['calls']
        check(f"{job['id']} goals", [g['frame'] for g in task['goals']] == [80, 160, 240, 320], [g['frame'] for g in task['goals']])
        check(f"{job['id']} one reference run", calls.get('native_reference_run') == 1, calls.get('native_reference_run'))
        guide = task['sampling_guidance']
        check(f"{job['id']} guided windows and steps", guide['guided_windows'] == 4 and guide['guided_steps'] == 40,
              [guide['guided_windows'], guide['guided_steps']])
        base = p0_run(info['base'], info['task'], job['seed'], info['offset_m'])
        check(f"{job['id']} P0 base run", base is not None)
        if base is not None:
            gap = float(np.abs(np.load(base/'motion.npz')['joints'][:80]-joints[:80]).max())
            check(f"{job['id']} = P0 {info['base']} before the first goal window", gap <= 1e-6, gap)
    passed = all(c['passed'] for c in checks) and runs == len(plan['jobs'])
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'checks': checks, 'audited_at': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'failed': [c for c in checks if not c['passed']]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'audit', 'batch', 'continuation'], required=True)
    parser.add_argument('--tag', help='smoke folder suffix')
    parser.add_argument('--tasks', default='10', help='smoke tasks')
    parser.add_argument('--seeds', default='20', help='smoke seeds')
    parser.add_argument('--arms', help='smoke arms, e.g. S0@100,X0@1')
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
                        {a.replace('@', 's'): smoke_arm(a) for a in args.arms.split(',')}, [float(o) for o in args.offsets.split(',')])
        json_write(destination, {'devices': devices, 'deadline': '2026-11-05T12:00:00Z', 'report_reserve_seconds': 0,
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
        jobs = jobs_for({PILOT_TASK}, [PILOT_SEED], spec['arms'], spec['offsets_m'])
    else:
        jobs = jobs_for(None, spec['seeds'], spec['arms'], spec['offsets_m'])
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
