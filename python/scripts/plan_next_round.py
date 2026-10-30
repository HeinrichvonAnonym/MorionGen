#!/usr/bin/env python
"""Register 20261030 next-round queue plans (M0: ARDY protocol, V1: reach geometry).

Pilot plans cover every condition on a few tasks; batch plans require a passed
pilot audit and reuse its successful runs. Existing plans are never replaced.
"""
import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen import task_geometry as tg
from motiogen.config import ROOT, environment_root, load_config
from motiogen.runtime import checksum, json_write

SPEC_PATH = ROOT/'config/next_round_20261030.json'
OUT = ROOT/'develop_logs/test_runs/20261030_next_round'
LEGACY_ARDY = ROOT/'develop_logs/test_runs/20261024_ardy_matrix'
LEGACY_REACH = ROOT/'develop_logs/test_runs/20261024_control_matrix/plan.json'
REVIEWED_PAIR = {'A person walks forward.', 'A person walks backward.'}
LEGACY_REPLAYS = ['ardy_task1_seed21_paused_condition0', 'ardy_task1_seed21_paused_condition1']
PILOT = {'m0a': [2], 'm0c': [10], 'v1': None}
AMENDMENTS = {'amend1': {'path': ROOT/'config/next_round_20261030_amend1.json', 'blocks': ['v1'],
                         'body': tg.MEASURED_BODY_20261031},
              'amend2': {'path': ROOT/'config/next_round_20261030_amend2.json', 'blocks': ['m0'], 'body': None}}


def ardy_job(template, job_id, task_id, seed, prompt, update, path, condition, estimated):
    job = copy.deepcopy(template)
    scene = job['scene']
    scene['prompt'] = prompt
    scene['updates'] = [] if update is None else [{'time_seconds': update[0], 'prompt': update[1]}]
    scene['native_root_path'] = path
    scene['clock'] = 'paused'
    job['required_outputs'] = [p for p in job['required_outputs'] if not p.endswith('/clock_trace.json')]
    config = job['config']
    config.pop('_frozen_prompt_cache', None)
    prompts = {prompt, *([update[1]] if update else [])}
    config['_frozen_prompt_cache' if prompts == REVIEWED_PAIR else '_registered_prompt_cache'] = True
    config['_intervention'] = condition
    job.update(id=job_id, seed=seed, task_id=task_id, estimated_seconds=estimated, timeout_seconds=1800)
    return job


def m0_jobs(stage):
    plan = json.loads((LEGACY_ARDY/'plan.json').read_text())
    template = copy.deepcopy(plan['jobs'][0])
    jobs, tasks = [], {'m0a': tg.ardy_path_tasks(), 'm0c': tg.ardy_offset_tasks()}
    for task in tasks['m0a']:
        if stage == 'pilot' and task['id'] not in PILOT['m0a']:
            continue
        path = {k: task[k] for k in ['direction_xz', 'speed_before_m_s', 'speed_after_m_s', 'switch_seconds']}
        for seed in ([20] if stage == 'pilot' else tg_seeds()):
            block = []
            for text, waypoints in [('conflict', 'sparse'), ('aligned', 'sparse'), ('aligned', 'dense')]:
                for correction in (0, 1):
                    condition = {'history_frames': 32, 'correction': correction, 'text': text,
                                 'waypoints': waypoints, 'target_mode': 'path'}
                    block.append(ardy_job(template, f"m0a_task{task['id']}_seed{seed}_{text}_{waypoints}_c{correction}",
                        task['id'], seed, task[text+'_prompt'], (5., task[text+'_update']), path, condition, 60))
            random.Random(1000*task['id']+seed).shuffle(block); jobs.extend(block)
    for task in tasks['m0c']:
        if stage == 'pilot' and task['id'] not in PILOT['m0c']:
            continue
        path = {'direction_xz': task['direction_xz'], 'lateral_xz': task['lateral_xz'],
                'speed_before_m_s': 0., 'speed_after_m_s': 0., 'switch_seconds': 5.}
        update = None if task['update_seconds'] is None else (task['update_seconds'], task['aligned_update'])
        for seed in ([20] if stage == 'pilot' else tg_seeds()):
            block = []
            for offset in [.1, .25, .5, 1.]:
                for correction in (0, 1):
                    condition = {'history_frames': 32, 'correction': correction, 'text': 'aligned',
                                 'waypoints': 'sparse', 'target_mode': 'relative_offset', 'offset_m': offset}
                    block.append(ardy_job(template, f"m0c_task{task['id']}_seed{seed}_d{offset:g}_c{correction}",
                        100+task['id'], seed, task['aligned_prompt'], update, path, condition, 90))
            random.Random(1000*(100+task['id'])+seed).shuffle(block); jobs.extend(block)
    if stage == 'pilot':
        state = json.loads((LEGACY_ARDY/'queue/state.json').read_text())['jobs']
        for key in LEGACY_REPLAYS:
            job = copy.deepcopy(next(j for j in plan['jobs'] if j['id'] == key))
            job['legacy_run_dir'] = state[key]['run_dir']
            job['id'] = 'm0_legacy_replay_'+key
            jobs.append(job)
    tables = {'m0a': tasks['m0a'], 'm0c': tasks['m0c']}
    return jobs, tables, plan['identity_files']


def tg_seeds():
    return json.loads(SPEC_PATH.read_text())['seeds']


def v1_jobs(stage, body=None):
    cfg = load_config()
    legacy = json.loads(LEGACY_REACH.read_text())
    base = json.loads((ROOT/'config/scenarios/reach.json').read_text())
    tasks = tg.reach_tasks(body)
    pilot_ids = [t['id'] for t in tasks if t['side'] == 'right' and t['height'] == 'mid' and t['tier'] in ('reachable', 'legacy', 'mirror')]
    conditions = [('motionlcm', {'control_residual': v}) for v in (0, 1)]
    conditions += [('omnicontrol', {'realism': r, 'spatial': s}) for r in (0, 1) for s in (0, 1)]
    jobs = []
    for block_name in ('v1', 'v1b'):
        for task in tasks:
            if stage == 'pilot' and (block_name == 'v1b' or task['id'] not in pilot_ids):
                continue
            prompt = task['prompt'] if block_name == 'v1' else tg.reach_direction_prompt(task)
            if block_name == 'v1b' and (task['tier'] != 'reachable' or prompt is None):
                continue
            scene = copy.deepcopy(base)
            scene['prompt'] = prompt
            scene['constraints'][0].update(joint=task['joint'], position_m=task['position_m'])
            scene['next_round_task'] = {k: task[k] for k in ['id', 'tier', 'height', 'side', 'target_body_m', 'nominal_shoulder_distance_ratio']}
            for seed in ([20] if stage == 'pilot' else tg_seeds()):
                block = []
                for method, condition in conditions:
                    block.append({'id': f"{block_name}_{method}_task{task['id']}_seed{seed}_"+''.join(str(v) for v in condition.values()),
                        'method': method, 'scene': scene, 'seed': seed, 'task_id': task['id'], 'config': {'_intervention': condition},
                        'command': [str(environment_root(cfg, method)/'bin/python'), str(ROOT/'python/scripts/baseline_worker.py'), '--run-dir', '{run_dir}'],
                        'required_outputs': ['{run_dir}/motion.npz', '{run_dir}/metrics.json', '{run_dir}/intervention.json'],
                        'estimated_seconds': 150 if method == 'omnicontrol' else 60, 'timeout_seconds': 900})
                random.Random(1000*task['id']+seed+(7 if block_name == 'v1b' else 0)).shuffle(block); jobs.extend(block)
    return jobs, {'reach': tasks}, legacy['identity_files']


def environment_difference(reference, run_dir):
    path = Path(run_dir)/'environment_packages.json'
    if not path.exists():
        return {'environment_packages.json': 'missing'}
    packages = json.loads(path.read_text())['packages']
    return {n: [reference.get(n), packages.get(n)] for n in sorted(set(reference) | set(packages)) if reference.get(n) != packages.get(n)}


def continuation(folder, spec):
    """Remaining batch work after an interrupted or environment-contaminated batch segment.

    Kept: batch successes whose package set equals the pilot run of the same method.
    Rerun: everything else (pending, failed, running-at-stop, or package drift), with the
    user site directory disabled so only the method environment is importable.
    """
    import psutil
    batch = json.loads((folder/'batch_plan.json').read_text())
    state = json.loads((folder/'batch_queue/state.json').read_text())
    try:
        alive = abs(psutil.Process(state['pid']).create_time()-state['process_create_time']) < 2
    except psutil.NoSuchProcess:
        alive = False
    if alive:
        raise ValueError('Batch queue is still running')
    pilot_plan = {j['id']: j for j in json.loads((folder/'pilot_plan.json').read_text())['jobs']}
    reference = {}
    for job_id, row in json.loads((folder/'pilot_queue/state.json').read_text())['jobs'].items():
        if row.get('status') == 'success' and 'legacy_run_dir' not in pilot_plan[job_id]:
            reference.setdefault(pilot_plan[job_id]['method'], json.loads((Path(row['run_dir'])/'environment_packages.json').read_text())['packages'])
    jobs, superseded = [], []
    for job in batch['jobs']:
        row = state['jobs'][job['id']]
        drift = environment_difference(reference[job['method']], row['run_dir']) if row.get('run_dir') else {}
        if row['status'] == 'success' and not drift:
            continue
        if row['status'] != 'pending':
            superseded.append({'id': job['id'], 'batch_status': row['status'], 'run_dir': row.get('run_dir'),
                               'reason': row.get('reason'), 'package_drift_vs_pilot': drift})
        job = copy.deepcopy(job)
        job['environment'] = {**job.get('environment', {}), 'PYTHONNOUSERSITE': '1'}
        jobs.append(job)
    return {'devices': batch['devices'], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': batch['reused'], 'superseded': superseded, 'continues': str(folder/'batch_plan.json'),
            'tasks': batch['tasks'], 'identity_files': batch['identity_files'], 'specification_sha256': checksum(SPEC_PATH),
            'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': spec['protocol'], 'block': batch['block'],
            'stage': 'continuation', 'isolation': 'PYTHONNOUSERSITE=1: the user site directory is not importable by workers',
            **({k: batch[k] for k in ('amendment', 'amendment_sha256') if k in batch})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['m0', 'v1'], required=True)
    parser.add_argument('--stage', choices=['pilot', 'batch', 'continuation'], required=True)
    parser.add_argument('--amendment', choices=sorted(AMENDMENTS), help='registered revision; writes to <block>_<amendment>/')
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    folder = OUT/(args.block+(f'_{args.amendment}' if args.amendment else ''))
    destination = folder/f'{args.stage}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    spec_files = [SPEC_PATH]
    if args.amendment:
        amendment = AMENDMENTS[args.amendment]
        if args.block not in amendment['blocks']:
            raise ValueError(f'{args.amendment} does not amend {args.block}')
        spec_files.append(amendment['path'])
    if args.stage == 'continuation':
        plan = continuation(folder, spec)
        json_write(destination, plan)
        print(json.dumps({'plan': str(destination), 'jobs': len(plan['jobs']), 'superseded': len(plan['superseded'])}))
        return
    if args.amendment:
        jobs, tables, identity = m0_jobs(args.stage) if args.block == 'm0' else v1_jobs(args.stage, amendment['body'])
    else:
        jobs, tables, identity = (m0_jobs if args.block == 'm0' else v1_jobs)(args.stage)
    reused = []
    if args.stage == 'batch':
        audit = json.loads((folder/'pilot_audit.json').read_text())
        if not audit.get('passed'):
            raise ValueError('Batch admission requires a passed pilot audit')
        state = json.loads((folder/'pilot_queue/state.json').read_text())['jobs']
        keep = []
        for job in jobs:
            row = state.get(job['id'])
            if row and row.get('status') == 'success':
                reused.append({'id': job['id'], 'run': row['run_dir'], 'motion_sha256': checksum(Path(row['run_dir'])/'motion.npz')})
            else:
                keep.append(job)
        jobs = keep
    plan = {'devices': [0, 1], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'tasks': tables, 'identity_files': identity + [str(p) for p in spec_files],
            'specification_sha256': checksum(SPEC_PATH), 'registered_at': datetime.now(timezone.utc).isoformat(),
            'protocol': spec['protocol'], 'block': args.block, 'stage': args.stage}
    if args.amendment:
        plan.update(amendment=args.amendment, amendment_sha256=checksum(spec_files[-1]))
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
