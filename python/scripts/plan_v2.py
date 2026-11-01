#!/usr/bin/env python
"""Register the V2 CHOIS guidance decomposition plans (config/v2_chois_20261101.json).

pilot: task 1, seed 21, every arm plus the x 0 development check; batch: requires a
passed pilot audit and reuses the pilot successes of registered arms.
"""
import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT, load_config
from motiogen.runtime import checksum, json_write

SPEC_PATH = ROOT/'config/v2_chois_20261101.json'
LEGACY = ROOT/'develop_logs/test_runs/20261024_native_followups'
OUT = ROOT/'develop_logs/test_runs/20261101_v2_chois'
ARMS = ['N0', 'G', 'HO', 'FF', 'W49', 'S05', 'S2', 'RND']
NATIVE = {'terms': 'both', 'scale': 1., 'last_step': 10, 'direction': 'loss'}
ESTIMATED = {'N0': 60, 'W49': 400}
PILOT_TASK, PILOT_SEED = 1, 21
DEV_CHECK = 'Z0'


def condition(arm):
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))['arms']
    if arm == DEV_CHECK:
        return {'guidance': 1, **NATIVE, 'scale': 0.}
    row = {k: v for k, v in spec[arm].items() if k != 'role'}
    return row if not row['guidance'] else {**NATIVE, **row}


def source_windows():
    processed = Path(load_config()['data_root'])/'outputs/assets/chois/processed_data'
    return processed, processed/'cano_test_diffusion_manip_window_120_joints24.p'


def tasks():
    processed, source = source_windows()
    windows = joblib.load(source)
    # Same text and first-full-window filters as plan_native_followups (native dataset order).
    filtered = [v for v in windows.values() if (processed/'omomo_text_anno_json_data'/(v['seq_name']+'.json')).exists()]
    filtered = [v for v in filtered if len(v['motion']) >= 120 and v['start_t_idx'] == 0]
    by_object = {}
    for index, entry in enumerate(filtered):
        obj = entry['seq_name'].split('_')[1]
        if len(by_object.setdefault(obj, [])) < 2:
            by_object[obj].append({'native_window_index': index, 'sequence': entry['seq_name'], 'object': obj})
    first = [rows[0] for rows in by_object.values()]
    second = [rows[1] for rows in by_object.values()]
    result = first+second
    if len(by_object) != 13 or any(len(rows) != 2 for rows in by_object.values()):
        raise ValueError('Expected two eligible windows for each of 13 objects')
    legacy = json.loads((LEGACY/'plan.json').read_text(encoding='utf-8'))['tasks']
    if [t['native_window_index'] for t in result[:12]] != [t['native_window_index'] for t in legacy]:
        raise ValueError('The first 12 tasks must be the 20261024 matrix windows')
    return [{**t, 'task_id': i} for i, t in enumerate(result)]


def jobs_for(stage, seeds):
    legacy = json.loads((LEGACY/'plan.json').read_text(encoding='utf-8'))
    template = next(j for j in legacy['jobs'] if j['id'].startswith('chois_'))
    jobs = []
    arms = ARMS+([DEV_CHECK] if stage == 'pilot' else [])
    for task in tasks():
        if stage == 'pilot' and task['task_id'] != PILOT_TASK:
            continue
        for seed in seeds:
            block = []
            for arm in arms:
                job = copy.deepcopy(template)
                job.update(id=f"v2_{arm}_task{task['task_id']}_seed{seed}", seed=seed, task_id=400+task['task_id'],
                           estimated_seconds=ESTIMATED.get(arm, 120), timeout_seconds=1800,
                           environment={'PYTHONNOUSERSITE': '1'})
                job['scene'].update(native_window_index=task['native_window_index'], native_sequence=task['sequence'])
                job['config'] = {'_intervention': condition(arm)}
                job['v2'] = {'arm': arm, 'task': task['task_id'], 'object': task['object'],
                             'development_check': arm == DEV_CHECK}
                if task['task_id'] < 12 and arm in ('N0', 'G'):
                    job['v2']['legacy_job'] = f"chois_task{task['task_id']}_seed{seed}_guidance{int(arm == 'G')}"
                block.append(job)
            random.Random(1000*(400+task['task_id'])+seed).shuffle(block); jobs.extend(block)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'batch'], required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    destination = OUT/f'{args.stage}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    jobs = jobs_for(args.stage, [PILOT_SEED] if args.stage == 'pilot' else spec['seeds'])
    reused = []
    if args.stage == 'batch':
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
    legacy = json.loads((LEGACY/'plan.json').read_text(encoding='utf-8'))
    _, source = source_windows()
    plan = {'devices': [0, 1], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'tasks': tasks(),
            'identity_files': sorted(set(legacy['identity_files']+[str(SPEC_PATH), str(source)])),
            'source_windows_sha256': checksum(source), 'specification_sha256': checksum(SPEC_PATH),
            'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
