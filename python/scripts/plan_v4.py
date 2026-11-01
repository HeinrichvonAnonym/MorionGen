#!/usr/bin/env python
"""Register the V4 queue plans (config/v4_20261101.json): correction writeback x history length.

pilot: offset task 10, seed 20, every cell; batch: requires a passed pilot audit,
reuses the M1 C0 / C1 runs for history 32 and the pilot successes.
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
from plan_m1 import OUT as M1_OUT, PILOT_SEED, PILOT_TASK, condition as m1_condition
from plan_next_round import LEGACY_ARDY, ardy_job

SPEC_PATH = ROOT/'config/v4_20261101.json'
OUT = ROOT/'develop_logs/test_runs/20261101_v4'
ARMS = {'C0': 'C0', 'W1': 'C1', 'W0': 'C1'}
HISTORY = [32, 160]
OFFSETS = [.1, .5, 1.]


def condition(arm, history, offset):
    row = m1_condition(ARMS[arm], .05, offset)
    row['history_frames'] = history
    if arm == 'W0':
        row['correction_writeback'] = False
    return row


def m1_id(arm, task, seed, offset):
    return f"m1_m1_{ARMS[arm]}_e0.05_task{task}_seed{seed}_d{offset:g}"


def jobs_for(stage, seeds):
    template = copy.deepcopy(json.loads((LEGACY_ARDY/'plan.json').read_text())['jobs'][0])
    jobs = []
    for task in tg.ardy_offset_tasks():
        if stage == 'pilot' and task['id'] != PILOT_TASK:
            continue
        path = {'direction_xz': task['direction_xz'], 'lateral_xz': task['lateral_xz'],
                'speed_before_m_s': 0., 'speed_after_m_s': 0., 'switch_seconds': 5.}
        update = None if task['update_seconds'] is None else (task['update_seconds'], task['aligned_update'])
        for seed in seeds:
            block = []
            for arm in ARMS:
                for history in HISTORY:
                    for offset in OFFSETS:
                        job = ardy_job(template, f"v4_{arm}_h{history}_task{task['id']}_seed{seed}_d{offset:g}", 300+task['id'],
                                       seed, task['aligned_prompt'], update, path, condition(arm, history, offset), 90)
                        job['v4'] = {'arm': arm, 'history_frames': history, 'offset_m': offset, 'task': task['id']}
                        if history == 32 and arm != 'W0':
                            job['v4']['m1_job'] = m1_id(arm, task['id'], seed, offset)
                        block.append(job)
            random.Random(1000*(300+task['id'])+seed).shuffle(block); jobs.extend(block)
    return jobs


def m1_runs():
    found = {}
    for stage in ('pilot', 'batch', 'continuation'):
        path = M1_OUT/f'{stage}_queue/state.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text())['jobs']
        found.update({k: v['run_dir'] for k, v in state.items() if v.get('status') == 'success'})
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'batch'], required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    destination = OUT/f'{args.stage}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    jobs = jobs_for(args.stage, [PILOT_SEED] if args.stage == 'pilot' else spec['seeds'])
    reused = []
    if args.stage == 'batch':
        audit = json.loads((OUT/'pilot_audit.json').read_text())
        if not audit.get('passed'):
            raise ValueError('Batch admission requires a passed pilot audit')
        pilot = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
        m1 = m1_runs()
        keep = []
        for job in jobs:
            source = m1.get(job['v4'].get('m1_job')) or (pilot.get(job['id'], {}).get('run_dir')
                                                        if pilot.get(job['id'], {}).get('status') == 'success' else None)
            if source:
                reused.append({'id': job['id'], 'run': source, 'job': job, 'motion_sha256': checksum(Path(source)/'motion.npz')})
            elif job['v4'].get('m1_job'):
                raise ValueError(f"Missing M1 run {job['v4']['m1_job']}")
            else:
                keep.append(job)
        jobs = keep
    for job in jobs:
        job['environment'] = {'PYTHONNOUSERSITE': '1'}
    identity = json.loads((LEGACY_ARDY/'plan.json').read_text())['identity_files']
    plan = {'devices': [0, 1], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'identity_files': identity+[str(SPEC_PATH)],
            'specification_sha256': checksum(SPEC_PATH), 'registered_at': datetime.now(timezone.utc).isoformat(),
            'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
