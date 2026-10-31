#!/usr/bin/env python
"""Register the V3-PRIMAL omega sweep (config/v3_primal_20261031.json) on the 20261024 PRIMAL matrix scenes."""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import checksum, json_write

SPEC_PATH = ROOT/'config/v3_primal_20261031.json'
LEGACY = ROOT/'develop_logs/test_runs/20261024_primal_matrix'
OUT = ROOT/'develop_logs/test_runs/20261031_v3_primal'
OMEGAS = [2.5, 5., 10., 20., 40.]
VARIANTS = [f'omega{w:g}' for w in OMEGAS]+['inertialize0']
PILOT = 'primal_initial0_seed21'


def sweep_job(job, legacy_state):
    job = copy.deepcopy(job)
    command = job['command']
    command[command.index('--ground-ablation')] = '--omega-sweep'
    command.insert(command.index('--omega-sweep')+1, ','.join(f'{w:g}' for w in OMEGAS))
    job['required_outputs'] = ['{run_dir}/factorial/timing.json', '{run_dir}/factorial/initial_history.npz']+[
        f'{{run_dir}}/factorial/{v}_{suffix}' for v in VARIANTS for suffix in ('canonical.npz', 'metrics.json')]
    row = legacy_state.get(job['id'], {})
    job['legacy_matrix_run_dir'] = row['run_dir'] if row.get('status') == 'success' else None
    job['id'] = 'v3p_'+job['id']
    job.update(estimated_seconds=120, timeout_seconds=1800, environment={'PYTHONNOUSERSITE': '1'})
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'batch'], required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    destination = OUT/f'{args.stage}_plan.json'
    if destination.exists():
        raise ValueError('Refusing to replace a registered plan; use a new revision')
    legacy = json.loads((LEGACY/'plan.json').read_text())
    state = json.loads((LEGACY/'queue/state.json').read_text())['jobs']
    jobs = [sweep_job(j, state) for j in legacy['jobs'] if args.stage == 'batch' or j['id'] == PILOT]
    reused = []
    if args.stage == 'batch':
        audit = json.loads((OUT/'pilot_audit.json').read_text())
        if not audit.get('passed'):
            raise ValueError('Batch admission requires a passed pilot audit')
        pilot_state = json.loads((OUT/'pilot_queue/state.json').read_text())['jobs']
        keep = []
        for job in jobs:
            row = pilot_state.get(job['id'])
            if row and row.get('status') == 'success':
                reused.append({'id': job['id'], 'run': row['run_dir']})
            else:
                keep.append(job)
        jobs = keep
    plan = {'devices': [0, 1], 'deadline': spec['deadline'], 'report_reserve_seconds': spec['report_reserve_seconds'],
            'jobs': jobs, 'reused': reused, 'tasks': legacy['tasks'], 'sampling_scope': legacy['sampling_scope'],
            'identity_files': legacy['identity_files']+[str(SPEC_PATH)], 'specification_sha256': checksum(SPEC_PATH),
            'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': spec['protocol'], 'stage': args.stage}
    json_write(destination, plan)
    print(json.dumps({'plan': str(destination), 'jobs': len(jobs), 'reused': len(reused)}))


if __name__ == '__main__':
    main()
