#!/usr/bin/env python
"""X3 E criterion: the E4 OpenSim evaluation (run_e4_cross_method.evaluate, unchanged) on the X3 E arms at delta 1.0 m.

--plan STAGE writes develop_logs/test_runs/20261104_x3/e/plan.json for the stage's finished generations;
without it, the runner evaluates the plan with a process pool (one process per run) and skips finished runs.
Runs in the OpenSim 4.6 environment; the guard entry is e1_guard.py --protocol x3.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
OUT = ROOT/'develop_logs/test_runs/20261104_x3/e'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/x3_e_20261104/runs')


def e4():
    spec = importlib.util.spec_from_file_location('run_e4_cross_method', ROOT/'python/scripts/run_e4_cross_method.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evaluate(job, out, data):
    return e4().evaluate(job, out, data)


def plan(stage):
    from motiogen import x3_tasks as xt
    reg = xt.prereg()
    jobs, missing = [], []
    for method, arms in reg['metrics']['E_arms'].items():
        for t in xt.job_tasks(method, stage):
            for s in xt.seeds(stage):
                for arm in arms:
                    path = xt.run_dir(method, t['id'], s, arm, None if arm == 'C0' else 1.0)
                    job = {'id': f'{method}_t{t["id"]:02d}_s{s}_{arm}', 'method': method, 'task': t['id'], 'seed': s,
                           'arm': arm, 'offset_m': None if arm == 'C0' else 1.0, 'source': str(path)}
                    (jobs if (path/'receipt.json').exists() else missing).append(job)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((OUT/'plan.json').read_text())['jobs'] if (OUT/'plan.json').exists() else []
    known = {j['id'] for j in jobs}
    jobs = jobs+[j for j in previous if j['id'] not in known]
    (OUT/'plan.json').write_text(json.dumps({'stage': stage, 'jobs': jobs, 'missing_generation': missing,
                                             'written_at': datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f'{len(jobs)} jobs, {len(missing)} generations missing')


def run_pool(args, jobs, returned, counts):
    submitted = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        while True:
            for job_id in [j for j, f in submitted.items() if f.done()]:
                counts[submitted.pop(job_id).result()[1]] += 1
                returned.add(job_id)
            todo = [j for j in jobs if j['id'] not in submitted and j['id'] not in returned and not (OUT/'runs'/j['id']/'metrics.json').exists()]
            for job in todo[:max(0, 2*args.workers-len(submitted))]:
                submitted[job['id']] = pool.submit(evaluate, job, OUT, DATA)
            (OUT/'progress.json').write_text(json.dumps({'jobs': len(jobs), 'evaluated': sum(1 for _ in (OUT/'runs').glob('*/metrics.json')),
                                                         'in_flight': len(submitted), 'waiting': len(todo), 'this_session': counts,
                                                         'updated_at': datetime.now(timezone.utc).isoformat()}))
            if not submitted and not todo:
                return
            time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--poll', type=float, default=20.)
    parser.add_argument('--plan', choices=['pilot', 'confirm', 'top_up'])
    args = parser.parse_args()
    if args.plan:
        return plan(args.plan)
    jobs = json.loads((OUT/'plan.json').read_text())['jobs']
    DATA.mkdir(parents=True, exist_ok=True); (OUT/'runs').mkdir(parents=True, exist_ok=True)
    returned, counts = set(), {'success': 0, 'failed': 0, 'pool_restarts': 0}
    while True:
        try:
            run_pool(args, jobs, returned, counts)
            return
        except BrokenProcessPool as exc:
            counts['pool_restarts'] += 1
            print(f'{datetime.now(timezone.utc).isoformat()} process pool broken ({exc}); restarting', flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
