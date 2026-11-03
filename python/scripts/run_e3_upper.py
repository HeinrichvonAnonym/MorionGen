#!/usr/bin/env python
"""E3 worker: upper-body ergonomic evaluation of every run already evaluated by E1 and E2.

Preregistration: config/e3_upper_body_20261103.json. Runs in the OpenSim 4.6 environment. One task per run and
region: 'neck' (generated-skeleton posture plus the HYOID head-neck model) and 'lumbar' (FBLS full-body model with
the E1 contact pipeline). Finished tasks are skipped on restart.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
RUNS = ROOT/'develop_logs/test_runs'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs')
OUT = RUNS/'20261103_e3_upper'
DATA = DISK/'e3_upper_20261103/runs'
SOURCES = {'e1': (RUNS/'20261101_e1_biomech', 'm2'), 'e2': (RUNS/'20261102_e2_biomech', 'p0')}
PARTS = ('neck', 'lumbar')
PRIMARY_ARMS = {'e1': ('W0X', 'C0'), 'e2': ('W0g', 'G4')}
MASS = 75.


def priority(experiment, info):
    """Runs of the primary contrasts first, so that a partial batch already answers the registered hypotheses."""
    return 0 if info['arm'] in PRIMARY_ARMS[experiment] else 1


def source_runs():
    """Run id -> (experiment, design cell, generated run directory) for every successful E1/E2 evaluation."""
    found = {}
    for experiment, (source, key) in SOURCES.items():
        for path in sorted((source/'runs').glob('*/metrics.json')):
            record = json.loads(path.read_text())
            found[record['id']] = (experiment, record[key], record['run'])
    return found


def evaluate(job_id, part, experiment, info, run, step):
    import opensim as osim
    import upper_body
    osim.Logger.setLevelString('error')
    work = DATA/job_id/part; record = OUT/'runs'/job_id
    record.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if work.exists():
            shutil.rmtree(work)
        result = {'id': job_id, 'part': part, 'experiment': experiment, 'cell': info, 'run': str(run), 'work': str(work)}
        if part == 'neck':
            result['posture'] = upper_body.posture_metrics(run)
            receipt, metrics = upper_body.neck(run, work, MASS, step)
        else:
            receipt, metrics = upper_body.lumbar(run, work, MASS, step)
        result.update(receipt=receipt, metrics=metrics, validity=upper_body.validity(part, receipt, metrics))
        result.update(valid=all(result['validity'].values()), seconds=time.time()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        (record/f'{part}.json').write_text(json.dumps(result, indent=1))
        (record/f'{part}_error.json').unlink(missing_ok=True)
        return job_id, part, 'success'
    except Exception as exc:
        (record/f'{part}_error.json').write_text(json.dumps({'id': job_id, 'part': part, 'error': repr(exc),
                                                             'traceback': traceback.format_exc(),
                                                             'at': datetime.now(timezone.utc).isoformat()}, indent=1))
        return job_id, part, 'failed'


def run_pool(args, returned, counts):
    """One process per task: OpenSim workers grow in memory with every run."""
    submitted = {}
    runs = source_runs()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        while True:
            for task in [t for t, f in submitted.items() if f.done()]:
                counts[submitted.pop(task).result()[2]] += 1
                returned.add(task)
            todo = sorted(((j, p) for j in runs for p in args.parts
                           if (j, p) not in submitted and (j, p) not in returned and not (OUT/'runs'/j/f'{p}.json').exists()),
                          key=lambda t: (priority(*runs[t[0]][:2]), runs[t[0]][1]['task'], t[0], t[1]))
            for task in todo[:max(0, 2*args.workers-len(submitted))]:
                submitted[task] = pool.submit(evaluate, *task, *runs[task[0]], args.neck_step if task[1] == 'neck' else args.lumbar_step)
            (OUT/'progress.json').write_text(json.dumps({
                'runs': len(runs), 'tasks': len(runs)*len(args.parts),
                'evaluated': {p: sum(1 for _ in (OUT/'runs').glob(f'*/{p}.json')) for p in args.parts},
                'in_flight': len(submitted), 'waiting': len(todo), 'this_session': counts,
                'updated_at': datetime.now(timezone.utc).isoformat()}))
            if not submitted and not todo:
                return
            time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--poll', type=float, default=20.)
    parser.add_argument('--neck-step', type=int, default=2)
    parser.add_argument('--lumbar-step', type=int, default=10)
    parser.add_argument('--parts', nargs='+', choices=PARTS, default=list(PARTS))
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); (OUT/'runs').mkdir(parents=True, exist_ok=True)
    returned, counts = set(), {'success': 0, 'failed': 0, 'pool_restarts': 0}
    while True:
        try:
            run_pool(args, returned, counts)
            return
        except BrokenProcessPool as exc:
            counts['pool_restarts'] += 1
            print(f'{datetime.now(timezone.utc).isoformat()} process pool broken ({exc}); restarting', flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
