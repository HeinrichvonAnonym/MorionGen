#!/usr/bin/env python
"""E1 worker (config/e1_biomech_20261101.json): OpenSim evaluation of every successful M2 run.

Runs in the OpenSim 4.6 environment. Follows the M2 queues until they finish, so it can
start while the M2 batch is still generating. Finished runs are skipped on restart.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
SPEC = ROOT/'config/e1_biomech_20261101.json'
M2 = ROOT/'develop_logs/test_runs/20261101_m2'
OUT = ROOT/'develop_logs/test_runs/20261101_e1_biomech'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e1_biomech_20261101/runs')
MASS = 75.
TRIM = .5


def m2_runs():
    """Successful M2 runs by job id, and whether any M2 queue is still open."""
    found, open_queue = {}, False
    stages = ['pilot', 'batch', *sorted(p.name[:-len('_plan.json')] for p in M2.glob('continuation*_plan.json'))]
    for stage in stages:
        plan_path, state_path = M2/f'{stage}_plan.json', M2/f'{stage}_queue/state.json'
        if not plan_path.exists():
            continue
        if not state_path.exists():
            open_queue = True; continue
        plan = json.loads(plan_path.read_text()); state = json.loads(state_path.read_text())['jobs']
        for item in plan.get('reused', []):
            found[item['id']] = ({**item['job']['m2'], 'seed': item['job']['seed']}, item['run'])
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') in ('pending', 'running'):
                open_queue = True
            if row.get('status') == 'success':
                found[job['id']] = ({**job['m2'], 'seed': job['seed']}, row['run_dir'])
    if not (M2/'analysis.out').exists():
        open_queue = True
    return found, open_queue


def evaluate(job_id, info, run):
    import opensim as osim
    import biomech
    osim.Logger.setLevelString('error')
    work = DATA/job_id; record = OUT/'runs'/job_id
    record.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if work.exists():
            shutil.rmtree(work)
        receipt = biomech.prepare(run, work, MASS)
        metrics = biomech.dynamics(work, MASS)
        t1 = (receipt['frames']-1)/receipt['fps']-TRIM
        metrics.update(biomech.static_optimization(work, TRIM, t1))
        for name in ('so/so_so_force.sto', 'so/so_so_controls.xml', 'calibration.trc', 'scaled.osim'):
            (work/name).unlink(missing_ok=True)
        ik = receipt['ik_marker_error']
        valid = {'ik_rms_ok': ik['rms_mean_m'] <= .03, 'ik_max_ok': ik['max_m'] <= .10,
                 'vertical_force_ok': abs(metrics['required_vertical_force_mean_bw']-1) <= .02}
        result = {'id': job_id, 'm2': info, 'run': str(run), 'work': str(work), 'ik': ik, 'scaling': receipt['scaling'],
                  'metrics': metrics, 'validity': valid, 'valid': all(valid.values()), 'seconds': time.time()-started,
                  'finished_at': datetime.now(timezone.utc).isoformat()}
        (record/'metrics.json').write_text(json.dumps(result, indent=1))
        (record/'error.json').unlink(missing_ok=True)
        return job_id, 'success'
    except Exception as exc:
        (record/'error.json').write_text(json.dumps({'id': job_id, 'error': repr(exc), 'traceback': traceback.format_exc(),
                                                     'at': datetime.now(timezone.utc).isoformat()}, indent=1))
        return job_id, 'failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--poll', type=float, default=120.)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); (OUT/'runs').mkdir(parents=True, exist_ok=True)
    submitted, counts = {}, {'success': 0, 'failed': 0}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        while True:
            runs, open_queue = m2_runs()
            todo = sorted((job_id for job_id in runs if job_id not in submitted and not (OUT/'runs'/job_id/'metrics.json').exists()),
                          key=lambda j: (runs[j][0]['task'], j))
            for job_id in todo:
                submitted[job_id] = pool.submit(evaluate, job_id, *runs[job_id])
            done = [j for j, f in submitted.items() if f is not None and f.done()]
            for j in done:
                counts[submitted.pop(j).result()[1]] += 1
                submitted[j] = None
            pending = sum(1 for f in submitted.values() if f is not None)
            finished = sum(1 for p in (OUT/'runs').glob('*/metrics.json'))
            (OUT/'progress.json').write_text(json.dumps({'m2_successful_runs': len(runs), 'm2_open': open_queue, 'evaluated': finished,
                                                         'in_flight': pending, 'this_session': counts,
                                                         'updated_at': datetime.now(timezone.utc).isoformat()}))
            if not open_queue and pending == 0 and not todo:
                break
            time.sleep(args.poll)


if __name__ == '__main__':
    main()
