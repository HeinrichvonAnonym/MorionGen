#!/usr/bin/env python
"""E4 worker: OpenSim evaluation of the experiment-1 outputs of all eight methods (config/e4_cross_method_20261103.json).

Runs in the OpenSim 4.6 environment. Jobs come from develop_logs/test_runs/20261103_e4_cross_method/plan.json
(python/scripts/plan_e4.py). --smoke evaluates only the first (task, seed) of each method. Finished runs are
skipped on restart. One process per run, as in run_e1_biomech.py.
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
OUT = ROOT/'develop_logs/test_runs/20261103_e4_cross_method'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e4_cross_method_20261103/runs')
MASS = 75.
TRIM = .5


def contact_validation(job, work):
    """CLoSD only: estimated vertical ground reaction force against the simulator contact forces."""
    import numpy as np
    with np.load(Path(job['source'])/'motion.npz') as m:
        sim = np.asarray(m['contact_forces'], float)[..., 2].sum(1); fps = float(m['fps'])
    d = np.load(work/'dynamics.npz')
    times = d['times']; est = d['grf_r'][:, 1]+d['grf_l'][:, 1]
    keep = (times >= TRIM) & (times <= times[-1]-TRIM)
    sim_t = np.arange(len(sim))/fps
    sim_i = np.interp(times[keep], sim_t, sim)
    e, s = est[keep], sim_i
    return {'pearson_r_vertical': float(np.corrcoef(e/e.mean(), s/s.mean())[0, 1]),
            'simulator_mean_vertical_n': float(s.mean()), 'simulator_flight_fraction': float(np.mean(s < .05*s.mean())),
            'estimate_flight_fraction': float(np.mean(e < .05*e.mean()))}


def evaluate(job, out, data):
    import opensim as osim
    import biomech
    import biomech_sources
    biomech.load = biomech_sources.load
    osim.Logger.setLevelString('error')
    work = data/job['id']; record = out/'runs'/job['id']
    record.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if work.exists():
            shutil.rmtree(work)
        receipt = biomech.prepare(job['source'], work, MASS, free_yaw=True)
        metrics = biomech.dynamics(work, MASS)
        t1 = (receipt['frames']-1)/receipt['fps']-TRIM
        metrics.update(biomech.static_optimization(work, TRIM, t1))
        for name in ('so/so_so_force.sto', 'so/so_so_controls.xml', 'calibration.trc', 'scaled.osim'):
            (work/name).unlink(missing_ok=True)
        ik = receipt['ik_marker_error']
        valid = {'ik_rms_ok': ik['rms_mean_m'] <= .03, 'ik_max_ok': ik['max_m'] <= .10,
                 'vertical_force_ok': abs(metrics['required_vertical_force_mean_bw']-1) <= .02}
        result = {**job, 'work': str(work), 'frames': receipt['frames'], 'fps': receipt['fps'], 'ik': ik, 'scaling': receipt['scaling'],
                  'metrics': metrics, 'validity': valid, 'valid': all(valid.values()), 'seconds': time.time()-started,
                  'finished_at': datetime.now(timezone.utc).isoformat()}
        if job['method'] == 'closd':
            result['contact_validation'] = contact_validation(job, work)
        (record/'metrics.json').write_text(json.dumps(result, indent=1))
        (record/'error.json').unlink(missing_ok=True)
        return job['id'], 'success'
    except Exception as exc:
        (record/'error.json').write_text(json.dumps({'id': job['id'], 'error': repr(exc), 'traceback': traceback.format_exc(),
                                                     'at': datetime.now(timezone.utc).isoformat()}, indent=1))
        return job['id'], 'failed'


def select(jobs, smoke):
    if not smoke:
        return jobs
    first = {}
    for j in jobs:
        first.setdefault(j['method'], (j['task'], j['seed']))
    return [j for j in jobs if (j['task'], j['seed']) == first[j['method']]]


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
                                                         'in_flight': len(submitted), 'waiting': len(todo),
                                                         'this_session': counts, 'smoke': args.smoke,
                                                         'updated_at': datetime.now(timezone.utc).isoformat()}))
            if not submitted and not todo:
                return
            time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--poll', type=float, default=20.)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    jobs = select(json.loads((OUT/'plan.json').read_text())['jobs'], args.smoke)
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
