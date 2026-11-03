#!/usr/bin/env python
"""E1/E2 worker: OpenSim evaluation of every successful source run.

E1 (config/e1_biomech_20261101.json): every M2 run. E2 (config/e2_biomech_20261102.json): the P0 runs
of the registered arms. Runs in the OpenSim 4.6 environment. Follows the source queues until they
finish, so it can start while the source batch is still generating. Finished runs are skipped on restart.
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
PROTOCOLS = {
    'e1': {'source': RUNS/'20261101_m2', 'key': 'm2', 'arms': None, 'out': RUNS/'20261101_e1_biomech',
           'data': DISK/'e1_biomech_20261101/runs', 'done_marker': 'analysis.out'},
    'e2': {'source': RUNS/'20261102_p0', 'key': 'p0', 'arms': ('G0', 'G4', 'W0g', 'G4W'), 'out': RUNS/'20261102_e2_biomech',
           'data': DISK/'e2_biomech_20261102/runs', 'done_marker': 'analysis.json'},
}
# Post hoc (20261103): the Rajagopal model clamps pelvis_rotation (global heading) to +-90 degrees. Re-evaluate,
# with the heading unclamped, every run whose IK hit that limit, plus two runs that never did as controls.
for name, base, day in (('e1y', 'e1', '20261103'), ('e2y', 'e2', '20261103')):
    PROTOCOLS[name] = {**PROTOCOLS[base], 'out': RUNS/f'{day}_{name}_biomech', 'data': DISK/f'{name}_biomech_{day}/runs',
                       'free_yaw': True, 'select_from': PROTOCOLS[base]['out']}
CONTROLS = 2


def yaw_selection(evaluated):
    hit, clear = [], []
    for path in sorted((evaluated/'runs').glob('*/markers_posthoc.json')):
        limits = json.loads(path.read_text())['frames_at_coordinate_limit']
        (hit if limits.get('pelvis_rotation', 0) else clear).append(path.parent.name)
    return set(hit+clear[:CONTROLS])
MASS = 75.
TRIM = .5


def source_runs(protocol):
    """Successful source runs by job id, and whether any source queue is still open."""
    source, key, arms = protocol['source'], protocol['key'], protocol['arms']
    found, open_queue = {}, False
    stages = ['pilot', 'batch', *sorted(p.name[:-len('_plan.json')] for p in source.glob('continuation*_plan.json'))]
    for stage in stages:
        plan_path, state_path = source/f'{stage}_plan.json', source/f'{stage}_queue/state.json'
        if not plan_path.exists():
            continue
        plan = json.loads(plan_path.read_text())
        if not state_path.exists():
            open_queue = open_queue or bool(plan['jobs']); continue
        state = json.loads(state_path.read_text())['jobs']
        for item in plan.get('reused', []):
            found[item['id']] = ({**item['job'][key], 'seed': item['job']['seed']}, item['run'])
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') in ('pending', 'running'):
                open_queue = True
            if row.get('status') == 'success':
                found[job['id']] = ({**job[key], 'seed': job['seed']}, row['run_dir'])
    if not (source/protocol['done_marker']).exists():
        open_queue = True
    if arms is not None:
        found = {j: v for j, v in found.items() if v[0]['arm'] in arms}
    if 'select_from' in protocol:
        selected = protocol.setdefault('selected', yaw_selection(protocol['select_from']))
        found = {j: v for j, v in found.items() if j in selected}
    return found, open_queue


def evaluate(job_id, info, run, out, data, key, free_yaw=False):
    import opensim as osim
    import biomech
    osim.Logger.setLevelString('error')
    work = data/job_id; record = out/'runs'/job_id
    record.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        if work.exists():
            shutil.rmtree(work)
        receipt = biomech.prepare(run, work, MASS, free_yaw=free_yaw)
        metrics = biomech.dynamics(work, MASS)
        t1 = (receipt['frames']-1)/receipt['fps']-TRIM
        metrics.update(biomech.static_optimization(work, TRIM, t1))
        for name in ('so/so_so_force.sto', 'so/so_so_controls.xml', 'calibration.trc', 'scaled.osim'):
            (work/name).unlink(missing_ok=True)
        ik = receipt['ik_marker_error']
        valid = {'ik_rms_ok': ik['rms_mean_m'] <= .03, 'ik_max_ok': ik['max_m'] <= .10,
                 'vertical_force_ok': abs(metrics['required_vertical_force_mean_bw']-1) <= .02}
        result = {'id': job_id, key: info, 'run': str(run), 'work': str(work), 'ik': ik, 'scaling': receipt['scaling'],
                  'metrics': metrics, 'validity': valid, 'valid': all(valid.values()), 'seconds': time.time()-started,
                  'finished_at': datetime.now(timezone.utc).isoformat()}
        (record/'metrics.json').write_text(json.dumps(result, indent=1))
        (record/'error.json').unlink(missing_ok=True)
        return job_id, 'success'
    except Exception as exc:
        (record/'error.json').write_text(json.dumps({'id': job_id, 'error': repr(exc), 'traceback': traceback.format_exc(),
                                                     'at': datetime.now(timezone.utc).isoformat()}, indent=1))
        return job_id, 'failed'


def run_pool(args, protocol, returned, counts):
    """One process per run: OpenSim workers grow by about 0.4 GB per run, which exhausted memory after 5 h."""
    submitted = {}; out = protocol['out']
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        while True:
            runs, open_queue = source_runs(protocol)
            for job_id in [j for j, f in submitted.items() if f.done()]:
                counts[submitted.pop(job_id).result()[1]] += 1
                returned.add(job_id)
            todo = sorted((j for j in runs if j not in submitted and j not in returned and not (out/'runs'/j/'metrics.json').exists()),
                          key=lambda j: (runs[j][0]['task'], j))
            for job_id in todo[:max(0, 2*args.workers-len(submitted))]:
                submitted[job_id] = pool.submit(evaluate, job_id, *runs[job_id], out, protocol['data'], protocol['key'],
                                                protocol.get('free_yaw', False))
            (out/'progress.json').write_text(json.dumps({'m2_successful_runs': len(runs), 'm2_open': open_queue,
                                                         'evaluated': sum(1 for _ in (out/'runs').glob('*/metrics.json')),
                                                         'in_flight': len(submitted), 'waiting': len(todo), 'this_session': counts,
                                                         'updated_at': datetime.now(timezone.utc).isoformat()}))
            if not open_queue and not submitted and not todo:
                return
            time.sleep(args.poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--poll', type=float, default=20.)
    parser.add_argument('--protocol', choices=sorted(PROTOCOLS), default='e1')
    args = parser.parse_args()
    protocol = PROTOCOLS[args.protocol]
    protocol['data'].mkdir(parents=True, exist_ok=True); (protocol['out']/'runs').mkdir(parents=True, exist_ok=True)
    returned, counts = set(), {'success': 0, 'failed': 0, 'pool_restarts': 0}
    while True:
        try:
            run_pool(args, protocol, returned, counts)
            return
        except BrokenProcessPool as exc:
            counts['pool_restarts'] += 1
            print(f'{datetime.now(timezone.utc).isoformat()} process pool broken ({exc}); restarting', flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
